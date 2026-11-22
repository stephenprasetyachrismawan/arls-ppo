"""Rugi dan gradien PPO untuk pemilih pasangan (GAE, rugi terpotong, gradien minibatch).

Satu keputusan memilih satu pasangan (tugas, pengembang) dari ``K`` kandidat. Skor tiap
kandidat dari jaringan ``scorer``; peluang pilihan adalah softmax atas skor itu. Critic
menilai keadaan. ``samples`` adalah daftar ``DecisionSample`` (lihat ``_pair_ppo``).
"""

from __future__ import annotations

import numpy as np

EPS = 1e-8
# Batas baris pasangan per satu forward/backward. Rugi adalah jumlah per keputusan, jadi
# memotong minibatch per kelompok keputusan lalu menjumlahkan gradien memberi hasil yang
# sama; yang dibatasi hanya memori aktivasi (sekitar 32.768 x (17 + 2 x 64) x 8 byte).
# Tanpa batas, varian tanpa mask pada MK10 (hingga 3.600 pasangan per keputusan) dapat
# membuat aktivasi satu minibatch lebih dari 1 GB.
PAIR_CHUNK_ROWS = 32768


def compute_gae(rewards, values, dones, last_value: float, gamma: float, lam: float):
    """Advantage GAE dan target critic untuk satu kumpulan data.

    Returns
    -------
    advantages, returns : array, array
        Advantage dan target critic. GAE tidak menyambung melewati akhir episode.

    Examples
    --------
    >>> from dfjsp.selectors._pair_loss import compute_gae
    >>> adv, ret = compute_gae([1.0], [0.5], [True], 0.0, 0.99, 0.95)
    >>> float(adv[0]), float(ret[0])
    (0.5, 1.0)
    """
    rewards = np.asarray(rewards, dtype=float)
    values = np.asarray(values, dtype=float)
    dones = np.asarray(dones, dtype=bool)
    n = len(rewards)
    adv = np.zeros(n, dtype=float)
    running = 0.0
    for t in reversed(range(n)):
        if dones[t]:
            next_value, continues = 0.0, 0.0
        else:
            next_value = values[t + 1] if t + 1 < n else last_value
            continues = 1.0
        delta = rewards[t] + gamma * next_value - values[t]
        running = delta + gamma * lam * continues * running
        adv[t] = running
    return adv, adv + values


def log_softmax(scores: np.ndarray) -> np.ndarray:
    """Log peluang softmax dari ``scores``."""
    z = scores - scores.max()
    return z - np.log(np.exp(z).sum())


def clipped_policy_loss(new_logp: float, old_logp: float, advantage: float,
                        clip_range: float) -> tuple[float, float]:
    """Rugi kebijakan PPO terpotong untuk satu keputusan.

    Returns
    -------
    loss, d_loss_d_logp : float, float
        Rugi dan turunannya terhadap ``new_logp``. Turunannya nol ketika suku
        terpotong yang terpilih, karena suku itu tidak bergantung pada rasio.
    """
    ratio = float(np.exp(new_logp - old_logp))
    clipped_ratio = float(np.clip(ratio, 1.0 - clip_range, 1.0 + clip_range))
    unclipped = ratio * advantage
    clipped = clipped_ratio * advantage
    if unclipped <= clipped:
        return -unclipped, -ratio * advantage
    return -clipped, 0.0


def entropy_and_grad(logp: np.ndarray) -> tuple[float, np.ndarray]:
    """Entropi distribusi pasangan dan turunannya terhadap skor tiap pasangan."""
    p = np.exp(logp)
    entropy = float(-(p * logp).sum())
    return entropy, -p * (logp + entropy)


def minibatch_grads_reference(scorer, critic, samples, advantages, returns, hyper):
    """ACUAN lambat: gradien satu minibatch, satu keputusan demi satu keputusan.

    Dipakai tes untuk membuktikan ``minibatch_grads`` (versi batch) memberi hasil sama.

    Returns
    -------
    grad_scorer, grad_critic, policy_loss, value_loss, entropy : dict, dict, float, float, float
    """
    grad_scorer = {k: np.zeros_like(v) for k, v in scorer.params.items()}
    grad_critic = {k: np.zeros_like(v) for k, v in critic.params.items()}
    policy_loss = value_loss = entropy_total = 0.0
    m = max(1, len(samples))
    for sample, adv, ret in zip(samples, advantages, returns):
        if sample.chosen >= 0:
            scores, cache = scorer.forward(sample.pair_feats)
            logp = log_softmax(scores[:, 0])
            loss, d_logp = clipped_policy_loss(
                float(logp[sample.chosen]), sample.old_logp, float(adv), hyper.clip_range)
            entropy, d_entropy = entropy_and_grad(logp)
            policy_loss += loss / m
            entropy_total += entropy / m

            p = np.exp(logp)
            d_scores = np.zeros_like(logp)
            d_scores[sample.chosen] += d_logp
            d_scores -= d_logp * p                       # turunan log-softmax
            d_scores += hyper.ent_coef * (-d_entropy)    # meminimalkan -entropi
            g = scorer.backward(cache, (d_scores / m)[:, None])
            for k in grad_scorer:
                grad_scorer[k] += g[k]

        value, value_cache = critic.forward(sample.state[None, :])
        diff = float(value[0, 0]) - float(ret)
        value_loss += 0.5 * diff * diff / m
        g_value = critic.backward(value_cache, np.array([[hyper.vf_coef * diff / m]]))
        for k in grad_critic:
            grad_critic[k] += g_value[k]
    return grad_scorer, grad_critic, policy_loss, value_loss, entropy_total


def _actor_groups(samples, actors, max_rows):
    """Bagi keputusan beraktor menjadi kelompok berurutan dengan <= max_rows pasangan."""
    groups, current, rows = [], [], 0
    for i in actors:
        k = samples[i].pair_feats.shape[0]
        if current and rows + k > max_rows:
            groups.append(current)
            current, rows = [], 0
        current.append(i)
        rows += k
    if current:
        groups.append(current)
    return groups


def minibatch_grads(scorer, critic, samples, advantages, returns, hyper, *,
                    max_rows: int | None = None):
    """Gradien satu minibatch dengan forward/backward batch per jaringan.

    Returns
    -------
    grad_scorer, grad_critic, policy_loss, value_loss, entropy : dict, dict, float, float, float
    """
    grad_scorer = {k: np.zeros_like(v) for k, v in scorer.params.items()}
    m = max(1, len(samples))
    actors = [i for i, s in enumerate(samples) if s.chosen >= 0]
    policy_loss = entropy_total = 0.0
    all_adv = np.asarray(advantages, dtype=float)
    for group in _actor_groups(samples, actors, max_rows or PAIR_CHUNK_ROWS):
        g, loss, entropy = _actor_grad(scorer, samples, group, all_adv[group], m, hyper)
        for k in grad_scorer:
            grad_scorer[k] += g[k]
        policy_loss += loss
        entropy_total += entropy
    states = np.stack([np.asarray(s.state, dtype=float) for s in samples])
    values, value_cache = critic.forward(states)
    diff = values[:, 0] - np.asarray(returns, dtype=float)
    value_loss = float(0.5 * (diff * diff).sum() / m)
    grad_critic = critic.backward(value_cache, (hyper.vf_coef * diff / m)[:, None])
    return grad_scorer, grad_critic, policy_loss, value_loss, entropy_total


def _actor_grad(scorer, samples, group, adv, m, hyper):
    """Gradien scorer, rugi, dan entropi satu kelompok keputusan beraktor (dibagi ``m``)."""
    sizes = np.array([samples[i].pair_feats.shape[0] for i in group], dtype=np.int64)
    if (sizes <= 0).any():
        raise ValueError("keputusan terpilih tanpa kandidat pasangan")
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1])).astype(np.int64)
    segment = np.repeat(np.arange(len(group)), sizes)
    chosen = np.array([samples[i].chosen for i in group], dtype=np.int64)
    if (chosen >= sizes).any():
        raise ValueError("indeks pasangan terpilih di luar daftar kandidat")
    features = np.concatenate([samples[i].pair_feats for i in group], axis=0)
    scores, cache = scorer.forward(features)
    s = scores[:, 0]
    z = s - np.maximum.reduceat(s, starts)[segment]
    logp = z - np.log(np.add.reduceat(np.exp(z), starts))[segment]
    p = np.exp(logp)
    pick = starts + chosen
    old = np.array([samples[i].old_logp for i in group], dtype=float)
    ratio = np.exp(logp[pick] - old)
    unclipped = ratio * adv
    clipped = np.clip(ratio, 1.0 - hyper.clip_range, 1.0 + hyper.clip_range) * adv
    use_unclipped = unclipped <= clipped
    loss = -np.where(use_unclipped, unclipped, clipped)
    d_logp = np.where(use_unclipped, -ratio * adv, 0.0)
    entropy = -np.add.reduceat(p * logp, starts)
    d_entropy = -p * (logp + entropy[segment])
    d_scores = -d_logp[segment] * p
    d_scores[pick] += d_logp
    d_scores += hyper.ent_coef * (-d_entropy)
    grad = scorer.backward(cache, (d_scores / m)[:, None])
    return grad, float(loss.sum() / m), float(entropy.sum() / m)


def clip_gradient_norm(*groups_and_limit) -> float:
    """Potong norma gabungan semua gradien ke batas. Argumen terakhir adalah batasnya.

    Returns
    -------
    float
        Norma sebelum dipotong.
    """
    *groups, limit = groups_and_limit
    total = np.sqrt(sum(float((g * g).sum()) for grp in groups for g in grp.values()))
    if limit and total > limit:
        scale = limit / (total + EPS)
        for grp in groups:
            for k in grp:
                grp[k] *= scale
    return total
