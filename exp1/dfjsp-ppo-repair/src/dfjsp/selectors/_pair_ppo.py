"""Inti PPO pasangan: kebijakan yang memilih pasangan (tugas, pengembang) langsung.

Dipindah dari ``agents/external/pair_ppo_v2.py``. Perilaku sama; yang berubah hanya nama,
gerbang pelatihan lama dibuang, dan fitur pasangan kini 8 angka pasangan ditambah 9 angka
keadaan.

Satu keputusan: semua pasangan kandidat diberi skor oleh jaringan ``scorer``, lalu satu
pasangan dipilih menurut softmax skor. Jaringan ``critic`` menilai keadaan untuk GAE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.due_date import e_nominal
from ..env.schedule import developer_load, partition
from ..env.state import STATE_DIM
from ..repair.actions import MAINTAIN, REASSIGN
from ..repair.masks import low_level_mask
from ._nn import MLP, Adam
from ._pair_loss import (
    EPS, clip_gradient_norm, compute_gae, log_softmax, minibatch_grads,
)

PAIR_FEATURE_DIM = 8                       # 4 angka tugas + 4 angka pengembang
PAIR_INPUT_DIM = STATE_DIM + PAIR_FEATURE_DIM   # keadaan | tugas | pengembang


@dataclass(frozen=True)
class PairHyper:
    """Hiperparameter PPO pasangan (nilai bawaan sama dengan PPOSelector)."""

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    learning_rate: float = 3e-4
    n_epochs: int = 10
    n_steps: int = 1024        # langkah yang dikumpulkan sebelum satu update
    batch_size: int = 256      # keputusan per minibatch
    ent_coef: float = 0.03
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


@dataclass
class DecisionSample:
    """Satu keputusan beserta semua yang diperlukan untuk menghitung ulang rasio."""

    state: np.ndarray
    pair_feats: np.ndarray
    chosen: int
    old_logp: float
    old_value: float
    reward: float
    done: bool


# ----------------------------------------------------------------- kandidat dan fitur
def candidate_pairs(env, use_mask: bool) -> list[tuple[int, int]]:
    """Pasangan (tugas, pengembang) yang dapat dipilih, urut id tugas lalu id pengembang.

    Dengan ``use_mask`` hanya pasangan yang lolos mask tingkat bawah. Tanpa mask semua
    tugas yang belum mulai dipasangkan dengan semua pengembang; pasangan yang tidak
    layak ditolak operator saat dijalankan.
    """
    inst, sched, t = env.inst, env.sched, env.t_now
    if use_mask:
        feasible = low_level_mask(inst, sched, t)
        return [(i, j) for i in sorted(feasible) for j in sorted(feasible[i])]
    _, _, not_started = partition(inst, sched, t)
    devs = sorted(inst.developers, key=lambda d: d.dev_id)
    return [(i, d.dev_id) for i in sorted(not_started) for d in devs]


def pair_feature(state, task, dev, load: dict, horizon: float) -> np.ndarray:
    """Fitur satu pasangan: 9 angka keadaan, 4 angka tugas, 4 angka pengembang."""
    effort = task.eligible_devs.get(dev.dev_id, e_nominal(task))
    h = max(1e-8, horizon)
    cap = max(1e-8, dev.capacity)
    task_part = [task.priority, task.due_date / h, effort / h, task.progress]
    dev_part = [load.get(dev.dev_id, 0.0) / cap, dev.capacity / cap,
                1.0 if dev.available else 0.0,
                len(dev.skills) / max(1, len(dev.skills) + 1)]
    return np.concatenate([np.asarray(state, float), task_part, dev_part])


def pair_features(env, pairs, state) -> np.ndarray:
    """Matriks fitur ``(jumlah pasangan, 17)`` untuk ``pairs`` pada keadaan sekarang."""
    inst = env.inst
    load = developer_load(inst, env.sched, env.t_now)
    tasks = {t.task_id: t for t in inst.tasks}
    devs = {d.dev_id: d for d in inst.developers}
    return np.array([pair_feature(state, tasks[i], devs[j], load, inst.horizon)
                     for i, j in pairs], dtype=float).reshape(len(pairs), PAIR_INPUT_DIM)


# ------------------------------------------------------------------------ kebijakan
class PairPolicy:
    """Jaringan dan algoritma PPO untuk pemilih pasangan.

    Parameters
    ----------
    use_mask : bool
        Kandidat hanya pasangan yang lolos mask tingkat bawah.
    hidden : int
        Lebar lapisan tersembunyi ``scorer`` dan ``critic``.
    hyper : PairHyper
    seed : int
        Seed bobot awal dan pengambilan sampel. ``critic`` memakai ``seed + 1``.
    """

    def __init__(self, use_mask: bool, hidden: int, hyper: PairHyper, seed: int):
        self.use_mask = bool(use_mask)
        self.hidden = int(hidden)
        self.hyper = hyper
        self.scorer = MLP([PAIR_INPUT_DIM, hidden, 1], seed=seed)
        self.critic = MLP([STATE_DIM, hidden, 1], seed=seed + 1)
        self.opt_scorer = Adam(self.scorer.params, lr=hyper.learning_rate)
        self.opt_critic = Adam(self.critic.params, lr=hyper.learning_rate)
        self.rng = np.random.default_rng(seed)
        self.pending: tuple[int, int] | None = None
        self.buffer: list[DecisionSample] = []
        self.n_env_interactions = 0
        self.n_optimizer_updates = 0
        self.rollout_batch_sizes: list[int] = []
        self._last: dict | None = None

    # ------------------------------------------------------------------ keputusan
    def choose(self, state, env, *, sample: bool) -> int:
        """Pilih pasangan, simpan di ``pending``, dan kembalikan strategi untuk ``env.step``.

        Mengembalikan Reassign bila ada pasangan, Maintain bila tidak ada kandidat.
        Dengan ``sample`` pasangan diambil menurut peluang (untuk mengumpulkan data
        latih); tanpa itu diambil yang peluangnya terbesar.
        """
        self.pending = None
        self._last = None
        state = np.asarray(state, float)
        pairs = candidate_pairs(env, self.use_mask)
        value, _ = self.critic.forward(state[None, :])
        if not pairs:
            # Maintain paksa tetap satu transisi: critic dan GAE harus melihat
            # reward dan tanda akhirnya.
            self._last = {"state": state.copy(), "pair_feats": np.empty((0, PAIR_INPUT_DIM)),
                          "chosen": -1, "old_logp": 0.0, "old_value": float(value[0, 0])}
            return MAINTAIN
        features = pair_features(env, pairs, state)
        scores, _ = self.scorer.forward(features)
        logp = log_softmax(scores[:, 0])
        if sample:
            p = np.exp(logp)
            k = int(self.rng.choice(len(pairs), p=p / p.sum()))
        else:
            k = int(np.argmax(logp))
        self.pending = pairs[k]
        self._last = {"state": state.copy(), "pair_feats": features.copy(), "chosen": k,
                      "old_logp": float(logp[k]), "old_value": float(value[0, 0])}
        return REASSIGN

    def record(self, reward: float, done: bool) -> None:
        """Simpan hasil satu keputusan. Dipanggil sesudah ``env.step``."""
        if self._last is None:
            return
        self.buffer.append(DecisionSample(reward=float(reward), done=bool(done), **self._last))
        self.n_env_interactions += 1
        self._last = None

    def value(self, state) -> float:
        """Nilai keadaan dari critic, tanpa memakai RNG."""
        value, _ = self.critic.forward(np.asarray(state, float)[None, :])
        return float(value[0, 0])

    # ----------------------------------------------------------------- pelatihan
    def train(self, new_episode, *, max_env_steps: int, on_step=None, on_update=None) -> None:
        """Latih PPO dengan kumpulan data lintas episode, berhenti TEPAT di anggaran.

        Parameters
        ----------
        new_episode : callable
            ``new_episode() -> (env, state)``, episode baru yang punya titik keputusan.
        max_env_steps : int
            Anggaran langkah lingkungan.
        on_step : callable, optional
            ``on_step(reward, episode_ended)`` sesudah tiap langkah lingkungan.
        on_update : callable, optional
            ``on_update(stats)`` sesudah tiap update bobot.
        """
        budget = int(max_env_steps)
        if budget < 0:
            raise ValueError("max_env_steps harus >= 0")
        collected = 0
        env = state = None
        need_reset = True
        while collected < budget:
            while len(self.buffer) < self.hyper.n_steps and collected < budget:
                if need_reset:
                    env, state = new_episode()
                    need_reset = False
                action = self.choose(state, env, sample=True)
                next_state, reward, done, info = env.step(action)
                collected += 1
                done = bool(done)
                truncated = (not done) and bool((info or {}).get("truncated", False))
                train_reward = float(reward)
                if truncated:
                    train_reward += self.hyper.gamma * self.value(next_state)
                self.record(train_reward, done or truncated)
                if on_step is not None:
                    on_step(float(reward), done or truncated)
                state = next_state
                need_reset = done or truncated
            if not self.buffer:
                break
            last_value = 0.0 if need_reset else self.value(state)
            stats = self.update(last_value)
            if on_update is not None:
                on_update(stats)
        self.pending = None
        self._last = None

    def update(self, last_value: float = 0.0) -> dict:
        """Satu update PPO atas seluruh isi buffer, lalu buffer dikosongkan."""
        if not self.buffer:
            return {"n_samples": 0}
        h = self.hyper
        adv, ret = compute_gae(
            [s.reward for s in self.buffer], [s.old_value for s in self.buffer],
            [s.done for s in self.buffer], last_value, h.gamma, h.gae_lambda)
        samples = list(self.buffer)
        self.buffer = []
        n = len(samples)
        self.rollout_batch_sizes.append(int(n))
        order = np.arange(n)
        policy_loss = value_loss = entropy_total = 0.0
        n_minibatches = 0
        for _ in range(h.n_epochs):
            self.rng.shuffle(order)
            for start in range(0, n, h.batch_size):
                part = order[start:start + h.batch_size]
                a = adv[part]
                if len(part) > 1 and float(np.std(a)) > EPS:
                    a = (a - a.mean()) / (a.std() + EPS)
                pk, pv, ent = self._minibatch_step([samples[i] for i in part], a, ret[part])
                policy_loss += pk
                value_loss += pv
                entropy_total += ent
                n_minibatches += 1
        return {"n_samples": n, "policy_loss": policy_loss, "value_loss": value_loss,
                "entropy": entropy_total, "entropy_mean": entropy_total / max(1, n_minibatches)}

    def _minibatch_step(self, samples, advantages, returns):
        """Satu langkah optimizer atas satu minibatch keputusan."""
        grad_scorer, grad_critic, policy_loss, value_loss, entropy = minibatch_grads(
            self.scorer, self.critic, samples, advantages, returns, self.hyper)
        clip_gradient_norm(grad_scorer, grad_critic, self.hyper.max_grad_norm)
        self.opt_scorer.step(self.scorer.params, grad_scorer)
        self.opt_critic.step(self.critic.params, grad_critic)
        self.n_optimizer_updates += 1
        return policy_loss, value_loss, entropy

    # ------------------------------------------------------------------ bobot
    def weights(self) -> dict[str, np.ndarray]:
        """Semua bobot dengan awalan ``scorer.`` dan ``critic.``."""
        out = {f"scorer.{k}": v for k, v in self.scorer.params.items()}
        out.update({f"critic.{k}": v for k, v in self.critic.params.items()})
        return out

    def load_weights(self, weights: dict[str, np.ndarray]) -> None:
        """Isi bobot dari ``weights``; bentuk harus sama persis."""
        for prefix, net in (("scorer", self.scorer), ("critic", self.critic)):
            for name, current in net.params.items():
                new = np.asarray(weights[f"{prefix}.{name}"], dtype=float)
                if new.shape != current.shape:
                    raise ValueError(
                        f"bentuk bobot {prefix}.{name} {new.shape} berbeda dari {current.shape}")
                current[...] = new

    def param_count(self) -> int:
        """Jumlah semua bobot scorer dan critic."""
        return int(self.scorer.param_count() + self.critic.param_count())
