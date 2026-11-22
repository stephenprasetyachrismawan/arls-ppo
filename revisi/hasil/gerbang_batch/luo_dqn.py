"""E7: pembanding DRL dari literatur, double DQN yang disesuaikan dari Luo (2020).

Luo, S. (2020). Dynamic scheduling for flexible job shop with new job insertions by deep
reinforcement learning. Applied Soft Computing, 91, 106208.

Yang diikuti dari makalah:
- tujuh fitur keadaan di [0, 1]: U_ave, U_std, CRO_ave, CRJ_ave, CRJ_std, Tard_e, Tard_a
  (Persamaan 6-10, Algoritma 2 dan 3);
- reward bertingkat ±1 dari Tard_a, lalu Tard_e, lalu U_ave (Algoritma 10);
- jaringan 7 -> 5 lapisan × 30 neuron tanh -> 6, keluaran linear (Bagian 5.4);
- double DQN dengan soft target update τ = 0,01, γ = 0,9, buffer 1000, batch 32, ε turun
  linear dari 0,5 ke 0,1 (Tabel 3, Algoritma 11).

Yang disesuaikan (wajib disebut di naskah):
- aksi: enam aturan pustaka ``RuleRepair`` (FIFO, SPT, LPT, EDD, MOF, SkillFit) yang
  menyusun ulang semua tugas belum mulai, bukan enam aturan gabungan Luo;
- titik keputusan: setiap kejadian gangguan, bukan setiap operasi selesai atau job datang;
- "operasi yang sudah dikirim" = tugas yang sudah mulai pada saat keputusan; tenggat job =
  tenggat tugas terakhirnya;
- lingkungan, gangguan, dan instans sama dengan PPO, jadi perbandingannya pada masalah yang
  sama. Luo tidak punya pilihan "biarkan"; versi berpenjaga (``luo_g``) menolak aturan yang
  memperburuk J, sama seperti penjaga PPO;
- saat dipakai, aksi dipilih serakah (Q terbesar), bukan softmax (Bagian 5.6), supaya hasil
  bisa diulang;
- anggaran latih: episode di 7 instans latih × 3 kapasitas × 3 intensitas, seed gangguan
  7000 ke atas (bukan seed validasi atau uji).

Pemakaian (dari folder ini)::

    $PY luo_dqn.py $SRC keluaran.pt [banyak_episode] [seed]
"""
from __future__ import annotations

import penjaga_uji  # noqa: F401  (memblokir pembukaan data uji)

import copy
import math
import sys

import numpy as np
import torch

RULES = ("fifo", "spt", "lpt", "edd", "mof", "skillfit")
GAMMA, TAU, BUFFER, BATCH = 0.9, 0.01, 1000, 32
EPS0, EPS1 = 0.5, 0.1


def luo_features(env) -> np.ndarray:
    """Tujuh fitur Luo (2020) dari jadwal pada saat keputusan, semua di [0, 1]."""
    inst, sched, t = env.inst, env.sched, env.t_now
    jobs = {}
    for task in inst.tasks:
        jobs.setdefault(task.job_id, []).append(task)
    for ops in jobs.values():
        ops.sort(key=lambda x: x.op_index)
    started = {i for i, e in sched.entries.items() if e.start <= t + 1e-9}
    ct = {d.dev_id: 0.0 for d in inst.developers}
    busy = {d.dev_id: 0.0 for d in inst.developers}
    for i in started:
        e = sched.entries[i]
        ct[e.dev_id] = max(ct[e.dev_id], e.end)
        busy[e.dev_id] += e.end - e.start
    u = np.array([busy[k] / ct[k] if ct[k] > 0 else 0.0 for k in ct])
    u_ave, u_std = float(u.mean()), float(u.std())
    op = {j: sum(1 for x in ops if x.task_id in started) for j, ops in jobs.items()}
    n = {j: len(ops) for j, ops in jobs.items()}
    cro = sum(op.values()) / max(1, sum(n.values()))
    crj = np.array([op[j] / n[j] for j in jobs])
    t_cur = float(np.mean(list(ct.values())))
    due = {j: max(x.due_date for x in ops) for j, ops in jobs.items()}
    # Algoritma 2: perkiraan keterlambatan.
    n_tard = n_left = 0
    for j, ops in jobs.items():
        if op[j] < n[j]:
            n_left += n[j] - op[j]
            t_left = 0.0
            for idx in range(op[j], n[j]):
                eff = ops[idx].eligible_devs
                t_left += sum(eff.values()) / len(eff)
                if t_cur + t_left > due[j]:
                    n_tard += n[j] - idx
                    break
    tard_e = n_tard / n_left if n_left else 0.0
    # Algoritma 3: keterlambatan nyata.
    n_tard = 0
    for j, ops in jobs.items():
        if op[j] < n[j] and op[j] > 0:
            last = ops[op[j] - 1]
            if sched.entries[last.task_id].end > due[j]:
                n_tard += n[j] - op[j]
    tard_a = n_tard / n_left if n_left else 0.0
    return np.array([u_ave, u_std, cro, float(crj.mean()), float(crj.std()), tard_e, tard_a],
                    dtype=np.float32)


def luo_reward(f0, f1) -> float:
    """Algoritma 10 Luo (2020): urutan Tard_a, Tard_e, lalu U_ave."""
    U, TE, TA = 0, 5, 6
    if f1[TA] < f0[TA]:
        return 1.0
    if f1[TA] > f0[TA]:
        return -1.0
    if f1[TE] < f0[TE]:
        return 1.0
    if f1[TE] > f0[TE]:
        return -1.0
    if f1[U] > f0[U]:
        return 1.0
    if f1[U] > f0[U] * 0.95:
        return 0.0
    return -1.0


def make_net():
    layers, d = [], 7
    for _ in range(5):
        layers += [torch.nn.Linear(d, 30), torch.nn.Tanh()]
        d = 30
    layers.append(torch.nn.Linear(d, len(RULES)))
    return torch.nn.Sequential(*layers)


def apply_rule(env, a: int, lam: float, guard: bool):
    """Pasang aturan ke-a pada jadwal sekarang. Kembalikan (jadwal baru atau None, perubahan)."""
    from cek_b1_b2 import score, try_level
    from dfjsp.env.state import total_tardiness
    from dfjsp.reschedulers._operators import RuleRepair
    inst, base, t, E = env.inst, env.sched, env.t_now, env.mean_effort
    tr = try_level(inst, base, t, base.affected, RuleRepair(RULES[a]))
    if tr is None:
        return None, 0
    jj, c = score(inst, base, tr, t, E, lam)
    if guard and jj > total_tardiness(inst, base) / E + 1e-9:
        return None, 0
    return tr, c


def _commit(env, sched, chg):
    from dfjsp.env.schedule import recompute_timeline
    from dfjsp.env.state import total_tardiness
    recompute_timeline(env.inst, env.sched, env.t_now)
    if sched is None:
        env.step(0)
    else:
        env.probes[3] = {"tard": total_tardiness(env.inst, sched), "changes": chg,
                         "entries": dict(sched.entries), "moves": []}
        env.step(3)


def train(src, out, n_episodes=300, seed=0):
    sys.path.insert(0, src)
    # cek_episode membaca sys.argv saat diimpor (SRC, model, keluaran); model tidak dipakai.
    sys.argv = [sys.argv[0], src, "/dev/null", "/dev/null"]
    from cek_episode import EpProbe
    from dfjsp.core.domain import set_capacity
    from dfjsp.datasets import BRANDIMARTE_TRAIN, load_brandimarte
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    insts = load_brandimarte(list(BRANDIMARTE_TRAIN))
    pool = [(n, c, i) for n in BRANDIMARTE_TRAIN for c in (1.0, 1.5, 2.0)
            for i in (0.05, 0.15, 0.30)]
    q, qt = make_net(), make_net()
    qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=1e-3)
    buf, log = [], []
    total = 0
    for ep in range(n_episodes):
        name, cap, inten = pool[rng.integers(len(pool))]
        inst = copy.deepcopy(insts[name])
        set_capacity(inst, cap)
        env = EpProbe(lam=1.0)
        # Seed gangguan 7000-9999: di luar seed validasi (5000-5001) dan uji (10000-10009).
        ep_seed = 7000 + 1000 * seed + ep
        assert 7000 <= ep_seed < 10000
        env.reset(inst, intensity=inten, seed=ep_seed)
        eps = EPS0 + (EPS1 - EPS0) * ep / max(1, n_episodes - 1)
        prev, n_dec, ret = None, 0, 0.0
        while not env.done:
            f = luo_features(env)
            if prev is not None:
                r = luo_reward(prev[0], f)
                buf.append((prev[0], prev[1], r, f, 0.0))
                ret += r
            if rng.random() < eps:
                a = int(rng.integers(len(RULES)))
            else:
                with torch.no_grad():
                    a = int(q(torch.tensor(f)[None]).argmax())
            sched, chg = apply_rule(env, a, 1.0, guard=False)
            _commit(env, sched, chg)
            prev = (f, a)
            n_dec += 1
            total += 1
            buf = buf[-BUFFER:]
            if len(buf) >= BATCH:
                b = [buf[k] for k in rng.integers(len(buf), size=BATCH)]
                s = torch.tensor(np.stack([x[0] for x in b]))
                a_ = torch.tensor([x[1] for x in b])
                r_ = torch.tensor([x[2] for x in b], dtype=torch.float32)
                s1 = torch.tensor(np.stack([x[3] for x in b]))
                done = torch.tensor([x[4] for x in b], dtype=torch.float32)
                with torch.no_grad():
                    a1 = q(s1).argmax(1)                         # double DQN: pilih dengan Q
                    y = r_ + GAMMA * (1 - done) * qt(s1).gather(1, a1[:, None])[:, 0]
                loss = torch.nn.functional.mse_loss(q(s).gather(1, a_[:, None])[:, 0], y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                with torch.no_grad():                            # soft target update
                    for p, pt in zip(q.parameters(), qt.parameters()):
                        pt.mul_(1 - TAU).add_(TAU * p)
        if prev is not None:   # kejadian terakhir: episode selesai
            buf.append((prev[0], prev[1], luo_reward(prev[0], luo_features(env)), prev[0], 1.0))
        log.append(dict(ep=ep, instans=name, keputusan=n_dec, return_=ret, eps=eps))
        if ep % 20 == 0 or ep == n_episodes - 1:
            print(f"episode {ep}: {name} cap {cap} int {inten} keputusan {n_dec} "
                  f"return {ret:+.0f} eps {eps:.2f} langkah {total}", flush=True)
    torch.save({"net": q.state_dict(), "log": log, "episodes": n_episodes}, out)


_NET = {}


def policy_action(path, env) -> int:
    if path not in _NET:
        torch.set_num_threads(1)
        net = make_net()
        net.load_state_dict(torch.load(path)["net"])
        net.eval()
        _NET[path] = net
    with torch.no_grad():
        return int(_NET[path](torch.tensor(luo_features(env))[None]).argmax())


if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 300,
          int(sys.argv[4]) if len(sys.argv) > 4 else 0)
