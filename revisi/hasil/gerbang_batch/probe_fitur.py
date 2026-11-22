"""Fitur hasil probe untuk PPO per gangguan (dipakai latih_ppo_probe.py dan cek_episode.py).

Modul ini tidak membaca argumen, jadi aman diimpor dari skrip mana pun sesudah cek_b3.
"""
import numpy as np

import cek_b3
from cek_b1_b2 import score, try_level

N_PROBE = 8


def probe_features(dj):
    """dj: daftar 4 angka J(tingkat) - J(cur), nan = tidak sah. Kembalikan 8 angka."""
    dj = np.asarray(dj, dtype=float)
    valid = ~np.isnan(dj)
    v = np.where(valid, dj, 0.0)
    f1 = np.sign(v) * np.log1p(np.abs(v))
    # pilihan satu langkah: J terkecil, seri dimenangkan tingkat 0 lalu tingkat kecil
    key = np.where(valid, dj - np.array([1e-9, 0, 0, 0]), np.inf)
    f2 = np.zeros(4)
    f2[int(np.argmin(key))] = 1.0
    return np.concatenate([f1, f2])


def probe_all(env, cur, cur_j, base_sched, A_j, mask, lam):
    """Coba tiap tingkat sah dari jadwal cur. Kembalikan dj dan percobaan per tingkat."""
    dj = [0.0, np.nan, np.nan, np.nan]
    trials = {0: (cur_j, cur, None)}
    for lv in (1, 2, 3):
        if not mask[lv - 1]:
            continue
        tr = try_level(env.inst, cur, env.t_now, A_j, cek_b3.OPS[lv])
        if tr is None:
            continue
        jj, c = score(env.inst, base_sched, tr, env.t_now, env.mean_effort, lam)
        dj[lv] = jj - cur_j
        trials[lv] = (jj, tr, c)
    return dj, trials
