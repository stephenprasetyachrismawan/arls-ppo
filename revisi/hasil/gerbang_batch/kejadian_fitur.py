"""Fitur dan langkah untuk ablasi "PPO per kejadian" (satu tingkat untuk seluruh kejadian).

Dipakai latih_ppo_kejadian.py dan cek_episode.py. Tidak membaca argumen, jadi aman diimpor
sesudah cek_b3. Fiturnya sama dengan fitur per gangguan (dimensi dan skala sama), tetapi
dihitung untuk gabungan semua tugas terdampak dalam kejadian; angka k diisi banyak gangguan
yang sebenarnya, posisi j = 0.
"""
from collections import Counter

import cek_b3
from cek_b1_b2 import ops_for, score, try_level
from dfjsp.env.repair_env import RepairEnv
from dfjsp.env.state import total_tardiness


def event_features(env, lam):
    """Kembalikan (fitur, mask lapis atas, gabungan A, k) untuk kejadian saat ini."""
    A = [a for _, a in env.per_sub]
    types = [ty for ty, _ in env.per_sub]
    union = set().union(*A) if A else set()
    k = len(A)
    dominant = Counter(types).most_common(1)[0][0]
    x, mask = cek_b3.features(env, env.sched, RepairEnv._state(env), 0, [union],
                              [dominant], lam, 0)
    # urutan ujung fitur: k, j/k, sisa gangguan, tumpang tindih, log sisa, log ubah, log lam
    x[-7], x[-6], x[-5] = k, 0.0, k - 1
    return x, mask, union, k


def apply_level(env, lam, union, k, level):
    """Jalankan satu tingkat untuk seluruh kejadian (batas tugas x k). None = tidak berubah."""
    if level == 0:
        return None, 0, total_tardiness(env.inst, env.sched) / env.mean_effort
    tr = try_level(env.inst, env.sched, env.t_now, union, ops_for(k)[level])
    if tr is None:
        return None, 0, total_tardiness(env.inst, env.sched) / env.mean_effort
    jj, c = score(env.inst, env.sched, tr, env.t_now, env.mean_effort, lam)
    return tr, c, jj


def teacher_level(env, lam, union, k, mask):
    """Guru BC: Lookahead satu tingkat per kejadian dengan batas tugas x k (b1w)."""
    best_lv, best_j = 0, total_tardiness(env.inst, env.sched) / env.mean_effort
    for lv in (1, 2, 3):
        if not mask[lv - 1]:
            continue
        tr, c, jj = apply_level(env, lam, union, k, lv)
        if tr is not None and jj < best_j - 1e-9:
            best_lv, best_j = lv, jj
    return best_lv
