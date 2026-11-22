"""Gerbang B1 dan B2, satu langkah, pada jalur AOR (seperti uji oracle).

Hanya instans latih, seed validasi 5000-5001. Tidak ada pelatihan.

Per kejadian batch dan per lambda, nilai J = telat/E + lambda x perubahan (dibanding jadwal
AOR saat itu) untuk:
- BATCH1     : satu tingkat untuk seluruh batch (v3), tingkat 2 memindah <= 3 tugas,
               tingkat 3 <= 8 tugas;
- BATCH1LEBAR: sama, tetapi batas tugas dikali k (kontrol yang adil: per gangguan juga
               bisa menyentuh sampai 3k / 8k tugas);
- SEQ        : Sequential Lookahead per gangguan (urutan datang), dasar = tingkat 0;
- JOINT      : semua kombinasi 4^k, hanya untuk k <= 3.
Untung = J_AOR - J_metode (besar lebih baik), satuan E.
"""
from __future__ import annotations

import penjaga_uji  # noqa: F401  (memblokir pembukaan data uji)

import copy
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool

SRC, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, SRC)
sys.path.insert(0, sys.argv[3])

from cek_b0_mask import CAPS, INTENSITIES, SEEDS, Probe  # noqa: E402
from dfjsp.core.domain import set_capacity  # noqa: E402
from dfjsp.datasets import BRANDIMARTE_TRAIN, load_brandimarte  # noqa: E402
from dfjsp.env.schedule import recompute_timeline  # noqa: E402
from dfjsp.env.state import total_tardiness  # noqa: E402
from dfjsp.repair.ladder import (LadderReassign, PartialReschedule, Resequence,  # noqa: E402
                                 _changes, _light_copy)

# Nilai lambda bisa diganti lewat LAMS (misalnya "0.1,1,10") untuk menyapu Pareto front.
LAMS = tuple(float(v) for v in os.environ.get("LAMS", "0.3,3,30").split(","))
JOINT_MAX_K = 3


def ops_for(scale: int = 1):
    return {1: Resequence(), 2: LadderReassign(max_tasks=3 * scale),
            3: PartialReschedule(max_tasks=8 * scale)}


def try_level(inst, sched, t, affected, op):
    trial = _light_copy(sched)
    trial.affected = set(affected)
    res = op.apply(inst, trial, t)
    return None if res.invalid_attempt else trial


def score(inst, base, trial, t, E, lam):
    md, mp = _changes(base, trial, t, inst)
    return total_tardiness(inst, trial) / E + lam * (md + mp), md + mp


def decide(env):
    inst, base, t, E = env.inst, env.sched, env.t_now, env.mean_effort
    A = [a for _, a in env.per_sub]
    k = len(A)
    union = set(base.affected)
    j_aor = total_tardiness(inst, base) / E
    out = {"k": k, "j_aor": j_aor}
    ops = ops_for(1)

    # BATCH1 dan BATCH1LEBAR: hasil per tingkat tidak bergantung lambda.
    for tag, o in (("b1", ops), ("b1w", ops_for(k))):
        t0 = time.perf_counter()
        cand = {0: base}
        for lv in (1, 2, 3):
            tr = try_level(inst, base, t, union, o[lv])
            if tr is not None:
                cand[lv] = tr
        pre = {lv: score(inst, base, tr, t, E, 0.0) for lv, tr in cand.items()}
        out[f"ms_{tag}"] = 1000 * (time.perf_counter() - t0)
        for lam in LAMS:
            vals = {lv: pre[lv][0] + lam * pre[lv][1] for lv in pre}
            best = min(vals, key=lambda v: (vals[v], v))
            out[f"{tag}_{lam}"] = j_aor - vals[best]

    # SEQ per lambda.
    for lam in LAMS:
        t0 = time.perf_counter()
        cur, cur_j, choice = base, j_aor, []
        for a_j in A:
            best = (cur_j, 0, cur)
            for lv in (1, 2, 3):
                tr = try_level(inst, cur, t, a_j, ops[lv])
                if tr is None:
                    continue
                j, _ = score(inst, base, tr, t, E, lam)
                if j < best[0] - 1e-9:
                    best = (j, lv, tr)
            cur_j, lv, cur = best
            choice.append(lv)
        out[f"seq_{lam}"] = j_aor - cur_j
        out[f"seqpilih_{lam}"] = choice
        out[f"ms_seq_{lam}"] = 1000 * (time.perf_counter() - t0)

    # JOINT untuk k kecil: pohon awalan supaya awalan yang sama tidak dihitung ulang.
    if k <= JOINT_MAX_K:
        t0 = time.perf_counter()
        leaves = []

        def walk(j, sched, path):
            if j == k:
                leaves.append((path, sched))
                return
            walk(j + 1, sched, path + (0,))
            for lv in (1, 2, 3):
                tr = try_level(inst, sched, t, A[j], ops[lv])
                if tr is not None:
                    walk(j + 1, tr, path + (lv,))

        walk(0, base, ())
        pre = [(p, *score(inst, base, s, t, E, 0.0)) for p, s in leaves]
        out["ms_joint"] = 1000 * (time.perf_counter() - t0)
        out["n_leaves"] = len(leaves)
        for lam in LAMS:
            best = min(pre, key=lambda x: x[1] + lam * x[2])
            out[f"joint_{lam}"] = j_aor - (best[1] + lam * best[2])
            out[f"jointpilih_{lam}"] = list(best[0])
    recompute_timeline(inst, base, t)
    return out


def run(job):
    name, cap, inten, seed = job
    inst = copy.deepcopy(load_brandimarte([name])[name])
    set_capacity(inst, cap)
    env = Probe(lam=1.0)
    env.reset(inst, intensity=inten, seed=seed)
    rows = []
    while not env.done:
        if getattr(env, "per_sub", None):
            r = decide(env)
            r.update(instans=name, kapasitas=cap, intensitas=inten, seed=seed)
            rows.append(r)
        env.step(0)
    return rows


if __name__ == "__main__":
    jobs = [(n, c, i, s) for n in BRANDIMARTE_TRAIN for c in CAPS for i in INTENSITIES
            for s in SEEDS]
    with Pool(8) as pool:
        res = pool.map(run, jobs, chunksize=1)
    rows = [r for rs in res for r in rs]
    json.dump(rows, open(OUT, "w"))
    print(len(rows))
