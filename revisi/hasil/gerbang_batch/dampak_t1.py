"""Dampak perbaikan hitungan perubahan (T1, PR #164) pada label guru.

Jalur Sequential dengan hitungan baru (seed guru 7000, bukan validasi 5000-5001 atau uji
10000-10009). Di setiap sub-keputusan dengan lebih dari satu tingkat sah, pilihan guru
dihitung dua kali pada keadaan yang sama: dengan hitungan perubahan lama (disalin dari
``ladder.py`` sebelum perbaikan) dan dengan hitungan baru. Gerbang di
``docs/ARSITEKTUR_PIPELINE.md`` 5.6: latih ulang kalau label berbeda di ≥ 5% sub-keputusan
pada salah satu λ.

Pemakaian (dari folder ini)::

    $PY dampak_t1.py $SRC $B3 > dampak_t1.md
"""
from __future__ import annotations

import penjaga_uji  # noqa: F401  (memblokir pembukaan data uji)

import copy
import os
import sys
from multiprocessing import Pool

if "_T1_ARGS" not in os.environ:          # pekerja forkserver membaca argumen yang sama
    os.environ["_T1_ARGS"] = "\t".join(sys.argv[1:])
SRC, B3 = os.environ["_T1_ARGS"].split("\t")[:2]
sys.argv = [sys.argv[0], SRC, B3, "/dev/null"]   # dibaca gerbang_rollout saat diimpor

import gerbang_rollout as gr  # noqa: E402
from gerbang_rollout import (CAPS, INTENSITIES, BRANDIMARTE_TRAIN, RepairEnv,  # noqa: E402
                             ce, load_brandimarte, set_capacity, total_tardiness)
from dfjsp.env.schedule import queue_order  # noqa: E402
from dfjsp.repair.ladder import _lis_length  # noqa: E402

SEED_EP = 7000
LAMS = (0.3, 3.0, 30.0)


def changes_lama(before, after, t_now, inst):
    """Hitungan perubahan sebelum perbaikan T1 (salinan apa adanya, untuk pembanding)."""
    moved_dev = sum(1 for tid, e in after.entries.items()
                    if tid in before.entries and before.entries[tid].dev_id != e.dev_id)
    qb = queue_order(inst, before, t_now)
    qa = queue_order(inst, after, t_now)
    moved_pos = 0
    for j, new in qa.items():
        old_pos = {t: k for k, t in enumerate(qb.get(j, []))}
        common = [old_pos[t] for t in new if t in old_pos]
        moved_pos += len(common) - _lis_length(common)
    return moved_dev, moved_pos


def run(job):
    name, cap, inten, lam = job
    assert 7000 <= SEED_EP < 10000
    inst = copy.deepcopy(load_brandimarte([name])[name])
    set_capacity(inst, cap)
    env = ce.EpProbe(lam=lam)
    env.reset(inst, intensity=inten, seed=SEED_EP)
    E = env.mean_effort
    n = beda = 0
    while not env.done:
        base = env.sched
        A = [a for _, a in env.per_sub]
        types = [ty for ty, _ in env.per_sub]
        bs = RepairEnv._state(env)
        cur, cur_j, chg = base, total_tardiness(env.inst, base) / E, 0
        for j in range(len(A)):
            cand = gr.candidates(env, cur, cur_j, chg, base, j, A, types, bs, lam)
            baru = min(cand, key=lambda lv: (cand[lv][0], lv))
            if len(cand) > 1:
                lama_j = {}
                for lv, (jj, sc, c) in cand.items():
                    # Tingkat 0 = jadwal sekarang, yang perubahannya dari sub-langkah sebelumnya
                    # juga harus dihitung dengan cara lama supaya kedua sisi sebanding.
                    md, mp = (0, 0) if sc is base else changes_lama(base, sc, env.t_now, env.inst)
                    lama_j[lv] = total_tardiness(env.inst, sc) / E + lam * (md + mp)
                lama = min(lama_j, key=lambda lv: (lama_j[lv], lv))
                n += 1
                beda += lama != baru
            cur_j, cur, chg = cand[baru]
        gr.commit(env, None if cur is base else cur, chg)
    return name, lam, n, beda


if __name__ == "__main__":
    only = os.environ.get("T1_INSTANS")          # uji asap: satu instans saja
    names = [only] if only else list(BRANDIMARTE_TRAIN)
    jobs = [(nm, c, i, lam) for nm in names for c in CAPS for i in INTENSITIES for lam in LAMS]
    jobs.sort(key=lambda jb: jb[0] != "MK10")
    tot = {lam: [0, 0] for lam in LAMS}
    with Pool(8, initializer=ce._init, initargs=(B3,)) as pool:
        for k, (nm, lam, n, b) in enumerate(pool.imap_unordered(run, jobs, chunksize=1), 1):
            tot[lam][0] += n
            tot[lam][1] += b
            print(f"{k}/{len(jobs)} episode", file=sys.stderr, flush=True)
    print("# Dampak perbaikan T1 pada label guru (seed guru 7000)\n")
    print("| λ | sub-keputusan dengan > 1 tingkat sah | label lama ≠ baru | porsi |")
    print("|---|---|---|---|")
    for lam in LAMS:
        n, b = tot[lam]
        print(f"| {lam:g} | {n} | {b} | {b / max(n, 1):.1%} |".replace(".", ","))
    lewat = any(tot[lam][1] / max(tot[lam][0], 1) >= 0.05 for lam in LAMS)
    print(f"\nGerbang (≥ 5% di salah satu λ): {'LATIH ULANG' if lewat else 'tidak perlu dari sisi label'}.")
