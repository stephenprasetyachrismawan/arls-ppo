"""E9c: kumpulkan data DAgger dari keadaan yang dikunjungi PPO + penjaga.

Data guru lama (``bc_data_7000.pkl``) dikumpulkan di jalur AOR: di antara kejadian, jadwal
tidak diperbaiki, dan di dalam kejadian jadwal mengikuti pilihan guru. Saat dipakai, PPO
berjalan di jadwal hasil pilihannya sendiri, jadi keadaan yang ia lihat berbeda. DAgger
(Ross, Gordon, dan Bagnell, 2011) menutup selisih ini: murid dijalankan sepanjang episode,
dan di setiap keadaan yang ia kunjungi guru (Sequential Lookahead) memberi label.

Satu episode per (instans, kapasitas, intensitas, seed, λ), karena jalur episode bergantung
pada λ. Seed gangguan 7002-7003 (rentang guru 7000-9999), bukan seed validasi 5000-5001 atau
uji 10000-10009. Format keluaran sama dengan ``cek_b3.py kumpul``: {"rows", "inst"}.

Pemakaian (dari folder ini)::

    PPO_DIR=folder_model_ppo [PPO_NAME=bc_ppo] $PY dagger_kumpul.py $SRC $B3 keluaran.pkl
    DAGGER_HGB=hgb7.pkl $PY dagger_kumpul.py $SRC $B3 keluaran.pkl    # murid boosting (E10)
"""
from __future__ import annotations

import penjaga_uji  # noqa: F401  (memblokir pembukaan data uji)

import copy
import os
import pickle
import sys
from multiprocessing import Pool

SEEDS_DAGGER = tuple(int(v) for v in os.environ.get("SEED_DAGGER", "7002,7003").split(","))
assert all(7000 <= s < 10000 for s in SEEDS_DAGGER), "seed DAgger harus di rentang guru"
SRC, B3, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
NAME = os.environ.get("PPO_NAME", "bc_ppo")   # lewat lingkungan, supaya pekerja Pool ikut
HGB = os.environ.get("DAGGER_HGB")            # E10: murid boosting dari latih_hgb.py
_HGB = {}


def student(name):
    """Murid + penjaga: PPO (PPO_DIR) atau boosting (DAGGER_HGB)."""
    if not HGB:
        return PPOPol(NAME, guard=True)
    if "m" not in _HGB:
        _HGB["m"] = pickle.load(open(HGB, "rb"))["models"]
    return cek_b3.Pol(instans_acak.model_instans(_HGB["m"], name)["hgb"], True)
sys.argv = [sys.argv[0], SRC, B3, "/dev/null"]   # dibaca cek_episode saat diimpor

import cek_episode as ce  # noqa: E402
from cek_episode import (CAPS, INTENSITIES, LAMS, BRANDIMARTE_TRAIN, EpProbe,  # noqa: E402
                         PPOPol, RepairEnv, cek_b3, load_brandimarte, recompute_timeline,
                         set_capacity, total_tardiness)
import instans_acak  # noqa: E402


def run(job):
    name, cap, inten, seed, lam = job
    inst = copy.deepcopy(instans_acak.load_any(name))
    set_capacity(inst, cap)
    env = EpProbe(lam=lam)
    env.reset(inst, intensity=inten, seed=seed)
    pol = student(name)
    rows = []
    while not env.done:
        bs = RepairEnv._state(env)
        *_, cur, chg, rs = cek_b3.run_batch(env, lam, pol, bs, record=True, want_sched=True,
                                            follow=True)
        rows += [(x, y, d, lam) for x, y, d in rs]
        recompute_timeline(env.inst, env.sched, env.t_now)
        if cur is env.sched:
            env.step(0)
        else:
            env.probes[3] = {"tard": total_tardiness(env.inst, cur), "changes": chg,
                             "entries": dict(cur.entries), "moves": []}
            env.step(3)
    return name, rows


def pekerjaan():
    """Episode DAgger: 7 MK latih, ditambah instans W bila ``DAGGER_ACAK="W0-59"`` (E16)."""
    nama = instans_acak.daftar_instans(BRANDIMARTE_TRAIN, tambah=os.environ.get("DAGGER_ACAK"))
    return [(n, c, i, s, lam) for n in nama for c in CAPS for i in INTENSITIES
            for s in SEEDS_DAGGER for lam in LAMS]


if __name__ == "__main__":
    assert ce.PPO_DIR or HGB, "isi PPO_DIR atau DAGGER_HGB"
    jobs = pekerjaan()
    jobs.sort(key=lambda j: j[0] != "MK10")
    rows, inst = [], []
    with Pool(8) as pool:
        for k, (name, rs) in enumerate(pool.imap_unordered(run, jobs, chunksize=1), 1):
            rows += rs
            inst += [name] * len(rs)
            if k % 50 == 0 or k == len(jobs):
                print(f"{k}/{len(jobs)} episode, {len(rows)} baris", flush=True)
    pickle.dump({"rows": rows, "inst": inst}, open(OUT, "wb"))
