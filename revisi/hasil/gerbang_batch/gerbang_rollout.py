"""Gerbang rollout: apakah ada nilai jangka panjang di masalah ini?

Sequential Lookahead (seq) memilih tingkat dengan J terkecil saat ini. Rollout memilih tingkat
dengan J terkecil SESUDAH beberapa kejadian ke depan disimulasikan, dengan Sequential sebagai
kebijakan dasar (Bertsekas, rollout). Kalau rollout jauh lebih baik dari seq, keputusan
sekarang memang memengaruhi masa depan, sehingga ada ruang untuk PPO jangka panjang.

Masa depan yang disimulasikan memakai gangguan CONTOH: gangguan sesudah waktu sekarang diganti
dengan timeline dari seed lain (20000 ke atas, bukan seed validasi 5000-5001 atau uji
10000-10009), dengan instans, kapasitas, dan intensitas yang sama. Varian "intip" memakai
gangguan yang sebenarnya; itu hanya batas atas nilai antisipasi, bukan metode yang sah.

Metode (variabel METODE, dipisah koma):
- seq          : Sequential Lookahead biasa;
- roll_h<H>_s<S>: rollout H kejadian ke depan, rata-rata atas S masa depan contoh;
- intip_h<H>   : rollout H kejadian ke depan dengan gangguan sebenarnya (batas atas).
roll_h0_s1 adalah kontrol: sama seperti seq, tetapi pilihan dinilai sesudah sisa kejadian
ini diselesaikan dengan seq.

J episode = telat akhir / E + lambda x total perubahan (sama dengan cek_episode.py).

Pemakaian: $PY gerbang_rollout.py $SRC b3_model.pkl keluaran.json
Variabel: METODE, LAMS (bawaan "3"), INSTANS (bawaan semua instans latih), SEEDS_EP
(bawaan "5000,5001"), PROSES (bawaan 8).
"""
from __future__ import annotations

import penjaga_uji  # noqa: F401  (memblokir pembukaan data uji)

import copy
import json
import os
import sys
import time
from multiprocessing import Pool

SRC, MODEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.environ.setdefault("LAMS", "3")
sys.argv = [sys.argv[0], SRC, MODEL, OUT]
sys.path.insert(0, SRC)

import cek_b3  # noqa: E402
import cek_episode as ce  # noqa: E402
from cek_b0_mask import CAPS, INTENSITIES  # noqa: E402
from cek_b1_b2 import LAMS, score, try_level  # noqa: E402
from dfjsp.core.domain import set_capacity  # noqa: E402
from dfjsp.datasets import BRANDIMARTE_TRAIN, load_brandimarte  # noqa: E402
from dfjsp.env.repair_env import RepairEnv  # noqa: E402
from dfjsp.env.schedule import recompute_timeline  # noqa: E402
from dfjsp.env.state import total_tardiness  # noqa: E402

METHODS = tuple(os.environ.get("METODE", "seq,roll_h0_s1,roll_h3_s4,intip_h3").split(","))
INSTANCES = tuple(os.environ.get("INSTANS", ",".join(BRANDIMARTE_TRAIN)).split(","))
EP_SEEDS = tuple(int(s) for s in os.environ.get("SEEDS_EP", "5000,5001").split(","))
FUTURE_SEED0 = 20000
FORBIDDEN = set(range(5000, 5002)) | set(range(10000, 10010))


def commit(env, sched, chg):
    """Pasang keputusan satu kejadian lalu maju ke kejadian berikutnya (seperti cek_episode)."""
    recompute_timeline(env.inst, env.sched, env.t_now)
    if sched is None:
        env.step(0)
        return 0
    env.probes[3] = {"tard": total_tardiness(env.inst, sched), "changes": chg,
                     "entries": dict(sched.entries), "moves": []}
    env.step(3)
    return chg


def candidates(env, cur, cur_j, chg, base, j, A, types, bs, lam):
    """Semua tingkat sah untuk gangguan j: {tingkat: (J, jadwal, perubahan)}."""
    inst, t, E = env.inst, env.t_now, env.mean_effort
    _, mask = cek_b3.features(env, cur, bs, j, A, types, lam, chg)
    out = {0: (cur_j, cur, chg)}
    for lv in (1, 2, 3):
        if not mask[lv - 1]:
            continue
        tr = try_level(inst, cur, t, A[j], cek_b3.OPS[lv])
        if tr is None:
            continue
        jj, c = score(inst, base, tr, t, E, lam)
        out[lv] = (jj, tr, c)
    return out


def finish_event_seq(env, cur, cur_j, chg, base, j0, A, types, bs, lam):
    """Selesaikan gangguan j0, j0+1, ... di kejadian ini dengan Sequential."""
    for j in range(j0, len(A)):
        cand = candidates(env, cur, cur_j, chg, base, j, A, types, bs, lam)
        cur_j, cur, chg = min(cand.values(), key=lambda v: v[0])
    return cur, chg


def simulate(env, sched, chg, j, A, types, bs, lam, horizon, future):
    """Nilai satu pilihan: J sesudah sisa kejadian ini dan `horizon` kejadian berikutnya."""
    e, s = copy.deepcopy((env, sched))
    base = e.sched
    cur_j = total_tardiness(e.inst, s) / e.mean_effort  # hanya untuk urutan seq di sisa kejadian
    s, chg = finish_event_seq(e, s, cur_j, chg, base, j + 1, A, types, bs, lam)
    if future is not None:
        # Ganti gangguan sesudah kejadian ini SEBELUM step(), karena step() langsung
        # menerapkan kejadian berikutnya. Dengan begitu tidak ada gangguan asli yang terintip.
        e._events = e._events[: e._event_idx + 1] + [ev for ev in future if ev.time > e.t_now]
    total = commit(e, None if s is base else s, chg)
    for _ in range(horizon):
        if e.done:
            break
        sc, c = ce.decide(e, "seq", lam)
        total += commit(e, sc, c)
    return total_tardiness(e.inst, e.sched) / e.mean_effort + lam * total


def rollout_batch(env, lam, horizon, futures):
    """Rollout per gangguan di atas Sequential. futures = daftar timeline (None = intip)."""
    base = env.sched
    A = [a for _, a in env.per_sub]
    types = [ty for ty, _ in env.per_sub]
    bs = RepairEnv._state(env)
    cur, cur_j, chg = base, total_tardiness(env.inst, base) / env.mean_effort, 0
    for j in range(len(A)):
        cand = candidates(env, cur, cur_j, chg, base, j, A, types, bs, lam)
        if len(cand) == 1:
            continue
        val = {}
        for lv, (jj, sc, c) in cand.items():
            vs = [simulate(env, sc, c, j, A, types, bs, lam, horizon, f) for f in futures]
            val[lv] = sum(vs) / len(vs)
        best = min(val, key=lambda lv: (val[lv], cand[lv][0]))
        cur_j, cur, chg = cand[best]
    return (None if cur is base else cur), chg


def future_timelines(name, cap, inten, n):
    """n timeline gangguan contoh dari seed 20000+ untuk instans dan kondisi yang sama."""
    out = []
    for k in range(n):
        seed = FUTURE_SEED0 + k
        assert seed not in FORBIDDEN
        inst = copy.deepcopy(load_brandimarte([name])[name])
        set_capacity(inst, cap)
        env = ce.EpProbe(lam=1.0)
        env.reset(inst, intensity=inten, seed=seed)
        out.append(list(env._events))
    return out


def run(job):
    name, cap, inten, seed, method, lam = job
    inst = copy.deepcopy(load_brandimarte([name])[name])
    set_capacity(inst, cap)
    env = ce.EpProbe(lam=lam)
    env.reset(inst, intensity=inten, seed=seed)
    futures = None
    horizon = 0
    if method.startswith("roll_"):
        h, s = method[5:].split("_")
        horizon, n = int(h[1:]), int(s[1:])
        futures = future_timelines(name, cap, inten, n)
    elif method.startswith("intip_"):
        horizon, futures = int(method[7:]), [None]
    changes, n_dec, spent = 0, 0, 0.0
    while not env.done:
        t0 = time.perf_counter()
        if method == "seq":
            sched, chg = ce.decide(env, "seq", lam)
        else:
            sched, chg = rollout_batch(env, lam, horizon, futures)
        spent += time.perf_counter() - t0
        changes += commit(env, sched, chg)
        n_dec += 1
    telat = total_tardiness(env.inst, env.sched) / env.mean_effort
    return {"instans": name, "kapasitas": cap, "intensitas": inten, "seed": seed,
            "metode": method, "lam": lam, "telat": telat, "ubah": changes,
            "J": telat + lam * changes, "keputusan": n_dec, "ms": 1000 * spent / max(1, n_dec)}


if __name__ == "__main__":
    jobs = [(nm, c, i, s, m, lam) for nm in INSTANCES for c in CAPS for i in INTENSITIES
            for s in EP_SEEDS for m in METHODS for lam in LAMS]
    jobs.sort(key=lambda j: (not j[4].startswith(("roll", "intip")), j[0] != "MK10"))
    done = set()
    try:
        for line in open(OUT + "l"):
            r = json.loads(line)
            done.add((r["instans"], r["kapasitas"], r["intensitas"], r["seed"], r["metode"],
                      r["lam"]))
    except FileNotFoundError:
        pass
    jobs = [j for j in jobs if j not in done]
    t0 = time.time()
    with Pool(int(os.environ.get("PROSES", "8")), initializer=ce._init,
              initargs=(MODEL,)) as pool, open(OUT + "l", "a") as f:
        for n, r in enumerate(pool.imap_unordered(run, jobs, chunksize=1), 1):
            f.write(json.dumps(r) + "\n")
            f.flush()
            el = time.time() - t0
            print(f"{n}/{len(jobs)} {r['instans']} {r['metode']} J {r['J']:.1f} "
                  f"{el / 60:.1f} menit, sisa kira-kira {el / n * (len(jobs) - n) / 60:.1f} menit",
                  flush=True)
    res = [json.loads(x) for x in open(OUT + "l")]
    json.dump(res, open(OUT, "w"))
    print(len(res))
