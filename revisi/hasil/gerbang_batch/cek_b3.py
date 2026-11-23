"""Gerbang B3: bisakah pilihan Sequential Lookahead (SEQ) ditebak dari ringkasan keadaan?

Tanpa pelatihan PPO. Hanya instans latih, seed validasi 5000-5001, jalur AOR seperti
cek_b1_b2.py. Fitur hanya memakai keadaan teragregasi, sifat gangguan d_j, konteks batch,
dan bit mask dua lapis (lapis atas: tingkat punya calon; lapis bawah: banyak pengembang
penerima yang sah). Tidak ada *probe*: penggolong tidak melihat hasil uji-coba tingkat.

Pemakaian (dari folder ini)::

    $PY cek_b3.py kumpul $SRC b3_data.pkl     # fitur, label SEQ, J tiap pilihan
    SEED_KUMPUL=7000,7001 $PY cek_b3.py kumpul $SRC bc_data_7000.pkl   # data guru BC PPO
    $PY cek_b3.py latih  $SRC b3_data.pkl b3_model.pkl
    $PY cek_b3.py nilai  $SRC b3_model.pkl b3_nilai.json
"""
from __future__ import annotations

import penjaga_uji  # noqa: F401  (memblokir pembukaan data uji)

import copy
import math
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np

MODE, SRC = sys.argv[1], sys.argv[2]
sys.path.insert(0, SRC)
sys.argv = [sys.argv[0], SRC, "/dev/null"] + sys.argv[3:]  # cek_b0_mask membaca argv[1:3]
ARGS = sys.argv[3:]

from cek_b0_mask import CAPS, INTENSITIES, SEEDS, Probe  # noqa: E402

# Seed gangguan untuk mode kumpul bisa diganti lewat SEED_KUMPUL (misalnya "7000,7001").
# Data guru BC untuk PPO harus dari seed ini, bukan 5000-5001 yang dipakai validasi
# episode; kalau sama, PPO dinilai pada skenario yang jawabannya sudah ia tiru.
if __import__("os").environ.get("SEED_KUMPUL"):
    SEEDS = tuple(int(s) for s in __import__("os").environ["SEED_KUMPUL"].split(","))
from cek_b1_b2 import LAMS, ops_for, score, try_level  # noqa: E402
from dfjsp.core.domain import set_capacity  # noqa: E402
from dfjsp.datasets import BRANDIMARTE_TRAIN, load_brandimarte  # noqa: E402
from dfjsp.env.repair_env import RepairEnv  # noqa: E402
from dfjsp.env.schedule import developer_load, partition, recompute_timeline  # noqa: E402
from dfjsp.env.state import total_tardiness  # noqa: E402
from dfjsp.repair.ladder import _changes  # noqa: E402
from dfjsp.repair.masks import is_pair_feasible  # noqa: E402

TYPES = ("urgent_task", "effort_revision", "rework_review_fail", "priority_change",
         "developer_unavailable")
OPS = ops_for(1)


# FITUR=v2 memakai fitur tanpa satuan dari fitur_v2.py (E2). Bawaan: 38 fitur lama.
FITUR = __import__("os").environ.get("FITUR", "v1")


def features(env, cur, base_state, j, A, types, lam, changes_so_far):
    """Fitur ringkas untuk gangguan ke-j, dihitung pada jadwal ``cur`` (tanpa probe)."""
    if FITUR == "v2":
        import fitur_v2
        return fitur_v2.features(env, cur, base_state, j, A, types, lam, changes_so_far, OPS)
    inst, t, E = env.inst, env.t_now, env.mean_effort
    _, in_prog, not_started = partition(inst, cur, t)
    ns = set(not_started)
    by_id = {x.task_id: x for x in inst.tasks}
    dev_by_id = {d.dev_id: d for d in inst.developers}
    loads = developer_load(inst, cur, t)
    a_j = A[j]
    a_ns = [x for x in a_j if x in ns]
    late = [x for x in a_j if cur.entries[x].end > by_id[x].due_date + 1e-9]
    tard_a = sum(max(0.0, cur.entries[x].end - by_id[x].due_date) for x in a_j) / E
    slack = [(by_id[x].due_date - cur.entries[x].end) / E for x in a_ns] or [0.0]
    n_elig = [len(by_id[x].eligible_devs) for x in a_ns] or [0]
    recv = set()
    n_pairs = 0
    for x in a_ns:
        curdev = cur.entries[x].dev_id
        for d in by_id[x].eligible_devs:
            if d != curdev and is_pair_feasible(x, d, inst, cur, t, loads=loads,
                                                by_id=by_id, dev_by_id=dev_by_id):
                recv.add(d)
                n_pairs += 1
    trial = copy.copy(cur)
    trial.affected = set(a_j)
    mask = [int(OPS[lv].inspect(inst, trial, t).feasible) for lv in (1, 2, 3)]
    util = [loads.get(d.dev_id, 0.0) / max(1e-9, d.capacity) for d in inst.developers]
    others = set().union(*[A[i] for i in range(len(A)) if i != j]) if len(A) > 1 else set()
    rest = set().union(*A[j + 1:]) if j + 1 < len(A) else set()
    unfinished = max(1, len(in_prog) + len(not_started))
    k = len(A)
    f = list(base_state) + [
        *[1.0 if types[j] == ty else 0.0 for ty in TYPES],
        math.log1p(len(a_j)), math.log1p(len(a_ns)), len(a_ns) / unfinished,
        len(late) / max(1, len(a_j)), math.log1p(tard_a),
        float(np.mean(slack)), float(np.min(slack)),
        float(np.mean(n_elig)), math.log1p(len(recv)), math.log1p(n_pairs),
        *mask,
        float(np.mean(util)), float(np.max(util)), float(np.std(util)),
        math.log1p(total_tardiness(inst, cur) / E),
        k, j / k, (k - j - 1),
        len(set(a_j) & others) / max(1, len(a_j)), math.log1p(len(rest)),
        math.log1p(changes_so_far), math.log(lam),
    ]
    return np.asarray(f, dtype=np.float32), mask


def catatan_tingkat_baru():
    """Kamus kosong untuk argumen ``catat`` di :func:`run_batch`."""
    return {"diminta": [0, 0, 0, 0], "dipakai": [0, 0, 0, 0], "gagal": 0,
            "ditolak_penjaga": 0}


def run_batch(env, lam, policy, base_state, record, want_sched=False, follow=False,
              catat=None, waktu_pemilih=None):
    """Satu batch, satu lambda. ``policy`` None = SEQ; kalau tidak, fungsi fitur -> tingkat.

    ``catat`` (opsional, kamus hasil ``catatan_tingkat_baru``) ditambah per sub-keputusan:
    tingkat yang diminta, tingkat yang dipakai, dan sebab jatuh ke 0 (operator gagal atau
    ditolak penjaga). Hanya mencatat; jalannya batch tidak berubah.

    ``waktu_pemilih`` (opsional, daftar) ditambah ms satu panggilan ``policy`` per
    sub-keputusan di jalur pemilih, tanpa fitur dan operator (I-1). Hanya mencatat.

    Kembalikan (untung, pilihan, ms, banyak apply). Bila ``record``, simpan baris data.
    ``follow`` (E9c, DAgger): catat label SEQ di setiap keadaan, tetapi jalan mengikuti
    ``policy`` (dengan penjaganya), supaya data berasal dari keadaan yang dikunjungi murid.
    """
    inst, base, t, E = env.inst, env.sched, env.t_now, env.mean_effort
    A = [a for _, a in env.per_sub]
    types = [ty for ty, _ in env.per_sub]
    j_aor = total_tardiness(inst, base) / E
    cur, cur_j, chg = base, j_aor, 0
    choice, rows, n_apply = [], [], 0
    t0 = time.perf_counter()
    feat_ms = 0.0
    for j in range(len(A)):
        tf = time.perf_counter()
        x, mask = features(env, cur, base_state, j, A, types, lam, chg)
        feat_ms += time.perf_counter() - tf
        if policy is None or record:
            # SEQ: coba ketiga tingkat; catat J tiap pilihan.
            opts = {0: (cur_j, cur, chg)}
            for lv in (1, 2, 3):
                tr = try_level(inst, cur, t, A[j], OPS[lv])
                n_apply += 1
                if tr is not None:
                    jj, c = score(inst, base, tr, t, E, lam)
                    opts[lv] = (jj, tr, c)
            best = min(opts, key=lambda v: (opts[v][0] - (1e-9 if v == 0 else 0), v))
            if record:
                rows.append((x, best, [opts.get(lv, (np.nan,))[0] - cur_j for lv in range(4)]))
            lv = best
            if catat is not None:
                # Jalur SEQ tanpa ``follow``: tingkat yang diminta = tingkat terbaik yang dipakai.
                catat["diminta"][best] += 1
            if follow and policy is not None:
                lv = policy(x, mask)
                if lv not in opts or (policy.guard and opts[lv][0] > cur_j + 1e-9):
                    lv = 0
            cur_j, cur, chg = opts[lv]
        else:
            tp = time.perf_counter()
            lv = diminta = policy(x, mask)
            if waktu_pemilih is not None:
                waktu_pemilih.append(1000 * (time.perf_counter() - tp))
            if lv > 0:
                tr = try_level(inst, cur, t, A[j], OPS[lv])
                n_apply += 1
                if tr is None:
                    lv = 0
                    if catat is not None:
                        catat["gagal"] += 1
                else:
                    jj, c = score(inst, base, tr, t, E, lam)
                    if policy.guard and jj > cur_j + 1e-9:
                        lv = 0          # penjaga: tolak bila J memburuk (tanpa probe tambahan)
                        if catat is not None:
                            catat["ditolak_penjaga"] += 1
                    else:
                        cur_j, cur, chg = jj, tr, c
            if catat is not None:
                catat["diminta"][diminta] += 1
        if catat is not None:
            catat["dipakai"][lv] += 1
        choice.append(lv)
    ms = 1000 * (time.perf_counter() - t0)
    if want_sched:
        return (j_aor - cur_j, choice, ms, cur, chg) + ((rows,) if record else ())
    return j_aor - cur_j, choice, ms, 1000 * feat_ms, n_apply, rows


def episodes():
    # INSTANS_ACAK="0-49" memakai instans acak R000-R049 (E3), "W0-59" instans rentang lebar
    # W000-W059 (E16), bukan MK. Tanpa variabel: 7 MK latih.
    import instans_acak
    names = instans_acak.daftar_instans(BRANDIMARTE_TRAIN,
                                        ganti=__import__("os").environ.get("INSTANS_ACAK"))
    return [(n, c, i, s) for n in names for c in CAPS for i in INTENSITIES for s in SEEDS]


def walk(job, fn):
    name, cap, inten, seed = job
    import instans_acak
    inst = copy.deepcopy(instans_acak.load_any(name))   # MK, R (E3), atau W (E16)
    set_capacity(inst, cap)
    env = Probe(lam=1.0)
    env.reset(inst, intensity=inten, seed=seed)
    out = []
    while not env.done:
        if getattr(env, "per_sub", None):
            base_state = RepairEnv._state(env)
            out.append(fn(env, base_state))
            recompute_timeline(env.inst, env.sched, env.t_now)
        env.step(0)
    return out


def collect(job):
    def fn(env, bs):
        rows = []
        for lam in LAMS:
            *_, r = run_batch(env, lam, None, bs, record=True)
            rows += [(x, y, d, lam) for x, y, d in r]
        return rows
    return job, [r for rs in walk(job, fn) for r in rs]


class Pol:
    def __init__(self, model, guard):
        self.model, self.guard = model, guard

    def __call__(self, x, mask):
        p = self.model.predict_proba(x[None, :])[0]
        cls = list(self.model.classes_)
        allowed = [1] + list(mask)
        best = max(range(4), key=lambda lv: (p[cls.index(lv)] if lv in cls and allowed[lv]
                                             else -1.0))
        return best


_MODELS = None


def _init_models(path):
    global _MODELS
    _MODELS = pickle.load(open(path, "rb"))["models"]


def evaluate(job):
    name = job[0]
    models = _MODELS[name]

    def fn(env, bs):
        r = {"k": len(env.per_sub)}
        for lam in LAMS:
            for tag, pol in [("seq", None)] + [
                    (f"{m}{'_g' if g else ''}", Pol(models[m], g))
                    for m in ("hgb", "mlp") for g in (False, True)]:
                gain, ch, ms, fms, na, _ = run_batch(env, lam, pol, bs, record=False)
                r[f"{tag}_{lam}"] = gain
                r[f"ms_{tag}_{lam}"] = ms
                r[f"fms_{tag}_{lam}"] = fms
                r[f"na_{tag}_{lam}"] = na
                r[f"pilih_{tag}_{lam}"] = ch
        return r
    rows = walk(job, fn)
    for r in rows:
        r.update(instans=job[0], kapasitas=job[1], intensitas=job[2], seed=job[3])
    return rows


def train(data_path, model_path):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    data = pickle.load(open(data_path, "rb"))
    X = np.stack([x for x, *_ in data["rows"]])
    y = np.array([r[1] for r in data["rows"]])
    D = np.array([r[2] for r in data["rows"]])     # J(tingkat) - J(cur), nan = tidak sah
    g = np.array(data["inst"])
    makers = {
        "tree4": lambda: DecisionTreeClassifier(max_depth=4, min_samples_leaf=50,
                                                random_state=0),
        "logreg": lambda: make_pipeline(StandardScaler(),
                                        LogisticRegression(max_iter=2000, C=1.0)),
        "mlp": lambda: make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(64, 64), max_iter=300, early_stopping=True,
            random_state=0)),
        "hgb": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                      random_state=0),
    }
    models, report = {}, []
    for test in sorted(set(g)):
        tr, te = g != test, g == test
        models[test] = {}
        for name, mk in makers.items():
            m = mk().fit(X[tr], y[tr])
            models[test][name] = m
            pred = m.predict(X[te])
            acc = float((pred == y[te]).mean())
            # Penyesalan satu langkah di jalur SEQ: J(pilihan) - J(pilihan SEQ).
            dp = D[te][np.arange(te.sum()), pred]
            dp = np.where(np.isnan(dp), 0.0, dp)
            ds = D[te][np.arange(te.sum()), y[te]]
            report.append((test, name, acc, float(np.mean(dp - ds))))
        maj = np.bincount(y[tr]).argmax()
        report.append((test, f"mayoritas({maj})", float((y[te] == maj).mean()), float("nan")))
    pickle.dump({"models": models, "report": report}, open(model_path, "wb"))
    import pandas as pd
    rep = pd.DataFrame(report, columns=["uji", "model", "akurasi", "sesal_langkah"])
    print(rep.groupby("model")[["akurasi", "sesal_langkah"]].mean().round(3))
    print("sebaran label:", np.bincount(y) / len(y))


if __name__ == "__main__":
    if MODE == "kumpul":
        with Pool(8) as pool:
            res = pool.map(collect, episodes(), chunksize=1)
        rows, inst = [], []
        for job, rs in res:
            rows += rs
            inst += [job[0]] * len(rs)
        pickle.dump({"rows": rows, "inst": inst}, open(ARGS[0], "wb"))
        print(len(rows))
    elif MODE == "latih":
        train(ARGS[0], ARGS[1])
    elif MODE == "nilai":
        import json
        with Pool(8, initializer=_init_models, initargs=(ARGS[0],)) as pool:
            res = pool.map(evaluate, episodes(), chunksize=1)
        json.dump([r for rs in res for r in rs], open(ARGS[1], "w"))
        print("selesai")
