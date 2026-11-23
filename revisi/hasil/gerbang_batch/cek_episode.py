"""Uji tingkat episode: setiap metode memasang keputusannya sendiri sampai episode selesai.

Berbeda dengan cek_b1_b2.py (satu langkah di jalur AOR), di sini jalur episode mengikuti
pilihan metode, jadi akibat jangka panjang ikut terhitung. Hanya instans latih, seed
validasi 5000-5001. Keputusan tunggal (k = 1) diperlakukan sebagai batch berisi satu.

Metode:
- aor      : selalu tingkat 0;
- b1       : satu tingkat per kejadian, J terkecil (v3, batas tugas 3/8);
- b1w      : sama, batas tugas dikali k;
- seq      : Sequential Lookahead per gangguan, urutan datang;
- seq_parah: sama, urutan dari gangguan dengan keterlambatan A_j terbesar;
- fix1..fix3 (_g): aturan tetap tanpa PPO, selalu tingkat L bila sah;
- thr        : Threshold tesis lama per kejadian; edd, ga: jadwal ulang total (EDD, GA-PR);
- acak (_g)  : tingkat acak di antara tingkat sah (tanpa belajar);
- il, il_g : penggolong HGB; mlp, mlp_g: penggolong MLP 64x64 (bentuk jaringan PPO);
            keduanya dari B3 (model yang tidak melihat instans ini), tanpa probe;
             _g = tolak pilihan yang memperburuk J.

J episode = telat akhir / E + lambda x total perubahan.

Pemakaian: $PY cek_episode.py $SRC b3_model.pkl keluaran.json
"""
from __future__ import annotations

import penjaga_uji  # noqa: F401  (memblokir pembukaan data uji)

import copy
import json
import pickle
import sys
import time
from multiprocessing import Pool

SRC, MODEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sys.argv = [sys.argv[0], "nilai", SRC, MODEL, OUT]
sys.path.insert(0, SRC)

import cek_b3  # noqa: E402
from cek_b0_mask import CAPS, INTENSITIES, SEEDS, Probe  # noqa: E402
from cek_b1_b2 import LAMS, ops_for, score, try_level  # noqa: E402
from dfjsp.core.domain import set_capacity  # noqa: E402
from dfjsp.datasets import BRANDIMARTE_TRAIN, load_brandimarte  # noqa: E402

import instans_acak  # noqa: E402
from dfjsp.env.repair_env import RepairEnv  # noqa: E402
from dfjsp.env.schedule import recompute_timeline  # noqa: E402
from dfjsp.env.state import total_tardiness  # noqa: E402

import os  # noqa: E402

METHODS = tuple(os.environ.get("METODE", "aor,b1,b1w,seq,seq_parah,mlp,mlp_g").split(","))
PPO_DIR = os.environ.get("PPO_DIR")  # folder berisi norm.pkl dan model .zip
_PPO = {}


class PPOPol:
    """Kebijakan MaskablePPO terlatih (deterministik) di atas fitur cek_b3."""

    def __init__(self, name, guard, nomask=False):
        self.nomask = nomask
        if name not in _PPO:
            import torch
            from sb3_contrib import MaskablePPO
            torch.set_num_threads(1)
            nrm = pickle.load(open(f"{PPO_DIR}/norm.pkl", "rb"))
            _PPO[name] = (MaskablePPO.load(f"{PPO_DIR}/{name}.zip", device="cpu"), nrm)
        self.model, self.nrm = _PPO[name]
        self.guard = guard

    def __call__(self, x, mask):
        import numpy as np
        obs = np.clip((x - self.nrm["mu"]) / self.nrm["sd"], -5, 5).astype("float32")
        allowed = (np.ones(4, dtype=bool) if self.nomask
                   else np.array([1] + list(mask), dtype=bool))
        a, _ = self.model.predict(obs, action_masks=allowed, deterministic=True)
        return int(a)
class RulePol:
    """Pembanding tanpa PPO: tingkat tetap L (fix<L>) atau acak di antara tingkat sah (acak).

    Bila tingkat tetap tidak sah menurut mask lapis atas, pakai tingkat 0.
    """

    def __init__(self, kind, guard, rng=None):
        self.kind, self.guard, self.rng = kind, guard, rng

    def __call__(self, x, mask):
        allowed = [1] + list(mask)
        if self.kind == "acak":
            return int(self.rng.choice([lv for lv in range(4) if allowed[lv]]))
        lv = int(self.kind[3:])
        return lv if allowed[lv] else 0


_MODELS = None


def _init(path):
    global _MODELS
    _MODELS = pickle.load(open(path, "rb"))["models"]
    cek_b3._MODELS = _MODELS


class EpProbe(Probe):
    pengamat = None   # diisi run() bila PERIKSA=1 (gerbang 2.5)

    def _apply_event(self, event):
        if self.pengamat is not None:
            self.pengamat.catat_gangguan(event)
        super()._apply_event(event)
        if self.per_sub is None:
            self.per_sub = [(event.type, set(self.sched.affected))]


def one_level(env, lam, scale):
    inst, base, t, E = env.inst, env.sched, env.t_now, env.mean_effort
    o = ops_for(scale)
    best = (total_tardiness(inst, base) / E, None, 0)
    for lv in (1, 2, 3):
        tr = try_level(inst, base, t, base.affected, o[lv])
        if tr is None:
            continue
        j, c = score(inst, base, tr, t, E, lam)
        if j < best[0] - 1e-9:
            best = (j, tr, c)
    return best[1], best[2]


def prune_batch(env, lam, model, thr):
    """SEQ dengan pemangkasan: lewati probe untuk d_j bila P(SEQ pilih 0) >= thr.

    Model hanya memutuskan kapan probe boleh dilewati; pilihan tingkat tetap dari probe.
    """
    inst, base, t, E = env.inst, env.sched, env.t_now, env.mean_effort
    A = [a for _, a in env.per_sub]
    types = [ty for ty, _ in env.per_sub]
    bs = RepairEnv._state(env)
    cur, cur_j, chg = base, total_tardiness(inst, base) / E, 0
    cls = list(model.classes_)
    for j in range(len(A)):
        x, mask = cek_b3.features(env, cur, bs, j, A, types, lam, chg)
        if model.predict_proba(x[None, :])[0][cls.index(0)] >= thr:
            continue
        best = (cur_j, cur, chg)
        for lv in (1, 2, 3):
            if not mask[lv - 1]:
                continue
            tr = try_level(inst, cur, t, A[j], cek_b3.OPS[lv])
            if tr is None:
                continue
            jj, c = score(inst, base, tr, t, E, lam)
            if jj < best[0] - 1e-9:
                best = (jj, tr, c)
        cur_j, cur, chg = best
    return (None if cur is base else cur), chg


def ppo_probe_batch(env, lam, name):
    """PPO yang diberi hasil probe (latih_ppo_probe.py): coba tiap tingkat, PPO memilih."""
    import numpy as np
    from probe_fitur import probe_all, probe_features
    if name not in _PPO:
        import torch
        from sb3_contrib import MaskablePPO
        torch.set_num_threads(1)
        nrm = pickle.load(open(f"{PPO_DIR}/norm.pkl", "rb"))
        _PPO[name] = (MaskablePPO.load(f"{PPO_DIR}/{name}.zip", device="cpu"), nrm)
    model, nrm = _PPO[name]
    inst, base, E = env.inst, env.sched, env.mean_effort
    A = [a for _, a in env.per_sub]
    types = [ty for ty, _ in env.per_sub]
    bs = RepairEnv._state(env)
    cur, cur_j, chg = base, total_tardiness(inst, base) / E, 0
    for j in range(len(A)):
        x, mask = cek_b3.features(env, cur, bs, j, A, types, lam, chg)
        dj, trials = probe_all(env, cur, cur_j, base, A[j], mask, lam)
        obs = np.clip((np.concatenate([x, probe_features(dj)]) - nrm["mu"]) / nrm["sd"],
                      -5, 5).astype("float32")
        allowed = np.array([True] + [lv in trials for lv in (1, 2, 3)])
        a, _ = model.predict(obs, action_masks=allowed, deterministic=True)
        if int(a) > 0:
            cur_j, cur, chg = trials[int(a)]
    return (None if cur is base else cur), chg


def threshold_level(env, tau1=0.2, tau2=0.5, tau3=0.3, tau4=0.8):
    """Aturan Threshold tesis lama (Persamaan 3.13), strategi dipetakan ke tingkat tangga:
    Maintain -> 0, Delay -> 1, Reassign -> 2, Escalate -> 3. Ambang = nilai bawaan."""
    from dfjsp.env.state import INTENSITY, OVERLOAD
    st = RepairEnv._state(env)
    inten, over = float(st[INTENSITY]), float(st[OVERLOAD])
    if inten >= tau4:
        return 3
    if over > tau3:
        return 2
    if tau1 <= inten < tau2:
        return 1
    return 0


def rule_batch(env, lam, method, taus=None):
    """Pembanding satu keputusan per kejadian tanpa probe.

    thr     : Threshold, tingkat dari aturan ambang, batas tugas dikali k;
    edd, ga : penjadwalan ulang semua tugas yang belum mulai (heuristik EDD, GA-PR).
    """
    inst, base, t, E = env.inst, env.sched, env.t_now, env.mean_effort
    if method == "thr":
        lv = threshold_level(env, *taus) if taus else threshold_level(env)
        if lv == 0:
            return None, 0
        op = ops_for(len(env.per_sub))[lv]
    elif method == "edd":
        from dfjsp.reschedulers._operators import RuleRepair
        op = RuleRepair("edd")
    else:
        from dfjsp.reschedulers._operators import GARepair
        op = GARepair(random_state=0)
    tr = try_level(inst, base, t, base.affected, op)
    if tr is None:
        return None, 0
    _, c = score(inst, base, tr, t, E, lam)
    return tr, c


def ppo_event_batch(env, lam, name):
    """Ablasi PPO per kejadian: satu tingkat untuk seluruh kejadian (kejadian_fitur.py)."""
    import numpy as np
    from kejadian_fitur import apply_level, event_features
    if name not in _PPO:
        import torch
        from sb3_contrib import MaskablePPO
        torch.set_num_threads(1)
        nrm = pickle.load(open(f"{PPO_DIR}/norm.pkl", "rb"))
        _PPO[name] = (MaskablePPO.load(f"{PPO_DIR}/{name}.zip", device="cpu"), nrm)
    model, nrm = _PPO[name]
    x, mask, union, k = event_features(env, lam)
    obs = np.clip((x - nrm["mu"]) / nrm["sd"], -5, 5).astype("float32")
    a, _ = model.predict(obs, action_masks=np.array([1] + list(mask), dtype=bool),
                         deterministic=True)
    tr, c, _ = apply_level(env, lam, union, k, int(a))
    return tr, c


def decide(env, method, lam):
    if method.startswith("ppok_"):
        # ppok_<model>, misalnya ppok_kejadian/bc_ppo
        return ppo_event_batch(env, lam, method[5:])
    if method in ("thr", "edd", "ga"):
        return rule_batch(env, lam, method)
    if method.startswith("thr:"):
        # thr:<tau1>;<tau2>;<tau3>;<tau4>, Threshold dengan ambang yang disetel (E7)
        return rule_batch(env, lam, "thr", [float(v) for v in method[4:].split(";")])
    if method.startswith("ppop_"):
        # ppop_<model>, misalnya ppop_bc, ppop_bc_ppo
        return ppo_probe_batch(env, lam, method[5:])
    if method.startswith("prune"):
        # prune_<ambang x100>, misalnya prune_80
        return prune_batch(env, lam, instans_acak.model_instans(_MODELS, env.inst.name)["mlp"], int(method[6:]) / 100)
    if method == "aor":
        return None, 0
    if method in ("luo", "luo_g"):
        # DQN gaya Luo (2020), luo_dqn.py; LUO_PATH = berkas .pt hasil latih
        from luo_dqn import apply_rule, policy_action
        return apply_rule(env, policy_action(os.environ["LUO_PATH"], env), lam,
                          guard=method == "luo_g")
    if method in ("b1", "b1w"):
        return one_level(env, lam, len(env.per_sub) if method == "b1w" else 1)
    if method == "seq_parah":
        by_id = {x.task_id: x for x in env.inst.tasks}
        sev = lambda a: sum(max(0.0, env.sched.entries[x].end - by_id[x].due_date)  # noqa: E731
                            for x in a[1] if x in env.sched.entries)
        env.per_sub = sorted(env.per_sub, key=sev, reverse=True)
    bs = RepairEnv._state(env)
    pol = None
    if method.startswith(("ppolam", "ppon_")):
        # ppolam[_g]: model per lambda di PPO_DIR/lam<lambda>/bc_ppo.zip
        # ppon_<model>[_g]: model tanpa mask lapis atas (semua tingkat boleh dipilih)
        guard = method.endswith("_g")
        if method.startswith("ppolam"):
            pol = PPOPol(f"lam{lam:g}/bc_ppo", guard)
        else:
            name = method[5:]
            pol = PPOPol(name[:-2] if guard else name, guard, nomask=True)
    elif method.startswith("ppo"):
        # ppo_<model>[_g], misalnya ppo_bc, ppo_bc_ppo_g
        name = method[4:]
        guard = name.endswith("_g")
        pol = PPOPol(name[:-2] if guard else name, guard)
    elif method.startswith(("fix", "acak")):
        guard = method.endswith("_g")
        kind = method[:-2] if guard else method
        pol = RulePol(kind, guard, getattr(env, "rule_rng", None))
    elif method in ("il", "il_g", "mlp", "mlp_g"):
        name = "mlp" if method.startswith("mlp") else "hgb"
        pol = cek_b3.Pol(instans_acak.model_instans(_MODELS, env.inst.name)[name],
                         method.endswith("_g"))
    gain, ch, ms, cur, chg = cek_b3.run_batch(env, lam, pol, bs, record=False,
                                             want_sched=True,
                                             catat=getattr(env, "catat_tingkat", None),
                                             waktu_pemilih=getattr(env, "waktu_pemilih", None))
    return (None if cur is env.sched else cur), chg


def run(job):
    name, cap, inten, seed, method, lam = job
    inst = copy.deepcopy(instans_acak.load_any(name))
    set_capacity(inst, cap)
    env = EpProbe(lam=lam)
    pengamat = None
    if os.environ.get("PERIKSA"):
        # Gerbang 2.5: periksa setiap keputusan dengan dfjsp.verify (pengamat.py).
        from pengamat import Pengamat
        pengamat = env.pengamat = Pengamat(method.endswith("_g") or method == "seq")
    env.reset(inst, intensity=inten, seed=seed)
    # Catatan tingkat dan penolakan penjaga untuk analisis sifat instans; hanya mencatat.
    env.catat_tingkat = cek_b3.catatan_tingkat_baru()
    env.waktu_pemilih = []   # ms satu panggilan pemilih per sub-keputusan (I-1); hanya mencatat
    import numpy as np
    env.rule_rng = np.random.default_rng(seed)   # untuk metode acak, sama tiap lambda
    changes, n, spent = 0, 0, 0.0
    waktu = []   # ms tiap keputusan, untuk p95 I-1; hanya mencatat
    while not env.done:
        sebelum = copy.deepcopy(env.sched) if pengamat is not None else None
        t0 = time.perf_counter()
        sched, chg = decide(env, method, lam)
        dt = time.perf_counter() - t0
        spent += dt
        waktu.append(1000 * dt)
        recompute_timeline(env.inst, env.sched, env.t_now)
        if pengamat is not None:
            sesudah, dipakai = (env.sched, 0) if sched is None else (sched, chg)
            E = env.mean_effort
            pengamat.amati(env.inst, sebelum, sesudah, env.t_now, dipakai,
                           total_tardiness(env.inst, sebelum) / E,
                           total_tardiness(env.inst, sesudah) / E + lam * dipakai)
        if sched is None:
            env.step(0)
        else:
            env.probes[3] = {"tard": total_tardiness(env.inst, sched), "changes": chg,
                             "entries": dict(sched.entries), "moves": []}
            env.step(3)
            changes += chg
        n += 1
    telat = total_tardiness(env.inst, env.sched) / env.mean_effort
    hasil = {"instans": name, "kapasitas": cap, "intensitas": inten, "seed": seed,
             "metode": method, "lam": lam, "telat": telat, "ubah": changes,
             "J": telat + lam * changes, "keputusan": n, "ms": 1000 * spent / max(1, n),
             "ms_keputusan": waktu, "ms_pemilih": env.waktu_pemilih,
             "tingkat": env.catat_tingkat}
    if pengamat is not None:
        hasil["periksa"] = pengamat.ringkas()
    return hasil


if __name__ == "__main__":
    # SEED_EP="7000" menjalankan episode di seed lain, misalnya seed guru untuk menyetel
    # ambang Threshold tanpa menyentuh seed validasi. Bawaan: seed validasi 5000-5001.
    if os.environ.get("SEED_EP"):
        SEEDS = tuple(int(v) for v in os.environ["SEED_EP"].split(","))
        assert not set(SEEDS) & set(range(10000, 10010)), "seed uji tidak boleh dipakai"
    # INSTANS_ACAK="W100-119" menjalankan episode di instans W validasi (E16), bukan 7 MK.
    nama = instans_acak.daftar_instans(BRANDIMARTE_TRAIN, ganti=os.environ.get("INSTANS_ACAK"))
    jobs = [(nm, c, i, s, m, lam) for nm in nama for c in CAPS
            for i in INTENSITIES for s in SEEDS for m in METHODS for lam in LAMS]
    # Pekerjaan berat (MK10, seq) duluan supaya pool tidak menunggu di akhir.
    jobs.sort(key=lambda j: (j[4] not in ("seq", "seq_parah"), j[0] != "MK10"))
    # Hasil ditulis per pekerjaan (jsonl) supaya kemajuan terlihat dan tidak hilang.
    done_keys = set()
    try:
        for line in open(OUT + "l"):
            r = json.loads(line)
            done_keys.add((r["instans"], r["kapasitas"], r["intensitas"], r["seed"],
                           r["metode"], r["lam"]))
    except FileNotFoundError:
        pass
    jobs = [j for j in jobs if j not in done_keys]
    t0 = time.time()
    with Pool(8, initializer=_init, initargs=(MODEL,)) as pool, open(OUT + "l", "a") as f:
        for n, r in enumerate(pool.imap_unordered(run, jobs, chunksize=1), 1):
            f.write(json.dumps(r) + "\n")
            f.flush()
            if n % 50 == 0 or n == len(jobs):
                el = time.time() - t0
                print(f"{n}/{len(jobs)} selesai, {el / 60:.1f} menit, sisa kira-kira "
                      f"{el / n * (len(jobs) - n) / 60:.1f} menit", flush=True)
    res = [json.loads(x) for x in open(OUT + "l")]
    json.dump(res, open(OUT, "w"))
    print(len(res))
