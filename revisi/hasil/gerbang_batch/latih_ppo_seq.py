"""B5 percobaan: PPO-Agg sekuensial, satu langkah = satu gangguan dalam kejadian.

Hanya instans latih. Seed episode latih diambil dari RNG pekerja (bukan 5000-5001 yang
dipakai validasi, bukan 10000-10009 milik uji). Keadaan = fitur cek_b3.features (tanpa
probe), dinormalisasi dengan rata-rata dan simpangan data B3. Mask = [1] + mask lapis atas
per gangguan. Reward langkah j = (J sebelum - J sesudah) / 10, jadi jumlah reward satu
kejadian = (J_AOR - J_akhir) / 10, sama dengan reward v3 untuk k = 1.

Dua tahap:
1. behaviour cloning dari label SEQ (data B3), pada jaringan kebijakan MaskablePPO;
2. PPO lanjutan dengan laju belajar kecil.

Pemakaian: $PY latih_ppo_seq.py $SRC b3_data.pkl folder_keluar langkah_ppo seed

Pilihan ablasi lewat variabel lingkungan (bawaan = metode usulan):
- NO_BC=1      : lewati behaviour cloning, PPO dari nol;
- NO_MASK=1    : tanpa mask lapis atas (semua tingkat boleh dipilih; tingkat tanpa calon
                 tidak mengubah jadwal);
- FIXED_LAM=x  : latih untuk satu lambda saja (BC hanya dari baris lambda itu).

Pilihan E1 (bawaan mati, supaya run lama tetap bisa diulang):
- BC_EPOCHS=n  : banyak epoch BC paling banyak (bawaan 8);
- BC_TAHAN=f   : sisihkan bagian f data guru untuk early stopping (misalnya 0.1);
                 berhenti kalau ketepatan di bagian itu tidak naik selama 10 epoch, lalu
                 kembali ke bobot terbaik;
- BC_BOBOT=rugi: setiap contoh ditimbang dengan selisih J tingkat terbaik dan terburuk pada
                 keputusan itu (dipotong di persentil 95, ditambah 0,05 supaya tidak nol),
                 sehingga keputusan yang semua tingkatnya sama baiknya hampir tidak dihitung;
- REWARD_LAM=1 : reward dibagi c(lambda), simpangan baku perbaikan J per langkah milik guru
                 pada lambda itu, diinterpolasi di skala log antara 0,3, 3, dan 30;
- NET=n        : lebar dua lapis jaringan (bawaan 64).

Pilihan E9 (perkuat PPO, bawaan mati):
- PPO_ACAK=a-b    : instans acak Ra-Rb, "Wa-b", atau "bench:latih" (instans_acak.py) ikut di tahap PPO bersama 7 MK,
                    masing-masing dengan 3 kapasitas x 3 intensitas;
- PENJAGA_LATIH=1 : penjaga di dalam lingkungan latih, sub-langkah yang memperburuk J
                    ditolak (jadwal tetap, reward 0), sama dengan penjaga saat dipakai;
- BC_TAMBAH=f     : data tambahan untuk BC (misalnya data DAgger), formatnya sama dengan
                    DATA; normalisasi tetap dari DATA.
Normalisasi selalu dari seluruh data B3, supaya semua varian memakai skala yang sama.
"""
from __future__ import annotations

import penjaga_uji  # noqa: F401  (memblokir pembukaan data uji)

import copy
import json
import math
import pickle
import sys
import time

import numpy as np

SRC, DATA, OUTDIR = sys.argv[1], sys.argv[2], sys.argv[3]
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 100_000
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 0
sys.argv = [sys.argv[0], "nilai", SRC, DATA, "/dev/null"]
sys.path.insert(0, SRC)

import gymnasium as gym  # noqa: E402
from gymnasium import spaces  # noqa: E402

from dfjsp.core.seeding import draw_training_episode_seed  # noqa: E402

import cek_b3  # noqa: E402
from cek_b1_b2 import score, try_level  # noqa: E402
from cek_episode import EpProbe  # noqa: E402
from dfjsp.core.domain import set_capacity  # noqa: E402
from dfjsp.datasets import BRANDIMARTE_TRAIN, load_brandimarte  # noqa: E402
from dfjsp.env.repair_env import RepairEnv  # noqa: E402
from dfjsp.env.schedule import recompute_timeline  # noqa: E402
from dfjsp.env.state import total_tardiness  # noqa: E402

LAM_RANGE = (0.1, 30.0)
REWARD_SCALE = 10.0
# Skala reward per lambda (E1, REWARD_LAM=1): c(lambda) diisi di __main__ dari data guru.
# Bawaan None berarti reward dibagi REWARD_SCALE yang tetap, seperti run lama.
REWARD_C = None
CLIP = 5.0
NO_MASK = bool(int(__import__("os").environ.get("NO_MASK", "0")))
PENJAGA_LATIH = __import__("os").environ.get("PENJAGA_LATIH", "0") == "1"


class Norm:
    def __init__(self, X):
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-6

    def __call__(self, x):
        return np.clip((x - self.mu) / self.sd, -CLIP, CLIP).astype(np.float32)


class SeqBatchEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, norm: Norm, worker_seed=0, lam=None, pool=None):
        super().__init__()
        self.norm = norm
        self.fixed_lam = lam
        self.rng = np.random.default_rng(worker_seed)
        insts = load_brandimarte(list(BRANDIMARTE_TRAIN))
        names = list(BRANDIMARTE_TRAIN)
        spec = __import__("os").environ.get("PPO_ACAK")
        if spec:
            import instans_acak
            for n in instans_acak.names_from_env(spec):
                insts[n] = instans_acak.load_any(n)
                names.append(n)
        self.pool = pool or [(n, c, i) for n in names for c in (1.0, 1.5, 2.0)
                             for i in (0.05, 0.15, 0.30)]
        self.insts = insts
        dim = len(norm.mu)
        self.observation_space = spaces.Box(-CLIP, CLIP, (dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self._mask = np.ones(4, dtype=bool)

    # ------------------------------------------------------------ bantuan
    def _start_decision(self):
        env = self.env
        self.A = [a for _, a in env.per_sub]
        self.types = [ty for ty, _ in env.per_sub]
        self.j = 0
        self.base = env.sched
        self.cur = env.sched
        self.chg = 0
        self.cur_j = total_tardiness(env.inst, env.sched) / env.mean_effort
        self.bs = RepairEnv._state(env)

    def _obs(self):
        x, mask = cek_b3.features(self.env, self.cur, self.bs, self.j, self.A, self.types,
                                  self.lam, self.chg)
        self._mask = (np.ones(4, dtype=bool) if NO_MASK
                      else np.array([1] + list(mask), dtype=bool))
        return self.norm(x)

    def _commit(self):
        env = self.env
        recompute_timeline(env.inst, env.sched, env.t_now)
        if self.cur is self.base:
            env.step(0)
        else:
            env.probes[3] = {"tard": total_tardiness(env.inst, self.cur),
                             "changes": self.chg, "entries": dict(self.cur.entries),
                             "moves": []}
            env.step(3)
            self.ep_changes += self.chg

    # ------------------------------------------------------------ gym
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        for _ in range(32):
            name, cap, inten = self.pool[int(self.rng.integers(len(self.pool)))]
            if self.fixed_lam is None:
                lo, hi = LAM_RANGE
                self.lam = float(math.exp(self.rng.uniform(math.log(lo), math.log(hi))))
            else:
                self.lam = float(self.fixed_lam)
            inst = copy.deepcopy(self.insts[name])
            set_capacity(inst, cap)
            self.env = EpProbe(lam=self.lam)
            self.env.reset(inst, intensity=inten, seed=draw_training_episode_seed(self.rng))
            if not self.env.done:
                break
        self.ep_changes = 0
        self._start_decision()
        return self._obs(), {"lam": self.lam}

    def step(self, action):
        a = int(action)
        env = self.env
        before = self.cur_j
        if a > 0 and self._mask[a]:
            tr = try_level(env.inst, self.cur, env.t_now, self.A[self.j], cek_b3.OPS[a])
            if tr is not None:
                jj, c = score(env.inst, self.base, tr, env.t_now, env.mean_effort, self.lam)
                if not (PENJAGA_LATIH and jj > self.cur_j + 1e-9):
                    self.cur, self.cur_j, self.chg = tr, jj, c
        scale = REWARD_SCALE if REWARD_C is None else REWARD_C(self.lam)
        reward = (before - self.cur_j) / scale
        self.j += 1
        done = False
        if self.j >= len(self.A):
            self._commit()
            if env.done:
                done = True
            else:
                self._start_decision()
        obs = np.zeros(self.observation_space.shape, np.float32) if done else self._obs()
        return obs, float(reward), done, False, {}

    def action_masks(self):
        return self._mask.copy()


def load_norm(data_path):
    data = pickle.load(open(data_path, "rb"))
    X = np.stack([r[0] for r in data["rows"]])
    y = np.array([r[1] for r in data["rows"]])
    return Norm(X), X, y


def load_rows(paths):
    """Baris data guru dari satu berkas atau lebih (dipisah koma)."""
    return [r for p in paths.split(",") for r in pickle.load(open(p, "rb"))["rows"]]


def regret_weights(data_path, keep=None):
    """Bobot rugi per contoh: selisih J tingkat terbaik dan terburuk yang sah."""
    rows = load_rows(data_path)
    D = np.array([[np.nan if v is None else v for v in r[2]] for r in rows], float)
    if keep is not None:
        D = D[keep]
    gap = np.nanmax(D, 1) - np.nanmin(D, 1)
    gap = np.nan_to_num(gap, nan=0.0)
    gap = np.minimum(gap, np.percentile(gap, 95))
    return (gap / max(gap.mean(), 1e-9) + 0.05).astype(np.float32)


def reward_scale_fn(data_path):
    """c(lambda) dari data guru: simpangan baku perbaikan J pada pilihan guru per lambda."""
    rows = pickle.load(open(data_path, "rb"))["rows"]
    lams = np.array([r[3] for r in rows])
    imp = np.array([-(r[2][r[1]] or 0.0) for r in rows], float)
    imp = np.nan_to_num(imp, nan=0.0)
    pts = sorted((float(l), float(imp[np.isclose(lams, l)].std()) + 1e-6)
                 for l in np.unique(lams))
    print("skala reward c(lambda):", [(l, round(c, 3)) for l, c in pts], flush=True)
    xl = np.log([p[0] for p in pts])
    yl = np.log([p[1] for p in pts])
    return lambda lam: float(np.exp(np.interp(math.log(lam), xl, yl)))


def regret_table(data_path, keep=None):
    """Selisih J tiap tingkat terhadap tingkat terbaik (0 = terbaik); tingkat tidak sah = 0."""
    rows = load_rows(data_path)
    D = np.array([[np.nan if v is None else v for v in r[2]] for r in rows], float)
    if keep is not None:
        D = D[keep]
    return np.nan_to_num(D - np.nanmin(D, 1, keepdims=True), nan=0.0)


def behaviour_cloning(model, Xn, y, epochs=8, lr=1e-3, seed=0, weights=None, hold=0.0,
                      regret=None):
    import copy as _copy

    import torch
    torch.manual_seed(seed)
    pol = model.policy
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    n_all = len(y)
    perm = rng.permutation(n_all)
    n_hold = int(n_all * hold)
    te, tr = perm[:n_hold], perm[n_hold:]
    X_t = torch.as_tensor(Xn)
    y_t = torch.as_tensor(y, dtype=torch.long)
    w_t = torch.as_tensor(weights if weights is not None else np.ones(n_all, np.float32))
    best, best_state, wait = -np.inf, None, 0
    for ep in range(epochs):
        idx = tr[rng.permutation(len(tr))]
        tot = 0.0
        for s in range(0, len(idx), 256):
            b = idx[s:s + 256]
            dist = pol.get_distribution(X_t[b])
            wb = w_t[b]
            loss = -(dist.log_prob(y_t[b]) * wb).sum() / wb.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        with torch.no_grad():
            pred = pol.get_distribution(X_t).distribution.probs.argmax(1)
            acc = float((pred[tr] == y_t[tr]).float().mean())
            acc_te = float((pred[te] == y_t[te]).float().mean()) if n_hold else acc
        # Ukuran berhenti: rugi J rata-rata di bagian yang disisihkan (kecil lebih baik).
        # Ketepatan menyesatkan, karena banyak keputusan semua tingkatnya sama baiknya.
        rg = (float(regret[te, pred[te].numpy()].mean()) if (n_hold and regret is not None)
              else None)
        print(f"BC epoch {ep}: loss {tot / len(tr):.3f} akurasi {acc:.3f} tahan {acc_te:.3f}"
              + (f" rugi tahan {rg:.3f}" if rg is not None else ""), flush=True)
        if n_hold:
            score_ = -rg if rg is not None else acc_te
            if score_ > best + 1e-4 or best_state is None:
                best, best_state, wait = score_, _copy.deepcopy(pol.state_dict()), 0
            else:
                wait += 1
                if wait >= 10:
                    print(f"BC berhenti di epoch {ep}, ukuran terbaik {best:.3f}", flush=True)
                    break
    if best_state is not None:
        pol.load_state_dict(best_state)


if __name__ == "__main__":
    import os

    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv

    torch.set_num_threads(1)
    os.makedirs(OUTDIR, exist_ok=True)
    norm, X, y = load_norm(DATA)
    BC_DATA = DATA
    if os.environ.get("BC_TAMBAH"):
        assert not os.environ.get("FIXED_LAM"), "BC_TAMBAH belum bisa dipakai dengan FIXED_LAM"
        # Normalisasi tetap dari DATA; BC memakai DATA + data tambahan.
        BC_DATA = DATA + "," + os.environ["BC_TAMBAH"]
        rows = load_rows(BC_DATA)
        X = np.stack([r[0] for r in rows])
        y = np.array([r[1] for r in rows])
        print(f"BC_TAMBAH: {len(rows)} baris BC", flush=True)
    fixed_lam = os.environ.get("FIXED_LAM")
    fixed_lam = float(fixed_lam) if fixed_lam else None
    if fixed_lam is not None:
        lams = np.array([r[3] for r in pickle.load(open(DATA, "rb"))["rows"]])
        keep = np.isclose(lams, fixed_lam)
        X, y = X[keep], y[keep]
        print(f"FIXED_LAM={fixed_lam}: {keep.sum()} baris BC", flush=True)
    pickle.dump({"mu": norm.mu, "sd": norm.sd}, open(f"{OUTDIR}/norm.pkl", "wb"))
    if os.environ.get("REWARD_LAM", "0") == "1":
        REWARD_C = reward_scale_fn(DATA)   # dipakai lingkungan (proses anak lewat fork)
    bc_w = None
    if os.environ.get("BC_BOBOT") == "rugi":
        bc_w = regret_weights(BC_DATA, keep if fixed_lam is not None else None)
    net = int(os.environ.get("NET", "64"))

    def make(i):
        return lambda: SeqBatchEnv(norm, worker_seed=1000 * SEED + i, lam=fixed_lam)

    venv = SubprocVecEnv([make(i) for i in range(8)], start_method="fork")
    model = MaskablePPO("MlpPolicy", venv, n_steps=256, batch_size=256, n_epochs=5,
                        learning_rate=1e-4, gamma=0.99, gae_lambda=0.95, ent_coef=0.003,
                        clip_range=0.1,
                        policy_kwargs=dict(net_arch=dict(pi=[net, net], vf=[net, net])),
                        seed=SEED, device="cpu", verbose=0)
    t0 = time.time()
    if os.environ.get("NO_BC", "0") != "1":
        behaviour_cloning(model, norm(X), y, seed=SEED,
                          epochs=int(os.environ.get("BC_EPOCHS", "8")), weights=bc_w,
                          hold=float(os.environ.get("BC_TAHAN", "0")),
                          regret=regret_table(BC_DATA, keep if fixed_lam is not None else None))
    model.save(f"{OUTDIR}/bc.zip")
    print(f"BC selesai {time.time() - t0:.0f} s", flush=True)

    class Curve(BaseCallback):
        def __init__(self):
            super().__init__()
            self.rows, self.buf = [], []

        def _on_step(self):
            self.buf.extend(float(r) for r in self.locals["rewards"])
            if self.num_timesteps % 4096 < 8:
                self.rows.append((self.num_timesteps, float(np.mean(self.buf[-4096:]))))
                print(f"langkah {self.num_timesteps}: reward rata {self.rows[-1][1]:.4f}",
                      flush=True)
            return True

    cb = Curve()
    if STEPS > 0:
        model.learn(total_timesteps=STEPS, callback=cb)
        model.save(f"{OUTDIR}/bc_ppo.zip")
    json.dump(cb.rows, open(f"{OUTDIR}/kurva.json", "w"))
    print(f"selesai {time.time() - t0:.0f} s", flush=True)
