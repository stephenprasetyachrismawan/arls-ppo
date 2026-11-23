"""E2: leave-one-instance-out (LOIO) untuk membandingkan dua set fitur, tanpa episode.

Data guru (seed 7000-7001) sudah menyimpan J ketiga tingkat di setiap keputusan Sequential.
Untuk setiap instans latih: model dilatih di 6 instans lain, lalu diukur di instans itu.

- ketepatan: pilihan model sama dengan pilihan guru;
- rugi: J(pilihan model) - J(pilihan terbaik) di keputusan itu. Tingkat yang tidak sah
  dianggap tetap AOR (J sama dengan tingkat 0), seperti di episode.

Rugi dilaporkan per λ, rata-rata atas keputusan. Ukuran ini satu langkah di jalur guru, jadi
hanya dipakai untuk memilih rancangan fitur. Yang terpilih dicek ulang dengan episode.

Pemakaian (dari folder ini)::

    $PY loio_e2.py v1=bc_data_7000.pkl v2=bc_data_7000_v2.pkl > loio_e2.md
"""
from __future__ import annotations

import pickle
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

SEEDS = (0, 1, 2)
NET = 128


def load(path):
    d = pickle.load(open(path, "rb"))
    X = np.stack([r[0] for r in d["rows"]]).astype(np.float32)
    y = np.array([r[1] for r in d["rows"]])
    D = np.array([[np.nan if v is None else v for v in r[2]] for r in d["rows"]], float)
    lam = np.array([r[3] for r in d["rows"]])
    return X, y, D, lam, np.array(d["inst"])


def regret_of(D, pred):
    Dz = np.where(np.isnan(D), 0.0, D)          # tidak sah = tetap AOR (J sama dengan tingkat 0)
    return Dz[np.arange(len(D)), pred] - Dz.min(1)


def weights_of(D):
    """Bobot rugi seperti E1: selisih tingkat terburuk dan terbaik, dipotong p95."""
    Dz = np.where(np.isnan(D), 0.0, D)
    w = Dz.max(1) - Dz.min(1)
    w = np.minimum(w, np.percentile(w, 95))
    return w / max(1e-9, w.mean()) + 0.05


# MLP_NORM=kuantil: tiap fitur diubah ke peringkatnya lalu dipetakan ke sebaran normal
# (QuantileTransformer), seperti praproses jaringan saraf pada Grinsztajn dkk. (2022).
# Nilai di luar rentang data latih menjadi ujung sebaran, bukan angka besar yang ditebak
# sembarangan oleh jaringan. Bawaan: z-score seperti PPO sekarang.
# MLP_NORM=ple (E11): piecewise-linear encoding (Gorishniy dkk., 2022). Setiap fitur dipecah
# menjadi PLE_SEKAT sekat dengan batas kuantil data latih; tiap sekat bernilai 0 (di bawah),
# 1 (di atas), atau bagian linear di antaranya. Fitur dengan sedikit nilai berbeda (misalnya
# bit mask) memakai sekat sebanyak nilai uniknya.
MLP_NORM = __import__("os").environ.get("MLP_NORM", "z")
PLE_SEKAT = int(__import__("os").environ.get("PLE_SEKAT", "16"))


def _ple(X, k):
    edges = []
    for j in range(X.shape[1]):
        e = np.unique(np.quantile(X[:, j], np.linspace(0, 1, k + 1)))
        if len(e) < 2:
            e = np.array([e[0], e[0] + 1.0])
        edges.append(e)

    def enc(q):
        out = []
        for j, e in enumerate(edges):
            lo, hi = e[:-1], e[1:]
            t = (q[:, j:j + 1] - lo) / (hi - lo)
            out.append(np.clip(t, 0.0, 1.0))
        return np.concatenate(out, 1).astype(np.float32)
    return enc


def _scaler(X, seed):
    if MLP_NORM == "ple":
        return _ple(X, PLE_SEKAT)
    if MLP_NORM == "kuantil":
        from sklearn.preprocessing import QuantileTransformer
        qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                                 subsample=200_000, random_state=seed).fit(X)
        return lambda q: qt.transform(q).astype(np.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    return lambda q: np.clip((q - mu) / sd, -5, 5).astype(np.float32)


def fit_mlp(X, y, D, seed, epochs=200, patience=10):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    scale = _scaler(X, seed)
    Xn = scale(X)
    idx = rng.permutation(len(X))
    n_hold = len(X) // 10
    te, tr = idx[:n_hold], idx[n_hold:]
    w = torch.tensor(weights_of(D), dtype=torch.float32)
    Xt, yt = torch.tensor(Xn, dtype=torch.float32), torch.tensor(y)
    net = torch.nn.Sequential(torch.nn.Linear(Xn.shape[1], NET), torch.nn.Tanh(),
                              torch.nn.Linear(NET, NET), torch.nn.Tanh(),
                              torch.nn.Linear(NET, 4))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    best, best_state, wait = np.inf, None, 0
    for _ in range(epochs):
        net.train()
        for b in np.array_split(rng.permutation(tr), max(1, len(tr) // 256)):
            loss = (torch.nn.functional.cross_entropy(net(Xt[b]), yt[b], reduction="none")
                    * w[b]).sum() / w[b].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            p = net(Xt[te]).argmax(1).numpy()
        rg = regret_of(D[te], p).mean()
        if rg < best - 1e-6:
            best, best_state, wait = rg, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(best_state)

    def predict(Xq):
        with torch.no_grad():
            q = torch.tensor(scale(Xq), dtype=torch.float32)
            return net(q).argmax(1).numpy()
    return predict


def fit_hgb(X, y, D, seed):
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed)
    m.fit(X, y, sample_weight=weights_of(D))
    return m.predict


DATA = {}


def _init(sets):
    """Setiap proses anak memuat data sendiri (Python 3.14 tidak lagi memakai fork)."""
    DATA.update({k: load(v) for k, v in sets.items()})


def _job(args):
    """Satu kali latih dan ukur: (instans uji, set fitur, model, seed)."""
    test, k, model, s = args
    torch.set_num_threads(1)
    X, y, D, lam, g = DATA[k]
    tr, te = g != test, g == test
    fit = fit_mlp if model == "mlp" else fit_hgb
    pred = fit(X[tr], y[tr], D[tr], s)(X[te])
    rg = regret_of(D[te], pred)
    print(f"{test} {k} {model} s{s}: tepat {(pred == y[te]).mean():.3f} rugi {rg.mean():.3f}",
          file=sys.stderr, flush=True)
    return [dict(fitur=k, model=model, seed=s, uji=test, lam=l,
                 tepat=float((pred[lam[te] == l] == y[te][lam[te] == l]).mean()),
                 rugi=float(rg[lam[te] == l].mean())) for l in sorted(set(lam))]


def main():
    from multiprocessing import Pool
    sets = dict(a.split("=") for a in sys.argv[1:])
    DATA.update({k: load(v) for k, v in sets.items()})
    ref = next(iter(DATA.values()))
    for k, (X, y, D, lam, g) in DATA.items():
        # Jalur guru tidak bergantung pada fitur, jadi label dan J harus sama persis.
        assert (y == ref[1]).all() and (g == ref[4]).all() and np.allclose(
            np.nan_to_num(D, nan=1e9), np.nan_to_num(ref[2], nan=1e9)), k
    jobs = [(t, k, m, s) for t in sorted(set(ref[4])) for k in sets
            for m, ss in (("mlp", SEEDS), ("hgb", (0,))) for s in ss]
    with Pool(8, initializer=_init, initargs=(sets,)) as pool:
        rows = [r for rs in pool.map(_job, jobs, chunksize=1) for r in rs]
    R = pd.DataFrame(rows)
    R.to_csv("loio_e2.csv", index=False)
    report(R, list(sets))


def report(R, names):
    print("# E2: fitur lama lawan fitur tanpa satuan, leave-one-instance-out\n")
    print("Data guru seed 7000-7001, 7 instans latih. Model dilatih di 6 instans, diukur di "
          "instans ketujuh. Rugi = J(pilihan model) - J(pilihan terbaik) per keputusan, "
          "rata-rata. MLP: rata-rata 3 seed.\n")
    G = R.groupby(["fitur", "model", "uji", "lam"])[["tepat", "rugi"]].mean().reset_index()
    for model in ("mlp", "hgb"):
        M = G[G.model == model]
        print(f"## {model.upper()}\n")
        print("| fitur | " + " | ".join(f"tepat λ {l:g}" for l in sorted(M.lam.unique()))
              + " | " + " | ".join(f"rugi λ {l:g}" for l in sorted(M.lam.unique())) + " |")
        print("|---" * (1 + 2 * M.lam.nunique()) + "|")
        for f in names:
            a = M[M.fitur == f].groupby("lam")[["tepat", "rugi"]].mean()
            print(f"| {f} | " + " | ".join(f"{v:.3f}" for v in a.tepat) + " | "
                  + " | ".join(f"{v:.3f}" for v in a.rugi) + " |")
        if len(names) == 2:
            a, b = names
            P = M.pivot_table(index=["uji", "lam"], columns="fitur", values="rugi")
            rel = (P[b] - P[a]) / P[a].where(P[a] > 0) * 100
            rng = np.random.default_rng(0)
            print(f"\nSelisih rugi {b} terhadap {a} per λ (negatif = {b} lebih baik), "
                  "selang 95% bootstrap atas 7 instans:\n")
            for l, v in rel.groupby(level="lam"):
                v = v.dropna().to_numpy()
                bt = v[rng.integers(0, len(v), (2000, len(v)))].mean(1)
                lo, hi = np.percentile(bt, [2.5, 97.5])
                print(f"- λ {l:g}: {v.mean():+.1f}% [{lo:+.1f}; {hi:+.1f}]"
                      f"{' *' if lo > 0 or hi < 0 else ''}".replace(".", ","))
            print("\nPer instans (rugi rata-rata ketiga λ):\n")
            Q = M.groupby(["uji", "fitur"]).rugi.mean().unstack()
            print("| instans | " + " | ".join(names) + " |")
            print("|---" * (1 + len(names)) + "|")
            for i, r in Q.iterrows():
                print(f"| {i} | " + " | ".join(f"{r[n]:.3f}" for n in names) + " |")
        print()


if __name__ == "__main__":
    main()
