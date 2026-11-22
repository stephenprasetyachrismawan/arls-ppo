"""E7: latih boosting (HGB) berbobot rugi sebagai pembanding "klasifikasi biasa" untuk PPO.

Protokolnya sama dengan PPO: dilatih di data guru seed 7000-7001 pada 7 instans latih
(boleh ditambah instans acak dari E3), lalu diuji di episode dengan seed validasi 5000-5001
lewat ``cek_episode.py`` metode ``il`` / ``il_g``. Model yang sama dipakai untuk semua
instans (tidak LOIO), sama seperti PPO.

Bobot contoh sama dengan BC di E1 (selisih J tingkat terburuk dan terbaik, dipotong p95).

Pemakaian (dari folder ini)::

    $PY latih_hgb.py keluaran.pkl bc_data_7000.pkl [bc_data_acak50.pkl ...]
"""
from __future__ import annotations

import pickle
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from loio_e2 import load, regret_of, weights_of

MK_TRAIN = ("MK01", "MK02", "MK05", "MK06", "MK07", "MK08", "MK10")


def main():
    out, paths = sys.argv[1], sys.argv[2:]
    parts = [load(p) for p in paths]
    X = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    D = np.concatenate([p[2] for p in parts])
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)
    m.fit(X, y, sample_weight=weights_of(D))
    pred = m.predict(X)
    print(f"{len(X)} contoh, tepat latih {(pred == y).mean():.3f}, "
          f"rugi latih {regret_of(D, pred).mean():.3f}")
    # cek_episode.py membaca _MODELS[nama instans]["hgb"].
    pickle.dump({"models": {n: {"hgb": m} for n in MK_TRAIN}, "sumber": paths},
                open(out, "wb"))


if __name__ == "__main__":
    main()
