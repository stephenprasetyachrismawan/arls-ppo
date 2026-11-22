"""Perbandingan gabungan untuk data uji (``banding_umum.bandingkan_gabungan``).

Keputusan pemilik 18 November 2026 (docs/KUNCI_UJI.md bagian 6-8): tiap
instans berbobot sama untuk kedua ukuran; bootstrap gugus (instans, intensitas, seed)
diambil ulang di dalam tiap instans; tingkat selang bisa 98,75% untuk keluarga 4 uji
utama; kondisi dengan J pembanding = 0 dikeluarkan dari ukuran 1 dan dihitung.

    PYTHONPATH=../../../exp1/dfjsp-ppo-repair/src $PY -m pytest -p no:cacheprovider -q test_banding_gabungan.py
"""
from __future__ import annotations

import pandas as pd

from banding_umum import bandingkan_gabungan


def _df(baris):
    """baris: (instans, intensitas, seed, J_a, J_b) pada kapasitas 1,0 dan λ 3."""
    out = []
    for inst, inten, seed, ja, jb in baris:
        for metode, j in (("a", ja), ("b", jb)):
            out.append(dict(instans=inst, kapasitas=1.0, intensitas=inten, seed=seed,
                            lam=3.0, metode=metode, J=j))
    return pd.DataFrame(out)


def test_tiap_instans_berbobot_sama_untuk_kedua_ukuran():
    # Instans A: 2 kondisi, a 20% lebih buruk, J kecil. Instans B: 4 kondisi, a sama, J besar.
    # Bobot sama per instans: kedua ukuran = rata-rata (1,2; 1,0) - 1 = +10%.
    # (Tanpa bobot per instans, uk. 1 = 6,67% dan uk. 2 = 0,05%.)
    df = _df([("A", 0.05, s, 12.0, 10.0) for s in (1, 2)]
             + [("B", 0.05, s, 1000.0, 1000.0) for s in (1, 2, 3, 4)])
    h = bandingkan_gabungan(df, "a", "b", n_resamples=200)[3.0]
    assert abs(h.rasio_rata.estimate - 10.0) < 1e-9
    assert abs(h.rasio_total.estimate - 10.0) < 1e-9


def test_gugus_diambil_ulang_di_dalam_tiap_instans():
    # Di dalam tiap instans rasionya sama semua (A 1,2; B 1,0). Bila setiap sampel ulang
    # selalu memuat kedua instans, rata-rata per instans tidak pernah berubah: selang nol.
    # A hanya punya satu gugus dari sepuluh; tanpa strata, sekitar 35% sampel ulang tidak
    # memuat A sama sekali (0,9 pangkat 10), dan batas bawah selang jatuh ke 0%.
    df = _df([("A", 0.05, 1, 12.0, 10.0)]
             + [("B", i, s, 50.0, 50.0) for i in (0.05, 0.15, 0.30) for s in (1, 2, 3)])
    h = bandingkan_gabungan(df, "a", "b", n_resamples=300)[3.0]
    for u in (h.rasio_rata, h.rasio_total):
        assert abs(u.low - 10.0) < 1e-9 and abs(u.high - 10.0) < 1e-9


def test_selang_98_75_lebih_lebar_dari_95():
    import numpy as np

    rng = np.random.default_rng(1)
    df = _df([(f"I{k}", i, s, 100 * rng.uniform(0.8, 1.3), 100.0)
              for k in range(4) for i in (0.05, 0.15) for s in range(5)])
    h95 = bandingkan_gabungan(df, "a", "b", n_resamples=2000)[3.0]
    h99 = bandingkan_gabungan(df, "a", "b", tingkat_selang=0.9875, n_resamples=2000)[3.0]
    for u95, u99 in ((h95.rasio_rata, h99.rasio_rata), (h95.rasio_total, h99.rasio_total)):
        assert u99.low < u95.low and u99.high > u95.high


def test_j_pembanding_nol_dikeluarkan_dari_ukuran_1_dan_dihitung():
    # A: satu kondisi biasa (rasio 1,2) dan satu kondisi J_b = 0. B: rasio 1,0.
    # C: semua J_b = 0, jadi dikeluarkan dari kedua ukuran.
    # uk. 1 = rata-rata (1,2; 1,0) - 1 = +10%. uk. 2: A = 17/10 = 1,7, B = 1,0, jadi +35%.
    df = _df([("A", 0.05, 1, 12.0, 10.0), ("A", 0.05, 2, 5.0, 0.0),
              ("B", 0.05, 1, 50.0, 50.0), ("B", 0.05, 2, 50.0, 50.0),
              ("C", 0.05, 1, 3.0, 0.0), ("C", 0.05, 2, 4.0, 0.0)])
    h = bandingkan_gabungan(df, "a", "b", n_resamples=200)[3.0]
    assert abs(h.rasio_rata.estimate - 10.0) < 1e-9
    assert abs(h.rasio_total.estimate - 35.0) < 1e-9
    assert h.dikeluarkan_uk1 == 3
    assert h.instans_dikeluarkan == ("C",)


def test_celah_gabungan_berbobot_sama_per_instans_dan_konstan_tanpa_ragam():
    """Butir 1 di data uji: celah a ke b di λ sasaran dikurangi celah di λ acuan, tiap
    instans berbobot sama. A punya 4 kondisi, B 2. Celah λ 0,3: A dan B +10%; celah λ 0,1:
    A +20%, B +40%. Bobot sama: (20 + 40) / 2 - 10 = +20 poin (tanpa bobot: +16,7).
    Di dalam tiap instans rasionya sama, jadi selangnya nol (strata instans)."""
    from banding_umum import bandingkan_celah_gabungan

    rasio = {("A", 0.3): 1.1, ("B", 0.3): 1.1, ("A", 0.1): 1.2, ("B", 0.1): 1.4}
    out = []
    for inst, seeds in (("A", (1, 2, 3, 4)), ("B", (1, 2))):
        for s in seeds:
            for lam in (0.1, 0.3):
                for m, j in (("a", 100.0 * rasio[inst, lam]), ("b", 100.0)):
                    out.append(dict(instans=inst, kapasitas=1.0, intensitas=0.05, seed=s,
                                    lam=lam, metode=m, J=j))
    h = bandingkan_celah_gabungan(pd.DataFrame(out), "a", "b", {0.1: 1.0}, {0.3: 1.0},
                                  n_resamples=200)
    for x in (h.rasio_rata, h.rasio_total):
        assert abs(x.estimate - 20.0) < 1e-9
        assert abs(x.low - 20.0) < 1e-9 and abs(x.high - 20.0) < 1e-9
