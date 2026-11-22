"""Perbandingan berpasangan dua metode dengan bootstrap bergugus."""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from dfjsp.stats import BootstrapResult
from dfjsp.stats._bootstrap import make_column_sampler

KONDISI = ["instans", "kapasitas", "intensitas", "seed"]
# Kapasitas yang berbeda memakai urutan gangguan yang sama, jadi tidak bebas satu sama lain.
GUGUS = ["instans", "intensitas", "seed"]


class Banding(NamedTuple):
    """Selisih J metode a terhadap b dalam persen, dengan dua ukuran agregat.

    ``rasio_rata``: rata-rata (J_a / J_b - 1) per kondisi. ``rasio_total``: total J_a dibagi
    total J_b, dikurangi 1. Positif berarti a lebih buruk.
    """

    rasio_rata: BootstrapResult
    rasio_total: BootstrapResult


def _ukuran(ja, jb):
    return (np.mean(ja / jb) - 1) * 100, (ja.sum() / jb.sum() - 1) * 100


def bandingkan(df, a, b, *, n_resamples=10_000, random_state=0):
    rng = np.random.default_rng(random_state)
    hasil = {}
    for lam, g in df.groupby("lam"):
        t = g.pivot_table(index=KONDISI, columns="metode", values="J")[[a, b]]
        if t.isna().any().any():
            raise ValueError(f"λ {lam:g}: {int(t.isna().any(axis=1).sum())} kondisi hanya "
                             f"ada di salah satu metode ({a}, {b})")
        ja, jb = t[a].to_numpy(), t[b].to_numpy()
        if (jb <= 0).any():
            raise ValueError(f"λ {lam:g}: {int((jb <= 0).sum())} kondisi dengan J = 0 atau "
                             f"kurang untuk pembanding {b}; rasio tidak terdefinisi")
        gugus = t.index.to_frame()[GUGUS].astype(str).agg("|".join, axis=1).to_numpy()
        ambil = make_column_sampler(len(ja), groups=gugus)
        titik = _ukuran(ja, jb)
        boot = np.array([_ukuran(ja[i], jb[i]) for i in
                         (ambil(rng) for _ in range(n_resamples))])
        lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
        hasil[lam] = Banding(*(BootstrapResult(float(titik[k]), float(lo[k]), float(hi[k]))
                               for k in (0, 1)))
    return hasil


def _tabel(df, a, b, lam):
    t = df[df.lam == lam].pivot_table(index=KONDISI, columns="metode", values="J")[[a, b]]
    if t.isna().any().any():
        raise ValueError(f"λ {lam:g}: {int(t.isna().any(axis=1).sum())} kondisi hanya "
                         f"ada di salah satu metode ({a}, {b})")
    return t.sort_index()


def bandingkan_celah(df, a, b, sasaran, acuan, *, n_resamples=10_000, random_state=0):
    """Celah a ke b di λ sasaran dikurangi celah di λ acuan, dalam poin persen.

    Celah di satu λ = ukuran ``bandingkan`` (rata-rata rasio dan rasio total). ``sasaran`` dan
    ``acuan`` berupa {λ: bobot}; celah tiap sisi = jumlah bobot × celah. Gugus diambil ulang
    sekali untuk semua λ, karena kondisi yang sama dipakai di semua λ. Positif berarti celah
    di sasaran lebih besar (a lebih tertinggal dari b di sana).
    """
    rng = np.random.default_rng(random_state)
    bobot = dict(sasaran)
    for lam, w in acuan.items():
        bobot[lam] = bobot.get(lam, 0.0) - w
    tabel = {lam: _tabel(df, a, b, lam) for lam in bobot}
    indeks = next(iter(tabel.values())).index
    for lam, t in tabel.items():
        if not t.index.equals(indeks):
            raise ValueError(f"kondisi di λ {lam:g} tidak sama dengan kondisi di λ lain")
    ja = {lam: t[a].to_numpy() for lam, t in tabel.items()}
    jb = {lam: t[b].to_numpy() for lam, t in tabel.items()}

    def nilai(i):
        return sum(w * np.array(_ukuran(ja[lam][i], jb[lam][i])) for lam, w in bobot.items())

    gugus = indeks.to_frame()[GUGUS].astype(str).agg("|".join, axis=1).to_numpy()
    ambil = make_column_sampler(len(indeks), groups=gugus)
    titik = nilai(np.arange(len(indeks)))
    boot = np.array([nilai(ambil(rng)) for _ in range(n_resamples)])
    lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
    return Banding(*(BootstrapResult(float(titik[k]), float(lo[k]), float(hi[k]))
                     for k in (0, 1)))


class BandingGabungan(NamedTuple):
    """Hasil :func:`bandingkan_gabungan` di satu λ."""

    rasio_rata: BootstrapResult
    rasio_total: BootstrapResult
    #: Banyak kondisi dengan J pembanding = 0, dikeluarkan dari uk. 1.
    dikeluarkan_uk1: int
    #: Instans dengan total J pembanding = 0, dikeluarkan dari kedua ukuran.
    instans_dikeluarkan: tuple


def _ukuran_per_instans(ja, jb, inst):
    """Kedua ukuran dengan bobot sama per instans, dalam persen.

    ``inst`` boleh nama instans atau kode bilangan bulat per kondisi. Kondisi dengan J
    pembanding = 0 tidak ikut uk. 1; instans dengan total J pembanding = 0 tidak ikut kedua
    ukuran. Dihitung dengan ``np.bincount`` supaya cepat di dalam bootstrap.
    """
    inst = np.asarray(inst)
    kode = inst if np.issubdtype(inst.dtype, np.integer) else np.unique(inst, return_inverse=True)[1]
    n = int(kode.max()) + 1
    sah = jb > 0
    rasio = np.where(sah, ja / np.where(sah, jb, 1.0), 0.0)
    total_b = np.bincount(kode, jb, n)
    pakai = total_b > 0
    r1 = np.bincount(kode, rasio, n)[pakai] / np.bincount(kode, sah, n)[pakai]
    r2 = np.bincount(kode, ja, n)[pakai] / total_b[pakai]
    return (r1.mean() - 1) * 100, (r2.mean() - 1) * 100


def bandingkan_gabungan(df, a, b, *, tingkat_selang=0.95, n_resamples=10_000, random_state=0):
    """Selisih J metode a terhadap b dengan bobot sama per instans (data uji gabungan).

    uk. 1 = rata-rata atas instans dari rata-rata rasio J per kondisi di instans itu, dikurangi
    1. uk. 2 = rata-rata atas instans dari (total J a / total J b di instans itu), dikurangi 1.
    Keduanya dalam persen; positif berarti a lebih buruk. Bootstrap mengambil ulang gugus
    (instans, intensitas, seed) di dalam tiap instans, sehingga setiap instans selalu ada.
    Kondisi dengan J pembanding = 0 dikeluarkan dari uk. 1 dan banyaknya dilaporkan; instans
    dengan total J pembanding = 0 dikeluarkan dari kedua ukuran dan namanya dilaporkan.
    """
    rng = np.random.default_rng(random_state)
    hasil = {}
    for lam, g in df.groupby("lam"):
        t = g.pivot_table(index=KONDISI, columns="metode", values="J")[[a, b]]
        if t.isna().any().any():
            raise ValueError(f"λ {lam:g}: {int(t.isna().any(axis=1).sum())} kondisi hanya "
                             f"ada di salah satu metode ({a}, {b})")
        ja, jb = t[a].to_numpy(), t[b].to_numpy()
        kondisi = t.index.to_frame()
        inst = kondisi["instans"].astype(str).to_numpy()
        gugus = kondisi[GUGUS].astype(str).agg("|".join, axis=1).to_numpy()
        ambil = make_column_sampler(len(ja), groups=gugus, strata=inst)
        kode = np.unique(inst, return_inverse=True)[1]
        titik = _ukuran_per_instans(ja, jb, kode)
        boot = np.array([_ukuran_per_instans(ja[i], jb[i], kode[i]) for i in
                         (ambil(rng) for _ in range(n_resamples))])
        ekor = (1 - tingkat_selang) / 2
        lo, hi = np.quantile(boot, [ekor, 1 - ekor], axis=0)
        total_b = pd.Series(jb).groupby(inst).sum()
        hasil[lam] = BandingGabungan(*(BootstrapResult(float(titik[k]), float(lo[k]),
                                                       float(hi[k])) for k in (0, 1)),
                                     dikeluarkan_uk1=int((jb <= 0).sum()),
                                     instans_dikeluarkan=tuple(total_b[total_b <= 0].index))
    return hasil


def bandingkan_celah_gabungan(df, a, b, sasaran, acuan, *, tingkat_selang=0.95,
                              n_resamples=10_000, random_state=0):
    """Butir 1 di data uji: :func:`bandingkan_celah` dengan bobot sama per instans.

    Celah di satu λ = kedua ukuran :func:`bandingkan_gabungan` (rata-rata atas instans).
    ``sasaran`` dan ``acuan`` berupa {λ: bobot}; hasil = celah sasaran dikurangi celah acuan,
    dalam poin persen (positif = a lebih tertinggal dari b di λ sasaran). Gugus (instans,
    intensitas, seed) diambil ulang sekali untuk semua λ, di dalam tiap instans.
    """
    rng = np.random.default_rng(random_state)
    bobot = dict(sasaran)
    for lam, w in acuan.items():
        bobot[lam] = bobot.get(lam, 0.0) - w
    tabel = {lam: _tabel(df, a, b, lam) for lam in bobot}
    indeks = next(iter(tabel.values())).index
    for lam, t in tabel.items():
        if not t.index.equals(indeks):
            raise ValueError(f"kondisi di λ {lam:g} tidak sama dengan kondisi di λ lain")
    ja = {lam: t[a].to_numpy() for lam, t in tabel.items()}
    jb = {lam: t[b].to_numpy() for lam, t in tabel.items()}
    kondisi = indeks.to_frame()
    inst = kondisi["instans"].astype(str).to_numpy()
    kode = np.unique(inst, return_inverse=True)[1]

    def nilai(i):
        return sum(w * np.array(_ukuran_per_instans(ja[lam][i], jb[lam][i], kode[i]))
                   for lam, w in bobot.items())

    gugus = kondisi[GUGUS].astype(str).agg("|".join, axis=1).to_numpy()
    ambil = make_column_sampler(len(indeks), groups=gugus, strata=inst)
    titik = nilai(np.arange(len(indeks)))
    boot = np.array([nilai(ambil(rng)) for _ in range(n_resamples)])
    ekor = (1 - tingkat_selang) / 2
    lo, hi = np.quantile(boot, [ekor, 1 - ekor], axis=0)
    return Banding(*(BootstrapResult(float(titik[k]), float(lo[k]), float(hi[k]))
                     for k in (0, 1)))


def bandingkan_selisih_gabungan(df, a, b, kolom="ubah", *, tingkat_selang=0.95,
                                n_resamples=10_000, random_state=0):
    """I-3 deskriptif: selisih rata-rata ``kolom`` metode a dikurangi b, per λ.

    Bobot sama per instans: selisih dirata-rata per instans, lalu dirata-rata lintas instans.
    Baris yang sama kondisinya (misalnya tiga seed model PPO) dirata-rata dulu. Selisih,
    bukan rasio, karena pembanding bisa punya nol perubahan. Bootstrap sama dengan
    :func:`bandingkan_gabungan`: gugus (instans, intensitas, seed), berstrata instans.
    """
    rng = np.random.default_rng(random_state)
    hasil = {}
    for lam, g in df.groupby("lam"):
        t = g.pivot_table(index=KONDISI, columns="metode", values=kolom)[[a, b]]
        if t.isna().any().any():
            raise ValueError(f"λ {lam:g}: {int(t.isna().any(axis=1).sum())} kondisi hanya "
                             f"ada di salah satu metode ({a}, {b})")
        d = (t[a] - t[b]).to_numpy()
        kondisi = t.index.to_frame()
        inst = kondisi["instans"].astype(str).to_numpy()
        kode = np.unique(inst, return_inverse=True)[1]
        n = int(kode.max()) + 1

        def nilai(i):
            return (np.bincount(kode[i], d[i], n) / np.bincount(kode[i], minlength=n)).mean()

        gugus = kondisi[GUGUS].astype(str).agg("|".join, axis=1).to_numpy()
        ambil = make_column_sampler(len(d), groups=gugus, strata=inst)
        titik = nilai(np.arange(len(d)))
        boot = np.array([nilai(ambil(rng)) for _ in range(n_resamples)])
        ekor = (1 - tingkat_selang) / 2
        lo, hi = np.quantile(boot, [ekor, 1 - ekor])
        hasil[lam] = BootstrapResult(float(titik), float(lo), float(hi))
    return hasil
