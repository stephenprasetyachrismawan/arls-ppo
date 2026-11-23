"""Analisis hasil uji: semua tabel Bab V dari baris episode.

Menjalankan keputusan pemilik di draf kunci uji (docs/KUNCI_UJI.md bagian
6-9). Dipakai dua kali: dulu sebagai gladi pada hasil validasi, lalu pada data uji.
"""
from __future__ import annotations

import pandas as pd

import math

import numpy as np

from banding_umum import (KONDISI, _ukuran_per_instans, bandingkan_celah_gabungan,
                          bandingkan_gabungan, bandingkan_selisih_gabungan)

#: Keluarga uji utama (draf kunci uji bagian 7): butir 2 lawan dua pembanding di dua λ,
#: selang Bonferroni 98,75%.
UTAMA = ("fix3_g", "acak_g")
LAM_UTAMA = (0.3, 3.0)
SELANG_UTAMA = 0.9875



def _log_interp(lam, kiri, kanan):
    """Bobot interpolasi linear dalam log λ antara dua λ berlabel (sama dengan banding_e15)."""
    w = (math.log(lam) - math.log(kiri)) / (math.log(kanan) - math.log(kiri))
    return {kiri: 1 - w, kanan: w}


#: Pasangan λ butir 1, keputusan pemilik 16 November 2026 (docs/ARSITEKTUR_PIPELINE.md 5.6),
#: sama dengan ``e15/banding_e15.PASANGAN_BUTIR1``. Yang terakhir pelengkap, tidak menentukan.
PASANGAN_BUTIR1 = {
    "0,1 lawan 0,3": ({0.1: 1.0}, {0.3: 1.0}),
    "1 lawan interpolasi log 0,3 dan 3": ({1.0: 1.0}, _log_interp(1.0, 0.3, 3.0)),
    "10 lawan interpolasi log 3 dan 30": ({10.0: 1.0}, _log_interp(10.0, 3.0, 30.0)),
    "pelengkap: rata-rata tanpa label lawan rata-rata berlabel":
        ({0.1: 1 / 3, 1.0: 1 / 3, 10.0: 1 / 3}, {0.3: 1 / 3, 3.0: 1 / 3, 30.0: 1 / 3}),
}

_KELUARGA = {"brandimarte": "Brandimarte mk11-15", "barnes": "Barnes",
             "dauzere": "Dauzère-Pérès"}


def keluarga(nama: str) -> str:
    """Keluarga benchmark dari nama instans."""
    if "/" not in nama:
        return "MK"
    bagian = nama.split("/")
    if bagian[0] == "hurink":
        return f"Hurink {bagian[1]}"
    return _KELUARGA[bagian[0]]


def _tabel(df, a, b):
    t = df.pivot_table(index=KONDISI, columns="metode", values="J")[[a, b]]
    if t.isna().any().any():
        raise ValueError(f"ada kondisi yang hanya punya salah satu metode ({a}, {b})")
    return t


def _titik(df, a, b):
    """Titik kedua ukuran berbobot sama per instans, per λ."""
    hasil = {}
    for lam, g in df.groupby("lam"):
        t = _tabel(g, a, b)
        inst = t.index.get_level_values("instans").astype(str).to_numpy()
        r1, r2 = _ukuran_per_instans(t[a].to_numpy(), t[b].to_numpy(), inst)
        hasil[lam] = {"rasio_rata": float(r1), "rasio_total": float(r2)}
    return hasil


def _ringkas_tingkat(g):
    """Porsi tingkat diminta dan dipakai, serta laju penolakan, dari kolom ``tingkat``."""
    diminta = [sum(t["diminta"][k] for t in g["tingkat"]) for k in range(4)]
    dipakai = [sum(t["dipakai"][k] for t in g["tingkat"]) for k in range(4)]
    minta_ubah = sum(diminta[1:])
    return {"porsi_diminta": [v / sum(diminta) for v in diminta],
            "porsi_dipakai": [v / sum(dipakai) for v in dipakai],
            "laju_tolak_penjaga": sum(t["ditolak_penjaga"] for t in g["tingkat"]) / minta_ubah
            if minta_ubah else 0.0,
            "laju_gagal": sum(t["gagal"] for t in g["tingkat"]) / minta_ubah
            if minta_ubah else 0.0}


def _pelanggaran(baris):
    """I-2: pelanggaran keras pemeriksa jadwal per metode, dijumlah atas semua episode."""
    hasil = {}
    for r in baris:
        h = hasil.setdefault(r["metode"], {"keras": 0, "episode": 0, "tanpa_periksa": 0})
        h["episode"] += 1
        if "periksa" not in r:
            h["tanpa_periksa"] += 1
            continue
        h["keras"] += sum(n for k, n in r["periksa"].get("hitung", {}).items()
                          if k.endswith(":keras"))
    return hasil


#: Ambang I-1 (KERANGKA_NASKAH.md): p95 waktu pemilih di bawah 50 ms di setiap instans.
AMBANG_MS = 50.0


def _p95_per_instans(baris, kolom):
    per = {}
    for r in baris:
        per.setdefault(r["instans"], []).extend(r.get(kolom, []))
    return {n: float(np.percentile(v, 95)) for n, v in sorted(per.items()) if v}


def _waktu(baris):
    """I-1 per metode: p95 per instans (semua episode instans digabung).

    Ambang 50 ms hanya untuk waktu pemilih (``ms_pemilih``, satu panggilan pemilih per
    sub-keputusan); waktu total (``ms_keputusan``) dilaporkan tanpa ambang, sama dengan
    definisi lama. Metode tanpa pemilih terpisah (Sequential, tingkat 0) tidak dinilai:
    ``di_bawah_50ms`` = None.
    """
    per_metode = {}
    for r in baris:
        if "ms_keputusan" in r or "ms_pemilih" in r:
            per_metode.setdefault(r["metode"], []).append(r)
    hasil = {}
    for m, rs in per_metode.items():
        pem = _p95_per_instans(rs, "ms_pemilih")
        tot = _p95_per_instans(rs, "ms_keputusan")
        maks = max(pem.values()) if pem else None
        hasil[m] = {"p95_pemilih_per_instans": pem, "p95_pemilih_maks": maks,
                    "di_bawah_50ms": None if maks is None else maks < AMBANG_MS,
                    "p95_total_per_instans": tot,
                    "p95_total_maks": max(tot.values()) if tot else None}
    return hasil


def analisis(baris, usulan, pembanding, sifat, *, n_resamples=10_000):
    """Semua tabel analisis uji sebagai kamus.

    ``baris`` = baris episode semua metode (baris PPO dari tiga seed model dirata-rata per
    kondisi). ``usulan`` = nama metode usulan; ``pembanding`` = daftar nama metode lain;
    ``sifat`` = {instans: kamus sifat_instans}.
    """
    df = pd.DataFrame(baris)
    df["keluarga"] = df["instans"].map(keluarga)
    kw = dict(n_resamples=n_resamples)
    bench = df[df.keluarga != "MK"]
    gabungan = {b: bandingkan_gabungan(df, usulan, b, **kw) for b in pembanding}
    benchmark_saja = {b: bandingkan_gabungan(bench, usulan, b, **kw) for b in pembanding}
    sub = df[df.lam.isin(LAM_UTAMA)]
    utama = {b: bandingkan_gabungan(sub, usulan, b, tingkat_selang=SELANG_UTAMA, **kw)
             for b in pembanding if b in UTAMA}
    butir1 = {}
    if "seq" in pembanding:
        d = df[df.metode.isin([usulan, "seq"])]
        ada = set(d.lam)
        # Pasangan yang λ-nya tidak lengkap di baris dilewati (misalnya gladi satu λ).
        butir1 = {judul: bandingkan_celah_gabungan(d, usulan, "seq", sasaran, acuan, **kw)
                  for judul, (sasaran, acuan) in PASANGAN_BUTIR1.items()
                  if set(sasaran) | set(acuan) <= ada}
    per_keluarga = {}
    for kel, g in df.groupby("keluarga"):
        per_keluarga[kel] = {b: _titik(g, usulan, b) for b in pembanding}
    baris_sifat = []
    for (nama, lam), g in df.groupby(["instans", "lam"]):
        for b in pembanding:
            titik = _titik(g, usulan, b)[lam]
            baris_sifat.append({"instans": nama, "keluarga": keluarga(nama), "lam": lam,
                                "pembanding": b, **titik, **sifat.get(nama, {})})
    tingkat = {}
    u = df[df.metode == usulan]
    if "tingkat" in u:
        for (kel, lam), g in u.groupby(["keluarga", "lam"]):
            tingkat.setdefault(kel, {})[lam] = _ringkas_tingkat(g)
    # I-3 deskriptif (KERANGKA_NASKAH bagian 8, catatan 19 November 2026): selisih banyak
    # perubahan jadwal per episode, usulan dikurangi pembanding, tanpa ambang.
    stabil = ({b: bandingkan_selisih_gabungan(df, usulan, b, "ubah", **kw) for b in pembanding}
              if "ubah" in df else {})
    return {"pelanggaran": _pelanggaran(baris), "butir1": butir1, "waktu": _waktu(baris),
            "stabil": stabil,
            "per_sifat": pd.DataFrame(baris_sifat), "tingkat": tingkat, "gabungan": gabungan, "benchmark_saja": benchmark_saja, "utama": utama,
            "per_keluarga": per_keluarga}


def _sel(x):
    tanda = "*" if x.high < 0 or x.low > 0 else ""
    return (f"{x.estimate:+.2f} [{x.low:+.2f}; {x.high:+.2f}]{tanda}"
            .replace(".", ",").replace("-", "−"))


def _lam(lam):
    return f"{lam:g}".replace(".", ",")


def _angka(v):
    return f"{v:+.2f}".replace(".", ",").replace("-", "−")


def tulis_laporan(h, usulan, judul):
    """Laporan Markdown dari hasil :func:`analisis`."""
    out = [f"# {judul}\n",
           f"Metode usulan: `{usulan}`. Angka = selisih J usulan terhadap pembanding dalam "
           "persen (negatif = usulan lebih baik), tiap instans berbobot sama. uk. 1 rata-rata "
           "rasio per kondisi, uk. 2 rasio J total. * = selang tidak memuat nol.\n"]
    for kunci, teks in (("utama", "Keluarga uji utama (selang 98,75%)"),
                        ("gabungan", "Bukti utama: gabungan (selang 95%)"),
                        ("benchmark_saja", "Kepekaan: benchmark saja (selang 95%)")):
        out.append(f"## {teks}\n")
        if not any(h[kunci].values()):
            out.append("Tidak ada baris.\n")
            continue
        out.append("| pembanding | λ | uk. 1 | uk. 2 | kondisi J=0 |\n|---|---|---|---|---|")
        for b, per_lam in h[kunci].items():
            for lam, x in per_lam.items():
                out.append(f"| {b} | {lam:g} | {_sel(x.rasio_rata)} | {_sel(x.rasio_total)} | "
                           f"{x.dikeluarkan_uk1} |".replace(f"| {lam:g} |",
                                                          f"| {_lam(lam)} |"))
        out.append("")
    out.append("## Butir 1: celah usulan ke Sequential, λ tanpa label dikurangi acuan "
               "(selang 95%)\n")
    out.append("Poin persen, tiap instans berbobot sama; positif = celah di λ tanpa label lebih "
               "besar. Kalimat naskah: \"celah paling banyak X poin lebih besar dari acuan\", "
               "X = batas atas selang.\n")
    if not h["butir1"]:
        out.append("Tidak ada baris Sequential.\n")
    else:
        out.append("| pasangan λ | uk. 1 | uk. 2 |\n|---|---|---|")
        for judul, x in h["butir1"].items():
            out.append(f"| {judul} | {_sel(x.rasio_rata)} | {_sel(x.rasio_total)} |")
        out.append("")
    out.append("## Per keluarga (titik, uk. 1 / uk. 2)\n")
    out.append("| keluarga | pembanding | λ | uk. 1 | uk. 2 |\n|---|---|---|---|---|")
    for kel, per_b in h["per_keluarga"].items():
        for b, per_lam in per_b.items():
            for lam, x in per_lam.items():
                out.append(f"| {kel} | {b} | {_lam(lam)} | "
                           f"{_angka(x['rasio_rata'])} | {_angka(x['rasio_total'])} |")
    out.append("\n## Waktu keputusan (I-1)\n")
    if not h["waktu"]:
        out.append("Baris episode tidak punya kolom waktu.\n")
    else:
        out.append("p95 per instans, semua episode instans digabung; yang ditampilkan p95 "
                   "terbesar atas instans. Ambang 50 ms hanya untuk waktu pemilih (satu "
                   "panggilan pemilih per sub-keputusan, tanpa operator); waktu total "
                   "(fitur, pemilih, operator, dan penjaga) tanpa ambang. Metode tanpa pemilih "
                   "terpisah ditandai \"-\".\n")
        out.append("| metode | p95 pemilih (ms) | pemilih di bawah 50 ms | p95 total (ms) |\n"
                   "|---|---|---|---|")
        ms = lambda v: "-" if v is None else f"{v:.2f}".replace(".", ",")  # noqa: E731
        for m, x in sorted(h["waktu"].items()):
            ya = {None: "-", True: "ya", False: "tidak"}[x["di_bawah_50ms"]]
            out.append(f"| {m} | {ms(x['p95_pemilih_maks'])} | {ya} | "
                       f"{ms(x['p95_total_maks'])} |")
    out.append("\n## Pelanggaran keras pemeriksa jadwal (I-2)\n")
    out.append("| metode | episode | pelanggaran keras | episode tanpa pemeriksa |\n"
               "|---|---|---|---|")
    for m, x in sorted(h["pelanggaran"].items()):
        out.append(f"| {m} | {x['episode']} | {x['keras']} | {x['tanpa_periksa']} |")
    out.append("\n## Perubahan jadwal (I-3)\n")
    if not h["stabil"]:
        out.append("Baris episode tidak punya kolom `ubah`.\n")
    else:
        out.append("Selisih banyak perubahan jadwal per episode, usulan dikurangi pembanding "
                   "(positif = usulan lebih banyak mengubah jadwal), tiap instans berbobot "
                   "sama, selang 95%. Deskriptif, tanpa ambang lolos atau gagal; * = selang "
                   "tidak memuat nol.\n")
        out.append("| pembanding | λ | selisih perubahan |\n|---|---|---|")
        for b, per_lam in h["stabil"].items():
            for lam, x in per_lam.items():
                out.append(f"| {b} | {_lam(lam)} | {_sel(x)} |")
    out.append("\n## Tingkat yang dipilih usulan dan penolakan penjaga\n")
    if not h["tingkat"]:
        out.append("Baris episode tidak punya kolom `tingkat`.\n")
    else:
        out.append("| keluarga | λ | porsi diminta 0/1/2/3 | porsi dipakai 0/1/2/3 | "
                   "tolak penjaga | gagal |\n|---|---|---|---|---|---|")
        for kel, per_lam in h["tingkat"].items():
            for lam, x in per_lam.items():
                p = lambda v: " / ".join(f"{a:.2f}" for a in v).replace(".", ",")  # noqa: E731
                out.append(f"| {kel} | {_lam(lam)} | {p(x['porsi_diminta'])} | "
                           f"{p(x['porsi_dipakai'])} | "
                           f"{x['laju_tolak_penjaga']:.2f} | {x['laju_gagal']:.2f} |"
                           .replace(".", ","))
    return "\n".join(out) + "\n"


def main():
    """Pemakaian (dari folder gerbang_batch)::

        PYTHONPATH=../../../exp1/dfjsp-ppo-repair/src $PY analisis_uji.py USULAN JUDUL \\
            KELUAR_MD KELUAR_CSV BERKAS_EPISODE...

    Sifat instans dihitung lewat ``instans_acak.load_any``. Instans uji tidak bisa dimuat di
    luar proses pembukaan, jadi sifatnya dibaca dari ``SIFAT=<keluar>/sifat_uji.json`` yang
    ditulis ``buka_uji.py``.
    """
    import json
    import sys

    import penjaga_uji  # noqa: F401  (memblokir pembukaan data uji)
    import instans_acak
    from sifat_instans import sifat_instans

    usulan, judul, keluar_md, keluar_csv, *berkas = sys.argv[1:]
    baris = [r for f in berkas for r in json.load(open(f))]
    pembanding = sorted({r["metode"] for r in baris} - {usulan})
    nama = sorted({r["instans"] for r in baris})
    # Sifat instans uji ditulis buka_uji.py ke sifat_uji.json (penjaga memblokir berkas uji di
    # luar proses pembukaan); instans lain dihitung di sini.
    import os
    sifat = json.load(open(os.environ["SIFAT"])) if os.environ.get("SIFAT") else {}
    sifat.update({n: sifat_instans(instans_acak.load_any(n)) for n in nama if n not in sifat})
    h = analisis(baris, usulan, pembanding, sifat)
    open(keluar_md, "w").write(tulis_laporan(h, usulan, judul))
    h["per_sifat"].to_csv(keluar_csv, index=False)


if __name__ == "__main__":
    main()
