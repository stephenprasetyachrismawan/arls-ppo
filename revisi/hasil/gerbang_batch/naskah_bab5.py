"""Pembuat tabel Bab V (ARLS-PPO, resep E15 E9c): enam fungsi murni, seam S1 sampai S6.

Seam disetujui pemilik 21 November 2026. Setiap fungsi hanya mengolah masukan yang sudah ada
(hasil analisis, baris episode, teks log, bobot jaringan, data guru, catatan satu episode);
tidak ada yang melatih model atau membuka data uji. Pemanggil yang memuat berkas dari mesin
penelitian ada di ``bab5_jalan.py``.

- S1 :func:`tabel_analisis`     hasil ``analisis_uji.analisis`` -> teks tabel LaTeX
- S2 :func:`uraian_j`           baris episode -> suku J per metode per λ (instans berbobot sama)
     :func:`kuartil_rasio`      baris episode -> rasio J dua metode per kelompok besar J
- S3 :func:`baca_log`           teks log ``latih_ppo_seq.py`` -> epoch BC dan reward PPO
- S4 :func:`banding_bobot`      dua kumpulan bobot jaringan -> perubahan per lapis
     :func:`kesamaan_pilihan`   pilihan dua jaringan (dan guru) -> porsi sama dan matriks
- S5 :func:`ringkas_guru`       baris data guru (+ bobot contoh BC) -> ringkasan per λ
- S6 :func:`catatan_kejadian`   catatan mentah satu episode -> baris per kejadian
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

KONDISI = ["instans", "kapasitas", "intensitas", "seed"]

#: Nama metode di naskah (docs/KERANGKA_NASKAH.md bagian 11 butir 3).
NAMA = {
    "ppo_bc_ppo_g": "ARLS-PPO",
    "ppo_bc_ppo": "ARLS-PPO tanpa penjaga",
    "ppo_bc_g": "ARLS-BC",
    "aor": "AOR",
    "seq": "Sequential Lookahead",
    "fix3_g": "Fixed-L3 + penjaga",
    "acak_g": "Random + penjaga",
    "il_g": "Boosting-IL + penjaga",
}

#: Urutan baris pembanding di tabel.
URUTAN = ("aor", "acak_g", "fix3_g", "il_g", "ppo_bc_g", "ppo_bc_ppo", "seq")


# --------------------------------------------------------------------- bantuan format

def angka(v, d=2, tanda=False):
    """Angka LaTeX dengan koma desimal dan tanda minus matematika: -1.5 -> ``$-1{,}50$``."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    if round(v, d) == 0:
        v = 0.0     # hindari "-0,00"
    teks = f"{v:+.{d}f}" if tanda else f"{v:.{d}f}"
    teks = teks.replace("-", "-").replace(".", "{,}")
    return f"${teks}$"


def _lam(lam):
    return f"{lam:g}".replace(".", "{,}")


def _selang(x, tanda="*"):
    """``estimate [low; high]`` dari BootstrapResult, * bila selang tidak memuat nol."""
    bintang = "$^{*}$" if (x.high < 0 or x.low > 0) else ""
    return f"{angka(x.estimate, tanda=True)} [{angka(x.low, tanda=True)}; " \
           f"{angka(x.high, tanda=True)}]{bintang}"


def _tabel(caption, label, kolom, kepala, baris, catatan=None, kecil=True):
    out = ["% Dibangkitkan oleh revisi/hasil/gerbang_batch/bab5_jalan.py. Jangan diubah dengan "
           "tangan.",
           "\\begin{table}[htbp]", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
           "\\centering" + ("\\footnotesize" if kecil else ""),
           "\\setlength{\\tabcolsep}{4pt}",
           "\\begin{adjustbox}{max width=\\textwidth}",
           f"\\begin{{tabular}}{{{kolom}}}", "\\toprule", kepala + " \\\\", "\\midrule"]
    for b in baris:
        out.append(b if b.startswith("\\") else b + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}"]
    if catatan:
        out.append("\\par\\smallskip\\footnotesize\\textit{Catatan:} " + catatan)
    out.append("\\end{table}")
    return "\n".join(out) + "\n"


def _nama(m):
    return NAMA.get(m, m)


def _urut(kunci):
    return sorted(kunci, key=lambda m: (URUTAN.index(m) if m in URUTAN else 99, m))


# --------------------------------------------------------------------- S1

_CAT_SELANG = ("Angka adalah selisih $J$ ARLS-PPO terhadap pembanding dalam persen; negatif "
               "berarti ARLS-PPO lebih baik. Uk.~1 rata-rata rasio per kondisi, uk.~2 rasio "
               "$J$ total; setiap instans berbobot sama. Tanda $^{*}$: selang tidak memuat nol.")


def _kalimat_utama(x):
    """Kalimat klaim menurut kunci uji bagian 7: hanya bila kedua ukuran searah dan tegas."""
    r1, r2 = x.rasio_rata, x.rasio_total
    if r1.high < 0 and r2.high < 0:
        return "lebih baik"
    if r1.low > 0 and r2.low > 0:
        return "lebih buruk"
    return "tidak tegas"


def tabel_analisis(h):
    """S1: hasil ``analisis_uji.analisis`` -> {nama berkas: teks tabel LaTeX}.

    Nama berkas tanpa akhiran. Semua angka diambil dari ``h`` apa adanya; fungsi ini hanya
    memilih, mengurutkan, dan memformat.
    """
    t = {}
    baris = []
    for b in _urut(h["utama"]):
        for lam, x in sorted(h["utama"][b].items()):
            baris.append(f"{_nama(b)} & {_lam(lam)} & {_selang(x.rasio_rata)} & "
                         f"{_selang(x.rasio_total)} & {_kalimat_utama(x)}")
    t["uji_utama"] = _tabel(
        "Keluarga uji utama: ARLS-PPO terhadap dua pembanding (selang 98{,}75\\%)",
        "tab:uji-utama", "llllc",
        "Pembanding & $\\lambda$ & Uk.~1 (\\%) & Uk.~2 (\\%) & Kesimpulan", baris,
        _CAT_SELANG + " Kesimpulan ditulis \\emph{lebih baik} atau \\emph{lebih buruk} hanya "
        "bila kedua selang berada di sisi yang sama dari nol.")
    # Kepekaan benchmark saja untuk baris uji utama (λ dan pembanding yang sama).
    baris = []
    for b in _urut(h["utama"]):
        for lam in sorted(h["utama"][b]):
            x = h["benchmark_saja"][b][lam]
            baris.append(f"{_nama(b)} & {_lam(lam)} & {_selang(x.rasio_rata)} & "
                         f"{_selang(x.rasio_total)}")
    t["uji_kepekaan"] = _tabel(
        "Analisis kepekaan: 35 instans benchmark saja (selang 95\\%)", "tab:uji-kepekaan",
        "llll", "Pembanding & $\\lambda$ & Uk.~1 (\\%) & Uk.~2 (\\%)", baris, _CAT_SELANG)
    # Gabungan semua pembanding di semua λ.
    baris = []
    for i, b in enumerate(_urut(h["gabungan"])):
        if i:
            baris.append("\\addlinespace")
        for lam, x in sorted(h["gabungan"][b].items()):
            baris.append(f"{_nama(b)} & {_lam(lam)} & {_selang(x.rasio_rata)} & "
                         f"{_selang(x.rasio_total)}")
    t["uji_gabungan"] = _tabel(
        "ARLS-PPO terhadap semua pembanding pada data uji gabungan (selang 95\\%)",
        "tab:uji-gabungan", "llll", "Pembanding & $\\lambda$ & Uk.~1 (\\%) & Uk.~2 (\\%)",
        baris, _CAT_SELANG + " $\\lambda$ 0{,}3, 3, dan 30 berlabel guru; 0{,}1, 1, dan 10 "
        "tanpa label.")
    # Butir 1.
    baris = [f"{judul} & {_selang(x.rasio_rata)} & {_selang(x.rasio_total)}"
             for judul, x in h["butir1"].items()]
    t["uji_butir1"] = _tabel(
        "Celah ARLS-PPO ke Sequential Lookahead: $\\lambda$ tanpa label dikurangi acuan "
        "(selang 95\\%)", "tab:uji-butir1", "p{6cm}ll",
        "Pasangan $\\lambda$ & Uk.~1 (poin) & Uk.~2 (poin)", baris,
        "Positif berarti celah ke Sequential Lookahead di $\\lambda$ tanpa label lebih besar "
        "daripada di acuan berlabel. Batas atas selang adalah $X$ pada kalimat klaim butir~1.")
    # I-1 waktu.
    baris = []
    for m in _urut(h["waktu"]) if "ppo_bc_ppo_g" not in h["waktu"] else \
            ["ppo_bc_ppo_g"] + _urut(set(h["waktu"]) - {"ppo_bc_ppo_g"}):
        x = h["waktu"][m]
        ya = {None: "--", True: "ya", False: "tidak"}[x["di_bawah_50ms"]]
        baris.append(f"{_nama(m)} & {angka(x['p95_pemilih_maks'])} & {ya} & "
                     f"{angka(x['p95_total_maks'])}")
    t["uji_waktu"] = _tabel(
        "Waktu keputusan pada data uji (I-1)", "tab:uji-waktu", "lrcr",
        "Metode & \\shortstack[r]{p95 pemilih\\\\(ms)} & \\shortstack{Di bawah\\\\50 ms} & "
        "\\shortstack[r]{p95 total\\\\(ms)}", baris,
        "p95 dihitung per instans dari semua episode instans itu; yang ditampilkan p95 terbesar "
        "atas 38 instans. Ambang 50 ms hanya untuk waktu pemilih. Waktu total memuat fitur, "
        "pemilih, operator, dan penjaga. Tanda -- : metode tanpa pemilih terpisah. Diukur pada "
        "8 proses paralel.")
    # I-2 layak.
    baris = [f"{_nama(m)} & {x['episode']:,} & {x['keras']} & {x['tanpa_periksa']}"
             .replace(",", ".") for m, x in
             ((m, h["pelanggaran"][m]) for m in
              (["ppo_bc_ppo_g"] if "ppo_bc_ppo_g" in h["pelanggaran"] else [])
              + _urut(set(h["pelanggaran"]) - {"ppo_bc_ppo_g"}))]
    t["uji_layak"] = _tabel(
        "Pelanggaran keras pemeriksa jadwal pada data uji (I-2)", "tab:uji-layak", "lrrr",
        "Metode & Episode & \\shortstack[r]{Pelanggaran\\\\keras} & "
        "\\shortstack[r]{Episode tanpa\\\\pemeriksa}", baris,
        "Episode metode PPO dihitung untuk tiga \\emph{seed} model.")
    # I-3 perubahan: pembanding x λ.
    if h.get("stabil"):
        lams = sorted({lam for v in h["stabil"].values() for lam in v})
        baris = [f"{_nama(b)} & " + " & ".join(
            angka(h["stabil"][b][lam].estimate, tanda=True)
            + ("$^{*}$" if (h["stabil"][b][lam].high < 0 or h["stabil"][b][lam].low > 0)
               else "") for lam in lams) for b in _urut(h["stabil"])]
        t["uji_perubahan"] = _tabel(
            "Selisih banyak perubahan per episode, ARLS-PPO dikurangi pembanding (I-3)",
            "tab:uji-perubahan", "l" + "r" * len(lams),
            "Pembanding & " + " & ".join(f"$\\lambda={_lam(lam)}$" for lam in lams), baris,
            "Negatif berarti ARLS-PPO membuat lebih sedikit perubahan. Setiap instans berbobot "
            "sama. Tanda $^{*}$: selang 95\\% tidak memuat nol. Selang lengkap ada di "
            "Lampiran~\\ref{lamp:hasil-rinci}.")
        baris = []
        for i, b in enumerate(_urut(h["stabil"])):
            if i:
                baris.append("\\addlinespace")
            baris += [f"{_nama(b)} & {_lam(lam)} & {_selang(h['stabil'][b][lam])}"
                      for lam in lams if lam in h["stabil"][b]]
        t["uji_perubahan_selang"] = _tabel(
            "Selisih banyak perubahan per episode dengan selang 95\\%, ARLS-PPO dikurangi "
            "pembanding", "tab:uji-perubahan-selang", "lll",
            "Pembanding & $\\lambda$ & Selisih perubahan [selang 95\\%]", baris,
            "Sama dengan Tabel~\\ref{tab:uji-perubahan}, dengan batas bawah dan atas selang "
            "\\emph{bootstrap} bergugus. Negatif berarti ARLS-PPO membuat lebih sedikit "
            "perubahan. Tanda $^{*}$: selang tidak memuat nol.")
    # Tingkat diminta, dipakai, ditolak per keluarga di λ berlabel.
    if h.get("tingkat"):
        baris = []
        for i, kel in enumerate(sorted(h["tingkat"])):
            if i:
                baris.append("\\addlinespace")
            for lam, x in sorted(h["tingkat"][kel].items()):
                if lam not in (0.3, 3.0, 30.0):
                    continue
                p = lambda v: " & ".join(angka(a) for a in v)  # noqa: E731
                baris.append(f"{kel} & {_lam(lam)} & {p(x['porsi_diminta'])} & "
                             f"{p(x['porsi_dipakai'])} & {angka(x['laju_tolak_penjaga'])} & "
                             f"{angka(x['laju_gagal'])}")
        t["uji_tingkat"] = _tabel(
            "Porsi tingkat yang diminta dan dipakai ARLS-PPO, serta laju penolakan, menurut "
            "keluarga instans", "tab:uji-tingkat", "llrrrrrrrrrr",
            "Keluarga & $\\lambda$ & \\multicolumn{4}{c}{Diminta 0/1/2/3} & "
            "\\multicolumn{4}{c}{Dipakai 0/1/2/3} & Tolak & Gagal", baris,
            "Porsi atas semua sub-keputusan. \\emph{Tolak}: porsi permintaan tingkat 1 sampai 3 "
            "yang ditolak penjaga. \\emph{Gagal}: porsi permintaan tingkat 1 sampai 3 yang "
            "operatornya tidak menghasilkan jadwal. Hanya $\\lambda$ berlabel yang ditampilkan.")
    # Per keluarga, dua pembanding uji utama di λ uji utama.
    baris = []
    kel_semua = sorted(h["per_keluarga"])
    for kel in kel_semua:
        sel = []
        for b in ("acak_g", "fix3_g"):
            for lam in (0.3, 3.0):
                x = h["per_keluarga"][kel].get(b, {}).get(lam)
                sel += ["--", "--"] if x is None else [angka(x["rasio_rata"], tanda=True),
                                                      angka(x["rasio_total"], tanda=True)]
        baris.append(f"{kel} & " + " & ".join(sel))
    t["uji_keluarga"] = _tabel(
        "Selisih $J$ ARLS-PPO menurut keluarga instans (titik, \\%)", "tab:uji-keluarga",
        "lrrrrrrrr",
        " & \\multicolumn{4}{c}{Random + penjaga} & \\multicolumn{4}{c}{Fixed-L3 + penjaga} "
        "\\\\\n\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
        "Keluarga & \\multicolumn{2}{c}{$\\lambda=0{,}3$} & \\multicolumn{2}{c}{$\\lambda=3$} & "
        "\\multicolumn{2}{c}{$\\lambda=0{,}3$} & \\multicolumn{2}{c}{$\\lambda=3$} \\\\\n"
        " & uk.~1 & uk.~2 & uk.~1 & uk.~2 & uk.~1 & uk.~2 & uk.~1 & uk.~2", baris,
        "Titik tanpa selang, setiap instans dalam keluarga berbobot sama. Negatif berarti "
        "ARLS-PPO lebih baik.")
    return t


def tabel_terhadap_aor(hasil):
    """S1 (pelengkap RM1): ARLS-PPO dan Sequential Lookahead terhadap AOR.

    ``hasil`` = {metode: {"J": {λ: BandingGabungan}, "telat": {λ: BootstrapResult},
    "ubah": {λ: BootstrapResult}}}, dihitung dengan fungsi bootstrap ``banding_umum`` yang
    sama dengan analisis uji.
    """
    baris = []
    for i, m in enumerate(hasil):
        if i:
            baris.append("\\addlinespace")
        for lam in sorted(hasil[m]["J"]):
            x = hasil[m]["J"][lam]
            baris.append(f"{_nama(m)} & {_lam(lam)} & {_selang(x.rasio_rata)} & "
                         f"{_selang(x.rasio_total)} & {_selang(hasil[m]['telat'][lam])} & "
                         f"{_selang(hasil[m]['ubah'][lam])}")
    return _tabel(
        "ARLS-PPO dan Sequential Lookahead terhadap AOR pada data uji (selang 95\\%)",
        "tab:uji-aor", "llllll",
        "Metode & $\\lambda$ & $J$ uk.~1 (\\%) & $J$ uk.~2 (\\%) & "
        "\\shortstack{Selisih\\\\keterlambatan ($E$)} & \\shortstack{Selisih\\\\perubahan}",
        baris,
        "Selisih $J$ dalam persen terhadap AOR (negatif = lebih baik). Selisih keterlambatan "
        "dan perubahan: rata-rata per episode metode dikurangi AOR, setiap instans berbobot "
        "sama. Tanda $^{*}$: selang tidak memuat nol.")


def _peringkat(v):
    return pd.Series(v).rank(method="average").to_numpy()


def ringkas_sifat(per_sifat, pembanding, lams, kolom):
    """S1 (analisis sifat, deskriptif): korelasi Spearman selisih J dengan sifat instans.

    ``per_sifat`` = DataFrame ``h["per_sifat"]`` (satu baris per instans, λ, pembanding).
    Kembalikan {sifat: {(λ, ukuran): rho}} untuk ukuran ``rasio_rata`` dan ``rasio_total``.
    Spearman = korelasi Pearson antara peringkat (peringkat sama dirata-rata).
    """
    out = {}
    for s in kolom:
        out[s] = {}
        for lam in lams:
            d = per_sifat[(per_sifat.pembanding == pembanding) & np.isclose(per_sifat.lam, lam)]
            for u in ("rasio_rata", "rasio_total"):
                a, b = _peringkat(d[s]), _peringkat(d[u])
                out[s][(lam, u)] = float(np.corrcoef(a, b)[0, 1])
    return out


#: Nama sifat instans di naskah (sifat_instans.py).
NAMA_SIFAT = {"n_tugas": "Banyak tugas", "n_dev": "Banyak pengembang",
              "tugas_per_dev": "Tugas per pengembang",
              "fleks_rata": "Fleksibilitas (pengembang per tugas)",
              "beban_per_dev": "Beban per pengembang",
              "ketimpangan_keahlian": "Ketimpangan keahlian", "keketatan": "Keketatan tenggat",
              "dep_rata": "Prasyarat per tugas"}


def tabel_sifat(rho, pembanding, lams):
    """Teks LaTeX untuk keluaran :func:`ringkas_sifat`."""
    baris = [f"{NAMA_SIFAT.get(s, s)} & " + " & ".join(
        angka(v[(lam, u)], tanda=True) for lam in lams for u in ("rasio_rata", "rasio_total"))
        for s, v in rho.items()]
    kepala = ("Sifat instans & " + " & ".join(
        f"\\multicolumn{{2}}{{c}}{{$\\lambda={_lam(lam)}$}}" for lam in lams) + " \\\\\n"
        + " & " + " & ".join("uk.~1 & uk.~2" for _ in lams))
    return _tabel(
        f"Korelasi Spearman selisih $J$ ARLS-PPO terhadap {_nama(pembanding)} dengan sifat "
        "instans uji", f"tab:uji-sifat-{pembanding.replace('_', '-')}",
        "l" + "r" * (2 * len(lams)), kepala, baris,
        "Satu titik per instans uji (38 instans). Positif berarti makin besar sifat itu, makin "
        "besar selisih $J$ (ARLS-PPO makin tertinggal). Deskriptif, tanpa uji hipotesis.")


# --------------------------------------------------------------------- S2

def _per_kondisi(baris):
    """Baris episode -> satu baris per (kondisi, metode, λ); seed model PPO dirata-rata."""
    df = pd.DataFrame(baris)
    beda = (df["telat"] + df["lam"] * df["ubah"] - df["J"]).abs()
    if (beda > 1e-6 * (1 + df["J"].abs())).any():
        raise ValueError("ada baris dengan J yang tidak sama dengan telat + λ x ubah")
    return df.groupby(KONDISI + ["metode", "lam"], as_index=False)[["telat", "ubah", "J"]].mean()


def uraian_j(baris):
    """S2: suku J per metode per λ, setiap instans berbobot sama.

    J = telat + λ x ubah. Baris PPO dari beberapa seed model dirata-rata per kondisi dulu,
    lalu dirata-rata per instans, lalu atas instans. Kembalikan DataFrame dengan kolom
    metode, lam, telat, ubah, harga_ubah (= λ x ubah), J, n_instans.
    """
    k = _per_kondisi(baris)
    per_inst = k.groupby(["metode", "lam", "instans"], as_index=False)[["telat", "ubah", "J"]] \
        .mean()
    out = per_inst.groupby(["metode", "lam"], as_index=False).agg(
        telat=("telat", "mean"), ubah=("ubah", "mean"), J=("J", "mean"),
        n_instans=("instans", "nunique"))
    out["harga_ubah"] = out["lam"] * out["ubah"]
    return out[["metode", "lam", "telat", "ubah", "harga_ubah", "J", "n_instans"]]


def tabel_uraian(df, urutan, caption, label, catatan):
    """Teks LaTeX untuk keluaran :func:`uraian_j`: baris = metode, dikelompokkan per λ."""
    baris = []
    for i, lam in enumerate(sorted(df.lam.unique())):
        if i:
            baris.append("\\addlinespace")
        for m in urutan:
            r = df[(df.metode == m) & np.isclose(df.lam, lam)]
            if r.empty:
                continue
            r = r.iloc[0]
            baris.append(f"{_lam(lam)} & {_nama(m)} & {angka(r.telat)} & {angka(r.ubah)} & "
                         f"{angka(r.harga_ubah)} & {angka(r.J)}")
    return _tabel(caption, label, "llrrrr",
                  "$\\lambda$ & Metode & Keterlambatan & Perubahan & $\\lambda\\times$perubahan"
                  " & $J$", baris, catatan)


def kuartil_rasio(baris, a, b, lam):
    """Rasio J metode ``a`` terhadap ``b`` per kelompok besar J ``b`` (pemeriksaan sesudah hasil).

    Di dalam setiap instans, kondisi diurutkan menurut J ``b`` lalu dibagi empat kelompok yang
    hampir sama banyak (Q1 terkecil); bila banyak kondisi tidak habis dibagi empat, sisanya
    masuk Q4 (``ceil(peringkat / n * 4)``). ``HASIL_UJI.md`` bagian 7 memakai ``pd.qcut``,
    yang menaruh sisa di Q1, sehingga angkanya sedikit berbeda dengan arah yang sama. Kembalikan DataFrame per kelompok: rata-rata dan median rasio, porsi kondisi
    dengan J ``a`` lebih kecil, porsi J total ``b``, rasio J total kelompok, dan rata-rata
    perubahan kedua metode.
    """
    k = _per_kondisi(baris)
    k = k[np.isclose(k.lam, lam)]
    t = k.pivot_table(index=KONDISI, columns="metode", values=["J", "ubah", "telat"])
    t = t.dropna()
    ja, jb = t[("J", a)], t[("J", b)]
    inst = t.index.get_level_values("instans")
    q = pd.Series(index=t.index, dtype=int)
    for n in inst.unique():
        pilih = inst == n
        urut = jb[pilih].rank(method="first")
        q[pilih] = np.ceil(urut / pilih.sum() * 4).astype(int).clip(1, 4).to_numpy()
    rasio = ja / jb
    out = []
    total_b = jb.sum()
    for g in (1, 2, 3, 4):
        s = q == g
        out.append({"kelompok": g, "rata_rasio": float(rasio[s].mean()),
                    "median_rasio": float(rasio[s].median()),
                    "porsi_a_lebih_kecil": float((ja[s] < jb[s]).mean()),
                    "porsi_J_b": float(jb[s].sum() / total_b),
                    "rasio_total": float(ja[s].sum() / jb[s].sum()),
                    "ubah_a": float(t[("ubah", a)][s].mean()),
                    "ubah_b": float(t[("ubah", b)][s].mean()),
                    "telat_a": float(t[("telat", a)][s].mean()),
                    "telat_b": float(t[("telat", b)][s].mean()),
                    "n": int(s.sum())})
    return pd.DataFrame(out)


# --------------------------------------------------------------------- S3

_RE_EPOCH = re.compile(r"BC epoch (\d+): loss ([\d.]+) akurasi ([\d.]+) tahan ([\d.]+)"
                       r"(?: rugi tahan ([\d.]+))?")
_RE_BERHENTI = re.compile(r"BC berhenti di epoch (\d+)")
_RE_LANGKAH = re.compile(r"langkah (\d+): reward rata (-?[\d.]+)")
_RE_SKALA = re.compile(r"skala reward c\(lambda\): (\[.*\])")
_RE_BARIS = re.compile(r"BC_TAMBAH: (\d+) baris BC")


def baca_log(teks):
    """S3: teks log ``latih_ppo_seq.py`` -> ringkasan pelatihan.

    Kembalikan kamus: ``bc`` (daftar epoch dengan loss, akurasi, tahan, rugi), ``berhenti``
    (epoch berhenti atau None), ``terbaik`` (epoch yang bobotnya dipakai, menurut aturan
    di ``behaviour_cloning``: rugi tahan turun lebih dari 1e-4), ``ppo`` (daftar
    (langkah, reward rata)), ``skala_c`` (daftar (λ, c)), ``bc_baris`` (banyak baris BC bila
    dicatat).
    """
    bc, ppo, skala, berhenti, n_baris = [], [], [], None, None
    for baris in teks.splitlines():
        m = _RE_EPOCH.search(baris)
        if m:
            bc.append({"epoch": int(m[1]), "loss": float(m[2]), "akurasi": float(m[3]),
                       "tahan": float(m[4]), "rugi": None if m[5] is None else float(m[5])})
            continue
        m = _RE_BERHENTI.search(baris)
        if m:
            berhenti = int(m[1])
            continue
        m = _RE_LANGKAH.search(baris)
        if m:
            ppo.append((int(m[1]), float(m[2])))
            continue
        m = _RE_SKALA.search(baris)
        if m:
            skala = [(float(a), float(c)) for a, c in
                     re.findall(r"\(([\d.]+), ([\d.]+)\)", m[1])]
            continue
        m = _RE_BARIS.search(baris)
        if m:
            n_baris = int(m[1])
    terbaik, skor = None, -math.inf
    for e in bc:
        s = -e["rugi"] if e["rugi"] is not None else e["tahan"]
        if terbaik is None or s > skor + 1e-4:
            terbaik, skor = e["epoch"], s
    return {"bc": bc, "berhenti": berhenti, "terbaik": terbaik, "ppo": ppo, "skala_c": skala,
            "bc_baris": n_baris}


# --------------------------------------------------------------------- S4

def banding_bobot(awal, akhir):
    """S4: perubahan bobot per lapis antara dua kumpulan bobot dengan nama yang sama.

    ``awal`` dan ``akhir`` = {nama lapis: array}. Kembalikan daftar kamus per lapis: nama,
    banyak angka, norma awal, norma perubahan, perubahan relatif (norma perubahan / norma
    awal; None bila norma awal nol, misalnya bias critic yang belum dilatih BC), dan porsi
    angka yang berubah lebih dari 1e-8.
    """
    if set(awal) != set(akhir):
        raise ValueError("nama lapis kedua jaringan berbeda")
    out = []
    for nama in awal:
        a, b = np.asarray(awal[nama], float), np.asarray(akhir[nama], float)
        if a.shape != b.shape:
            raise ValueError(f"bentuk {nama} berbeda: {a.shape} lawan {b.shape}")
        d = b - a
        na, nd = float(np.linalg.norm(a)), float(np.linalg.norm(d))
        out.append({"lapis": nama, "n": int(a.size), "norma_awal": na, "norma_ubah": nd,
                    "ubah_relatif": nd / na if na > 0 else None,
                    "porsi_berubah": float((np.abs(d) > 1e-8).mean())})
    return out


def kesamaan_pilihan(pa, pb, guru=None, lam=None):
    """S4: seberapa sering dua jaringan memilih tingkat yang sama pada keadaan yang sama.

    ``pa``, ``pb`` = tingkat pilihan (0-3) per keadaan; ``guru`` = label guru (opsional);
    ``lam`` = λ per keadaan (opsional, untuk rincian per λ). Kembalikan kamus: ``sama``,
    ``matriks`` (4x4, baris = pa, kolom = pb), ``sama_guru_a``, ``sama_guru_b``, dan
    ``per_lam`` bila ``lam`` diberikan.
    """
    pa, pb = np.asarray(pa), np.asarray(pb)

    def ringkas(s):
        r = {"n": int(s.sum()), "sama": float((pa[s] == pb[s]).mean())}
        if guru is not None:
            g = np.asarray(guru)
            r["sama_guru_a"] = float((pa[s] == g[s]).mean())
            r["sama_guru_b"] = float((pb[s] == g[s]).mean())
        return r

    semua = np.ones(len(pa), bool)
    out = ringkas(semua)
    m = np.zeros((4, 4), int)
    np.add.at(m, (pa, pb), 1)
    out["matriks"] = m
    if lam is not None:
        lam = np.asarray(lam, float)
        out["per_lam"] = {float(v): ringkas(np.isclose(lam, v)) for v in np.unique(lam)}
    return out


# --------------------------------------------------------------------- S5

def ringkas_guru(baris, bobot=None):
    """S5: baris data guru (x, label, D, λ) -> ringkasan per λ.

    D = J(tingkat) - J(sekarang) untuk tingkat 0-3, NaN (atau None) bila tingkat tidak sah.
    ``bobot`` = bobot contoh BC yang sudah dihitung fungsi asli (``regret_weights``), satu per
    baris, opsional. Per λ: n, porsi label, porsi sah per tingkat, rata-rata D per tingkat
    (hanya baris yang sah), porsi keputusan yang semua tingkat sahnya sama (selisih <= 1e-9),
    dan ringkasan bobot (rata-rata, median, p95, porsi bobot di bawah 0,1).
    """
    lams = np.array([float(r[3]) for r in baris])
    y = np.array([int(r[1]) for r in baris])
    D = np.array([[np.nan if v is None else float(v) for v in r[2]] for r in baris], float)
    w = None if bobot is None else np.asarray(bobot, float)
    if w is not None and len(w) != len(baris):
        raise ValueError("banyak bobot tidak sama dengan banyak baris")
    out = {}
    for lam in np.unique(lams):
        s = np.isclose(lams, lam)
        Ds = D[s]
        sah = ~np.isnan(Ds)
        rentang = np.nanmax(Ds, 1) - np.nanmin(Ds, 1)
        r = {"n": int(s.sum()),
             "porsi_label": [float((y[s] == k).mean()) for k in range(4)],
             "porsi_sah": [float(sah[:, k].mean()) for k in range(4)],
             "rata_selisih": [float(np.nanmean(Ds[:, k])) if sah[:, k].any() else math.nan
                              for k in range(4)],
             "porsi_seri": float((rentang <= 1e-9).mean())}
        if w is not None:
            ws = w[s]
            r["bobot"] = {"rata": float(ws.mean()), "median": float(np.median(ws)),
                          "p95": float(np.percentile(ws, 95)),
                          "porsi_kecil": float((ws < 0.1).mean())}
        out[float(lam)] = r
    return out


# --------------------------------------------------------------------- S6

def catatan_kejadian(langkah, lam):
    """S6: catatan mentah satu episode -> baris per kejadian.

    ``langkah`` = daftar kamus per kejadian dengan kunci ``t``, ``jenis`` (daftar jenis
    gangguan), ``diminta`` dan ``dipakai`` (daftar tingkat per sub-keputusan),
    ``telat_sebelum`` (keterlambatan/E sesudah gangguan diterapkan dengan tingkat 0),
    ``telat_sesudah`` (keterlambatan/E jadwal yang dipasang), dan ``ubah`` (perubahan
    kejadian itu). Kembalikan daftar kamus dengan tambahan ``no``, ``pulih`` (keterlambatan
    yang dipulihkan), ``harga`` (λ x ubah), ``untung`` (pulih - harga, sama dengan J_AOR -
    J_akhir kejadian, dasar reward), ``ubah_kumulatif``, dan ``J`` (telat_sesudah + λ x
    ubah kumulatif).
    """
    out, kum = [], 0
    for i, r in enumerate(langkah, 1):
        kum += r["ubah"]
        pulih = r["telat_sebelum"] - r["telat_sesudah"]
        harga = lam * r["ubah"]
        out.append({**r, "no": i, "pulih": pulih, "harga": harga, "untung": pulih - harga,
                    "ubah_kumulatif": kum, "J": r["telat_sesudah"] + lam * kum})
    return out


def tabel_kejadian(rows, lam, caption, label):
    """Teks LaTeX untuk keluaran :func:`catatan_kejadian`."""
    def tingkat(v):
        return "".join(str(int(a)) for a in v)

    baris = [f"{r['no']} & {angka(r['t'])} & {len(r['jenis'])} & {tingkat(r['diminta'])} & "
             f"{tingkat(r['dipakai'])} & {angka(r['telat_sebelum'])} & "
             f"{angka(r['telat_sesudah'])} & {r['ubah']} & {r['ubah_kumulatif']} & "
             f"{angka(r['pulih'])} & {angka(r['harga'])} & {angka(r['J'])}" for r in rows]
    return _tabel(caption, label, "rrcllrrrrrrr",
                  "No & $t$ & $k$ & Diminta & Dipakai & \\shortstack[r]{Telat\\\\sebelum} & "
                  "\\shortstack[r]{Telat\\\\sesudah} & $\\Delta$ & $\\Sigma$ & Pulih & "
                  "$\\lambda\\Delta$ & $J$", baris,
                  f"$\\lambda = {_lam(lam)}$. $k$: banyak gangguan dalam kejadian. Diminta dan "
                  "dipakai: tingkat per sub-keputusan berurutan. Telat dalam satuan $E$. "
                  "$\\Delta$: perubahan kejadian itu; $\\Sigma$: perubahan kumulatif. Pulih: "
                  "keterlambatan yang dipulihkan (telat sebelum dikurangi telat sesudah). "
                  "$J$: telat sesudah ditambah $\\lambda\\Sigma$.")
