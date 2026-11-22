"""Tes pembuat tabel Bab V (``naskah_bab5``, seam S1-S6, disetujui pemilik 21 November 2026).

Masukan kecil dibuat tangan, dan keluaran yang diharapkan dihitung tangan, bukan dengan rumus
yang sama dengan kode.

    $PY -m pytest -p no:cacheprovider -q test_naskah_bab5.py
"""
from __future__ import annotations

import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

from naskah_bab5 import (angka, baca_log, banding_bobot, catatan_kejadian, kesamaan_pilihan,
                         kuartil_rasio, ringkas_guru, tabel_analisis, tabel_kejadian,
                         tabel_uraian, uraian_j)


# ------------------------------------------------------------------ format

def test_angka_koma_desimal_dan_tanda():
    assert angka(-1.5) == "$-1{,}50$"
    assert angka(18.144, tanda=True) == "$+18{,}14$"
    assert angka(0.5, d=3) == "$0{,}500$"
    assert angka(None) == "--"
    assert angka(-0.001, tanda=True) == "$+0{,}00$"


# ------------------------------------------------------------------ S1

def _br(e, lo, hi):
    return NS(estimate=e, low=lo, high=hi)


def _h():
    bg = lambda r1, r2: NS(rasio_rata=_br(*r1), rasio_total=_br(*r2), dikeluarkan_uk1=0)  # noqa
    return {
        "utama": {"fix3_g": {0.3: bg((18.14, 7.86, 32.65), (-4.45, -6.29, -0.91))},
                  "acak_g": {0.3: bg((-24.01, -26.59, -21.38), (-23.62, -25.58, -21.74))}},
        "benchmark_saja": {"fix3_g": {0.3: bg((17.47, 8.03, 29.39), (-5.02, -6.36, -2.21))},
                           "acak_g": {0.3: bg((-23.96, -26.09, -21.73),
                                              (-23.18, -24.80, -21.60))}},
        "gabungan": {"fix3_g": {0.3: bg((18.14, 9.47, 29.33), (-4.45, -5.77, -1.74))}},
        "butir1": {"0,1 lawan 0,3": NS(rasio_rata=_br(9.14, 2.07, 17.75),
                                        rasio_total=_br(-0.62, -2.20, 1.05))},
        "waktu": {"ppo_bc_ppo_g": {"p95_pemilih_maks": 1.92, "di_bawah_50ms": True,
                                   "p95_total_maks": 466.42},
                  "seq": {"p95_pemilih_maks": None, "di_bawah_50ms": None,
                          "p95_total_maks": 1021.01}},
        "pelanggaran": {"ppo_bc_ppo_g": {"episode": 33210, "keras": 0, "tanpa_periksa": 0}},
        "stabil": {"fix3_g": {0.3: _br(-10.42, -11.87, -8.98), 3.0: _br(0.2, -0.1, 0.5)}},
        "tingkat": {"MK": {0.3: {"porsi_diminta": [0.31, 0.08, 0.17, 0.45],
                                 "porsi_dipakai": [0.54, 0.07, 0.13, 0.26],
                                 "laju_tolak_penjaga": 0.11, "laju_gagal": 0.22},
                           0.1: {"porsi_diminta": [1, 0, 0, 0], "porsi_dipakai": [1, 0, 0, 0],
                                 "laju_tolak_penjaga": 0, "laju_gagal": 0}}},
        "per_keluarga": {"MK": {"fix3_g": {0.3: {"rasio_rata": 25.96, "rasio_total": 2.13}}}},
    }


def test_s1_uji_utama_menulis_kalimat_menurut_kunci_uji():
    t = tabel_analisis(_h())["uji_utama"]
    assert ("Fixed-L3 + penjaga & 0{,}3 & $+18{,}14$ [$+7{,}86$; $+32{,}65$]$^{*}$ & "
            "$-4{,}45$ [$-6{,}29$; $-0{,}91$]$^{*}$ & tidak tegas \\\\") in t
    assert ("Random + penjaga & 0{,}3 & $-24{,}01$ [$-26{,}59$; $-21{,}38$]$^{*}$ & "
            "$-23{,}62$ [$-25{,}58$; $-21{,}74$]$^{*}$ & lebih baik \\\\") in t
    # Random sebelum Fixed-L3 (urutan URUTAN), label tabel ada.
    assert t.index("Random + penjaga") < t.index("Fixed-L3 + penjaga")
    assert "\\label{tab:uji-utama}" in t


def test_s1_butir1_waktu_layak_perubahan_tingkat_keluarga():
    t = tabel_analisis(_h())
    assert ("0,1 lawan 0,3 & $+9{,}14$ [$+2{,}07$; $+17{,}75$]$^{*}$ & "
            "$-0{,}62$ [$-2{,}20$; $+1{,}05$] \\\\") in t["uji_butir1"]
    assert "ARLS-PPO & $1{,}92$ & ya & $466{,}42$ \\\\" in t["uji_waktu"]
    assert "Sequential Lookahead & -- & -- & $1021{,}01$ \\\\" in t["uji_waktu"]
    assert "ARLS-PPO & 33.210 & 0 & 0 \\\\" in t["uji_layak"]
    assert "Fixed-L3 + penjaga & $-10{,}42$$^{*}$ & $+0{,}20$ \\\\" in t["uji_perubahan"]
    # λ 0,1 tidak berlabel, jadi tidak tampil di tabel tingkat.
    assert ("MK & 0{,}3 & $0{,}31$ & $0{,}08$ & $0{,}17$ & $0{,}45$ & $0{,}54$ & $0{,}07$ & "
            "$0{,}13$ & $0{,}26$ & $0{,}11$ & $0{,}22$ \\\\") in t["uji_tingkat"]
    assert "MK & 0{,}1" not in t["uji_tingkat"]
    assert "MK & -- & -- & -- & -- & $+25{,}96$ & $+2{,}13$ & -- & -- \\\\" in t["uji_keluarga"]


def test_s1_perubahan_selang_lengkap_untuk_lampiran():
    # Catatan Tabel uji-perubahan menjanjikan selang lengkap di Lampiran 6.
    t = tabel_analisis(_h())["uji_perubahan_selang"]
    assert "\\label{tab:uji-perubahan-selang}" in t
    assert "Fixed-L3 + penjaga & 0{,}3 & $-10{,}42$ [$-11{,}87$; $-8{,}98$]$^{*}$ \\\\" in t
    assert "Fixed-L3 + penjaga & 3 & $+0{,}20$ [$-0{,}10$; $+0{,}50$] \\\\" in t


# ------------------------------------------------------------------ S2

def _ep(instans, seed, metode, lam, telat, ubah, kap=1.0, inten=0.05):
    return dict(instans=instans, kapasitas=kap, intensitas=inten, seed=seed, metode=metode,
                lam=lam, telat=telat, ubah=ubah, J=telat + lam * ubah)


def test_s2_instans_berbobot_sama_dan_seed_model_dirata_rata():
    baris = [
        # Instans A: dua kondisi; PPO dua seed model pada kondisi pertama.
        _ep("A", 1, "ppo", 3.0, 10.0, 2), _ep("A", 1, "ppo", 3.0, 14.0, 4),
        _ep("A", 2, "ppo", 3.0, 6.0, 0),
        # Instans B: satu kondisi.
        _ep("B", 1, "ppo", 3.0, 30.0, 6),
    ]
    r = uraian_j(baris).iloc[0]
    # A: kondisi 1 -> telat 12, ubah 3; kondisi 2 -> telat 6, ubah 0; rata A: telat 9, ubah 1,5.
    # B: telat 30, ubah 6. Atas instans: telat 19,5, ubah 3,75, harga 11,25, J 30,75.
    assert r.telat == pytest.approx(19.5)
    assert r.ubah == pytest.approx(3.75)
    assert r.harga_ubah == pytest.approx(11.25)
    assert r.J == pytest.approx(30.75)
    assert r.n_instans == 2


def test_s2_menolak_baris_dengan_j_tidak_konsisten():
    b = _ep("A", 1, "ppo", 3.0, 10.0, 2)
    b["J"] = 99.0
    with pytest.raises(ValueError):
        uraian_j([b])


def test_s2_tabel_uraian():
    df = uraian_j([_ep("A", 1, "ppo_bc_ppo_g", 0.3, 2.0, 10), _ep("A", 1, "seq", 0.3, 1.0, 20)])
    t = tabel_uraian(df, ["ppo_bc_ppo_g", "seq"], "judul", "tab:x", "cat")
    assert "0{,}3 & ARLS-PPO & $2{,}00$ & $10{,}00$ & $3{,}00$ & $5{,}00$ \\\\" in t
    assert "0{,}3 & Sequential Lookahead & $1{,}00$ & $20{,}00$ & $6{,}00$ & $7{,}00$ \\\\" in t


def test_kuartil_rasio_per_instans():
    # Satu instans, empat kondisi (seed 1-4), λ 1. J b = 10, 20, 30, 40; J a = 20, 20, 15, 20.
    baris = []
    for s, (ja, jb) in enumerate([(20, 10), (20, 20), (15, 30), (20, 40)], 1):
        baris += [_ep("A", s, "a", 1.0, ja - 1, 1), _ep("A", s, "b", 1.0, jb - 2, 2)]
    q = kuartil_rasio(baris, "a", "b", 1.0)
    assert list(q.rata_rasio) == pytest.approx([2.0, 1.0, 0.5, 0.5])
    assert list(q.porsi_a_lebih_kecil) == [0.0, 0.0, 1.0, 1.0]
    assert list(q.porsi_J_b) == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert list(q.ubah_a) == [1, 1, 1, 1] and list(q.ubah_b) == [2, 2, 2, 2]


# ------------------------------------------------------------------ S3

LOG = """\
skala reward c(lambda): [(0.3, 1.234), (3.0, 2.5), (30.0, 9.75)]
BC_TAMBAH: 86832 baris BC
BC epoch 0: loss 1.100 akurasi 0.500 tahan 0.480 rugi tahan 0.900
BC epoch 1: loss 0.900 akurasi 0.600 tahan 0.590 rugi tahan 0.500
BC epoch 2: loss 0.800 akurasi 0.650 tahan 0.600 rugi tahan 0.49995
BC epoch 3: loss 0.700 akurasi 0.700 tahan 0.610 rugi tahan 0.400
BC berhenti di epoch 3, ukuran terbaik -0.400
BC selesai 12 s
langkah 4096: reward rata 0.1234
langkah 8192: reward rata -0.0500
selesai 99 s
"""


def test_s3_baca_log():
    r = baca_log(LOG)
    assert [e["epoch"] for e in r["bc"]] == [0, 1, 2, 3]
    assert r["bc"][1] == {"epoch": 1, "loss": 0.9, "akurasi": 0.6, "tahan": 0.59, "rugi": 0.5}
    assert r["berhenti"] == 3
    # Epoch 2 turun hanya 0,00005 (< 1e-4), jadi tidak dihitung lebih baik; epoch 3 terbaik.
    assert r["terbaik"] == 3
    assert r["ppo"] == [(4096, 0.1234), (8192, -0.05)]
    assert r["skala_c"] == [(0.3, 1.234), (3.0, 2.5), (30.0, 9.75)]
    assert r["bc_baris"] == 86832


def test_s3_terbaik_tanpa_perbaikan_kecil():
    teks = "\n".join(["BC epoch 0: loss 1 akurasi 0.5 tahan 0.5 rugi tahan 0.30",
                      "BC epoch 1: loss 1 akurasi 0.5 tahan 0.5 rugi tahan 0.29995"])
    assert baca_log(teks)["terbaik"] == 0


# ------------------------------------------------------------------ S4

def test_s4_banding_bobot():
    awal = {"w": np.array([[3.0, 0.0], [0.0, 4.0]]), "b": np.array([1.0, 0.0])}
    akhir = {"w": np.array([[3.0, 0.0], [0.0, 4.0]]), "b": np.array([1.0, 1.0])}
    r = {x["lapis"]: x for x in banding_bobot(awal, akhir)}
    assert r["w"]["norma_awal"] == pytest.approx(5.0)
    assert r["w"]["norma_ubah"] == 0.0 and r["w"]["porsi_berubah"] == 0.0
    assert r["b"]["n"] == 2
    assert r["b"]["ubah_relatif"] == pytest.approx(1.0)
    assert r["b"]["porsi_berubah"] == 0.5
    with pytest.raises(ValueError):
        banding_bobot(awal, {"w": awal["w"]})


def test_s4_banding_bobot_norma_awal_nol_tidak_terdefinisi():
    # Bias critic belum dilatih BC, jadi semua nol sebelum PPO: perubahan relatif tidak punya
    # pembagi. Hasilnya None (tercetak "--"), bukan inf (tercetak "$inf$" dan JSON tidak sah).
    awal = {"v": np.zeros(3)}
    akhir = {"v": np.array([0.0, 3.0, 4.0])}
    r = banding_bobot(awal, akhir)[0]
    assert r["norma_awal"] == 0.0
    assert r["norma_ubah"] == pytest.approx(5.0)
    assert r["ubah_relatif"] is None
    assert angka(r["ubah_relatif"]) == "--"


def test_s4_kesamaan_pilihan():
    pa = [0, 0, 1, 3, 3]
    pb = [0, 1, 1, 3, 0]
    guru = [0, 1, 1, 3, 3]
    lam = [0.3, 0.3, 3, 3, 3]
    r = kesamaan_pilihan(pa, pb, guru, lam)
    assert r["sama"] == pytest.approx(3 / 5)
    assert r["sama_guru_a"] == pytest.approx(4 / 5)
    assert r["sama_guru_b"] == pytest.approx(4 / 5)
    assert r["matriks"][3, 0] == 1 and r["matriks"][0, 1] == 1 and r["matriks"].sum() == 5
    assert r["per_lam"][0.3] == {"n": 2, "sama": 0.5, "sama_guru_a": 0.5, "sama_guru_b": 1.0}


# ------------------------------------------------------------------ S5

def test_s5_ringkas_guru():
    nan = float("nan")
    baris = [
        (None, 0, [0.0, 0.0, None, 0.0], 0.3),      # semua sah seri
        (None, 3, [0.0, -1.0, nan, -2.0], 0.3),
        (None, 1, [0.0, -4.0, -3.0, -4.0], 3.0),
    ]
    r = ringkas_guru(baris, bobot=[0.05, 1.0, 2.0])
    a = r[0.3]
    assert a["n"] == 2
    assert a["porsi_label"] == [0.5, 0.0, 0.0, 0.5]
    assert a["porsi_sah"] == [1.0, 1.0, 0.0, 1.0]
    assert a["rata_selisih"][1] == pytest.approx(-0.5)
    assert a["rata_selisih"][3] == pytest.approx(-1.0)
    assert math.isnan(a["rata_selisih"][2])
    assert a["porsi_seri"] == 0.5
    assert a["bobot"]["porsi_kecil"] == 0.5
    assert r[3.0]["porsi_label"] == [0.0, 1.0, 0.0, 0.0]
    assert r[3.0]["bobot"]["median"] == 2.0
    with pytest.raises(ValueError):
        ringkas_guru(baris, bobot=[1.0])


# ------------------------------------------------------------------ S6

def test_s6_catatan_kejadian():
    langkah = [
        {"t": 2.849, "jenis": ["a", "a"], "diminta": [1, 0], "dipakai": [1, 0],
         "telat_sebelum": 18.9373, "telat_sesudah": 0.0102, "ubah": 1},
        {"t": 5.0, "jenis": ["b"], "diminta": [3], "dipakai": [0],
         "telat_sebelum": 2.0102, "telat_sesudah": 2.0102, "ubah": 0},
        {"t": 7.5, "jenis": ["c"], "diminta": [2], "dipakai": [2],
         "telat_sebelum": 4.0, "telat_sesudah": 1.0, "ubah": 2},
    ]
    r = catatan_kejadian(langkah, 3.0)
    assert [x["no"] for x in r] == [1, 2, 3]
    assert [x["ubah_kumulatif"] for x in r] == [1, 1, 3]
    assert r[0]["pulih"] == pytest.approx(18.9271)
    assert r[0]["harga"] == 3.0 and r[0]["untung"] == pytest.approx(15.9271)
    # J akhir = telat sesudah kejadian terakhir + 3 x 3 = 10.
    assert r[2]["J"] == pytest.approx(10.0)
    assert r[1]["J"] == pytest.approx(5.0102)
    t = tabel_kejadian(r, 3.0, "judul", "tab:k")
    assert ("1 & $2{,}85$ & 2 & 10 & 10 & $18{,}94$ & $0{,}01$ & 1 & 1 & $18{,}93$ & "
            "$3{,}00$ & $3{,}01$ \\\\") in t


def test_s1_tabel_terhadap_aor():
    from naskah_bab5 import tabel_terhadap_aor
    hasil = {"ppo_bc_ppo_g": {
        "J": {0.3: NS(rasio_rata=_br(-76.32, -77.20, -75.42), rasio_total=_br(-78.64, -79.2, -77.75))},
        "telat": {0.3: _br(-300.5, -310.0, -290.0)}, "ubah": {0.3: _br(105.35, 98.16, 112.62)}}}
    t = tabel_terhadap_aor(hasil)
    assert ("ARLS-PPO & 0{,}3 & $-76{,}32$ [$-77{,}20$; $-75{,}42$]$^{*}$ & "
            "$-78{,}64$ [$-79{,}20$; $-77{,}75$]$^{*}$ & $-300{,}50$ [$-310{,}00$; $-290{,}00$]"
            "$^{*}$ & $+105{,}35$ [$+98{,}16$; $+112{,}62$]$^{*}$ \\\\") in t


def test_s1_ringkas_sifat_spearman():
    import pandas as pd
    from naskah_bab5 import ringkas_sifat, tabel_sifat
    d = pd.DataFrame({"pembanding": ["fix3_g"] * 4, "lam": [0.3] * 4,
                      "n_tugas": [10, 20, 30, 40],
                      "rasio_rata": [1.0, 2.0, 3.0, 9.0],      # naik terus -> +1
                      "rasio_total": [4.0, 3.0, 2.0, 1.0]})   # turun terus -> -1
    rho = ringkas_sifat(d, "fix3_g", [0.3], ["n_tugas"])
    assert rho["n_tugas"][(0.3, "rasio_rata")] == pytest.approx(1.0)
    assert rho["n_tugas"][(0.3, "rasio_total")] == pytest.approx(-1.0)
    t = tabel_sifat(rho, "fix3_g", [0.3])
    assert "Banyak tugas & $+1{,}00$ & $-1{,}00$ \\\\" in t
