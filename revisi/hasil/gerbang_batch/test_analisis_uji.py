"""Analisis hasil uji (``analisis_uji.analisis``): semua tabel Bab V dari baris episode.

Seam disetujui pemilik 18 November 2026: ``analisis(baris, usulan, pembanding, sifat)``.
Keputusan yang dijalankan: draf kunci uji bagian 6-9 (gabungan berbobot sama per instans,
kepekaan benchmark saja, keluarga utama 4 uji dengan selang 98,75%, analisis sifat).

    PYTHONPATH=../../../exp1/dfjsp-ppo-repair/src $PY -m pytest -p no:cacheprovider -q test_analisis_uji.py
"""
from __future__ import annotations

from analisis_uji import analisis, keluarga, tulis_laporan


def _baris(instans, faktor, lam=(0.3, 3.0), seeds=(1, 2), tingkat=None):
    """Baris sintetis: J pembanding 100 per kondisi, J metode = 100 x faktor[metode]."""
    out = []
    for inst in instans:
        for inten in (0.05, 0.15):
            for s in seeds:
                for l in lam:
                    for m, f in faktor.items():
                        r = dict(instans=inst, kapasitas=1.0, intensitas=inten, seed=s, lam=l,
                                 metode=m, J=100.0 * (f(inst) if callable(f) else f))
                        if tingkat and m in tingkat:
                            r["tingkat"] = tingkat[m]
                        out.append(r)
    return out


def test_nama_instans_dipetakan_ke_keluarga():
    assert keluarga("MK03") == "MK"
    assert keluarga("brandimarte/mk12") == "Brandimarte mk11-15"
    assert keluarga("hurink/rdata/la06") == "Hurink rdata"
    assert keluarga("barnes/seti5c12") == "Barnes"
    assert keluarga("dauzere/08a") == "Dauzère-Pérès"


def test_titik_per_keluarga():
    # PPO 10% lebih buruk di Hurink edata, 20% lebih baik di MK.
    f = {"ppo": lambda i: 1.1 if i.startswith("hurink") else 0.8, "acak_g": 1.0}
    h = analisis(_baris(["MK03", "MK04", "hurink/edata/la01"], f), "ppo", ["acak_g"], {},
                 n_resamples=100)
    per = h["per_keluarga"]
    assert abs(per["Hurink edata"]["acak_g"][3.0]["rasio_rata"] - 10.0) < 1e-9
    assert abs(per["MK"]["acak_g"][0.3]["rasio_total"] + 20.0) < 1e-9


def test_gabungan_benchmark_saja_dan_keluarga_utama():
    # Tiga instans berbobot sama: MK03 0,8; dua Hurink 1,1. Gabungan uk. 1 = 0%;
    # benchmark saja = +10%.
    f = {"ppo": lambda i: 1.1 if i.startswith("hurink") else 0.8, "acak_g": 1.0,
         "fix3_g": 1.0, "seq": 1.0}
    inst = ["MK03", "hurink/edata/la01", "hurink/rdata/la01"]
    h = analisis(_baris(inst, f, lam=(0.3, 3.0, 30.0)), "ppo", ["acak_g", "fix3_g", "seq"], {},
                 n_resamples=300)
    assert abs(h["gabungan"]["seq"][30.0].rasio_rata.estimate) < 1e-9
    assert abs(h["benchmark_saja"]["acak_g"][3.0].rasio_total.estimate - 10.0) < 1e-9
    # Keluarga utama: lawan tingkat 3 + penjaga dan acak + penjaga di λ 0,3 dan 3 saja.
    utama = h["utama"]
    assert set(utama) == {"acak_g", "fix3_g"}
    assert all(set(utama[b]) == {0.3, 3.0} for b in utama)
    g95, g99 = h["gabungan"]["acak_g"][3.0].rasio_rata, utama["acak_g"][3.0].rasio_rata
    assert g99.estimate == g95.estimate and g99.low <= g95.low and g99.high >= g95.high


def test_sebaran_tingkat_dan_penolakan_penjaga_per_keluarga():
    # Per episode: diminta 0,0,1,2; tingkat 2 ditolak penjaga, jadi dipakai 0,0,0,1.
    t = {"diminta": [2, 1, 1, 0], "dipakai": [3, 1, 0, 0], "gagal": 0, "ditolak_penjaga": 1}
    h = analisis(_baris(["MK03", "barnes/mt10c1"], {"ppo": 0.9, "acak_g": 1.0},
                        tingkat={"ppo": t}), "ppo", ["acak_g"], {}, n_resamples=50)
    b = h["tingkat"]["Barnes"][3.0]
    assert b["porsi_diminta"] == [0.5, 0.25, 0.25, 0.0]
    assert b["porsi_dipakai"] == [0.75, 0.25, 0.0, 0.0]
    # Dari 2 permintaan tingkat 1-3, satu ditolak penjaga.
    assert b["laju_tolak_penjaga"] == 0.5 and b["laju_gagal"] == 0.0


def test_tabel_per_instans_memasangkan_rasio_dengan_sifat():
    f = {"ppo": lambda i: 0.9 if i.startswith("barnes") else 1.2, "acak_g": 1.0}
    sifat = {"MK03": {"fleks_rata": 3.0}, "barnes/mt10c1": {"fleks_rata": 1.1}}
    h = analisis(_baris(["MK03", "barnes/mt10c1"], f), "ppo", ["acak_g"], sifat,
                 n_resamples=50)
    t = h["per_sifat"]
    r = t[(t.instans == "barnes/mt10c1") & (t.lam == 3.0) & (t.pembanding == "acak_g")]
    assert len(r) == 1
    assert r.iloc[0]["fleks_rata"] == 1.1 and r.iloc[0]["keluarga"] == "Barnes"
    assert abs(r.iloc[0]["rasio_total"] + 10.0) < 1e-9
    assert len(t) == 2 * 2 * 1    # 2 instans x 2 λ x 1 pembanding


def test_pelanggaran_keras_dijumlah_per_metode():
    """I-2: jumlah pelanggaran keras pemeriksa jadwal per metode, dari kolom ``periksa``;
    temuan lunak tidak dihitung, baris tanpa pemeriksa dilaporkan banyaknya."""
    baris = _baris(["MK03"], {"ppo": 0.9, "acak_g": 1.0})
    for r in baris:
        r["periksa"] = {"hitung": {"B4:lunak": 5}}
    ppo = [r for r in baris if r["metode"] == "ppo"]
    ppo[0]["periksa"] = {"hitung": {"B1:keras": 2, "B4:lunak": 1}}
    ppo[1]["periksa"] = {"hitung": {"B2:keras": 1}}
    del ppo[2]["periksa"]
    h = analisis(baris, "ppo", ["acak_g"], {}, n_resamples=50)
    assert h["pelanggaran"]["ppo"] == {"keras": 3, "episode": len(ppo), "tanpa_periksa": 1}
    assert h["pelanggaran"]["acak_g"] == {"keras": 0, "episode": len(ppo), "tanpa_periksa": 0}


def test_butir1_celah_ke_sequential_untuk_keempat_pasangan_lambda():
    """Butir 1 (draf kunci uji bagian 7): pasangan λ ARSITEKTUR_PIPELINE 5.6 dipakai apa
    adanya. Usulan 10% di atas Sequential di λ berlabel dan 30% di λ tanpa label: setiap
    pasangan (bobot acuan berjumlah 1) memberi celah +20 poin dengan kedua ukuran."""
    lams = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
    f = {"ppo": lambda i: 1.0, "seq": 1.0}
    baris = _baris(["MK03", "hurink/edata/la01"], f, lam=lams)
    for r in baris:
        if r["metode"] == "ppo":
            r["J"] = 100.0 * (1.1 if r["lam"] in (0.3, 3.0, 30.0) else 1.3)
    h = analisis(baris, "ppo", ["seq"], {}, n_resamples=50)
    assert len(h["butir1"]) == 4
    for x in h["butir1"].values():
        assert abs(x.rasio_rata.estimate - 20.0) < 1e-9
        assert abs(x.rasio_total.high - 20.0) < 1e-9


def test_i1_ambang_pada_waktu_pemilih_waktu_total_tanpa_ambang():
    """I-1 (keputusan pemilik 19 November 2026, sama dengan definisi lama): p95 per instans,
    semua episode instans digabung. Ambang 50 ms hanya untuk waktu pemilih (``ms_pemilih``);
    waktu total (``ms_keputusan``) dilaporkan tanpa ambang. Metode tanpa pemilih terpisah
    (Sequential) tidak dinilai. Waktu 1..100 ms di instans A: p95 = 95,05."""
    baris = _baris(["MK03", "MK04"], {"ppo": 0.9, "acak_g": 1.0, "seq": 1.0}, lam=(3.0,),
                   seeds=(1,))
    a = [r for r in baris if r["instans"] == "MK03" and r["metode"] == "ppo"]
    a[0]["ms_pemilih"] = [float(x) for x in range(1, 51)]
    a[1]["ms_pemilih"] = [float(x) for x in range(51, 101)]
    for r in baris:
        r.setdefault("ms_pemilih", [] if r["metode"] == "seq" else [2.0, 2.0])
        r["ms_keputusan"] = [200.0, 200.0]
    h = analisis(baris, "ppo", ["acak_g", "seq"], {}, n_resamples=50)
    w = h["waktu"]["ppo"]
    assert abs(w["p95_pemilih_per_instans"]["MK03"] - 95.05) < 1e-9
    assert w["p95_pemilih_per_instans"]["MK04"] == 2.0
    assert abs(w["p95_pemilih_maks"] - 95.05) < 1e-9 and w["di_bawah_50ms"] is False
    assert w["p95_total_maks"] == 200.0
    assert h["waktu"]["acak_g"]["di_bawah_50ms"] is True
    assert h["waktu"]["seq"]["di_bawah_50ms"] is None
    assert h["waktu"]["seq"]["p95_total_maks"] == 200.0


def _dengan_ubah(baris, ubah):
    """Tambah kolom ``ubah`` (banyak perubahan jadwal per episode) dari {metode: nilai}."""
    for r in baris:
        u = ubah[r["metode"]]
        r["ubah"] = float(u(r["instans"]) if callable(u) else u)
    return baris


def test_i3_selisih_perubahan_jadwal_per_lambda_tanpa_ambang():
    # Seam disetujui pemilik 19 November 2026 (I-3 deskriptif, KERANGKA_NASKAH bagian 8).
    # Usulan selalu 2 perubahan lebih banyak dari pembanding di setiap episode.
    baris = _dengan_ubah(_baris(["MK03", "hurink/edata/la01"], {"ppo": 1.0, "aor": 1.0}),
                         {"ppo": 5, "aor": 3})
    h = analisis(baris, "ppo", ["aor"], {}, n_resamples=200)
    for lam in (0.3, 3.0):
        s = h["stabil"]["aor"][lam]
        assert abs(s.estimate - 2.0) < 1e-12
        assert abs(s.low - 2.0) < 1e-12 and abs(s.high - 2.0) < 1e-12


def test_i3_bobot_sama_per_instans_bukan_per_episode():
    # MK03 punya 3 kali lebih banyak kondisi (6 seed) dengan selisih 1; Hurink 2 seed dengan
    # selisih 3. Bobot sama per instans: (1 + 3) / 2 = 2. Bobot per kondisi akan memberi 1,5.
    f = {"ppo": 1.0, "aor": 1.0}
    u = {"ppo": lambda i: 4 if i == "MK03" else 6, "aor": 3}
    baris = (_dengan_ubah(_baris(["MK03"], f, seeds=(1, 2, 3, 4, 5, 6)), u)
             + _dengan_ubah(_baris(["hurink/edata/la01"], f, seeds=(1, 2)), u))
    h = analisis(baris, "ppo", ["aor"], {}, n_resamples=200)
    s = h["stabil"]["aor"][3.0]
    assert abs(s.estimate - 2.0) < 1e-12
    assert abs(s.low - 2.0) < 1e-12 and abs(s.high - 2.0) < 1e-12


def test_i3_tertulis_di_laporan_sebagai_selisih_tanpa_ambang():
    baris = _dengan_ubah(_baris(["MK03", "hurink/edata/la01"], {"ppo": 1.0, "aor": 1.0}),
                         {"ppo": 5, "aor": 3})
    teks = tulis_laporan(analisis(baris, "ppo", ["aor"], {}, n_resamples=50), "ppo", "uji")
    assert "## Perubahan jadwal (I-3)" in teks
    assert "| aor | 0,3 | +2,00 [+2,00; +2,00]* |" in teks
    assert "| aor | 3 | +2,00 [+2,00; +2,00]* |" in teks
