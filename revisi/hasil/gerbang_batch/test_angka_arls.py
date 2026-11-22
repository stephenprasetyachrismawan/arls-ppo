"""Tes pemuat angka hasil ARLS-PPO untuk artikel dan dek (seam ``angka_arls``).

Seam disetujui pemilik 22 November 2026: satu pemuat bersama yang membaca JSON hasil Bab V
(``chapters/chapter5/generated/*.json``) dan mengembalikan angka bernama. Masukan tes dibuat
tangan; angka yang diharapkan dihitung tangan, bukan dengan rumus yang sama dengan kode.

    $PY -m pytest -p no:cacheprovider -q test_angka_arls.py
"""
from __future__ import annotations

import pytest

from angka_arls import HasilArls, Selang


def _data():
    return {
        "uraian": [
            {"metode": "ppo_bc_ppo_g", "lam": 0.1, "telat": 771.62, "ubah": 110.79, "J": 782.70},
            {"metode": "aor", "lam": 0.1, "telat": 4582.26, "ubah": 0.0, "J": 4582.26},
            {"metode": "ppo_bc_ppo_g", "lam": 30.0, "telat": 1500.0, "ubah": 32.0, "J": 2460.0},
            {"metode": "ppo_bc_ppo_g", "lam": 3.0, "telat": 900.0, "ubah": 80.0, "J": 1140.0},
        ],
        "aor": {"ppo_bc_ppo_g": {
            "J": {"0.3": {"uk1": [-76.32, -77.20, -75.42], "uk2": [-78.64, -79.20, -77.75]}},
            "telat": {"0.3": [-3802.09, -4194.41, -3422.77]},
            "ubah": {"0.3": [105.35, 98.16, 112.62]}}},
        "kuartil": {"0.3": [
            {"kelompok": 1, "rata_rasio": 1.641, "porsi_a_lebih_kecil": 0.3237, "porsi_J_b": 0.0299,
             "ubah_a": 26.99, "ubah_b": 34.82, "n": 451},
            {"kelompok": 4, "rata_rasio": 0.910, "porsi_a_lebih_kecil": 0.7382, "porsi_J_b": 0.6313,
             "ubah_a": 203.95, "ubah_b": 214.97, "n": 489}]},
        "latih": {
            "bobot": {"0": [{"lapis": "mlp_extractor.policy_net.0.weight", "ubah_relatif": 0.0301},
                            {"lapis": "action_net.weight", "ubah_relatif": 0.0182},
                            {"lapis": "mlp_extractor.value_net.0.weight", "ubah_relatif": 0.1808},
                            {"lapis": "value_net.bias", "ubah_relatif": None}],
                      "1": [{"lapis": "mlp_extractor.policy_net.0.weight", "ubah_relatif": 0.0297},
                            {"lapis": "action_net.weight", "ubah_relatif": 0.0100},
                            {"lapis": "mlp_extractor.value_net.0.weight", "ubah_relatif": 0.1270},
                            {"lapis": "value_net.bias", "ubah_relatif": None}]},
            "pilih": {"0": {"sama": 0.940, "sama_guru_a": 0.831, "sama_guru_b": 0.823},
                      "1": {"sama": 0.916, "sama_guru_a": 0.825, "sama_guru_b": 0.784}}},
        "episode": {"kondisi": ["MK01", 1.5, 0.15, 5000], "model_seed": 0, "lam": {"3.0": {
            "kejadian": [{"diminta": [3, 2], "dipakai": [0, 2], "pulih": 19.25},
                         {"diminta": [0], "dipakai": [0], "pulih": 0.0},
                         {"diminta": [1, 0, 3], "dipakai": [1, 0, 3], "pulih": 45.81}],
            "J_akhir": 284.89, "telat_akhir": 221.89, "ubah": 21}}},
        "validasi": [
            {"metode": "tahap_bc", "lam": 0.3, "telat": 487.30, "ubah": 79.14, "J": 511.04},
            {"metode": "tahap_dagger", "lam": 0.3, "telat": 466.52, "ubah": 78.16, "J": 489.96},
        ],
        "ringkas": {
            "utama": {
                "fix3_g": {"0.3": {"uk1": [18.14, 7.86, 32.65], "uk2": [-4.45, -6.29, -0.91]}},
                "acak_g": {"0.3": {"uk1": [-24.01, -26.59, -21.38],
                                   "uk2": [-23.62, -25.58, -21.74]}},
            },
            "waktu": {
                "ppo_bc_ppo_g": {"p95_pemilih_per_instans": {"MK03": 1.60, "la01": 1.49,
                                                             "la36": 1.92},
                                 "p95_pemilih_maks": 1.92, "p95_total_maks": 466.42},
                "seq": {"p95_pemilih_per_instans": {}, "p95_pemilih_maks": None,
                        "p95_total_maks": 1021.01},
            },
            "pelanggaran": {"aor": {"keras": 0, "episode": 11070, "tanpa_periksa": 0},
                            "ppo_bc_ppo_g": {"keras": 0, "episode": 33210, "tanpa_periksa": 0},
                            "il_g": {"keras": 2, "episode": 11070, "tanpa_periksa": 0}},
            "butir1": {"10 lawan interpolasi log 3 dan 30": {"uk1": [5.14, 3.64, 6.68],
                                                             "uk2": [4.31, 3.18, 5.47]}},
            "stabil": {"fix3_g": {"0.3": [-10.42, -11.87, -8.98], "30.0": [-2.56, -3.21, -1.92]}},
            "gabungan": {
                "aor": {"0.1": {"uk1": [-77.49, -78.39, -76.59], "uk2": [-79.46, -80.03, -78.57]},
                        "3.0": {"uk1": [-64.98, -65.97, -63.97], "uk2": [-69.90, -70.44, -68.78]},
                        "30.0": {"uk1": [-21.40, -22.43, -20.37], "uk2": [-30.59, -31.22, -28.87]}},
            },
        },
    }


def test_banding_membaca_selang_utama_dan_menandai_yang_tegas():
    h = HasilArls(_data())
    s = h.banding("fix3_g", 0.3, "uk1", keluarga="utama")
    assert s == Selang(18.14, 7.86, 32.65)
    assert s.tegas
    assert not Selang(0.49, -2.17, 4.20).tegas


def test_kalimat_mengikuti_aturan_kunci_uji():
    h = HasilArls(_data())
    assert h.kalimat("acak_g", 0.3) == "lebih baik"
    # uk. 1 tegas di atas nol, uk. 2 tegas di bawah nol: tidak searah.
    assert h.kalimat("fix3_g", 0.3) == "tidak tegas"


def test_rentang_titik_atas_semua_lambda():
    h = HasilArls(_data())
    assert h.rentang("aor", "uk1") == (-77.49, -21.40)
    assert h.rentang("aor", "uk2") == (-79.46, -30.59)


def test_waktu_pemilih_dan_total_dalam_milidetik():
    h = HasilArls(_data())
    assert h.waktu("ppo_bc_ppo_g") == {"pemilih_min": 1.49, "pemilih_maks": 1.92,
                                       "total_maks": 466.42}
    # Metode tanpa pemilih terpisah tidak punya waktu pemilih.
    assert h.waktu("seq") == {"pemilih_min": None, "pemilih_maks": None, "total_maks": 1021.01}


def test_pelanggaran_dijumlah_atas_semua_metode():
    h = HasilArls(_data())
    assert h.pelanggaran() == {"episode": 55350, "keras": 2}


def test_uraian_dan_perubahan_antartahap():
    h = HasilArls(_data())
    assert h.uraian("ppo_bc_ppo_g", 0.1) == {"telat": 771.62, "ubah": 110.79, "J": 782.70}
    # (489,96 / 511,04 - 1) x 100 = -4,1249...
    assert h.tahap("tahap_bc", "tahap_dagger", 0.3) == pytest.approx(-4.1249, abs=1e-3)
    with pytest.raises(KeyError):
        h.uraian("ppo_bc_ppo_g", 10)


def test_butir1_dan_selisih_perubahan():
    h = HasilArls(_data())
    assert h.butir1("10 lawan interpolasi log 3 dan 30", "uk2") == Selang(4.31, 3.18, 5.47)
    assert h.perubahan("fix3_g", 30) == Selang(-2.56, -3.21, -1.92)


def test_selisih_terhadap_aor():
    h = HasilArls(_data())
    assert h.aor("ppo_bc_ppo_g", 0.3, "uk2") == Selang(-78.64, -79.20, -77.75)
    assert h.aor("ppo_bc_ppo_g", 0.3, "ubah") == Selang(105.35, 98.16, 112.62)
    assert h.aor("ppo_bc_ppo_g", 0.3, "telat").titik == -3802.09


def test_baca_folder_json_hasil_bab5(tmp_path):
    import json
    d = _data()
    berkas = {"bab5_uji_ringkas.json": d["ringkas"], "bab5_uji_uraian.json": d["uraian"],
              "bab5_uji_aor.json": d["aor"], "bab5_validasi_tahap.json": d["validasi"],
              "bab5_uji_kuartil.json": d["kuartil"], "bab5_latih_bobot.json": d["latih"],
              "bab5_episode.json": d["episode"]}
    for nama, isi in berkas.items():
        (tmp_path / nama).write_text(json.dumps(isi), encoding="utf-8")
    h = HasilArls.baca(tmp_path)
    assert h.kalimat("acak_g", 0.3) == "lebih baik"
    assert h.uraian("aor", 0.1)["J"] == 4582.26
    (tmp_path / "bab5_uji_aor.json").unlink()
    with pytest.raises(FileNotFoundError, match="bab5_uji_aor.json"):
        HasilArls.baca(tmp_path)


def test_kurva_satu_metode_urut_menurut_lambda():
    h = HasilArls(_data())
    assert h.kurva("ppo_bc_ppo_g") == [(0.1, 771.62, 110.79), (3.0, 900.0, 80.0),
                                       (30.0, 1500.0, 32.0)]


def test_kuartil_per_kelompok():
    h = HasilArls(_data())
    q4 = h.kuartil(0.3, 4)
    assert q4["porsi_J_b"] == 0.6313 and q4["porsi_a_lebih_kecil"] == 0.7382
    assert h.kuartil(0.3, 1)["rata_rasio"] == 1.641


def test_rentang_perubahan_bobot_dan_kesamaan_pilihan():
    h = HasilArls(_data())
    # actor = policy_net dan action_net; critic = value_net; None (norma awal nol) dilewati.
    assert h.bobot("actor") == (0.0100, 0.0301)
    assert h.bobot("critic") == (0.1270, 0.1808)
    assert h.pilih("sama") == (0.916, 0.940)
    assert h.pilih("sama_guru_b") == (0.784, 0.823)


def test_ringkasan_satu_episode():
    h = HasilArls(_data())
    assert h.episode(3) == {"kejadian": 3, "sub_keputusan": 6, "diminta": 4, "tidak_dipakai": 1,
                            "pulih": pytest.approx(65.06), "ubah": 21, "telat_akhir": 221.89,
                            "J_akhir": 284.89}
