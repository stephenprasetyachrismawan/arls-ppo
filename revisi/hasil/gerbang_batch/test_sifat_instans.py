"""Sifat instans untuk analisis perilaku metode (``sifat_instans``).

Seam disetujui pemilik 18 November 2026: ``sifat_instans(inst)`` dan
``di_rentang_latih(sifat, sifat_latih)``. Sifat dihitung dari instans sesudah transformasi
domain, sebelum episode (revisi/rencana_benchmark_uji_20261118.md bagian 7).

    PYTHONPATH=../../../exp1/dfjsp-ppo-repair/src $PY -m pytest -p no:cacheprovider -q test_sifat_instans.py
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from dfjsp.core.domain import to_software_project
from dfjsp.core.parser import parse_brandimarte_json_text

from sifat_instans import di_rentang_latih, sifat_instans


def _kecil():
    # 2 job x 2 tugas, 2 pengembang. Upaya per pasangan: 2 | 4,4 | 3 | 1,3.
    obj = {"machines": 2, "jobs": [
        [[{"machine": 0, "processing": 2}],
         [{"machine": 0, "processing": 4}, {"machine": 1, "processing": 4}]],
        [[{"machine": 1, "processing": 3}],
         [{"machine": 0, "processing": 1}, {"machine": 1, "processing": 3}]]]}
    cfg = SimpleNamespace(k_tight=1.5,
                          domain=SimpleNamespace(capacity_mult=1.2, p_cross_dep=0.0, seed=0),
                          priority=SimpleNamespace(invert=False))
    base = parse_brandimarte_json_text(json.dumps(obj), "kecil")
    return to_software_project(base, cfg, np.random.default_rng(0))


def test_sifat_instans_kecil_sesuai_hitungan_tangan():
    s = sifat_instans(_kecil())
    assert (s["n_job"], s["n_tugas"], s["n_dev"]) == (2, 4, 2)
    assert s["tugas_per_dev"] == 2.0
    assert s["fleks_rata"] == 1.5 and s["fleks_sd"] == 0.5
    assert abs(s["upaya_rata"] - 17 / 6) < 1e-12
    # Upaya nominal per tugas (rata-rata antar pengembang sah): 2, 4, 3, 2; total 11.
    assert s["beban_per_dev"] == 5.5
    # Tiap pengembang sah untuk 3 tugas: tidak timpang.
    assert s["ketimpangan_keahlian"] == 0.0
    # Tenggat akhir job = 1,5 x total upaya job: 9 dan 7,5, rata-rata 8,25.
    assert abs(s["keketatan"] - 5.5 / 8.25) < 1e-12
    # Tanpa dependensi antar-job: hanya tugas kedua tiap job yang punya prasyarat.
    assert s["dep_rata"] == 0.5
    assert s["skala_upaya"] == 1.0


def test_posisi_terhadap_rentang_latih_per_sifat():
    latih = [{"n_dev": 4, "fleks_rata": 1.4}, {"n_dev": 15, "fleks_rata": 4.1}]
    assert di_rentang_latih({"n_dev": 18, "fleks_rata": 1.13}, latih) == \
        {"n_dev": "di atas", "fleks_rata": "di bawah"}
    # Batas rentang termasuk di dalam.
    assert di_rentang_latih({"n_dev": 15, "fleks_rata": 2.0}, latih) == \
        {"n_dev": "di dalam", "fleks_rata": "di dalam"}


def test_nilai_sama_dengan_batas_kecuali_pembulatan_tetap_di_dalam():
    # 10 job / (1,5 x 10) dan 15 job / (1,5 x 15) sama-sama 2/3, tetapi bilangan pecahannya
    # bisa berbeda di digit terakhir.
    latih = [{"keketatan": 10 / (1.5 * 10)}, {"keketatan": 2.0}]
    assert di_rentang_latih({"keketatan": (15 / 15) / 1.5 - 1e-15}, latih) == \
        {"keketatan": "di dalam"}
