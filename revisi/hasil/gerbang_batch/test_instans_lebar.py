"""Pembangkit instans acak rentang lebar untuk E16 (``instans_acak.generate_lebar``).

Seam disetujui pemilik 18 November 2026. Rentang mengikuti sifat benchmark yang dipilih
(revisi/rencana_benchmark_uji_20261118.md bagian 10): 10-30 job, 4-18 pengembang, paling
banyak 300 tugas, fleksibilitas rata-rata 1,0-7,0, upaya setara data latih (1-19).

    PYTHONPATH=../../../exp1/dfjsp-ppo-repair/src $PY -m pytest -p no:cacheprovider -q test_instans_lebar.py
"""
from __future__ import annotations

import statistics

from instans_acak import generate_lebar


def _sifat(obj):
    ops = [op for job in obj["jobs"] for op in job]
    return {"job": len(obj["jobs"]), "dev": obj["machines"], "tugas": len(ops),
            "fleks": statistics.mean(len(op) for op in ops),
            "upaya": [c["processing"] for op in ops for c in op],
            "mesin": [c["machine"] for op in ops for c in op],
            "dobel": any(len({c["machine"] for c in op}) != len(op) for op in ops)}


def test_instans_lebar_tetap_untuk_idx_sama_dan_patuh_rentang():
    assert generate_lebar(3) == generate_lebar(3)
    assert generate_lebar(3) != generate_lebar(4)
    for idx in range(60):
        s = _sifat(generate_lebar(idx))
        assert 10 <= s["job"] <= 30, (idx, s["job"])
        assert 4 <= s["dev"] <= 18, (idx, s["dev"])
        assert s["tugas"] <= 300, (idx, s["tugas"])
        assert 1.0 <= s["fleks"] <= 7.0, (idx, s["fleks"])
        assert min(s["upaya"]) >= 1 and max(s["upaya"]) <= 19, idx
        assert all(0 <= m < s["dev"] for m in s["mesin"]) and not s["dobel"], idx


def test_himpunan_latih_w_mencapai_sumbu_benchmark():
    # Tujuan E16: data latih menjangkau sumbu tempat benchmark keluar dari rentang 7 MK
    # (Hurink edata fleksibilitas 1,13; vdata sampai 6,70; Barnes 18 pengembang;
    # Dauzere-Peres 293 tugas). Batas di bawah ini ditulis dari angka itu.
    semua = [_sifat(generate_lebar(i)) for i in range(60)]
    # Tiap ujung fleksibilitas diwakili paling sedikit 5 dari 60 instans, bukan satu saja.
    assert sum(s["fleks"] <= 1.3 for s in semua) >= 5
    assert sum(s["fleks"] >= 5.0 for s in semua) >= 5
    assert max(s["dev"] for s in semua) >= 16
    assert max(s["tugas"] for s in semua) >= 280
    assert min(s["tugas"] for s in semua) <= 80


def test_nama_w_dimuat_sebagai_instans_proyek_dan_nama_r_tetap():
    from instans_acak import load_any, names_from_env

    assert names_from_env("W0-2") == ["W000", "W001", "W002"]
    assert names_from_env("100-101") == ["R100", "R101"]
    inst = load_any("W105")
    banyak = sum(len(job) for job in generate_lebar(105)["jobs"])
    assert inst.name == "W105"
    assert len(inst.tasks) == banyak
    assert inst.n_devs == generate_lebar(105)["machines"]
    assert all(t.due_date > 0 for t in inst.tasks)
