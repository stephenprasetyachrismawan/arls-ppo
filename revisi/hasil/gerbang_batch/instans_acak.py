"""E3: instans acak bergaya Brandimarte untuk menambah ragam data latih.

Brandimarte (1993) membuat MK01-MK10 dengan mengambil acak banyak job, mesin, operasi per job,
mesin yang sah per operasi, dan waktu proses dari rentang tertentu. Di sini caranya sama,
dengan rentang yang diambil HANYA dari 7 instans latih (MK01, 02, 05, 06, 07, 08, 10):

| ciri | rentang di instans latih | rentang pembangkit |
|---|---|---|
| banyak job | 10 sampai 20 | 10 sampai 20 |
| banyak mesin (pengembang) | 4 sampai 15 | 4 sampai 15 |
| operasi per job | 5 sampai 15 | batas bawah 5-15 condong ke kecil, batas atas paling banyak +4, maks 15 |
| mesin sah per operasi, maksimum | 2 sampai 6 | 2 sampai min(6, banyak mesin) |
| waktu proses | 1-6 sampai 5-19 | bawah 1 atau 5, atas 6 sampai 19 |

Jumlah operasi dibatasi 240 (sama dengan MK10, instans latih terbesar). Instans uji (MK03,
MK04, MK09) tidak dipakai untuk menetapkan rentang.

Sesudah dibangkitkan, instans melewati jalur yang sama dengan MK: ``parse_brandimarte_json_text``,
``validate_instance``, lalu ``to_software_project`` dengan parameter bawaan
``load_brandimarte``. Nama instans "R000", "R001", dan seterusnya; instans ke-i memakai seed
30000 + i, terpisah dari seed gangguan (validasi 5000-5001, guru 7000-7001, uji 10000-10009,
rollout 20000 ke atas).
"""
from __future__ import annotations

import penjaga_uji  # noqa: F401  (memblokir pembukaan data uji)

import json
import sys
from types import SimpleNamespace

import numpy as np

from dfjsp.core.domain import to_software_project
from dfjsp.core.parser import parse_brandimarte_json_text
from dfjsp.core.validate import validate_instance
from dfjsp.datasets import (BENCHMARK_LATIH, BENCHMARK_VALIDASI, load_benchmark,
                            load_brandimarte)

SEED0 = 30000
MAX_OPS = 240


def generate(idx: int) -> dict:
    """Instans acak ke-idx dalam skema JSON yang sama dengan ``data/raw/MK*.json``."""
    rng = np.random.default_rng(SEED0 + idx)
    while True:
        n_jobs = int(rng.integers(10, 21))
        n_mach = int(rng.integers(4, 16))
        # Batas bawah operasi per job condong ke kecil, supaya instans kecil seperti MK01
        # (55 operasi) ikut terwakili; tanpa ini hampir semua instans di atas 100 operasi.
        lo = 5 + int(rng.integers(0, 11) * rng.random())
        hi = min(15, lo + int(rng.integers(0, 5)))
        if n_jobs * (lo + hi) / 2 <= MAX_OPS:
            break
    fmax = int(rng.integers(2, min(6, n_mach) + 1))
    pmin = int(rng.choice([1, 5]))
    pmax = int(rng.integers(max(6, pmin + 4), 20))
    jobs = []
    for _ in range(n_jobs):
        ops = []
        for _ in range(int(rng.integers(lo, hi + 1))):
            k = int(rng.integers(1, fmax + 1))
            machines = rng.choice(n_mach, size=k, replace=False)
            ops.append([{"machine": int(m), "processing": int(rng.integers(pmin, pmax + 1))}
                        for m in sorted(machines)])
        jobs.append(ops)
    return {"machines": n_mach, "jobs": jobs}


SEED_LEBAR = 31000
MAX_OPS_LEBAR = 300


def generate_lebar(idx: int) -> dict:
    """Instans acak rentang lebar ke-idx untuk E16 (nama W, seed 31000 + idx).

    Cara sama dengan :func:`generate`, rentang diperlebar mengikuti sifat benchmark uji
    yang dipilih (revisi/rencana_benchmark_uji_20261118.md bagian 10): 10-30 job, 4-18
    pengembang, paling banyak 300 tugas, fleksibilitas rata-rata 1,0-7,0. Upaya tetap
    1-19 seperti data latih, karena upaya benchmark diskalakan ke rata-rata data latih.

    Banyak tugas sasaran diambil merata 40-300. Fleksibilitas sasaran diambil merata pada
    skala log antara 1 dan min(6,5, banyak pengembang). Banyak pengembang yang sah per tugas
    = 1 + Binomial(pengembang - 1, q), dengan q dipilih supaya rata-ratanya sama dengan
    sasaran itu.
    """
    rng = np.random.default_rng(SEED_LEBAR + idx)
    while True:
        n_jobs = int(rng.integers(10, 31))
        n_mach = int(rng.integers(4, 19))
        # Ukuran sasaran diambil merata 40-300 tugas, supaya instans besar seperti
        # Dauzere-Peres (293) ikut terwakili, lalu dibagi ke job.
        rata = int(rng.integers(40, MAX_OPS_LEBAR + 1)) / n_jobs
        if not 5 <= rata <= 20:
            continue
        lo = max(5, int(rata) - 2)
        hi = min(20, lo + 4)
        # Banyak tugas per job diambil dulu; bila totalnya lewat 300, ulangi seluruhnya.
        per_job = [int(rng.integers(lo, hi + 1)) for _ in range(n_jobs)]
        if sum(per_job) <= MAX_OPS_LEBAR:
            break
    # Sasaran fleksibilitas diambil merata pada skala log, supaya fleksibilitas rendah
    # (Hurink edata, sekitar 1,15) sama terwakilinya dengan yang tinggi (vdata, sampai 6,7).
    sasaran = float(np.exp(rng.uniform(0.0, np.log(min(6.5, n_mach)))))
    q = (sasaran - 1.0) / (n_mach - 1)
    pmin = int(rng.choice([1, 5]))
    pmax = int(rng.integers(max(6, pmin + 4), 20))
    jobs = []
    for n_ops in per_job:
        ops = []
        for _ in range(n_ops):
            k = 1 + int(rng.binomial(n_mach - 1, q))
            machines = rng.choice(n_mach, size=k, replace=False)
            ops.append([{"machine": int(m), "processing": int(rng.integers(pmin, pmax + 1))}
                        for m in sorted(machines)])
        jobs.append(ops)
    return {"machines": n_mach, "jobs": jobs}


def name_of(idx: int) -> str:
    return f"R{idx:03d}"


def load_random(name: str):
    """Muat instans acak "Rxxx" atau rentang lebar "Wxxx" sebagai instans proyek, sama
    dengan jalur MK."""
    idx = int(name[1:])
    pembangkit = generate_lebar if name.startswith("W") else generate
    base = parse_brandimarte_json_text(json.dumps(pembangkit(idx)), name)
    # Pemeriksaan "ada di Tabel 3.2" hanya berlaku untuk nama MK, jadi dilewati di sini.
    problems = [p for p in validate_instance(base) if "Tabel 3.2" not in p]
    if problems:
        raise ValueError(f"validasi gagal untuk {name}: {problems}")
    cfg = SimpleNamespace(
        k_tight=1.5,
        domain=SimpleNamespace(capacity_mult=1.2, p_cross_dep=0.10, seed=0),
        priority=SimpleNamespace(invert=False),
    )
    return to_software_project(base, cfg, np.random.default_rng(0))


def load_any(name: str):
    """MK lewat ``load_brandimarte``, Rxxx dan Wxxx lewat pembangkit, benchmark (nama berisi
    "/", misalnya "hurink/edata/la02") lewat ``load_benchmark``. Instans uji benchmark
    ditolak pemuat dengan ``PermissionError``."""
    if "/" in name:
        # Izin buka uji hanya bila penjaga_uji sudah dilepas di proses ini (buka_uji.py).
        penjaga = sys.modules.get("penjaga_uji")
        izin = bool(penjaga is not None and penjaga.diizinkan())
        return load_benchmark([name], buka_uji=izin)[name]
    if name.startswith(("R", "W")):
        return load_random(name)
    return load_brandimarte([name])[name]


def names_from_env(spec: str) -> list[str]:
    """"0-49" menjadi R000 sampai R049; "W0-59" menjadi W000 sampai W059 (rentang lebar);
    "bench:latih" dan "bench:validasi" menjadi daftar benchmark latih atau validasi (cara D).
    Daftar benchmark uji tidak bisa dipilih di sini."""
    if spec.startswith("bench:"):
        daftar = {"latih": BENCHMARK_LATIH, "validasi": BENCHMARK_VALIDASI}
        bagian = spec.split(":", 1)[1]
        if bagian not in daftar:
            raise ValueError(f"{spec}: hanya bench:latih atau bench:validasi yang boleh")
        return list(daftar[bagian])
    awalan = "W" if spec.startswith("W") else "R"
    a, b = spec.lstrip("W").split("-")
    return [f"{awalan}{i:03d}" for i in range(int(a), int(b) + 1)]


def daftar_instans(bawaan, ganti=None, tambah=None) -> list[str]:
    """Daftar nama instans untuk skrip gerbang.

    ``ganti`` (misalnya "W100-119") menggantikan ``bawaan``; ``tambah`` (misalnya "W0-59")
    ditambahkan di belakang. Keduanya kosong: ``bawaan`` apa adanya, perilaku lama.
    """
    nama = names_from_env(ganti) if ganti else list(bawaan)
    if tambah:
        nama += names_from_env(tambah)
    return nama


def model_instans(models, name):
    """Model untuk instans ``name`` dari kamus ``{nama MK: {jenis: model}}``.

    ``latih_hgb.py`` menyimpan satu model yang sama di bawah tiap nama MK. Instans R atau W
    tidak punya kunci sendiri, jadi dipakai model bersama itu. Bila model per MK berbeda
    (model LOIO), tidak ada model yang sah untuk instans lain dan pemanggilan ditolak.
    """
    if name in models:
        return models[name]
    isi = list(models.values())
    if all(v.keys() == isi[0].keys() and all(v[k] is isi[0][k] for k in v) for v in isi):
        return isi[0]
    raise ValueError(f"{name} tidak punya model, dan model per instans berbeda (LOIO?)")
