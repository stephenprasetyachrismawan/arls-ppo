"""Pemuat benchmark FJSP publik menjadi instans proyek perangkat lunak.

Sumber: ``SchedulingLab/fjsp-instances`` (MIT), commit
``ac4c3402312bfbeafcf4472d78be567d4e6b46ab``, disalin ke ``data/raw/benchmark``. Pembagian
latih, validasi, dan uji dibuat menurut instans dasar supaya tidak ada kebocoran
(revisi/rencana_benchmark_uji_20261118.md bagian 11).
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..core.domain import to_software_project
from ..core.parser import parse_brandimarte_json_text
from ..core.validate import validate_instance
from ._brandimarte import _PROJECT_ROOT, _check_sha256, _read_checksums
from ._instance import ProjectInstance

#: Rata-rata upaya acuan: rata-rata dari rata-rata upaya per pasangan tugas-pengembang di 7 MK
#: latih (MK01, 02, 05, 06, 07, 08, 10), dihitung dari berkas mentah. Gangguan berukuran mutlak
#: dalam jam, jadi instans dengan satuan waktu lain disamakan dulu ke acuan ini.
UPAYA_ACUAN = 7.525106985177461
#: Rentang rata-rata upaya per instans di 7 MK latih. Instans di dalam rentang ini tidak diskalakan.
RENTANG_UPAYA_LATIH = (3.819327731092437, 12.037267080745341)

_HURINK = ("edata", "rdata", "vdata")


def _hurink(nomor):
    return tuple(f"hurink/{s}/la{n:02d}" for n in nomor for s in _HURINK)


#: 60 instans latih tambahan (di samping 7 MK latih).
BENCHMARK_LATIH = _hurink((2, 3, 4, 7, 8, 9, 12, 13, 14, 17, 18, 19, 22, 23, 27, 28, 29,
                           37, 38)) + ("dauzere/05a", "dauzere/10a", "dauzere/11a")
#: 23 instans validasi, untuk memilih resep; tidak pernah dipakai latih.
BENCHMARK_VALIDASI = _hurink((5, 10, 15, 20, 25, 30, 39)) + ("dauzere/06a", "dauzere/12a")
#: 35 instans uji. Hanya dimuat dengan izin buka uji.
BENCHMARK_UJI = (tuple(f"brandimarte/mk{i}" for i in range(11, 16))
                 + _hurink((1, 6, 11, 16, 21, 26, 36))
                 + ("barnes/mt10c1", "barnes/setb4c9", "barnes/seti5c12")
                 + tuple(f"dauzere/{x}a" for x in ("01", "02", "03", "07", "08", "09")))


def _skala(obj):
    """Faktor skala upaya: acuan dibagi rata-rata instans, atau 1 bila sudah di rentang latih."""
    rata = statistics.mean(c["processing"] for job in obj["jobs"] for op in job for c in op)
    lo, hi = RENTANG_UPAYA_LATIH
    return 1.0 if lo <= rata <= hi else UPAYA_ACUAN / rata


def load_benchmark(names, *, buka_uji=False, k_tight=1.5, capacity_mult=1.2, p_cross_dep=0.10,
                   domain_seed=0, data_home=None) -> dict[str, ProjectInstance]:
    """Muat instans benchmark dan ubah jadi instans proyek perangkat lunak.

    Tiap instans dibaca dari ``<data_home>/raw/benchmark/<nama>.json`` dan diperiksa SHA-256-nya
    terhadap ``configs/checksums.yaml``. Upaya dikali satu faktor supaya rata-ratanya sama
    dengan :data:`UPAYA_ACUAN`; faktornya dicatat di ``meta["skala_upaya"]``. Transformasi
    domain sama dengan :func:`load_brandimarte`.

    Parameters
    ----------
    names : daftar str
        Nama instans, misalnya ``"hurink/edata/la02"``, dari :data:`BENCHMARK_LATIH`,
        :data:`BENCHMARK_VALIDASI`, atau :data:`BENCHMARK_UJI`.
    buka_uji : bool, default=False
        Izin memuat instans uji. Hanya untuk pembukaan resmi data uji sesudah pemilik setuju.
    k_tight, capacity_mult, p_cross_dep, domain_seed, data_home
        Sama dengan :func:`load_brandimarte`.

    Returns
    -------
    dict
        Nama instans ke ``ProjectInstance``.

    Raises
    ------
    PermissionError
        Ada instans uji di ``names`` tanpa ``buka_uji=True``. Berkasnya tidak dibuka.
    ValueError
        Nama tidak dikenal, SHA-256 tidak cocok, atau validasi gagal.
    FileNotFoundError
        Berkas mentah atau ``configs/checksums.yaml`` tidak ada.

    Examples
    --------
    >>> from dfjsp.datasets import load_benchmark
    >>> inst = load_benchmark(["hurink/edata/la02"])["hurink/edata/la02"]
    >>> len(inst.tasks)
    50
    >>> round(inst.meta["skala_upaya"], 3)
    0.145
    """
    chosen = list(names)
    semua = set(BENCHMARK_LATIH) | set(BENCHMARK_VALIDASI) | set(BENCHMARK_UJI)
    unknown = [n for n in chosen if n not in semua]
    if unknown:
        raise ValueError(f"nama instans benchmark tidak dikenal: {unknown}")
    uji = [n for n in chosen if n in BENCHMARK_UJI]
    if uji and not buka_uji:
        raise PermissionError(f"{uji} adalah data uji; muat dengan buka_uji=True hanya pada "
                              "pembukaan resmi sesudah pemilik setuju")

    home = Path(data_home) if data_home is not None else _PROJECT_ROOT / "data"
    checksums = _read_checksums(home)
    cfg = SimpleNamespace(
        k_tight=float(k_tight),
        domain=SimpleNamespace(capacity_mult=float(capacity_mult),
                               p_cross_dep=float(p_cross_dep), seed=int(domain_seed)),
        priority=SimpleNamespace(invert=False),
    )
    out: dict[str, ProjectInstance] = {}
    for name in chosen:
        path = home / "raw" / "benchmark" / f"{name}.json"
        if not path.is_file():
            raise FileNotFoundError(f"berkas benchmark {name} tidak ditemukan: {path}")
        _check_sha256(path, checksums.get(name))
        obj = json.loads(path.read_text(encoding="utf-8"))
        s = _skala(obj)
        for job in obj["jobs"]:
            for op in job:
                for c in op:
                    c["processing"] = c["processing"] * s
        base = parse_brandimarte_json_text(json.dumps(obj), name)
        # Pemeriksaan "ada di Tabel 3.2" hanya berlaku untuk nama MK.
        problems = [p for p in validate_instance(base) if "Tabel 3.2" not in p]
        if problems:
            raise ValueError(f"validasi gagal untuk {name}: {problems}")
        inst = to_software_project(base, cfg, np.random.default_rng(cfg.domain.seed))
        inst.meta["skala_upaya"] = s
        out[name] = inst
    return out
