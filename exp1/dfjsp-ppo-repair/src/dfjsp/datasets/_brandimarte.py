"""Pemuat instans Brandimarte MK01 sampai MK10 menjadi instans proyek perangkat lunak."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from ..core.domain import to_software_project
from ..core.parser import parse_brandimarte
from ..core.validate import validate_instance
from ._instance import ProjectInstance

BRANDIMARTE_NAMES = tuple(f"MK{i:02d}" for i in range(1, 11))

# src/dfjsp/datasets/_brandimarte.py -> naik empat tingkat = folder proyek.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _read_checksums(data_home: Path) -> dict[str, str]:
    """Baca ``configs/checksums.yaml`` di samping ``data_home``; berkas hilang adalah galat.

    Tanpa daftar checksum, berkas mentah yang berubah tidak akan ketahuan, jadi pemuatan
    berhenti (audit W7 F12) alih-alih melewati pemeriksaan diam-diam.
    """
    path = data_home.parent / "configs" / "checksums.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"berkas checksum tidak ada: {path}. Tanpa itu data mentah tidak bisa diperiksa; "
            "hentikan dulu.")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {str(k): str(v) for k, v in loaded.items()}


def _check_sha256(path: Path, expected: str | None) -> None:
    """Tolak berkas mentah yang isinya berbeda dari checksum terkunci, atau tanpa checksum."""
    if expected is None:
        raise ValueError(f"{path.stem} tidak punya checksum di configs/checksums.yaml; "
                         "tambahkan dulu SHA-256 terkunci untuk berkas ini.")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(
            f"SHA-256 {path.name} tidak cocok. Terkunci {expected[:12]}..., "
            f"terbaca {actual[:12]}.... Berkas data berubah, hentikan dulu."
        )


def load_brandimarte(names=None, *, k_tight=1.5, capacity_mult=1.2, p_cross_dep=0.10,
                     domain_seed=0, data_home=None) -> dict[str, ProjectInstance]:
    """Muat instans Brandimarte dan ubah jadi instans proyek perangkat lunak.

    Tiap instans dibaca dari ``<data_home>/raw/<nama>.json``, diperiksa, lalu diberi
    domain proyek: mesin jadi pengembang, kapasitas, dependensi antartugas, tenggat
    dari total beban kerja, dan prioritas. Hasilnya sama persis dengan jalur lama
    (``build_real_instances_legacy``) untuk parameter yang sama.

    Parameters
    ----------
    names : daftar str, default=None
        Nama instans yang dimuat, urutan dipertahankan. ``None`` berarti semua,
        MK01 sampai MK10.
    k_tight : float, default=1.5
        Keketatan tenggat.
    capacity_mult : float, default=1.2
        Pengali kapasitas pengembang (kelipatan rata-rata beban).
    p_cross_dep : float, default=0.10
        Peluang dependensi antar-job.
    domain_seed : int, default=0
        Seed keacakan transformasi domain. Tiap instans memakai aliran sendiri
        yang dimulai dari seed ini.
    data_home : str atau Path, default=None
        Folder ``data`` yang berisi ``raw/``. ``None`` berarti folder ``data`` di
        proyek ini. ``configs/checksums.yaml`` di samping folder itu wajib ada dan
        memuat SHA-256 tiap instans yang dimuat.

    Returns
    -------
    dict
        Nama instans ke ``ProjectInstance``.

    Raises
    ------
    ValueError
        Nama tidak dikenal, berkas gagal validasi, SHA-256 tidak cocok, atau instans
        tidak punya checksum.
    FileNotFoundError
        Berkas mentah atau ``configs/checksums.yaml`` tidak ada.

    Examples
    --------
    >>> from dfjsp.datasets import load_brandimarte
    >>> data = load_brandimarte(["MK01"])
    >>> sorted(data)
    ['MK01']
    >>> len(data["MK01"].tasks) > 0
    True
    """
    chosen = list(BRANDIMARTE_NAMES) if names is None else list(names)
    unknown = [n for n in chosen if n not in BRANDIMARTE_NAMES]
    if unknown:
        raise ValueError(
            f"nama instans tidak dikenal: {unknown}. Yang ada: {list(BRANDIMARTE_NAMES)}")
    if len(set(chosen)) != len(chosen):
        raise ValueError(f"nama instans dobel di names: {chosen}")

    home = Path(data_home) if data_home is not None else _PROJECT_ROOT / "data"
    for name in chosen:
        path = home / "raw" / f"{name}.json"
        if not path.is_file():
            raise FileNotFoundError(f"berkas data {name} tidak ditemukan: {path}")
    checksums = _read_checksums(home)
    cfg = SimpleNamespace(
        k_tight=float(k_tight),
        domain=SimpleNamespace(capacity_mult=float(capacity_mult),
                               p_cross_dep=float(p_cross_dep), seed=int(domain_seed)),
        priority=SimpleNamespace(invert=False),
    )

    out: dict[str, ProjectInstance] = {}
    for name in chosen:
        path = home / "raw" / f"{name}.json"
        _check_sha256(path, checksums.get(name))
        base = parse_brandimarte(str(path), name)
        problems = validate_instance(base)
        if problems:
            raise ValueError(f"validasi gagal untuk {name}: {problems}")
        out[name] = to_software_project(base, cfg, np.random.default_rng(cfg.domain.seed))
    return out
