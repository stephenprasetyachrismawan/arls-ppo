"""Penjaga data uji untuk skrip percobaan.

Cukup ``import penjaga_uji`` di bagian atas skrip. Setelah itu, setiap usaha membuka berkas
instans uji (MK03, MK04, MK09, lihat ``configs/split.yaml``) dari proses itu gagal dengan
pesan yang jelas. Penjaga ini memakai audit hook Python (PEP 578), jadi ikut menangkap
pembukaan berkas dari pustaka mana pun, termasuk ``load_brandimarte``.

Data uji hanya boleh dibuka lewat ``studi uji --buka-uji`` setelah pemilik setuju; skrip di
folder ini untuk pelatihan dan validasi saja.
"""
from __future__ import annotations

import os
import sys

INSTANS_UJI = frozenset({"MK03", "MK04", "MK09"})

# Audit hook tidak bisa dicabut, jadi pembukaan resmi (buka_uji.py --buka-uji) memberi izin
# lewat izinkan(). Hanya berlaku di proses itu.
_IZIN: list[str] = []


def izinkan(alasan: str) -> None:
    """Lepas penjaga di proses ini untuk pembukaan resmi data uji."""
    _IZIN.append(alasan)


def diizinkan() -> bool:
    """Apakah penjaga sudah dilepas di proses ini."""
    return bool(_IZIN)


def _nama_berkas(jalur) -> str | None:
    if isinstance(jalur, int):          # nomor berkas yang sudah terbuka, bukan nama
        return None
    try:
        jalur = os.fsdecode(os.fspath(jalur))
    except TypeError:
        return None
    return os.path.splitext(os.path.basename(jalur))[0].upper()


def _periksa(kejadian, argumen):
    if _IZIN:
        return
    if kejadian == "open" and argumen and _nama_berkas(argumen[0]) in INSTANS_UJI:
        raise PermissionError(f"penjaga_uji: {argumen[0]} adalah data uji; skrip percobaan "
                              "tidak boleh membukanya")


sys.addaudithook(_periksa)
