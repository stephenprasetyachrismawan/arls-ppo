"""Nama ``ProjectInstance`` dan operasi kapasitas untuk instans proyek perangkat lunak."""

from __future__ import annotations

import copy

from ..core.datatypes import FJSPInstance
from ..core.domain import set_capacity as _set_capacity

#: Satu instans proyek perangkat lunak siap simulasi. Ini nama baru yang lebih jelas
#: untuk ``dfjsp.core.datatypes.FJSPInstance`` (kelas yang sama, tidak disalin).
#:
#: Pemetaan dari masalah penjadwalan mesin ke proyek perangkat lunak:
#: job = fitur atau modul, mesin = pengembang, operasi = tugas, waktu proses = upaya
#: (jam), kelayakan mesin = keahlian pengembang.
#:
#: Atribut yang dibaca pemakai:
#:
#: * ``name``: nama instans, "MK01" sampai "MK10".
#: * ``n_jobs``: jumlah job (fitur).
#: * ``n_devs``: jumlah pengembang.
#: * ``tasks``: daftar ``Task``. Tiap ``Task`` punya ``task_id``, ``job_id``,
#:   ``op_index`` (urutan di job), ``eligible_devs`` ({pengembang: upaya jam}),
#:   ``deps`` (tugas prasyarat), ``due_date`` (tenggat), ``priority`` (0 sampai 1).
#: * ``developers``: daftar ``Developer`` dengan ``dev_id``, ``skills``, ``capacity``.
#: * ``horizon``: panjang horizon perencanaan, pembagi normalisasi keadaan.
#: * ``meta``: catatan asal data (sumber, parameter domain, tenggat).
ProjectInstance = FJSPInstance


def set_capacity(instance: ProjectInstance, multiplier: float) -> None:
    """Atur kapasitas semua pengembang, LANGSUNG pada ``instance`` (tanpa salinan).

    Kapasitas tiap pengembang menjadi ``multiplier`` dikali rata-rata beban, dengan
    rata-rata beban = jumlah upaya nominal semua tugas dibagi jumlah pengembang.
    Pelanggaran kapasitas ditolak oleh mask tingkat bawah. Untuk mendapat salinan
    tanpa mengubah aslinya, pakai :func:`with_capacity`.

    Parameters
    ----------
    instance : ProjectInstance
        Instans yang sudah punya ``developers`` (hasil ``load_brandimarte``).
    multiplier : float
        Pengali kapasitas. Makin besar, makin longgar (contoh: 1,0 / 1,5 / 2,0).

    Returns
    -------
    None

    Examples
    --------
    >>> from dfjsp.datasets import load_brandimarte, set_capacity
    >>> inst = load_brandimarte(["MK01"])["MK01"]
    >>> set_capacity(inst, 2.0)
    >>> inst.developers[0].capacity > 0
    True
    """
    _set_capacity(instance, float(multiplier))


def with_capacity(instance: ProjectInstance, multiplier: float) -> ProjectInstance:
    """Salinan ``instance`` dengan kapasitas ``multiplier``; aslinya tidak berubah.

    Parameters
    ----------
    instance : ProjectInstance
        Instans asal.
    multiplier : float
        Pengali kapasitas, lihat :func:`set_capacity`.

    Returns
    -------
    ProjectInstance
        Salinan dalam (deep copy) dengan kapasitas baru.

    Examples
    --------
    >>> from dfjsp.datasets import load_brandimarte, with_capacity
    >>> inst = load_brandimarte(["MK01"])["MK01"]
    >>> longgar = with_capacity(inst, 2.0)
    >>> longgar.developers[0].capacity > inst.developers[0].capacity
    True
    """
    salinan = copy.deepcopy(instance)
    set_capacity(salinan, multiplier)
    return salinan
