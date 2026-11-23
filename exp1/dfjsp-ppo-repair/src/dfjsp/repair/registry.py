"""Susunan operator bawaan: aksi -> operator."""

from __future__ import annotations

from .actions import DELAY, ESCALATE, MAINTAIN, REASSIGN
from .base import RepairOperator
from .delay import Delay
from .escalate import Escalate
from .maintain import Maintain
from .reassign import Reassign


def default_operators(enforce_capacity: bool = True) -> dict[int, RepairOperator]:
    """Operator kanonis: Maintain, Delay (D2), Reassign (R3), Escalate (E3).

    Parameters
    ----------
    enforce_capacity : bool, default=True
        Bila False, Reassign dan Escalate tidak memeriksa batas kapasitas tujuan
        (dipakai ablasi tanpa mask tingkat bawah).

    Examples
    --------
    >>> from dfjsp.repair import default_operators
    >>> sorted(default_operators())
    [0, 1, 2, 3]
    """
    return {
        MAINTAIN: Maintain(),
        DELAY: Delay(),
        REASSIGN: Reassign(enforce_capacity=enforce_capacity),
        ESCALATE: Escalate(enforce_capacity=enforce_capacity),
    }
