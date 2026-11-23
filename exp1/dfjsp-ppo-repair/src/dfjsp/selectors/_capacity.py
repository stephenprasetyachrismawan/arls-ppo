"""Penjadwal kapasitas seimbang menurut langkah aktual.

Saat melatih, tiap worker lingkungan memakai satu pengali kapasitas selama satu
rollout. Penjadwal memilih pengali itu supaya jumlah langkah per pengali tetap
seimbang. Yang diseimbangkan adalah langkah yang benar-benar terkumpul, bukan jumlah
episode, karena panjang episode berbeda antar kapasitas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..utils import check_random_state

_TIE_EPS = 1e-9


@dataclass
class CapacityExposureLedger:
    """Buku catatan langkah aktual per pengali kapasitas.

    Examples
    --------
    >>> from dfjsp.selectors._capacity import CapacityExposureLedger
    >>> ledger = CapacityExposureLedger()
    >>> ledger.record(1.5, 1024)
    >>> ledger.total()
    1024
    """

    actual_steps: dict[float, int] = field(default_factory=dict)

    def record(self, multiplier: float, steps: int) -> None:
        """Tambah ``steps`` langkah untuk ``multiplier``."""
        if steps < 0:
            raise ValueError(f"steps harus >= 0, diterima {steps}")
        key = float(multiplier)
        self.actual_steps[key] = self.actual_steps.get(key, 0) + int(steps)

    def total(self) -> int:
        """Jumlah langkah semua pengali."""
        return sum(self.actual_steps.values())


class CapacityScheduler:
    """Menetapkan pengali kapasitas tiap worker pada tiap update PPO.

    Satu pengali berarti jadwal tetap. Beberapa pengali berarti jadwal seimbang:
    tiap worker mengambil pengali yang paling kurang langkahnya dibanding target
    (porsi sama rata). Bila ada seri, dipilih acak.

    Parameters
    ----------
    levels : urutan float
        Pengali kapasitas, harus positif dan tidak berulang.
    n_envs : int
        Jumlah worker (lingkungan paralel).
    n_steps : int
        Langkah yang dikumpulkan tiap worker dalam satu rollout.
    random_state : None, int, atau numpy.random.Generator, default=None
        Dipakai hanya untuk memecah seri.

    Examples
    --------
    >>> from dfjsp.selectors._capacity import CapacityExposureLedger, CapacityScheduler
    >>> sch = CapacityScheduler((1.0, 1.5, 2.0), n_envs=6, n_steps=10, random_state=0)
    >>> ledger = CapacityExposureLedger()
    >>> len(sch.assignments(1, ledger))
    6
    """

    def __init__(self, levels, *, n_envs: int, n_steps: int, random_state=None):
        levels = tuple(float(x) for x in levels)
        if not levels:
            raise ValueError("levels tidak boleh kosong")
        if any(not math.isfinite(x) or x <= 0 for x in levels):
            raise ValueError(f"levels harus bilangan positif, diterima {levels}")
        if len(set(levels)) != len(levels):
            raise ValueError(f"levels tidak boleh berulang, diterima {levels}")
        if n_envs <= 0 or n_steps <= 0:
            raise ValueError("n_envs dan n_steps harus > 0")
        self.levels = levels
        self.n_envs = int(n_envs)
        self.n_steps = int(n_steps)
        self._rng = check_random_state(random_state)
        self._target = 1.0 / len(levels)

    def assignments(self, update_index: int, ledger: CapacityExposureLedger) -> tuple:
        """Pengali untuk tiap worker pada update ke-``update_index`` (mulai dari 1)."""
        if update_index < 1:
            raise ValueError(f"update harus >= 1, diterima {update_index}")
        if len(self.levels) == 1:
            return (self.levels[0],) * self.n_envs
        projected = {level: ledger.actual_steps.get(level, 0) for level in self.levels}
        chosen_levels = []
        for _ in range(self.n_envs):
            running_total = sum(projected.values()) + self.n_steps
            best, tied = -np.inf, []
            for level in self.levels:
                deficit = self._target * running_total - projected[level]
                if deficit > best + _TIE_EPS:
                    best, tied = deficit, [level]
                elif abs(deficit - best) <= _TIE_EPS:
                    tied.append(level)
            pick = tied[0] if len(tied) == 1 else float(self._rng.choice(tied))
            projected[pick] += self.n_steps
            chosen_levels.append(pick)
        return tuple(chosen_levels)

    def validate_exposure(self, ledger: CapacityExposureLedger, *,
                          expected_total_steps: int,
                          tolerance_points: float = 1.0) -> None:
        """Pastikan total langkah tepat dan porsi tiap pengali dekat targetnya.

        Parameters
        ----------
        tolerance_points : float, default=1.0
            Selisih porsi yang diizinkan, dalam poin persen. Penjadwal hanya menjamin
            selisih satu slot worker (``n_steps`` langkah), jadi pada anggaran sangat
            kecil toleransi naik ke satu slot. Pada anggaran penuh satu slot jauh
            di bawah 1 poin, sehingga toleransi tetap 1 poin.
        """
        total = ledger.total()
        if total != expected_total_steps:
            raise ValueError(f"total langkah {total} != harapan {expected_total_steps}")
        tolerance = max(tolerance_points / 100.0, self.n_steps / total)
        for level in self.levels:
            share = ledger.actual_steps.get(level, 0) / total
            if abs(share - self._target) > tolerance + _TIE_EPS:
                raise ValueError(
                    f"porsi langkah kapasitas {level} menyimpang "
                    f"{abs(share - self._target):.4f} dari target, melebihi "
                    f"toleransi {tolerance:.4f}")
