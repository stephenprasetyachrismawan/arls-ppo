"""Kisi evaluasi: kombinasi intensitas, kapasitas, dan seed episode."""

from __future__ import annotations

from typing import NamedTuple


class GridPoint(NamedTuple):
    """Satu titik kisi. Instans ditambahkan oleh ``evaluate``.

    Examples
    --------
    >>> from dfjsp.evaluation import GridPoint
    >>> titik = GridPoint(intensity=0.15, capacity=1.5, seed=0, k_tight=1.5, scenario="mixed")
    >>> titik.capacity, titik.seed
    (1.5, 0)
    """

    intensity: float
    capacity: float
    seed: int
    k_tight: float
    scenario: str


class EvaluationGrid:
    """Kisi kombinasi evaluasi (kecuali instans dan metode).

    Satu titik kisi + satu instans = satu urutan gangguan, yang sama untuk semua metode.

    Parameters
    ----------
    intensities : tuple of float, default=(0.05, 0.15, 0.30)
        Laju gangguan per satuan waktu.
    capacities : tuple of float, default=(1.0, 1.5, 2.0)
        Pengali kapasitas pengembang (kelipatan rata-rata beban).
    episode_seeds : iterable of int, default=range(10)
        Seed episode. Seed ini menentukan gangguan, bukan seed pelatihan.
    k_tight : float, default=1.5
        Keketatan tenggat.
    scenario : str, default="mixed"
        Skenario gangguan.

    Examples
    --------
    >>> from dfjsp.evaluation import EvaluationGrid
    >>> grid = EvaluationGrid()
    >>> len(grid)
    90
    >>> next(iter(grid))
    GridPoint(intensity=0.05, capacity=1.0, seed=0, k_tight=1.5, scenario='mixed')
    """

    def __init__(self, intensities=(0.05, 0.15, 0.30), capacities=(1.0, 1.5, 2.0),
                 episode_seeds=tuple(range(10)), k_tight=1.5, scenario="mixed"):
        self.intensities = intensities
        self.capacities = capacities
        self.episode_seeds = episode_seeds
        self.k_tight = k_tight
        self.scenario = scenario

    def _checked(self) -> tuple:
        """Validasi parameter dan kembalikan ``(intensitas, kapasitas, seed)`` sebagai tuple."""
        intensities = tuple(self.intensities)
        capacities = tuple(self.capacities)
        seeds = tuple(self.episode_seeds)
        for name, values in (("intensities", intensities), ("capacities", capacities),
                             ("episode_seeds", seeds)):
            if not values:
                raise ValueError(f"Parameter {name} tidak boleh kosong")
        if any(x <= 0 for x in intensities):
            raise ValueError(f"Parameter intensities harus > 0, diterima {intensities}")
        if any(x <= 0 for x in capacities):
            raise ValueError(f"Parameter capacities harus > 0, diterima {capacities}")
        if self.k_tight <= 0:
            raise ValueError(f"Parameter k_tight harus > 0, diterima {self.k_tight!r}")
        return intensities, capacities, seeds

    def __iter__(self):
        """Urutan tetap: intensitas terluar, lalu kapasitas, seed paling dalam."""
        intensities, capacities, seeds = self._checked()
        for intensity in intensities:
            for capacity in capacities:
                for seed in seeds:
                    yield GridPoint(float(intensity), float(capacity), int(seed),
                                    float(self.k_tight), self.scenario)

    def __len__(self) -> int:
        intensities, capacities, seeds = self._checked()
        return len(intensities) * len(capacities) * len(seeds)

    def __repr__(self) -> str:
        return (f"EvaluationGrid(intensities={self.intensities!r}, "
                f"capacities={self.capacities!r}, episode_seeds={self.episode_seeds!r}, "
                f"k_tight={self.k_tight!r}, scenario={self.scenario!r})")
