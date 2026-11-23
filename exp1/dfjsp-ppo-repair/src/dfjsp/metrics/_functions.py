"""Ukuran episode: fungsi murni dari ``EpisodeLog``.

Definisinya sama dengan recorder lama (``logging/recorder.py``, ``finalize_episode``),
dibuktikan dengan rekaman emas. Log kosong (tanpa langkah) memberi 0 untuk semua
ukuran, juga sama dengan recorder lama.
"""

from __future__ import annotations

import numpy as np

from ._log import EpisodeLog


def _column(log: EpisodeLog, name: str) -> np.ndarray:
    """Kolom ``name`` dari semua langkah; log kosong memberi ``[0.0]`` (seperti lama)."""
    if not log.steps:
        return np.array([0.0])
    return np.array([getattr(s, name) for s in log.steps], dtype=float)


def total_tardiness(log: EpisodeLog) -> float:
    """Total keterlambatan jadwal di akhir episode (nilai langkah terakhir).

    Parameters
    ----------
    log : EpisodeLog

    Returns
    -------
    float
        0.0 bila log kosong. Bukan jumlah semua langkah: ``tardiness_total`` tiap
        langkah sudah keterlambatan seluruh jadwal pada saat itu.

    Examples
    --------
    >>> from dfjsp.metrics import EpisodeLog, StepRecord, total_tardiness
    >>> log = EpisodeLog(steps=[StepRecord(0, 5.0, 0.0, 0.0), StepRecord(1, 3.0, 0.1, 0.0)])
    >>> total_tardiness(log)
    3.0
    """
    return float(log.steps[-1].tardiness_total) if log.steps else 0.0


def nervousness(log: EpisodeLog) -> float:
    """Rata-rata ``nervousness_step`` atas semua keputusan.

    Nervousness satu langkah mengukur seberapa jauh jadwal sekarang menyimpang dari
    jadwal awal (tugas pindah pengembang dan waktu mulai bergeser). Makin kecil
    makin stabil.

    Examples
    --------
    >>> from dfjsp.metrics import EpisodeLog, StepRecord, nervousness
    >>> log = EpisodeLog(steps=[StepRecord(0, 0.0, 0.0, 0.0), StepRecord(1, 0.0, 0.2, 0.0)])
    >>> round(nervousness(log), 2)
    0.1
    """
    return float(_column(log, "nervousness_step").mean())


def context_switches(log: EpisodeLog) -> float:
    """Jumlah ``context_switch_step`` sepanjang episode.

    Biaya alih konteks bertambah bila tugas yang sudah berjalan dipindah ke
    pengembang lain.

    Examples
    --------
    >>> from dfjsp.metrics import EpisodeLog, StepRecord, context_switches
    >>> log = EpisodeLog(steps=[StepRecord(2, 0.0, 0.0, 0.5), StepRecord(2, 0.0, 0.0, 1.0)])
    >>> context_switches(log)
    1.5
    """
    return float(_column(log, "context_switch_step").sum())


def workload_imbalance(log: EpisodeLog) -> float:
    """Rata-rata ketimpangan beban antar pengembang atas semua keputusan.

    Ketimpangan satu langkah adalah simpangan baku beban semua pengembang.

    Examples
    --------
    >>> from dfjsp.metrics import EpisodeLog, StepRecord, workload_imbalance
    >>> log = EpisodeLog(steps=[StepRecord(0, 0, 0, 0, workload_imbalance=1.0),
    ...                         StepRecord(0, 0, 0, 0, workload_imbalance=3.0)])
    >>> workload_imbalance(log)
    2.0
    """
    return float(_column(log, "workload_imbalance").mean())


def invalid_strategy_rate(log: EpisodeLog) -> float:
    """Porsi keputusan yang meminta strategi yang dilarang mask (0 sampai 1).

    Examples
    --------
    >>> from dfjsp.metrics import EpisodeLog, StepRecord, invalid_strategy_rate
    >>> log = EpisodeLog(steps=[StepRecord(2, 0, 0, 0, invalid_strategy=True),
    ...                         StepRecord(0, 0, 0, 0)])
    >>> invalid_strategy_rate(log)
    0.5
    """
    return float(_column(log, "invalid_strategy").mean())


def infeasible_assignments(log: EpisodeLog) -> int:
    """Jumlah perbaikan yang DITERIMA tetapi hasilnya melanggar batasan keras.

    Dihitung dari langkah dengan ``invalid_repair`` False (perbaikan diterima) dan
    ``hard_feasible`` False (pemeriksa jadwal terpisah menemukan pelanggaran).
    Sasarannya nol.

    Examples
    --------
    >>> from dfjsp.metrics import EpisodeLog, StepRecord, infeasible_assignments
    >>> log = EpisodeLog(steps=[StepRecord(2, 0, 0, 0, hard_feasible=False),
    ...                         StepRecord(2, 0, 0, 0, invalid_repair=True, hard_feasible=False)])
    >>> infeasible_assignments(log)
    1
    """
    return int(sum(1 for s in log.steps if not s.invalid_repair and not s.hard_feasible))


def rescheduling_rate(log: EpisodeLog) -> float:
    """Porsi keputusan yang benar-benar memindahkan tugas (``n_moved > 0``).

    Aksi yang gagal atau tidak menyentuh tugas bukan penjadwalan ulang, jadi aksi
    bukan Maintain belum tentu dihitung. Log kosong memberi 0.0.

    Examples
    --------
    >>> from dfjsp.metrics import EpisodeLog, StepRecord, rescheduling_rate
    >>> log = EpisodeLog(steps=[StepRecord(a, 0, 0, 0, n_moved=m)
    ...                         for a, m in ((0, 0), (1, 0), (2, 3), (3, 1))])
    >>> rescheduling_rate(log)
    0.5
    """
    if not log.steps:
        return 0.0
    return sum(1 for s in log.steps if s.n_moved > 0) / len(log.steps)
