"""Antarmuka operator perbaikan."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.datatypes import Schedule


@dataclass(frozen=True)
class OperatorFeasibility:
    """Hasil probe kelayakan operator, tanpa mengubah jadwal.

    Attributes
    ----------
    feasible : bool
        True bila operator punya minimal satu pilihan perbaikan yang layak pada keadaan
        masuk. Sama artinya dengan ``not RepairResult.invalid_attempt``.
    choice_count : int
        Jumlah pilihan perbaikan layak yang dilihat probe (>= 0).
    reason : str or None
        Alasan singkat bila tidak layak, misalnya tidak ada tugas yang belum mulai.

    Examples
    --------
    >>> from dfjsp.repair import OperatorFeasibility
    >>> OperatorFeasibility(feasible=False, choice_count=0, reason="no_not_started_task")
    OperatorFeasibility(feasible=False, choice_count=0, reason='no_not_started_task')
    """

    feasible: bool
    choice_count: int
    reason: str | None = None


@dataclass
class RepairResult:
    """Keluaran ``apply`` sebuah operator.

    Attributes
    ----------
    new_sched : Schedule
        Jadwal sesudah perbaikan (diubah di tempat, lalu dikembalikan).
    moved_tasks : list of dict
        Tugas yang disentuh, misalnya ``{"task_id", "from_dev", "to_dev"}``.
    invalid_attempt : bool
        True bila tidak ada pilihan layak, sehingga jadwal dibiarkan (jatuh ke Maintain).
    feasible_pairs : int
        Jumlah pasangan (tugas, pengembang) layak pada keadaan masuk, untuk log.

    Examples
    --------
    >>> from dfjsp.core.datatypes import Schedule
    >>> from dfjsp.repair import RepairResult
    >>> hasil = RepairResult(new_sched=Schedule())
    >>> hasil.moved_tasks, hasil.invalid_attempt, hasil.feasible_pairs
    ([], False, 0)
    """

    new_sched: Schedule
    moved_tasks: list[dict] = field(default_factory=list)
    invalid_attempt: bool = False
    feasible_pairs: int = 0


class RepairOperator:
    """Kelas dasar operator. Turunannya mengisi ``apply`` dan ``inspect``.

    ``__init__`` turunan hanya menyimpan parameter. Operator tidak menyimpan keadaan
    antar-pemanggilan, jadi satu objek aman dipakai ulang di banyak episode.

    Examples
    --------
    Operator buatan sendiri yang tidak mengubah apa pun:

    >>> from dfjsp.repair import RepairOperator, RepairResult
    >>> class Diam(RepairOperator):
    ...     name = "Diam"
    ...     def apply(self, inst, sched, t_now):
    ...         return RepairResult(new_sched=sched)
    >>> Diam().name
    'Diam'
    >>> Diam().apply(None, "jadwal", 0.0).new_sched
    'jadwal'
    """

    name: str = "base"

    def inspect(self, inst, sched: Schedule, t_now: float) -> OperatorFeasibility:
        """Cek kelayakan tanpa mengubah jadwal."""
        from ..env.schedule import partition

        _, _, not_started = partition(inst, sched, t_now)
        count = len(not_started)
        return OperatorFeasibility(
            feasible=count > 0,
            choice_count=count,
            reason=None if count else "no_not_started_task",
        )

    def apply(self, inst, sched: Schedule, t_now: float) -> RepairResult:
        """Terapkan perbaikan pada ``sched`` dan kembalikan ``RepairResult``."""
        raise NotImplementedError
