"""Operator Maintain: jadwal tidak berubah, nervousness nol."""

from __future__ import annotations

from ..core.datatypes import Schedule
from .base import OperatorFeasibility, RepairOperator, RepairResult
from .masks import count_feasible_pairs


class Maintain(RepairOperator):
    """Pertahankan jadwal apa adanya.

    Examples
    --------
    >>> from dfjsp.datasets import load_brandimarte
    >>> from dfjsp.env import RepairEnv
    >>> env = RepairEnv()
    >>> state, info = env.reset(load_brandimarte(["MK01"])["MK01"], intensity=0.05, seed=0)
    >>> from dfjsp.repair import Maintain
    >>> hasil = Maintain().apply(env.inst, env.sched, env.t_now)
    >>> hasil.moved_tasks, hasil.invalid_attempt
    ([], False)
    """

    name = "Maintain"

    def inspect(self, inst, sched: Schedule, t_now: float) -> OperatorFeasibility:
        """Maintain selalu layak, dengan satu pilihan tunggal."""
        return OperatorFeasibility(feasible=True, choice_count=1)

    def apply(self, inst, sched: Schedule, t_now: float) -> RepairResult:
        """Tidak mengubah jadwal sama sekali."""
        return RepairResult(
            new_sched=sched,
            moved_tasks=[],
            invalid_attempt=False,
            feasible_pairs=count_feasible_pairs(inst, sched, t_now),
        )
