"""Penerapan strategi perbaikan pada satu langkah."""

from __future__ import annotations

from ..core.datatypes import FJSPInstance, Schedule
from ..repair.base import RepairOperator, RepairResult


def apply_repair(operators: dict[int, RepairOperator], action: int, inst: FJSPInstance,
                 sched: Schedule, t_now: float) -> RepairResult:
    """Jalankan operator milik ``action``; aksi tanpa operator berarti Maintain."""
    if action in operators:
        return operators[action].apply(inst, sched, t_now)
    return RepairResult(sched, [], invalid_attempt=False, feasible_pairs=0)
