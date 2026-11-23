"""Operator Delay (D2 MaxSlack): tunda tugas dengan kelonggaran terbesar per prioritas."""

from __future__ import annotations

from ..core.datatypes import Schedule
from ..env.schedule import partition, recompute_timeline
from .base import OperatorFeasibility, RepairOperator, RepairResult
from .masks import count_feasible_pairs

EPS = 1e-8


def select_delay_task(inst, sched: Schedule, t_now: float) -> int | None:
    """Pilih tugas belum mulai yang ditunda: kelonggaran / (prioritas + EPS) terbesar.

    Kelonggaran = tenggat - waktu sekarang - upaya di pengembang yang ditugaskan.
    Bila seri, tugas dengan ``task_id`` terkecil menang.
    """
    _, _, not_started = partition(inst, sched, t_now)
    by_id = {t.task_id: t for t in inst.tasks}
    best: int | None = None
    best_val = float("-inf")
    for i in sorted(not_started):
        t = by_id[i]
        e = sched.entries.get(i)
        eff = t.eligible_devs.get(e.dev_id, 0.0) if e else 0.0
        slack = t.due_date - t_now - eff
        val = slack / (t.priority + EPS)
        if val > best_val:
            best_val = val
            best = i
    return best


class Delay(RepairOperator):
    """Tunda satu tugas sampai sesudah selesai dikerjakan di jadwal sekarang.

    Examples
    --------
    >>> from dfjsp.datasets import load_brandimarte
    >>> from dfjsp.env import RepairEnv
    >>> env = RepairEnv()
    >>> state, info = env.reset(load_brandimarte(["MK01"])["MK01"], intensity=0.05, seed=0)
    >>> from dfjsp.repair import Delay
    >>> Delay().inspect(env.inst, env.sched, env.t_now).feasible
    True
    >>> hasil = Delay().apply(env.inst, env.sched, env.t_now)
    >>> [m["task_id"] for m in hasil.moved_tasks]
    [37]
    """

    name = "Delay"

    def inspect(self, inst, sched: Schedule, t_now: float) -> OperatorFeasibility:
        """Layak bila ada tugas belum mulai untuk ditunda."""
        feasible = select_delay_task(inst, sched, t_now) is not None
        return OperatorFeasibility(
            feasible=feasible,
            choice_count=int(feasible),
            reason=None if feasible else "no_not_started_task",
        )

    def apply(self, inst, sched: Schedule, t_now: float) -> RepairResult:
        feasible = count_feasible_pairs(inst, sched, t_now)
        i_star = select_delay_task(inst, sched, t_now)
        if i_star is None:
            return RepairResult(sched, [], invalid_attempt=True, feasible_pairs=feasible)
        e = sched.entries[i_star]
        task = next(t for t in inst.tasks if t.task_id == i_star)
        delta = task.eligible_devs.get(e.dev_id, e.end - e.start)
        sched.release[i_star] = max(t_now, e.start) + delta
        recompute_timeline(inst, sched, t_now)
        return RepairResult(
            new_sched=sched,
            moved_tasks=[{"task_id": i_star, "delay": delta}],
            invalid_attempt=False,
            feasible_pairs=feasible,
        )
