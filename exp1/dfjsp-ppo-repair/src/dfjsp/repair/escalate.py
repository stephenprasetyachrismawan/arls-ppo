"""Operator Escalate (E3 EDD): pindahkan tugas paling mendesak ke pengembang paling longgar."""

from __future__ import annotations

from ..core.datatypes import Schedule, ScheduleEntry
from ..env.schedule import developer_load, partition, recompute_timeline
from ..env.work_calendar import working_time
from .base import OperatorFeasibility, RepairOperator, RepairResult
from .masks import count_feasible_pairs, is_pair_feasible


def escalate_order(inst, sched: Schedule, t_now: float) -> list[int]:
    """Urutan tugas belum mulai menurut tenggat terdekat (EDD); seri: ``task_id``."""
    _, _, not_started = partition(inst, sched, t_now)
    by_id = {t.task_id: t for t in inst.tasks}
    return sorted(not_started, key=lambda i: (by_id[i].due_date, i))


def base_load(inst, sched: Schedule, t_now: float, in_progress: set[int]) -> dict[int, float]:
    """Sisa pekerjaan tugas yang sedang berjalan, per pengembang."""
    load = {d.dev_id: 0.0 for d in inst.developers}
    for tid in in_progress:
        entry = sched.entries[tid]
        load[entry.dev_id] += working_time(
            t_now, entry.end, sched.pauses.get(entry.dev_id, [])
        )
    return load


def enumerate_escalate_choices(inst, sched: Schedule, t_now: float,
                               enforce_capacity: bool,
                               budget: int | None) -> list[tuple[int, int]]:
    """Pilihan eskalasi layak ``(tugas, j_star)`` pada keadaan masuk, tanpa mengubah jadwal."""
    _, _, not_started = partition(inst, sched, t_now)
    if not not_started:
        return []
    by_id = {t.task_id: t for t in inst.tasks}
    dev_by_id = {d.dev_id: d for d in inst.developers}
    loads = developer_load(inst, sched, t_now)
    eff_budget = max(1, int(budget if budget is not None else len(inst.developers)))
    choices: list[tuple[int, int]] = []
    for i in escalate_order(inst, sched, t_now)[:eff_budget]:
        # Lepaskan hanya beban tugas yang sedang diperiksa. Tugas lain, termasuk
        # yang di luar budget atau gagal dipindah, tetap menjadi tanggungan.
        old = sched.entries[i].dev_id
        effort_old = by_id[i].eligible_devs[old]
        loads[old] -= effort_old
        cands = [
            d.dev_id for d in inst.developers
            if is_pair_feasible(i, d.dev_id, inst, sched, t_now, loads=loads,
                                enforce_capacity=enforce_capacity, exclude_self_load=False,
                                by_id=by_id, dev_by_id=dev_by_id)
        ]
        if not cands:
            loads[old] += effort_old
            continue
        j_star = min(cands, key=lambda j: (loads.get(j, 0.0), j))
        choices.append((i, j_star))
        loads[j_star] += by_id[i].eligible_devs[j_star]
    return choices


class Escalate(RepairOperator):
    """Pindahkan beberapa tugas paling mendesak ke pengembang paling longgar (E3 EDD).

    Parameters
    ----------
    enforce_capacity : bool, default=True
        Bila False, batas kapasitas tujuan tidak diperiksa (ablasi tanpa mask bawah).
    budget : int, optional
        Jumlah tugas paling mendesak yang boleh dieskalasi per langkah. None berarti
        jumlah pengembang, yaitu satu slot per pengembang.

    Examples
    --------
    >>> from dfjsp.datasets import load_brandimarte
    >>> from dfjsp.env import RepairEnv
    >>> env = RepairEnv()
    >>> state, info = env.reset(load_brandimarte(["MK01"])["MK01"], intensity=0.05, seed=0)
    >>> from dfjsp.repair import Escalate
    >>> Escalate().inspect(env.inst, env.sched, env.t_now).choice_count
    6
    >>> hasil = Escalate().apply(env.inst, env.sched, env.t_now)
    >>> [(m["task_id"], m["from_dev"], m["to_dev"]) for m in hasil.moved_tasks]
    [(27, 5, 2)]
    """

    name = "Escalate"

    def __init__(self, enforce_capacity: bool = True, budget: int | None = None):
        self.enforce_capacity = enforce_capacity
        self.budget = budget

    def inspect(self, inst, sched: Schedule, t_now: float) -> OperatorFeasibility:
        """Layak bila minimal satu tugas dalam budget punya tujuan yang cocok."""
        choices = enumerate_escalate_choices(
            inst, sched, t_now, self.enforce_capacity, self.budget
        )
        reason = None
        if not choices:
            _, _, not_started = partition(inst, sched, t_now)
            reason = "no_not_started_task" if not not_started else "no_feasible_escalate"
        return OperatorFeasibility(
            feasible=bool(choices),
            choice_count=len(choices),
            reason=reason,
        )

    def apply(self, inst, sched: Schedule, t_now: float) -> RepairResult:
        feasible = count_feasible_pairs(inst, sched, t_now)
        _, _, not_started = partition(inst, sched, t_now)
        if not not_started:
            return RepairResult(sched, [], invalid_attempt=True, feasible_pairs=feasible)

        moved: list[dict] = []
        choices = enumerate_escalate_choices(
            inst, sched, t_now, self.enforce_capacity, self.budget
        )
        for i, j_star in choices:
            old = sched.entries[i].dev_id
            if j_star != old:
                e = sched.entries[i]
                sched.entries[i] = ScheduleEntry(i, j_star, e.start, e.end)
                moved.append({"task_id": i, "from_dev": old, "to_dev": j_star})

        recompute_timeline(inst, sched, t_now)
        return RepairResult(
            new_sched=sched,
            moved_tasks=moved,
            invalid_attempt=not choices,
            feasible_pairs=feasible,
        )
