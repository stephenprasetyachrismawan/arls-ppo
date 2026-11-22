"""Operator Reassign (R3 Balanced): pindahkan satu tugas dari pengembang yang bermasalah."""

from __future__ import annotations

from ..core.datatypes import Schedule, ScheduleEntry
from ..env.schedule import developer_load, partition, recompute_timeline
from .base import OperatorFeasibility, RepairOperator, RepairResult
from .masks import count_feasible_pairs, is_pair_feasible


def dev_available(inst, dev_id: int) -> bool:
    """True bila pengembang dengan id tersebut sedang tersedia."""
    for d in inst.developers:
        if d.dev_id == dev_id:
            return bool(d.available)
    return False


def _source_order(inst, loads: dict[int, float], tasks_on: dict[int, list[int]]) -> list[int]:
    """Urutan pengembang sumber yang dicoba, dari yang paling perlu dilepas bebannya.

    Pengembang yang tidak tersedia dicoba lebih dulu (yang menahan tugas paling banyak
    didahulukan), lalu sisanya dari beban terbesar.
    """
    unavailable = sorted(
        (j for j in tasks_on if not dev_available(inst, j)),
        key=lambda j: (-len(tasks_on[j]), j),
    )
    rest = sorted(
        (j for j in tasks_on if j not in set(unavailable)),
        key=lambda j: (-loads.get(j, 0.0), j),
    )
    return unavailable + rest


def _feasible_moves(inst, sched: Schedule, t_now: float, enforce_capacity: bool):
    """Bagian bersama ``inspect`` dan ``apply``.

    Mengembalikan ``(j_over, targets, by_id, dev_by_id, loads)`` untuk pengembang sumber
    pertama yang punya pemindahan layak, atau None bila tidak ada.
    """
    _, _, not_started = partition(inst, sched, t_now)
    loads = developer_load(inst, sched, t_now)
    by_id = {t.task_id: t for t in inst.tasks}
    dev_by_id = {d.dev_id: d for d in inst.developers}

    tasks_on: dict[int, list[int]] = {}
    for i in sorted(not_started):
        e = sched.entries.get(i)
        if e is not None:
            tasks_on.setdefault(e.dev_id, []).append(i)

    for j_over in _source_order(inst, loads, tasks_on):
        targets_of: dict[int, list[int]] = {}
        for i in tasks_on[j_over]:
            targets = [
                d.dev_id for d in inst.developers
                if d.dev_id != j_over
                and is_pair_feasible(i, d.dev_id, inst, sched, t_now, loads=loads,
                                     enforce_capacity=enforce_capacity, by_id=by_id,
                                     dev_by_id=dev_by_id)
            ]
            if targets:
                targets_of[i] = targets
        if targets_of:
            return j_over, targets_of, by_id, dev_by_id, loads
    return None


def enumerate_reassign_choices(inst, sched: Schedule, t_now: float,
                               enforce_capacity: bool = True) -> list[tuple[int, int, int]]:
    """Semua pilihan pemindahan layak ``(i, j_over, j)`` pada keadaan masuk."""
    found = _feasible_moves(inst, sched, t_now, enforce_capacity)
    if found is None:
        return []
    j_over, targets_of, *_ = found
    return [(i, j_over, j) for i in sorted(targets_of) for j in targets_of[i]]


def choose_reassign(inst, sched: Schedule, t_now: float, enforce_capacity: bool = True):
    """Pilih ``(i*, j_over, j*)``, atau None bila tidak ada pemindahan layak.

    i* = tugas dengan ``2 * progress`` terkecil (seri: ``task_id`` terkecil).
    j* = tujuan dengan beban sesudah pemindahan (beban + upaya i* di tujuan) terkecil,
    seri: ``dev_id`` terkecil.
    """
    found = _feasible_moves(inst, sched, t_now, enforce_capacity)
    if found is None:
        return None
    j_over, targets_of, by_id, _dev_by_id, loads = found

    i_star = min(targets_of, key=lambda i: (2.0 * by_id[i].progress, i))

    def load_after_move(j):
        return (loads.get(j, 0.0) + by_id[i_star].eligible_devs.get(j, 0.0), j)

    j_star = min(targets_of[i_star], key=load_after_move)
    return i_star, j_over, j_star


class Reassign(RepairOperator):
    """Pindahkan satu tugas ke pengembang lain (R3 Balanced).

    Parameters
    ----------
    enforce_capacity : bool, default=True
        Bila False, batas kapasitas tujuan tidak diperiksa (ablasi tanpa mask bawah).

    Examples
    --------
    >>> from dfjsp.datasets import load_brandimarte
    >>> from dfjsp.env import RepairEnv
    >>> env = RepairEnv()
    >>> state, info = env.reset(load_brandimarte(["MK01"])["MK01"], intensity=0.05, seed=0)
    >>> from dfjsp.repair import Reassign
    >>> Reassign().inspect(env.inst, env.sched, env.t_now).choice_count
    10
    >>> hasil = Reassign().apply(env.inst, env.sched, env.t_now)
    >>> pindah = hasil.moved_tasks[0]
    >>> pindah["from_dev"], pindah["to_dev"]
    (5, 0)
    """

    name = "Reassign"

    def __init__(self, enforce_capacity: bool = True):
        self.enforce_capacity = enforce_capacity

    def inspect(self, inst, sched: Schedule, t_now: float) -> OperatorFeasibility:
        """Layak bila ada pemindahan layak (enumerasi sama dengan ``apply``)."""
        choices = enumerate_reassign_choices(inst, sched, t_now, self.enforce_capacity)
        return OperatorFeasibility(
            feasible=bool(choices),
            choice_count=len(choices),
            reason=None if choices else "no_feasible_reassign",
        )

    def apply(self, inst, sched: Schedule, t_now: float) -> RepairResult:
        feasible = count_feasible_pairs(inst, sched, t_now)
        choice = choose_reassign(inst, sched, t_now, self.enforce_capacity)
        if choice is None:
            return RepairResult(sched, [], invalid_attempt=True, feasible_pairs=feasible)
        i_star, j_over, j_star = choice
        e = sched.entries[i_star]
        sched.entries[i_star] = ScheduleEntry(i_star, j_star, e.start, e.end)
        recompute_timeline(inst, sched, t_now)
        return RepairResult(
            new_sched=sched,
            moved_tasks=[{"task_id": i_star, "from_dev": j_over, "to_dev": j_star}],
            invalid_attempt=False,
            feasible_pairs=feasible,
        )
