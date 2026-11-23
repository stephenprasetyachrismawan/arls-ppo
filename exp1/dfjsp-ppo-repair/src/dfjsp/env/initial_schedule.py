"""Jadwal awal prediktif S0 (Subbab 5.1 rancangan; Subbab 3.3.2 Bab III)."""

from __future__ import annotations

import copy

from ..core.datatypes import STATUS_NOT_STARTED, FJSPInstance, Schedule, ScheduleEntry
from .schedule import default_keep_order


def build_initial_schedule(inst: FJSPInstance) -> Schedule:
    """Bangun S0 via list-scheduling deterministik (Subbab 5.1).

    Raises
    ------
    ValueError
        Bila terdeteksi deadlock (siklus deps / tidak ada dev tersedia).
    """
    by_id = {t.task_id: t for t in inst.tasks}
    w = {d.dev_id: 0.0 for d in inst.developers}
    cap = {d.dev_id: d.capacity for d in inst.developers}
    avail = {d.dev_id: d.available for d in inst.developers}
    dev_free = {d.dev_id: 0.0 for d in inst.developers}
    finish: dict[int, float] = {}
    entries: dict[int, ScheduleEntry] = {}
    scheduled: set[int] = set()
    remaining = set(by_id)

    while remaining:
        ready = [
            tid for tid in remaining if all(dep in scheduled for dep in by_id[tid].deps)
        ]
        if not ready:
            raise ValueError(f"{inst.name}: deadlock penjadwalan S0 (siklus deps?).")
        ready.sort(key=lambda tid: (by_id[tid].due_date, tid))

        progressed = False
        for tid in ready:
            t = by_id[tid]
            cands = [j for j in t.eligible_devs if avail.get(j, True)]
            if not cands:
                continue  # tidak ada dev tersedia sekarang; coba ronde berikutnya
            fit = [j for j in cands if w[j] + t.eligible_devs[j] <= cap[j] + 1e-9]
            pool = fit if fit else cands
            jstar = min(pool, key=lambda j: (w[j], j))
            eff = t.eligible_devs[jstar]
            ready_time = max((finish[d] for d in t.deps), default=0.0)
            start = max(ready_time, dev_free[jstar])
            fin = start + eff

            entries[tid] = ScheduleEntry(task_id=tid, dev_id=jstar, start=start, end=fin)
            finish[tid] = fin
            w[jstar] += eff
            dev_free[jstar] = fin
            scheduled.add(tid)
            remaining.discard(tid)

            t.assigned_dev = jstar
            t.start_time = start
            t.finish_time = fin
            t.status = STATUS_NOT_STARTED
            t.progress = 0.0
            progressed = True

        if not progressed:
            raise ValueError(f"{inst.name}: tidak ada task dapat dijadwalkan (deadlock).")

    for d in inst.developers:
        d.workload = w[d.dev_id]

    sched = Schedule(entries=entries, keep_order=default_keep_order())
    sched.baseline_ref = copy.deepcopy(entries)
    return sched
