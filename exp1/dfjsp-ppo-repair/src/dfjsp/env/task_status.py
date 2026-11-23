"""Status dan kemajuan tugas dari garis waktu jadwal."""

from __future__ import annotations

from ..core.datatypes import (
    STATUS_COMPLETED,
    STATUS_IN_PROGRESS,
    STATUS_NOT_STARTED,
    FJSPInstance,
    Schedule,
)
from .work_calendar import working_time


def advance_statuses(inst: FJSPInstance, sched: Schedule, t_now: float) -> None:
    """Perbarui ``available`` pengembang serta ``status`` dan ``progress`` tiap tugas.

    Tugas yang berakhir sebelum ``t_now`` selesai, yang sedang berjalan diberi
    kemajuan menurut jam kerja (jeda tidak dihitung), sisanya belum mulai.
    """
    for d in inst.developers:
        if d.dev_id in sched.pauses:
            d.available = not any(a <= t_now < b for a, b in sched.pauses[d.dev_id])
    for t in inst.tasks:
        e = sched.entries.get(t.task_id)
        if e is None:
            continue
        pauses = sched.pauses.get(e.dev_id, [])
        dur = working_time(e.start, e.end, pauses)
        if e.end <= t_now:
            t.status = STATUS_COMPLETED
            t.progress = 1.0
        elif e.start <= t_now < e.end:
            t.status = STATUS_IN_PROGRESS
            t.progress = min(1.0, working_time(e.start, t_now, pauses) / dur) if dur > 0 else 1.0
        else:
            t.status = STATUS_NOT_STARTED
            t.progress = 0.0
