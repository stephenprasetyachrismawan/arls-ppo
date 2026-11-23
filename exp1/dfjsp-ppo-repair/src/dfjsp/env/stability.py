"""Ukuran stabilitas jadwal terhadap jadwal acuan (jadwal awal S0)."""

from __future__ import annotations

from typing import NamedTuple

from ..core.datatypes import FJSPInstance, Schedule
from .schedule import partition


class StabilityStep(NamedTuple):
    """Ukuran stabilitas satu titik keputusan."""

    nervousness: float
    context_switch: float
    start_deviation: float


def stability_step(inst: FJSPInstance, sched: Schedule, t_now: float,
                   horizon: float) -> StabilityStep:
    """Bandingkan jadwal sekarang dengan ``sched.baseline_ref`` pada ``t_now``.

    Hanya tugas yang belum selesai dan ada di jadwal acuan yang dibandingkan.

    * ``nervousness`` = 0,5 * porsi tugas yang pindah pengembang
      + 0,5 * rata-rata pergeseran waktu mulai (dibagi ``horizon``).
    * ``context_switch`` = jumlah kemajuan tugas yang dipindah pengembang.
    * ``start_deviation`` = jumlah mentah pergeseran waktu mulai.
    """
    base = sched.baseline_ref or {}
    _, in_progress, not_started = partition(inst, sched, t_now)
    unfinished = [t for t in inst.tasks if t.task_id in (in_progress | not_started)]
    comparable = [t for t in unfinished if t.task_id in base]
    if not comparable:
        return StabilityStep(0.0, 0.0, 0.0)
    n_changed = 0
    start_dev_raw = 0.0
    ctx = 0.0
    for t in comparable:
        cur = sched.entries.get(t.task_id)
        old = base[t.task_id]
        if cur is None:
            continue
        if cur.dev_id != old.dev_id:
            n_changed += 1
            ctx += t.progress
        start_dev_raw += abs(cur.start - old.start)
    n = len(comparable)
    n_a = n_changed / n
    n_s = (start_dev_raw / horizon) / n
    return StabilityStep(0.5 * n_a + 0.5 * n_s, ctx, start_dev_raw)
