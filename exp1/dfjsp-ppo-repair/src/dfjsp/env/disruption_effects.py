"""Dampak tiap jenis gangguan pada instans dan jadwal."""

from __future__ import annotations

import numpy as np

from ..core.datatypes import (
    STATUS_COMPLETED,
    STATUS_IN_PROGRESS,
    STATUS_NOT_STARTED,
    FJSPInstance,
    Schedule,
    ScheduleEntry,
    Task,
)
from .disruptions import Event
from .schedule import partition, recompute_timeline
from .task_status import advance_statuses
from .work_calendar import finish_after_work, working_time


def apply_disruption(inst: FJSPInstance, sched: Schedule, event: Event, t_now: float,
                     horizon: float) -> None:
    """Terapkan satu gangguan pada ``inst`` dan ``sched`` (diubah di tempat).

    Melempar ``ValueError`` bila isi gangguan tidak masuk akal, misalnya durasi
    tidak positif. Pemanggil yang memutuskan cara menangani galat itu.
    """
    p = event.payload
    if event.type == "batch":
        # Beberapa gangguan dalam satu jendela: semuanya diterapkan pada waktu keputusan.
        for sub in p.get("events", []):
            apply_disruption(inst, sched, sub, t_now, horizon)
        return
    if event.type == "developer_unavailable":
        _developer_unavailable(inst, sched, p, t_now)
    elif event.type == "urgent_task":
        _inject_urgent_task(inst, sched, p, t_now, horizon)
        recompute_timeline(inst, sched, t_now)
    elif event.type == "effort_revision":
        _revise_effort(inst, sched, p.get("task_id"), p.get("factor", 1.0), t_now)
        recompute_timeline(inst, sched, t_now)
        advance_statuses(inst, sched, t_now)
    elif event.type == "priority_change":
        task = _find(inst, p.get("task_id"))
        if task is not None:
            task.priority = float(min(1.0, max(0.0, p.get("new_priority", task.priority))))
            task.due_date += p.get("due_delta", 0.0) * horizon
    elif event.type == "rework_review_fail":
        _rework(inst, sched, p.get("task_id"), p.get("extra_effort", 0.0), t_now)
        recompute_timeline(inst, sched, t_now)


def _find(inst: FJSPInstance, task_id) -> Task | None:
    for t in inst.tasks:
        if t.task_id == task_id:
            return t
    return None


def _developer_unavailable(inst, sched, p: dict, t_now: float) -> None:
    """Pengembang berhenti kerja selama ``duration``; tugas yang berjalan molor."""
    dev_id = p.get("dev_id")
    duration = float(p.get("duration", 5.0))
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("durasi ketidaktersediaan harus positif dan berhingga")
    if not any(d.dev_id == dev_id for d in inst.developers):
        raise ValueError("pengembang sasaran tidak ditemukan")
    _, running, _ = partition(inst, sched, t_now)
    remaining = {
        tid: working_time(t_now, sched.entries[tid].end, sched.pauses.get(dev_id, []))
        for tid in running if sched.entries[tid].dev_id == dev_id
    }
    sched.pauses.setdefault(dev_id, []).append((t_now, t_now + duration))
    for tid, effort in remaining.items():
        e = sched.entries[tid]
        e.end = finish_after_work(t_now, effort, sched.pauses[dev_id])
        _find(inst, tid).finish_time = e.end
    for d in inst.developers:
        if d.dev_id == dev_id:
            d.available = False
    recompute_timeline(inst, sched, t_now)
    advance_statuses(inst, sched, t_now)


def _revise_effort(inst, sched, task_id, factor: float, t_now: float) -> None:
    """Revisi estimasi total; waktu yang sudah berlalu tidak dihapus."""
    t = _find(inst, task_id)
    if t is None or t.status == STATUS_COMPLETED:
        return
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("faktor revisi upaya harus positif dan berhingga")
    t.eligible_devs = {j: e * factor for j, e in t.eligible_devs.items()}
    entry = sched.entries.get(task_id)
    if entry is not None and t.status == STATUS_IN_PROGRESS:
        finish = max(t_now, finish_after_work(
            entry.start, t.eligible_devs[entry.dev_id],
            sched.pauses.get(entry.dev_id, [])))
        sched.entries[task_id] = ScheduleEntry(task_id, entry.dev_id, entry.start, finish)
        t.finish_time = finish


def _rework(inst, sched, task_id, extra_effort: float, t_now: float) -> None:
    """Tugas gagal ditinjau: kembali ke belum mulai dan upayanya bertambah."""
    t = _find(inst, task_id)
    if t is None:
        return
    t.status = STATUS_NOT_STARTED
    t.progress = 0.0
    t.eligible_devs = {j: e + extra_effort for j, e in t.eligible_devs.items()}
    e = sched.entries.get(task_id)
    if e is not None:
        j = e.dev_id
        eff = t.eligible_devs.get(j, e.end - e.start)
        if getattr(sched, "keep_order", True):
            # Tingkat 0: tugas tetap di posisinya semula, yaitu paling depan antrean
            # pengembang yang sama (ia baru saja dikerjakan). Waktu mulai sedikit sesudah
            # t_now supaya tercatat "belum mulai"; recompute_timeline menghitung waktunya.
            start = float(np.nextafter(t_now, np.inf))
        else:
            # Perilaku lama, termasuk jeda buatan max(eff, 1) (D1, diagnosis).
            start = t_now + max(eff, 1.0)
        sched.entries[task_id] = ScheduleEntry(task_id, j, start, start + eff)


def _inject_urgent_task(inst, sched, p: dict, t_now: float, horizon: float) -> None:
    """Tambahkan satu tugas baru di akhir antrean pengembang ``dev_id`` (tingkat 0).

    Model lama: hanya satu pengembang yang mampu, tenggat ``t_now + due_factor * horizon``.
    Model TAWOS: payload memuat ``eligible`` (pengembang yang mampu dan upayanya) dan
    ``due_rule``. ``"segera"``: tenggat = ``t_now + (1 + due_slack) * upaya`` (diminta
    langsung). ``"antrean"``: tenggat = selesai di akhir antrean + ``due_slack * upaya``,
    sama dengan cara tenggat tugas terencana diturunkan dari S0.
    """
    dev = int(p.get("dev_id", 0))
    eff = float(p.get("effort", 1.0))
    eligible = {int(j): float(e) for j, e in p.get("eligible", {dev: eff}).items()}
    eligible.setdefault(dev, eff)
    eff = eligible[dev]
    new_id = (max((t.task_id for t in inst.tasks), default=-1)) + 1
    new_job = (max((t.job_id for t in inst.tasks), default=-1)) + 1
    rule = p.get("due_rule")
    slack = float(p.get("due_slack", 0.5))
    if rule == "segera":
        due = t_now + (1.0 + slack) * eff
    else:
        due = t_now + float(p.get("due_factor", 0.3)) * horizon
    task = Task(
        task_id=new_id, job_id=new_job, op_index=0,
        eligible_devs=eligible, deps=[], due_date=due,
        priority=float(p.get("priority", 1.0)), status=STATUS_NOT_STARTED,
    )
    inst.tasks.append(task)
    dev_free = t_now
    for e in sched.entries.values():
        if e.dev_id == dev:
            dev_free = max(dev_free, e.end)
    # Entri baru belum pernah dikerjakan. Tandai sementara belum mulai;
    # recompute_timeline akan menempatkannya menurut kalender pengembang.
    start = max(dev_free, float(np.nextafter(t_now, np.inf)))
    sched.entries[new_id] = ScheduleEntry(new_id, dev, start, start + eff)
    for j in eligible:
        if j < len(inst.developers):
            inst.developers[j].skills.add(new_job)
    if rule == "antrean":
        # Tenggat dihitung sesudah posisinya di akhir antrean diketahui.
        recompute_timeline(inst, sched, t_now)
        task.due_date = sched.entries[new_id].end + slack * eff
