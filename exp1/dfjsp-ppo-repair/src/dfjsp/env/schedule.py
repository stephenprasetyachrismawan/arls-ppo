"""Partisi jadwal dan garis waktu (Subbab 5.2 rancangan; Pers. 3.1)."""

from __future__ import annotations

from contextlib import contextmanager

from ..core.datatypes import FJSPInstance, Schedule, ScheduleEntry
from .work_calendar import working_time, finish_after_work

# Nilai keep_order untuk jadwal yang baru dibangun. Diubah hanya lewat legacy_queue_order().
_KEEP_ORDER_DEFAULT = True


def default_keep_order() -> bool:
    """Nilai ``keep_order`` untuk jadwal awal yang baru dibangun."""
    return _KEEP_ORDER_DEFAULT


@contextmanager
def legacy_queue_order():
    """Selama blok ini, jadwal baru memakai perilaku lama (antrean menurut nomor tugas).

    Hanya untuk memutar ulang rekaman emas dan hasil resmi1. Lihat
    ``revisi/diagnosis_metode_sekarang.md`` (D1) untuk alasan perilaku lama diganti.

    Examples
    --------
    >>> from dfjsp.env.schedule import default_keep_order, legacy_queue_order
    >>> with legacy_queue_order():
    ...     default_keep_order()
    False
    >>> default_keep_order()
    True
    """
    global _KEEP_ORDER_DEFAULT
    lama = _KEEP_ORDER_DEFAULT
    _KEEP_ORDER_DEFAULT = False
    try:
        yield
    finally:
        _KEEP_ORDER_DEFAULT = lama


def partition(inst: FJSPInstance, sched: Schedule, t_now: float) -> tuple[set[int], set[int], set[int]]:
    """Klasifikasi tiap task pada waktu t_now (Pers. 3.1), sapuan O(|T|).

    Returns
    -------
    tuple of set
        ``(T_completed, T_in_progress, T_not_started)``: tiga himpunan saling-lepas
        yang gabungannya = seluruh task instans.
    """
    completed: set[int] = set()
    in_progress: set[int] = set()
    not_started: set[int] = set()
    for t in inst.tasks:
        e = sched.entries.get(t.task_id)
        if e is None:
            not_started.add(t.task_id)
        elif e.end <= t_now:
            completed.add(t.task_id)
        elif e.start <= t_now < e.end:
            in_progress.add(t.task_id)
        else:
            not_started.add(t.task_id)
    return completed, in_progress, not_started


def developer_load(inst: FJSPInstance, sched: Schedule, t_now: float) -> dict[int, float]:
    """Beban kerja w_j = sisa efort task non-selesai yang ditugaskan ke dev j."""
    w = {d.dev_id: 0.0 for d in inst.developers}
    completed, in_progress, _ = partition(inst, sched, t_now)
    for t in inst.tasks:
        e = sched.entries.get(t.task_id)
        if e is None or t.task_id in completed:
            continue
        pauses = sched.pauses.get(e.dev_id, [])
        dur = working_time(e.start, e.end, pauses)
        if t.task_id in in_progress and dur > 0:
            frac_done = min(1.0, working_time(e.start, t_now, pauses) / dur)
            rem = dur * (1.0 - frac_done)
        else:
            rem = dur
        if e.dev_id in w:
            w[e.dev_id] += max(0.0, rem)
    return w


def _topo_order(node_set: set[int], by_id: dict) -> list[int]:
    """Urutan topologis deterministik atas subgraf node_set (edge intra-set)."""
    indeg = {n: 0 for n in node_set}
    adj: dict[int, list[int]] = {n: [] for n in node_set}
    for n in node_set:
        for d in by_id[n].deps:
            if d in node_set:
                adj[d].append(n)
                indeg[n] += 1
    queue = sorted([n for n in node_set if indeg[n] == 0])
    out: list[int] = []
    while queue:
        u = queue.pop(0)
        out.append(u)
        for v in sorted(adj[u]):
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
                queue.sort()
    if len(out) != len(node_set):
        raise ValueError("Siklus terdeteksi pada subgraf T_not_started.")
    return out


def _place(j, eff, ready_time, dev_free, release, t_now, sched):
    """Waktu mulai dan selesai satu tugas di pengembang j (melompati masa tidak tersedia)."""
    start = max(t_now, ready_time, dev_free, release)
    for a, b in sorted(sched.pauses.get(j, [])):
        if a <= start < b:
            start = b
    return start, finish_after_work(start, eff, sched.pauses.get(j, []))


def queue_order(inst: FJSPInstance, sched: Schedule, t_now: float) -> dict[int, list[int]]:
    """Antrean tugas belum mulai per pengembang, menurut posisinya di jadwal sekarang.

    Posisi dibaca dari waktu mulai yang tercatat (seri: nomor tugas). Operator yang ingin
    memindahkan tugas ke posisi tertentu cukup memberi waktu mulai di antara dua
    tetangganya, lalu memanggil ``recompute_timeline``.
    """
    _, _, not_started = partition(inst, sched, t_now)
    queues: dict[int, list[int]] = {d.dev_id: [] for d in inst.developers}
    for tid in not_started:
        entry = sched.entries.get(tid)
        if entry is not None:
            queues.setdefault(entry.dev_id, []).append(tid)
    for j in queues:
        queues[j].sort(key=lambda tid: (sched.entries[tid].start, tid))
    return queues


def recompute_timeline(inst: FJSPInstance, sched: Schedule, t_now: float) -> int:
    """Hitung ulang waktu mulai dan selesai tugas yang belum mulai (Subbab 5.2).

    Bila ``sched.keep_order`` True (bawaan), antrean tiap pengembang dipertahankan: tugas
    hanya bergeser ke kanan, seperti *right-shift* di literatur (ALP-PPO subbab 4.6).
    Waktu mulai = paling akhir dari (tugas sebelumnya di antrean selesai, semua pendahulu
    selesai, batas rilis, waktu sekarang). Tidak ada pengisian celah, sama dengan cara S0
    dibangun, sehingga menghitung ulang S0 tanpa gangguan menghasilkan S0 persis.

    Bila urutan antrean bertabrakan dengan dependensi antar-pengembang (bisa terjadi
    sesudah operator memindah tugas), tugas yang sudah siap dan posisinya paling awal
    dimajukan ke depan antreannya. Banyaknya kejadian ini dikembalikan.

    Bila ``sched.keep_order`` False, perilaku lama dipakai (antrean menurut nomor tugas).

    Returns
    -------
    int
        Banyak tugas yang terpaksa dimajukan karena tabrakan urutan (selalu 0 untuk
        perilaku lama).

    Raises
    ------
    ValueError
        Bila sebuah dependensi tidak punya entri jadwal sama sekali.
    """
    if not getattr(sched, "keep_order", True):
        _recompute_by_task_id(inst, sched, t_now)
        return 0

    completed, in_progress, not_started = partition(inst, sched, t_now)
    by_id = {t.task_id: t for t in inst.tasks}
    dev_free = {d.dev_id: t_now for d in inst.developers}
    finish: dict[int, float] = {}
    for tid in completed | in_progress:
        e = sched.entries[tid]
        dev_free[e.dev_id] = max(dev_free.get(e.dev_id, t_now), e.end)
        finish[tid] = e.end

    for tid in not_started:
        for d in by_id[tid].deps:
            if d not in finish and d not in not_started:
                raise ValueError(f"{inst.name}: dependensi {d} task {tid} tidak terjadwal.")

    queues = queue_order(inst, sched, t_now)
    key = {tid: (sched.entries[tid].start, tid) for tid in not_started}
    waiting = set(not_started)
    forced = 0

    def schedule_head(j: int) -> None:
        tid = queues[j].pop(0)
        t = by_id[tid]
        e = sched.entries[tid]
        eff = t.eligible_devs.get(j, e.end - e.start)
        ready = max((finish[d] for d in t.deps), default=t_now)
        start, fin = _place(j, eff, ready, dev_free.get(j, t_now),
                            sched.release.get(tid, t_now), t_now, sched)
        sched.entries[tid] = ScheduleEntry(task_id=tid, dev_id=j, start=start, end=fin)
        finish[tid] = fin
        dev_free[j] = fin
        t.start_time = start
        t.finish_time = fin
        waiting.discard(tid)

    while waiting:
        progressed = False
        for j in sorted(queues):
            while queues[j] and all(d in finish for d in by_id[queues[j][0]].deps):
                schedule_head(j)
                progressed = True
        if progressed:
            continue
        # Tabrakan: setiap kepala antrean menunggu tugas yang ada di belakang antrean lain.
        ready = [tid for tid in waiting if all(d in finish for d in by_id[tid].deps)]
        if not ready:
            raise ValueError(f"{inst.name}: siklus dependensi pada tugas belum mulai.")
        tid = min(ready, key=lambda x: key[x])
        j = sched.entries[tid].dev_id
        queues[j].remove(tid)
        queues[j].insert(0, tid)
        forced += 1
    return forced


def _recompute_by_task_id(inst: FJSPInstance, sched: Schedule, t_now: float) -> None:
    """Perilaku lama: antrean disusun ulang menurut urutan topologis dan nomor tugas.

    Dipertahankan hanya untuk memutar ulang rekaman emas dan hasil resmi1 (D1 di
    ``revisi/diagnosis_metode_sekarang.md``).
    """
    completed, in_progress, not_started = partition(inst, sched, t_now)
    by_id = {t.task_id: t for t in inst.tasks}

    dev_free = {d.dev_id: t_now for d in inst.developers}
    finish: dict[int, float] = {}
    for tid in completed | in_progress:
        e = sched.entries[tid]
        dev_free[e.dev_id] = max(dev_free.get(e.dev_id, t_now), e.end)
        finish[tid] = e.end

    for tid in _topo_order(not_started, by_id):
        t = by_id[tid]
        e = sched.entries[tid]
        j = e.dev_id
        eff = t.eligible_devs.get(j, e.end - e.start)
        dep_finishes = []
        for d in t.deps:
            if d not in finish:
                raise ValueError(
                    f"{inst.name}: dependensi {d} task {tid} belum terjadwal "
                    f"(pelanggaran precedence)."
                )
            dep_finishes.append(finish[d])
        ready_time = max(dep_finishes) if dep_finishes else t_now
        release = sched.release.get(tid, t_now)  # kendala Delay (Pers. 3.5) yang bertahan
        start = max(t_now, ready_time, dev_free.get(j, t_now), release)
        for a, b in sorted(sched.pauses.get(j, [])):
            if a <= start < b:
                start = b
        fin = finish_after_work(start, eff, sched.pauses.get(j, []))
        sched.entries[tid] = ScheduleEntry(task_id=tid, dev_id=j, start=start, end=fin)
        finish[tid] = fin
        dev_free[j] = fin
        t.start_time = start
        t.finish_time = fin
