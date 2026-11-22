"""Tangga empat tingkat perbaikan pada tugas terdampak (rancangan v2 bagian 4.4).

Tingkat 0 adalah *right-shift*: jadwal sesudah gangguan diterima apa adanya (urutan
antrean tetap, lihat ``recompute_timeline``). Tingkat 1 sampai 3 hanya menyentuh tugas
terdampak ``sched.affected`` yang dihitung lingkungan setiap ada gangguan, bukan tugas di
seluruh proyek seperti operator lama (D4 di ``revisi/diagnosis_metode_sekarang.md``).

====  ===================  ===============================================  ==========
Aksi  Nama                 Yang dilakukan                                   Diubah
====  ===================  ===============================================  ==========
0     RightShift           tidak ada                                        -
1     Resequence           majukan tugas terdampak ke posisi menurut        urutan
                           tenggat di antrean pengembang yang sama
2     LadderReassign       pindahkan sampai ``max_tasks`` tugas terdampak    penugasan,
                           paling terlambat ke pengembang dan posisi yang    posisi
                           paling mengurangi keterlambatan
3     PartialReschedule    keluarkan tugas di jendela dekat beserta tugas    penugasan,
                           terdampak, lalu sisipkan ulang satu per satu      urutan
                           (tenggat terdekat dulu) di tempat terbaik
====  ===================  ===============================================  ==========

Tingkat 2 dan 3 menilai setiap calon dengan menghitung ulang jadwal salinan, jadi
"terbaik" berarti total keterlambatan paling kecil, seri dipecah dengan perubahan paling
sedikit. Dasar: *affected operations rescheduling* dan *match-up* (dikutip di rancangan;
belum dibaca langsung).
"""

from __future__ import annotations

import copy

import numpy as np

from ..core.datatypes import Schedule, ScheduleEntry
from ..env.schedule import developer_load, partition, queue_order, recompute_timeline
from ..env.state import total_tardiness
from .base import OperatorFeasibility, RepairOperator, RepairResult
from .masks import is_pair_feasible

RIGHT_SHIFT, RESEQUENCE, LADDER_REASSIGN, PARTIAL_RESCHEDULE = 0, 1, 2, 3
LADDER_NAMES = ["RightShift", "Resequence", "Reassign", "PartialReschedule"]
EPS = 1e-9


# ------------------------------------------------------------------ bantuan
def _light_copy(sched: Schedule) -> Schedule:
    """Salinan cukup untuk uji-coba: entri dan rilis disalin, jeda dibagi pakai."""
    out = copy.copy(sched)
    out.entries = dict(sched.entries)
    out.release = dict(sched.release)
    return out


def set_queue(sched: Schedule, dev: int, order: list[int], t_now: float) -> None:
    """Tetapkan antrean pengembang ``dev`` menjadi ``order`` (semua belum mulai).

    Caranya memberi waktu mulai sementara yang naik menurut urutan; nilai sebenarnya
    dihitung ``recompute_timeline``. Semua tugas di ``order`` dipindah ke ``dev``.
    """
    for k, tid in enumerate(order):
        e = sched.entries[tid]
        key = t_now + 1e-6 * (k + 1)
        sched.entries[tid] = ScheduleEntry(tid, dev, key, key + max(EPS, e.end - e.start))


def affected_not_started(inst, sched: Schedule, t_now: float) -> list[int]:
    """Tugas terdampak yang belum mulai, urut dari yang paling terlambat."""
    _, _, not_started = partition(inst, sched, t_now)
    by_id = {t.task_id: t for t in inst.tasks}
    ids = [t for t in getattr(sched, "affected", set()) if t in not_started]
    return sorted(ids, key=lambda t: (-(sched.entries[t].end - by_id[t].due_date), t))


def _edd_insert(queue: list[int], tid: int, by_id: dict) -> list[int]:
    """Sisipkan ``tid`` sebelum tugas pertama yang tenggatnya lebih jauh."""
    due = by_id[tid].due_date
    for k, other in enumerate(queue):
        if by_id[other].due_date > due:
            return queue[:k] + [tid] + queue[k:]
    return queue + [tid]


def _lis_length(seq: list[int]) -> int:
    """Panjang barisan naik terpanjang (algoritma patience, O(n log n))."""
    import bisect
    tails: list[int] = []
    for x in seq:
        k = bisect.bisect_left(tails, x)
        if k == len(tails):
            tails.append(x)
        else:
            tails[k] = x
    return len(tails)


def count_changes(before: Schedule, after: Schedule, t_now: float, inst) -> tuple[int, int]:
    """(tugas pindah pengembang, tugas pindah posisi) dibanding ``before``.

    Pindah posisi dihitung sebagai banyak tugas paling sedikit yang harus dipindah supaya
    antrean lama menjadi antrean baru, pada tugas yang tetap di pengembang yang sama:
    n dikurangi panjang barisan naik terpanjang. Memajukan satu tugas melewati sepuluh
    tugas lain dihitung satu perubahan, bukan sebelas.

    Tugas yang dibandingkan adalah tugas yang belum mulai di ``before``. Antrean barunya
    dibaca dari ``after`` walaupun tugas itu kini mulai tepat pada ``t_now``, supaya tugas
    yang dimajukan ke depan antrean tetap terhitung.
    """
    moved_dev = sum(1 for tid, e in after.entries.items()
                    if tid in before.entries and before.entries[tid].dev_id != e.dev_id)
    qb = queue_order(inst, before, t_now)
    waiting = {t for q in qb.values() for t in q}
    qa: dict[int, list[int]] = {}
    for tid in waiting:
        e = after.entries.get(tid)
        if e is not None:
            qa.setdefault(e.dev_id, []).append(tid)
    moved_pos = 0
    for j, new in qa.items():
        new.sort(key=lambda tid: (after.entries[tid].start, tid))
        old_pos = {t: k for k, t in enumerate(qb.get(j, []))}
        common = [old_pos[t] for t in new if t in old_pos]
        moved_pos += len(common) - _lis_length(common)
    return moved_dev, moved_pos


_changes = count_changes          # nama lama, masih dipakai modul lain dan skrip revisi


def _moves(before: Schedule, after: Schedule) -> list[dict]:
    out = []
    for tid, e in after.entries.items():
        b = before.entries.get(tid)
        if b is not None and (b.dev_id != e.dev_id or abs(b.start - e.start) > EPS):
            out.append({"task_id": tid, "from_dev": b.dev_id, "to_dev": e.dev_id,
                        "shift": e.start - b.start})
    return out


def _score(inst, sched: Schedule) -> float:
    return total_tardiness(inst, sched)


# ------------------------------------------------------------------ tingkat 0
class RightShift(RepairOperator):
    """Tingkat 0: terima jadwal sesudah gangguan (antrean tetap, waktu bergeser)."""

    name = "RightShift"

    def inspect(self, inst, sched, t_now):
        return OperatorFeasibility(feasible=True, choice_count=1)

    def apply(self, inst, sched, t_now):
        return RepairResult(new_sched=sched)


# ------------------------------------------------------------------ tingkat 1
def _resequence_plan(inst, sched, t_now):
    by_id = {t.task_id: t for t in inst.tasks}
    targets = affected_not_started(inst, sched, t_now)
    if not targets:
        return None
    queues = queue_order(inst, sched, t_now)
    plan = {}
    for j, queue in queues.items():
        mine = [t for t in targets if sched.entries[t].dev_id == j]
        if not mine or len(queue) < 2:
            continue
        rest = [t for t in queue if t not in set(mine)]
        new = rest
        for tid in sorted(mine, key=lambda t: (by_id[t].due_date, t)):
            new = _edd_insert(new, tid, by_id)
        if new != queue:
            plan[j] = new
    return plan or None


class Resequence(RepairOperator):
    """Tingkat 1: majukan tugas terdampak menurut tenggat, tanpa pindah pengembang."""

    name = "Resequence"

    def inspect(self, inst, sched, t_now):
        plan = _resequence_plan(inst, sched, t_now)
        n = 0 if plan is None else len(plan)
        return OperatorFeasibility(feasible=n > 0, choice_count=n,
                                   reason=None if n else "no_order_change")

    def apply(self, inst, sched, t_now):
        plan = _resequence_plan(inst, sched, t_now)
        if plan is None:
            return RepairResult(sched, [], invalid_attempt=True)
        before = _light_copy(sched)
        for j, order in plan.items():
            set_queue(sched, j, order, t_now)
        recompute_timeline(inst, sched, t_now)
        return RepairResult(sched, _moves(before, sched), invalid_attempt=False)


# ------------------------------------------------------------------ tingkat 2
def _reassign_candidates(inst, sched, t_now, tid, enforce_capacity):
    """Calon (pengembang, antrean baru) untuk memindah ``tid``."""
    by_id = {t.task_id: t for t in inst.tasks}
    dev_by_id = {d.dev_id: d for d in inst.developers}
    loads = developer_load(inst, sched, t_now)
    queues = queue_order(inst, sched, t_now)
    cur = sched.entries[tid].dev_id
    out = []
    for j in sorted(by_id[tid].eligible_devs):
        if j == cur or not is_pair_feasible(tid, j, inst, sched, t_now, loads=loads,
                                            enforce_capacity=enforce_capacity,
                                            by_id=by_id, dev_by_id=dev_by_id):
            continue
        queue = [t for t in queues.get(j, []) if t != tid]
        options = {tuple(_edd_insert(queue, tid, by_id)), tuple([tid] + queue),
                   tuple(queue + [tid])}
        out.extend((j, list(o)) for o in options)
    return out


class LadderReassign(RepairOperator):
    """Tingkat 2: pindahkan tugas terdampak paling terlambat ke tempat terbaik.

    Parameters
    ----------
    max_tasks : int, default=3
        Paling banyak tugas yang dipindah per keputusan. Pemindahan berhenti bila tidak
        lagi mengurangi keterlambatan.
    enforce_capacity : bool, default=True
    """

    name = "Reassign"

    def __init__(self, max_tasks: int = 3, enforce_capacity: bool = True):
        self.max_tasks = max_tasks
        self.enforce_capacity = enforce_capacity

    def inspect(self, inst, sched, t_now):
        n = sum(len(_reassign_candidates(inst, sched, t_now, t, self.enforce_capacity))
                for t in affected_not_started(inst, sched, t_now)[: self.max_tasks])
        return OperatorFeasibility(feasible=n > 0, choice_count=n,
                                   reason=None if n else "no_feasible_reassign")

    def apply(self, inst, sched, t_now):
        before = _light_copy(sched)
        current = _score(inst, sched)
        moved_any = False
        for tid in affected_not_started(inst, sched, t_now)[: self.max_tasks]:
            best = None
            for j, order in _reassign_candidates(inst, sched, t_now, tid,
                                                 self.enforce_capacity):
                trial = _light_copy(sched)
                set_queue(trial, j, order, t_now)
                recompute_timeline(inst, trial, t_now)
                score = _score(inst, trial)
                if score < current - EPS and (best is None or score < best[0] - EPS):
                    best = (score, trial)
            if best is None:
                continue
            current = best[0]
            sched.entries = best[1].entries
            moved_any = True
        recompute_timeline(inst, sched, t_now)
        if not moved_any:
            return RepairResult(sched, [], invalid_attempt=True)
        return RepairResult(sched, _moves(before, sched), invalid_attempt=False)


# ------------------------------------------------------------------ tingkat 3
class PartialReschedule(RepairOperator):
    """Tingkat 3: susun ulang penugasan dan urutan tugas di sekitar gangguan.

    Tugas yang dikeluarkan: tugas terdampak ditambah tugas belum mulai yang mulainya
    dalam ``window`` x rata-rata upaya dari sekarang, paling banyak ``max_tasks``. Lalu
    satu per satu (tenggat terdekat dulu) disisipkan ke pengembang dan posisi yang
    memberi keterlambatan total paling kecil.

    Parameters
    ----------
    window : float, default=3.0
    max_tasks : int, default=8
    max_devs : int, default=3
        Per tugas hanya ``max_devs`` pengembang mampu yang bebannya paling ringan (ditambah
        pengembang asalnya) yang dicoba, supaya satu keputusan tetap cepat.
    enforce_capacity : bool, default=True
    """

    name = "PartialReschedule"

    def __init__(self, window: float = 3.0, max_tasks: int = 8, max_devs: int = 3,
                 enforce_capacity: bool = True):
        self.window = window
        self.max_tasks = max_tasks
        self.max_devs = max_devs
        self.enforce_capacity = enforce_capacity

    def _pool(self, inst, sched, t_now):
        _, _, not_started = partition(inst, sched, t_now)
        if not not_started:
            return []
        effs = [e.end - e.start for e in sched.entries.values()]
        horizon = t_now + self.window * (float(np.mean(effs)) if effs else 1.0)
        near = sorted((t for t in not_started if sched.entries[t].start <= horizon),
                      key=lambda t: (sched.entries[t].start, t))
        pool = affected_not_started(inst, sched, t_now)
        for t in near:
            if len(pool) >= self.max_tasks:
                break
            if t not in pool:
                pool.append(t)
        # Tugas yang pendahulunya juga dikeluarkan tetap boleh: penyisipan memakai
        # urutan tenggat, dan hitung ulang menyelesaikan tabrakan urutan.
        return pool[: self.max_tasks]

    def inspect(self, inst, sched, t_now):
        n = len(self._pool(inst, sched, t_now))
        return OperatorFeasibility(feasible=n > 0, choice_count=n,
                                   reason=None if n else "no_task_in_window")

    def apply(self, inst, sched, t_now):
        pool = self._pool(inst, sched, t_now)
        if not pool:
            return RepairResult(sched, [], invalid_attempt=True)
        before = _light_copy(sched)
        start_score = _score(inst, sched)
        by_id = {t.task_id: t for t in inst.tasks}
        dev_by_id = {d.dev_id: d for d in inst.developers}
        work = _light_copy(sched)
        queues = {j: [t for t in q if t not in set(pool)]
                  for j, q in queue_order(inst, work, t_now).items()}
        # Tugas yang dikeluarkan ditaruh sementara di akhir antrean asalnya, supaya
        # setiap uji-coba selalu memuat semua tugas.
        for tid in sorted(pool, key=lambda t: (by_id[t].due_date, t)):
            loads = developer_load(inst, work, t_now)
            best = None
            home = work.entries[tid].dev_id
            others = sorted((j for j in by_id[tid].eligible_devs if j != home),
                            key=lambda j: (loads.get(j, 0.0), j))
            for j in [home] + others[: self.max_devs]:
                if j != home and not is_pair_feasible(
                        tid, j, inst, work, t_now, loads=loads,
                        enforce_capacity=self.enforce_capacity, by_id=by_id,
                        dev_by_id=dev_by_id):
                    continue
                for order in (tuple(_edd_insert(queues.get(j, []), tid, by_id)),):
                    trial = _light_copy(work)
                    set_queue(trial, j, list(order), t_now)
                    recompute_timeline(inst, trial, t_now)
                    score = (_score(inst, trial), j != before.entries[tid].dev_id)
                    if best is None or score < best[0]:
                        best = (score, j, list(order), trial)
            if best is None:
                continue
            _, j, order, trial = best
            queues[j] = order
            work = trial
        if _score(inst, work) > start_score + EPS:
            # Penyisipan rakus bisa saja lebih buruk dari jadwal awal; jangan dipakai.
            return RepairResult(sched, [], invalid_attempt=True)
        sched.entries = work.entries
        recompute_timeline(inst, sched, t_now)
        return RepairResult(sched, _moves(before, sched), invalid_attempt=False)


# ------------------------------------------------------------------ susunan
def ladder_operators(enforce_capacity: bool = True) -> dict[int, RepairOperator]:
    """Empat tingkat perbaikan sebagai peta aksi -> operator.

    Examples
    --------
    >>> from dfjsp.repair.ladder import ladder_operators
    >>> [op.name for _, op in sorted(ladder_operators().items())]
    ['RightShift', 'Resequence', 'Reassign', 'PartialReschedule']
    """
    return {RIGHT_SHIFT: RightShift(), RESEQUENCE: Resequence(),
            LADDER_REASSIGN: LadderReassign(enforce_capacity=enforce_capacity),
            PARTIAL_RESCHEDULE: PartialReschedule(enforce_capacity=enforce_capacity)}


def ladder_mask(inst, sched, t_now) -> np.ndarray:
    """Mask (4,) untuk tangga: tingkat boleh bila punya perubahan untuk dicoba.

    Tingkat 2 dan 3 bisa tetap gagal memperbaiki (lalu jatuh ke tingkat 0); mask hanya
    menutup yang pasti tidak punya calon.
    """
    ops = ladder_operators()
    return np.array([1] + [int(ops[a].inspect(inst, sched, t_now).feasible)
                           for a in (RESEQUENCE, LADDER_REASSIGN, PARTIAL_RESCHEDULE)],
                    dtype=np.int64)
