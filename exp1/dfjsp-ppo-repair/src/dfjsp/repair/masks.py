"""Mask kelayakan dua tingkat.

Tingkat atas (``high_level_mask``) menyatakan strategi mana yang boleh dipilih.
Tingkat bawah (``is_pair_feasible``, ``low_level_mask``) menyatakan pasangan (tugas,
pengembang) mana yang boleh dipakai operator.
"""

from __future__ import annotations

import numpy as np

from ..core.datatypes import STATUS_COMPLETED, FJSPInstance, Schedule
from ..env.schedule import developer_load, partition
from ..env.work_calendar import working_time
from .actions import DELAY, ESCALATE, MAINTAIN, N_ACTIONS, REASSIGN

EPS = 1e-8


# --------------------------------------------------------------------- tingkat atas
def _reassign_available(inst, sched, not_started, loads) -> bool:
    """Reassign boleh bila ada tugas belum mulai pada pengembang bermasalah.

    Pengembang bermasalah = kelebihan beban atau tidak tersedia. Harus ada juga
    pengembang pengganti yang punya keahlian dan sedang tersedia.
    """
    problem = {
        d.dev_id for d in inst.developers
        if (loads.get(d.dev_id, 0.0) > d.capacity + EPS) or (not d.available)
    }
    if not problem:
        return False
    available_devs = {d.dev_id for d in inst.developers if d.available}
    by_id = {t.task_id: t for t in inst.tasks}
    for i in not_started:
        e = sched.entries.get(i)
        if e is None or e.dev_id not in problem:
            continue
        task = by_id.get(i)
        if task is None:
            continue
        for j in task.eligible_devs:
            if j != e.dev_id and j in available_devs:
                return True
    return False


def high_level_mask(inst: FJSPInstance, sched: Schedule, t_now: float) -> np.ndarray:
    """Mask strategi (4,) bertipe int64: 1 = boleh dipilih, 0 = dilarang.

    Maintain selalu boleh. Delay dan Escalate butuh minimal satu tugas yang belum
    mulai. Reassign butuh pengembang bermasalah dan pengganti yang cocok.

    Examples
    --------
    >>> from dfjsp.datasets import load_brandimarte
    >>> from dfjsp.env import RepairEnv
    >>> env = RepairEnv()
    >>> state, info = env.reset(load_brandimarte(["MK01"])["MK01"], intensity=0.05, seed=0)
    >>> from dfjsp.repair import high_level_mask
    >>> high_level_mask(env.inst, env.sched, env.t_now).tolist()    # Reassign dilarang
    [1, 1, 0, 1]
    """
    _, _, not_started = partition(inst, sched, t_now)
    loads = developer_load(inst, sched, t_now)

    mask = np.zeros(N_ACTIONS, dtype=np.int64)
    mask[MAINTAIN] = 1
    mask[DELAY] = 1 if len(not_started) > 0 else 0
    mask[ESCALATE] = 1 if len(not_started) > 0 else 0
    mask[REASSIGN] = 1 if _reassign_available(inst, sched, not_started, loads) else 0
    return mask


def allow_all_mask(inst: FJSPInstance, sched: Schedule, t_now: float) -> np.ndarray:
    """Mask yang mengizinkan semua strategi (ablasi tanpa mask tingkat atas).

    Examples
    --------
    >>> from dfjsp.datasets import load_brandimarte
    >>> from dfjsp.env import RepairEnv
    >>> env = RepairEnv()
    >>> state, info = env.reset(load_brandimarte(["MK01"])["MK01"], intensity=0.05, seed=0)
    >>> from dfjsp.repair import allow_all_mask
    >>> allow_all_mask(env.inst, env.sched, env.t_now).tolist()
    [1, 1, 1, 1]
    """
    return np.ones(N_ACTIONS, dtype=np.int64)


# -------------------------------------------------------------------- tingkat bawah
def _deps_scheduled(task, sched: Schedule) -> bool:
    """Setiap pendahulu punya entri jadwal, sehingga waktu selesainya diketahui."""
    return all(sched.entries.get(dep) is not None for dep in task.deps)


def _self_load(i: int, j: int, sched: Schedule, t_now: float) -> float:
    """Beban tugas i yang sudah masuk beban dev j karena i memang ditugaskan ke j."""
    entry = sched.entries.get(i)
    if entry is None or entry.dev_id != j or entry.end <= t_now:
        return 0.0
    pauses = sched.pauses.get(j, [])
    dur = working_time(entry.start, entry.end, pauses)
    if dur <= 0:
        return 0.0
    if entry.start <= t_now:                                # sedang berjalan
        selesai = min(1.0, working_time(entry.start, t_now, pauses) / dur)
        return max(0.0, dur * (1.0 - selesai))
    return dur


def is_pair_feasible(
    i: int,
    j: int,
    inst: FJSPInstance,
    sched: Schedule,
    t_now: float,
    *,
    loads: dict[int, float] | None = None,
    enforce_capacity: bool = True,
    exclude_self_load: bool = True,
    by_id: dict | None = None,
    dev_by_id: dict | None = None,
) -> bool:
    """True bila pasangan (tugas i, pengembang j) memenuhi semua kendala.

    Kendala: tugas belum selesai, j punya keahlian untuk i, j tersedia, semua
    pendahulu i terjadwal, dan (bila ``enforce_capacity``) beban j ditambah upaya i
    tidak melewati kapasitas j.

    Parameters
    ----------
    loads : dict, optional
        Beban tiap pengembang yang sudah dihitung, supaya tidak dihitung ulang.
    enforce_capacity : bool, default=True
        Bila False, batas kapasitas tidak diperiksa. Kendala lain tetap berlaku.
    exclude_self_load : bool, default=True
        Bila True, beban tugas i sendiri dikeluarkan dari beban j ketika i memang sedang
        ditugaskan ke j. Tanpa ini, mempertahankan penugasan yang sudah ada dianggap
        menambah beban baru. Pemanggil yang sudah mengurangi beban itu sendiri, seperti
        Escalate, harus memberi False agar tidak dikurangi dua kali.
    by_id, dev_by_id : dict, optional
        Peta tugas dan peta pengembang yang sudah dibangun, supaya tidak dibangun ulang
        pada setiap pemanggilan.

    Examples
    --------
    >>> from dfjsp.datasets import load_brandimarte
    >>> from dfjsp.env import RepairEnv
    >>> env = RepairEnv()
    >>> state, info = env.reset(load_brandimarte(["MK01"])["MK01"], intensity=0.05, seed=0)
    >>> from dfjsp.repair import is_pair_feasible
    >>> i, s, t = env.inst, env.sched, env.t_now
    >>> is_pair_feasible(2, 2, i, s, t), is_pair_feasible(2, 0, i, s, t)  # dev 0 tanpa keahlian
    (True, False)
    """
    if by_id is None:
        by_id = {t.task_id: t for t in inst.tasks}
    task = by_id.get(i)
    if task is None:
        return False
    if task.status == STATUS_COMPLETED:
        return False
    if j not in task.eligible_devs:
        return False
    dev = dev_by_id.get(j) if dev_by_id is not None else \
        next((d for d in inst.developers if d.dev_id == j), None)
    if dev is None or not dev.available:
        return False
    if not _deps_scheduled(task, sched):
        return False
    if enforce_capacity:
        if loads is None:
            loads = developer_load(inst, sched, t_now)
        w_j = loads.get(j, 0.0)
        if exclude_self_load:
            w_j -= _self_load(i, j, sched, t_now)
        if w_j + task.eligible_devs[j] > dev.capacity + EPS:
            return False
    return True


def low_level_mask(inst: FJSPInstance, sched: Schedule, t_now: float, *,
                   enforce_capacity: bool = True) -> dict[int, list[int]]:
    """Untuk tiap tugas yang belum mulai, daftar pengembang yang boleh menerimanya.

    Examples
    --------
    >>> from dfjsp.datasets import load_brandimarte
    >>> from dfjsp.env import RepairEnv
    >>> env = RepairEnv()
    >>> state, info = env.reset(load_brandimarte(["MK01"])["MK01"], intensity=0.05, seed=0)
    >>> from dfjsp.repair import low_level_mask
    >>> mask = low_level_mask(env.inst, env.sched, env.t_now)
    >>> len(mask), mask[2], mask[3]
    (41, [2, 5], [0, 1, 5])
    """
    _, _, not_started = partition(inst, sched, t_now)
    loads = developer_load(inst, sched, t_now)
    by_id = {t.task_id: t for t in inst.tasks}
    dev_by_id = {d.dev_id: d for d in inst.developers}
    dev_ids = sorted(dev_by_id)
    return {
        i: [
            j for j in dev_ids
            if is_pair_feasible(i, j, inst, sched, t_now, loads=loads,
                                enforce_capacity=enforce_capacity,
                                by_id=by_id, dev_by_id=dev_by_id)
        ]
        for i in sorted(not_started)
    }


def count_feasible_pairs(inst: FJSPInstance, sched: Schedule, t_now: float) -> int:
    """Jumlah semua pasangan (tugas, pengembang) layak, dengan batas kapasitas.

    Examples
    --------
    >>> from dfjsp.datasets import load_brandimarte
    >>> from dfjsp.env import RepairEnv
    >>> env = RepairEnv()
    >>> state, info = env.reset(load_brandimarte(["MK01"])["MK01"], intensity=0.05, seed=0)
    >>> from dfjsp.repair import count_feasible_pairs
    >>> count_feasible_pairs(env.inst, env.sched, env.t_now)
    85
    """
    return sum(len(v) for v in low_level_mask(inst, sched, t_now).values())
