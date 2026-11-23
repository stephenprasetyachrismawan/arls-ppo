"""Ukuran satu langkah: sumber tunggal untuk reward dan log."""

from __future__ import annotations

import numpy as np

from ..core.datatypes import FJSPInstance, Schedule
from ..repair.actions import ACTION_NAMES
from .schedule import developer_load
from .stability import StabilityStep
from .state import EpisodeRefs, makespan, total_tardiness
from .work_calendar import working_time

_NO_OVERLOAD = ("overload_devs_before", "overload_devs_after", "overload_excess_before",
                "overload_excess_after", "overload_excess_added")


def n_late_tasks(inst: FJSPInstance, sched: Schedule) -> int:
    """Jumlah tugas terlambat: selesai setelah tenggatnya."""
    n = 0
    for t in inst.tasks:
        e = sched.entries.get(t.task_id)
        if e is not None and e.end > t.due_date:
            n += 1
    return n


def load_imbalance(loads: dict[int, float]) -> float:
    """Simpangan baku beban kerja pengembang pada satu waktu keputusan (satuan upaya)."""
    if not loads:
        return 0.0
    return float(np.std(np.array(list(loads.values()), dtype=float)))


def utilization(sched: Schedule, n_dev: int, cur_makespan: float) -> float:
    """Pemakaian pengembang: jumlah waktu sibuk dibagi (pengembang x makespan), paling besar 1."""
    busy = sum(working_time(e.start, e.end, sched.pauses.get(e.dev_id, []))
               for e in sched.entries.values())
    denom = max(1, n_dev) * max(1e-8, cur_makespan)
    return float(min(1.0, busy / denom))


def build_step_metrics(inst: FJSPInstance, sched: Schedule, t_now: float, *,
                       refs: EpisodeRefs, step_index: int, action: int, event_type: str,
                       prev_tardiness: float, prev_makespan: float,
                       stability: StabilityStep, invalid_strategy: bool,
                       invalid_repair: bool, feasible_pairs: int, n_moved: int,
                       overload: dict | None) -> dict:
    """Kumpulkan semua ukuran langkah ini menjadi satu kamus.

    ``overload`` berisi lima kunci ``overload_*`` dari ``ringkas_kelebihan_beban``;
    bila None semuanya nol.
    """
    w = developer_load(inst, sched, t_now)
    n_dev = max(1, len(inst.developers))
    cur_makespan = makespan(inst, sched)
    cur_tard = total_tardiness(inst, sched)
    imbalance = load_imbalance(w)
    return {
        "step": step_index,
        "t_now": t_now,
        "event_type": event_type,
        "action": int(action),
        "action_name": ACTION_NAMES[int(action)],
        "tardiness_total": cur_tard,
        "d_tardiness": cur_tard - prev_tardiness,
        "makespan": cur_makespan,
        "makespan_decrease": max(0.0, prev_makespan - cur_makespan),
        "n_late_tasks": n_late_tasks(inst, sched),
        "nervousness_step": stability.nervousness,
        "context_switch_step": stability.context_switch,
        "start_deviation_step": stability.start_deviation,
        # Dua kunci berisi nilai yang sama dan keduanya sudah menjadi kontrak: reward
        # membaca "imbalance", log langkah membaca "workload_imbalance".
        "imbalance": imbalance,
        "workload_imbalance": imbalance,
        "utilization": utilization(sched, n_dev, cur_makespan),
        "invalid_strategy": bool(invalid_strategy),
        "invalid_repair": bool(invalid_repair),
        "feasible_pairs": int(feasible_pairs),
        # Horizon episode: reward membagi suku waktu dengan ini supaya sebanding antar instans.
        "reward_scale": float(refs.horizon),
        "n_moved": int(n_moved),
        # Beban Escalate = tugas dipindah / jumlah pengembang.
        "escalate_load": float(n_moved) / max(1, len(inst.developers)),
        **(overload or dict.fromkeys(_NO_OVERLOAD, 0.0)),
    }
