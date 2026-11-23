"""Ringkasan satu episode: semua ukuran dalam satu kamus."""

from __future__ import annotations

import numpy as np

from ._functions import (
    _column,
    context_switches,
    infeasible_assignments,
    invalid_strategy_rate,
    nervousness,
    rescheduling_rate,
    total_tardiness,
    workload_imbalance,
)
from ._log import EpisodeLog

#: Nama kunci ringkasan yang beda dari nama kolom ``episodes.csv`` lama.
#: Kunci: nama baru di ``episode_summary``; nilai: nama lama di recorder.
OLD_COLUMN_NAME = {
    "nervousness": "nervousness_mean",
    "context_switches": "context_switch_total",
    "workload_imbalance": "workload_imbalance_mean",
    "infeasible_assignments": "accepted_hard_infeasible_count",
}


def _action_share(log: EpisodeLog, action: int) -> float:
    n = len(log.steps)
    return sum(1 for s in log.steps if s.action == action) / n if n else 0.0


def episode_summary(log: EpisodeLog) -> dict:
    """Hitung semua ukuran episode dari ``log``.

    Parameters
    ----------
    log : EpisodeLog

    Returns
    -------
    dict
        Nama ukuran ke angka Python biasa (``float`` atau ``int``). Isinya:

        * Tujuan cepat tidak ada di sini (waktu diukur terpisah oleh ``evaluation``).
        * Layak: ``invalid_strategy_rate``, ``infeasible_assignments``,
          ``invalid_repair_rate``.
        * Stabil: ``nervousness`` (rata-rata per keputusan), ``context_switches``
          (jumlah), ``rescheduling_rate``, ``start_deviation_total``.
        * Mutu jadwal: ``total_tardiness``, ``makespan``, ``n_late_tasks``,
          ``workload_imbalance``, ``utilization_mean``.
        * Pola aksi: ``action_share_maintain``, ``action_share_delay``,
          ``action_share_reassign``, ``action_share_escalate``.
        * Beban berlebih: ``overload_devs_after_mean``, ``overload_excess_after_mean``,
          ``overload_excess_added_total``, ``overload_step_rate``.

        Empat nama berbeda dari kolom ``episodes.csv`` lama; lihat
        ``OLD_COLUMN_NAME``.

    Examples
    --------
    >>> from dfjsp.metrics import EpisodeLog, StepRecord, episode_summary
    >>> log = EpisodeLog(steps=[StepRecord(0, 5.0, 0.0, 0.0),
    ...                         StepRecord(1, 3.0, 0.2, 0.0, n_moved=2)])
    >>> s = episode_summary(log)
    >>> s["total_tardiness"], s["rescheduling_rate"]
    (3.0, 0.5)
    """
    steps = log.steps
    return {
        "total_tardiness": total_tardiness(log),
        "makespan": float(steps[-1].makespan) if steps else 0.0,
        "n_late_tasks": int(steps[-1].n_late_tasks) if steps else 0,
        "nervousness": nervousness(log),
        "context_switches": context_switches(log),
        "start_deviation_total": float(_column(log, "start_deviation_step").sum()),
        "workload_imbalance": workload_imbalance(log),
        "utilization_mean": float(_column(log, "utilization").mean()),
        "invalid_strategy_rate": invalid_strategy_rate(log),
        "invalid_repair_rate": float(_column(log, "invalid_repair").mean()),
        "rescheduling_rate": rescheduling_rate(log),
        "action_share_maintain": _action_share(log, 0),
        "action_share_delay": _action_share(log, 1),
        "action_share_reassign": _action_share(log, 2),
        "action_share_escalate": _action_share(log, 3),
        "overload_devs_after_mean": float(_column(log, "overload_devs_after").mean()),
        "overload_excess_after_mean": float(_column(log, "overload_excess_after").mean()),
        "overload_excess_added_total": float(_column(log, "overload_excess_added").sum()),
        "overload_step_rate": float(
            np.mean(_column(log, "overload_devs_after") > 0.0) if steps else 0.0),
        "infeasible_assignments": infeasible_assignments(log),
    }
