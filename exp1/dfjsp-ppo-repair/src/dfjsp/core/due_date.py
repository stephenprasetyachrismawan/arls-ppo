"""Tenggat TWK dan prioritas kuantil (Tahap D, Subbab 4.4 rancangan)."""

from __future__ import annotations

import numpy as np

from .datatypes import FJSPInstance, Task


def e_nominal(task: Task) -> float:
    """Effort nominal operasi: rata-rata waktu proses pada mesin yang eligible."""
    vals = list(task.eligible_devs.values())
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def _tasks_by_job(inst: FJSPInstance) -> dict[int, list[Task]]:
    """Kelompokkan task per job, terurut berdasar op_index (untuk precedence)."""
    groups: dict[int, list[Task]] = {}
    for t in inst.tasks:
        groups.setdefault(t.job_id, []).append(t)
    for job_id in groups:
        groups[job_id].sort(key=lambda t: t.op_index)
    return groups


def assign_due_dates_twk(
    inst: FJSPInstance,
    k_tight: float,
    rng: np.random.Generator | None = None,
    release: dict[int, float] | None = None,
) -> None:
    """Isi Task.due_date untuk semua task via TWK (Subbab 4.4).

    Parameters
    ----------
    inst : FJSPInstance
        Instans (dimodifikasi in-place).
    k_tight : float
        Faktor keketatan {1.0, 1.5, 2.0}; makin besar makin longgar.
    rng : numpy.random.Generator, optional
        Tidak dipakai (tenggat deterministik); parameter ada untuk keseragaman.
    release : dict, optional
        ``{job_id: waktu rilis}``; bawaan 0 untuk semua job.

    Notes
    -----
    Sifat yang selalu berlaku:

    - d_i monoton naik mengikuti op_index dalam satu job;
    - d_job(j) >= W_j untuk k_tight >= 1 (tenggat tidak lebih ketat dari upaya);
    - tenggat membesar seiring k_tight.
    """
    groups = _tasks_by_job(inst)
    for job_id, ops in groups.items():
        release_j = 0.0 if release is None else float(release.get(job_id, 0.0))
        nominals = [e_nominal(t) for t in ops]
        w_j = sum(nominals)
        cum = 0.0
        for t, e in zip(ops, nominals):
            cum += e
            frac = (cum / w_j) if w_j > 0 else 1.0
            t.due_date = release_j + k_tight * w_j * frac


def assign_priorities_quantile(
    inst: FJSPInstance,
    rng: np.random.Generator,
    invert: bool = False,
) -> None:
    """Isi Task.priority dari kuantil (ECDF) efort nominal, dijepit ke [0,1].

    Parameters
    ----------
    inst : FJSPInstance
        Instans (dimodifikasi in-place).
    rng : numpy.random.Generator
        Generator berseed untuk tie-break.
    invert : bool, default=False
        Balik arah pemetaan prioritas.
    """
    tasks = inst.tasks
    n = len(tasks)
    if n == 0:
        return
    nominals = np.array([e_nominal(t) for t in tasks], dtype=float)
    # Kunci tie-break deterministik dari rng; urutkan (efort, tie_key) menaik.
    tie_keys = rng.random(n)
    order = sorted(range(n), key=lambda i: (nominals[i], tie_keys[i]))
    denom = max(1, n - 1)
    for rank, idx in enumerate(order):
        p = rank / denom  # rank terkecil (efort terendah) -> 0, tertinggi -> 1
        if invert:
            p = 1.0 - p
        tasks[idx].priority = float(min(1.0, max(0.0, p)))
