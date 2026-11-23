"""Tabel ringkasan instans."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.due_date import e_nominal

COLUMNS = ["instance", "n_jobs", "n_tasks", "n_developers", "flexibility", "total_effort"]


def describe_instances(instances) -> pd.DataFrame:
    """Ringkas ukuran tiap instans dalam satu tabel.

    Parameters
    ----------
    instances : dict
        Nama instans ke ``ProjectInstance``.

    Returns
    -------
    pandas.DataFrame
        Satu baris per instans, urutan sama dengan ``instances``. Kolom:
        ``instance``, ``n_jobs`` (jumlah job), ``n_tasks`` (jumlah tugas),
        ``n_developers`` (jumlah pengembang), ``flexibility`` (rata-rata jumlah
        pengembang yang boleh mengerjakan satu tugas), ``total_effort`` (jumlah
        upaya nominal semua tugas, jam).

    Examples
    --------
    >>> from dfjsp.datasets import describe_instances, load_brandimarte
    >>> tabel = describe_instances(load_brandimarte(["MK01"]))
    >>> list(tabel.columns)[:3]
    ['instance', 'n_jobs', 'n_tasks']
    """
    rows = []
    for name, inst in instances.items():
        tasks = inst.tasks
        flexibility = float(np.mean([len(t.eligible_devs) for t in tasks])) if tasks else 0.0
        rows.append({
            "instance": name,
            "n_jobs": len({t.job_id for t in tasks}),
            "n_tasks": len(tasks),
            "n_developers": len(inst.developers),
            "flexibility": round(flexibility, 2),
            "total_effort": float(sum(e_nominal(t) for t in tasks)),
        })
    return pd.DataFrame(rows, columns=COLUMNS)
