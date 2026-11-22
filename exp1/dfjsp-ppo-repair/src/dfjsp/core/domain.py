"""Transformasi domain mesin -> human-centric (Tahap E, Subbab 4.5; Tabel 3.1)."""

from __future__ import annotations

import hashlib
import json
from statistics import median

import numpy as np

from .datatypes import Developer, FJSPInstance
from .due_date import assign_due_dates_twk, assign_priorities_quantile, e_nominal


def machine_to_developer(inst: FJSPInstance) -> None:
    """Langkah 1: tiap mesin m -> Developer dengan skills = {job j : ada operasi
    job j yang eligible pada m} (Tabel 3.1, eligibility -> skill match).
    """
    skills: dict[int, set[int]] = {m: set() for m in range(inst.n_devs)}
    for t in inst.tasks:
        for dev_id in t.eligible_devs:
            skills[dev_id].add(t.job_id)
    inst.developers = [
        Developer(dev_id=m, skills=skills[m], capacity=0.0, available=True, workload=0.0)
        for m in range(inst.n_devs)
    ]


def set_capacity(inst: FJSPInstance, capacity_mult: float) -> None:
    """Langkah 2: cap_j = capacity_mult * mean_load, dengan
    mean_load = (total efort nominal) / |D| (Subbab 4.5, atribut kapasitas).
    """
    total = sum(e_nominal(t) for t in inst.tasks)
    mean_load = total / inst.n_devs if inst.n_devs > 0 else 0.0
    cap = capacity_mult * mean_load
    for dev in inst.developers:
        dev.capacity = float(cap)


def build_precedence(inst: FJSPInstance, p_cross_dep: float, rng: np.random.Generator) -> int:
    """Langkah 3: dependensi intra-job (rantai op_index) + lintas-job opsional."""
    by_job: dict[int, list] = {}
    for t in inst.tasks:
        by_job.setdefault(t.job_id, []).append(t)
    for job_id in by_job:
        by_job[job_id].sort(key=lambda t: t.op_index)

    # Intra-job: operasi k bergantung pada operasi k-1 (rantai).
    for ops in by_job.values():
        for prev, cur in zip(ops, ops[1:]):
            if prev.task_id not in cur.deps:
                cur.deps.append(prev.task_id)

    # Lintas-job: edge dari task terakhir job a ke task pertama job b (a < b).
    n_cross = 0
    job_ids = sorted(by_job.keys())
    for bi in range(1, len(job_ids)):
        b = job_ids[bi]
        first_b = by_job[b][0]
        for ai in range(bi):
            if rng.random() < p_cross_dep:
                a = job_ids[ai]
                last_a = by_job[a][-1]
                if last_a.task_id not in first_b.deps:
                    first_b.deps.append(last_a.task_id)
                    n_cross += 1
    return n_cross


def assert_dag_acyclic(inst: FJSPInstance) -> None:
    """Langkah 6a: pastikan graf dependensi task asiklik (Kahn topological sort).

    Raises
    ------
    ValueError
        Bila terdeteksi siklus (gerbang dag_acyclic gagal).
    """
    indeg: dict[int, int] = {t.task_id: 0 for t in inst.tasks}
    adj: dict[int, list[int]] = {t.task_id: [] for t in inst.tasks}
    for t in inst.tasks:
        for d in t.deps:
            adj[d].append(t.task_id)
            indeg[t.task_id] += 1
    queue = [tid for tid, dg in indeg.items() if dg == 0]
    visited = 0
    while queue:
        u = queue.pop()
        visited += 1
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    if visited != len(inst.tasks):
        raise ValueError(f"{inst.name}: graf dependensi memuat siklus (DAG gagal).")


def is_dag_acyclic(inst: FJSPInstance) -> bool:
    """Versi boolean dari :func:`assert_dag_acyclic` (untuk kolom manifest)."""
    try:
        assert_dag_acyclic(inst)
        return True
    except ValueError:
        return False


def domain_hash(capacity_mult: float, p_cross_dep: float, invert: bool, seed: int) -> str:
    """Ringkasan stabil parameter domain efektif (Tahap F: nama cache & manifest)."""
    payload = json.dumps(
        {
            "capacity_mult": capacity_mult,
            "p_cross_dep": p_cross_dep,
            "invert": bool(invert),
            "seed": int(seed),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def finalize_meta(
    inst: FJSPInstance,
    *,
    capacity_mult: float,
    p_cross_dep: float,
    invert: bool,
    seed: int,
    k_tight: float,
    n_cross_dep_edges: int,
) -> None:
    """Langkah 6b: isi meta + horizon perencanaan untuk normalisasi state."""
    skill_counts = [len(d.skills) for d in inst.developers]
    med = median(skill_counts) if skill_counts else 0
    n_generalist = sum(1 for c in skill_counts if c > med)
    n_specialist = sum(1 for c in skill_counts if c < med)
    nominals = [e_nominal(t) for t in inst.tasks]
    horizon = float(sum(nominals))

    inst.horizon = horizon
    inst.meta.update(
        {
            "n_tasks": len(inst.tasks),
            "avg_eligible": (sum(len(t.eligible_devs) for t in inst.tasks) / len(inst.tasks))
            if inst.tasks
            else 0.0,
            "capacity_mult": float(capacity_mult),
            "p_cross_dep": float(p_cross_dep),
            "priority_invert": bool(invert),
            "domain_seed": int(seed),
            "k_tight": float(k_tight),
            "n_generalist": int(n_generalist),
            "n_specialist": int(n_specialist),
            "n_cross_dep_edges": int(n_cross_dep_edges),
            "mean_capacity": float(inst.developers[0].capacity) if inst.developers else 0.0,
            "horizon": horizon,
            "domain_hash": domain_hash(capacity_mult, p_cross_dep, invert, seed),
        }
    )


def to_software_project(
    inst: FJSPInstance,
    cfg,
    rng: np.random.Generator,
) -> FJSPInstance:
    """Orkestrasi transformasi domain (urutan deterministik, Subbab 4.5).

    Parameters
    ----------
    inst : FJSPInstance
        Instans netral hasil parser (dimodifikasi in-place).
    cfg : namespace
        Konfigurasi domain.
    rng : numpy.random.Generator
        Generator berseed; SATU sumber keacakan transformasi domain.

    Returns
    -------
    FJSPInstance
        Instans lengkap siap-simulasi.
    """
    capacity_mult = float(cfg.domain.capacity_mult)
    p_cross_dep = float(cfg.domain.p_cross_dep)
    seed = int(getattr(cfg.domain, "seed", 0))
    k_tight = float(cfg.k_tight)
    invert = bool(getattr(cfg.priority, "invert", False))

    machine_to_developer(inst)                       # 1
    set_capacity(inst, capacity_mult)                # 2
    n_cross = build_precedence(inst, p_cross_dep, rng)  # 3
    assign_due_dates_twk(inst, k_tight, rng)         # 4 (Tahap D)
    assign_priorities_quantile(inst, rng, invert)    # 5
    assert_dag_acyclic(inst)                         # 6a
    finalize_meta(                                   # 6b
        inst,
        capacity_mult=capacity_mult,
        p_cross_dep=p_cross_dep,
        invert=invert,
        seed=seed,
        k_tight=k_tight,
        n_cross_dep_edges=n_cross,
    )
    return inst
