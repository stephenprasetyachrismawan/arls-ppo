"""``evaluate``: jalankan banyak metode pada kisi yang sama, hasilnya satu tabel."""

from __future__ import annotations

import multiprocessing as mp
import os
import sys

import pandas as pd

from ._episode import run_episode

#: Konteks yang diwarisi proses anak lewat ``fork`` (menghindari pickling pemilih).
_CONTEXT: dict = {}


def training_seed_of(method):
    """Seed pelatihan sebuah metode, atau ``None`` bila metode tidak dilatih.

    Metode dianggap dilatih bila punya atribut ``policy_`` (hasil belajar berakhiran
    ``_``). Seed pelatihannya adalah ``training_seed_`` bila ada, kalau tidak
    ``random_state``.
    """
    if not hasattr(method, "policy_"):
        return None
    seed = getattr(method, "training_seed_", getattr(method, "random_state", None))
    return None if seed is None else int(seed)


def _row_of(task: tuple) -> tuple:
    """Jalankan satu episode; kembalikan ``(baris, log atau None)``."""
    name, instance_name, point = task
    method = _CONTEXT["methods"][name]
    log = run_episode(
        method, _CONTEXT["instances"][instance_name], intensity=point.intensity,
        capacity=point.capacity, seed=point.seed, k_tight=point.k_tight,
        scenario=point.scenario, record_steps=_CONTEXT["record_steps"], method_name=name)
    row = {
        "method": name, "training_seed": training_seed_of(method),
        "instance": instance_name, "intensity": point.intensity,
        "capacity": point.capacity, "k_tight": point.k_tight,
        "scenario": point.scenario, "seed": point.seed,
        "n_events": log.final["n_events"],
        "n_events_no_target": log.final["n_events_no_target"],
        "events_digest": log.final["events_digest"],
        **log.final["summary"],
    }
    return row, (log if _CONTEXT["record_steps"] else None)


def _one_thread_in_worker() -> None:
    """Pekerja memakai satu thread torch, bila torch sudah dimuat."""
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.set_num_threads(1)


def _resolve_jobs(n_jobs: int) -> int:
    if n_jobs is None or n_jobs <= 0:
        return max(1, os.cpu_count() or 1)
    return int(n_jobs)


def evaluate(methods: dict, instances: dict, grid, *, n_jobs: int = 1,
             record_steps: bool = False) -> pd.DataFrame:
    """Evaluasi semua metode pada semua instans dan titik kisi.

    Untuk satu (instans, intensitas, kapasitas, seed episode), SEMUA metode menghadapi
    urutan gangguan yang sama persis. Kolom ``events_digest`` membuktikannya.

    Parameters
    ----------
    methods : dict
        Nama metode ke objek pemilih (protokol di ``dfjsp.evaluation.run_episode``).
    instances : dict
        Nama instans ke ``ProjectInstance``.
    grid : EvaluationGrid
        Kombinasi intensitas, kapasitas, dan seed episode.
    n_jobs : int, default=1
        Jumlah proses. 1 berarti satu proses; nol atau negatif berarti semua inti.
        Hasilnya sama persis untuk ``n_jobs`` berapa pun. Proses paralel memakai
        ``fork``, jadi hanya jalan di Linux dan macOS.
    record_steps : bool, default=False
        Bila True, ``EpisodeLog`` lengkap tiap episode disimpan di
        ``frame.attrs["episode_logs"]`` dengan urutan yang sama dengan baris.

    Returns
    -------
    pandas.DataFrame
        Satu baris per (metode, instans, intensitas, kapasitas, seed episode). Urutan
        baris: metode, instans, lalu urutan ``grid`` (intensitas, kapasitas, seed).
        Kolom identitas: ``method``, ``training_seed`` (kosong untuk metode tanpa
        pelatihan), ``instance``, ``intensity``, ``capacity``, ``k_tight``,
        ``scenario``, ``seed``, ``n_events``, ``n_events_no_target``, ``events_digest``;
        lalu semua kolom ``dfjsp.metrics.episode_summary``.

    Examples
    --------
    >>> from dfjsp.datasets import load_brandimarte
    >>> from dfjsp.evaluation import EvaluationGrid, evaluate
    >>> from dfjsp.reschedulers import RuleRescheduler
    >>> inst = load_brandimarte(["MK01"])
    >>> grid = EvaluationGrid(intensities=(0.05,), capacities=(1.5,), episode_seeds=(0,))
    >>> frame = evaluate({"EDD": RuleRescheduler("edd").fit()}, inst, grid)
    >>> frame.loc[0, ["method", "instance"]].tolist(), int(frame.loc[0, "seed"])
    (['EDD', 'MK01'], 0)
    """
    if not methods:
        raise ValueError("Parameter methods tidak boleh kosong")
    if not instances:
        raise ValueError("Parameter instances tidak boleh kosong")
    points = list(grid)
    tasks = [(name, instance_name, point)
             for name in methods for instance_name in instances for point in points]

    _CONTEXT.update(methods=methods, instances=instances, record_steps=record_steps)
    try:
        jobs = min(_resolve_jobs(n_jobs), len(tasks))
        if jobs <= 1:
            results = [_row_of(task) for task in tasks]
        else:
            if "fork" not in mp.get_all_start_methods():
                raise RuntimeError("n_jobs > 1 butuh start method 'fork' (Linux atau macOS)")
            with mp.get_context("fork").Pool(jobs, initializer=_one_thread_in_worker) as pool:
                results = list(pool.imap(_row_of, tasks, chunksize=1))
    finally:
        _CONTEXT.clear()

    frame = pd.DataFrame([row for row, _ in results])
    frame["training_seed"] = frame["training_seed"].astype("Int64")
    if record_steps:
        frame.attrs["episode_logs"] = [log for _, log in results]
    return frame
