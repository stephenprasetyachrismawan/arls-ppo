"""Waktu keputusan: berapa milidetik pemilih dan operator memakan waktu per keputusan."""

from __future__ import annotations

import sys
from contextlib import contextmanager

import pandas as pd

from ..utils import check_random_state
from ._episode import decide_and_step, open_episode

_LATENCY_COLUMNS = ("selector_ms", "operator_ms", "total_ms")


@contextmanager
def _single_thread_torch():
    """Satu thread torch selama pengukuran, lalu kembalikan jumlah semula."""
    try:
        import torch
    except ImportError:
        yield
        return
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _timed_decisions(method, name, instance, *, n_total, rng, intensity, capacity,
                     k_tight, scenario):
    """Jalankan episode berurutan sampai ``n_total`` keputusan terukur terkumpul.

    Yield ``(selector_ns, operator_ns)`` per keputusan.
    """
    produced, empty_in_a_row = 0, 0
    while produced < n_total:
        env, state = open_episode(
            method, name, instance, intensity=intensity, capacity=capacity,
            seed=int(rng.integers(0, 2**31 - 1)), k_tight=k_tight, scenario=scenario)
        if env.done:
            empty_in_a_row += 1
            if empty_in_a_row >= 100:
                raise RuntimeError(
                    f"100 episode berturut-turut pada {instance.name} tanpa gangguan; "
                    "naikkan intensity")
            continue
        empty_in_a_row = 0
        while not env.done and produced < n_total:
            state, _, _, _, predict_ns = decide_and_step(method, env, state)
            produced += 1
            yield predict_ns, env.last_repair_ns


def measure_latency(method, instances: dict, *, n_decisions: int = 200, warmup: int = 20,
                    random_state=0, intensity: float = 0.30, capacity: float = 1.5,
                    k_tight: float = 1.5, scenario: str = "mixed",
                    method_name: str | None = None) -> pd.DataFrame:
    """Ukur waktu keputusan ``method`` pada tiap instans.

    Cara ukurnya meniru Eksperimen 7 lama: satu proses, jam ``time.perf_counter_ns``,
    dan satu thread torch (``torch.set_num_threads(1)``, dikembalikan sesudahnya). Tiap
    instans dijalankan lewat episode berurutan dengan gangguan campuran sampai
    ``warmup + n_decisions`` keputusan terkumpul. ``warmup`` keputusan pertama dibuang
    (panggilan pertama torch dan NumPy lebih lambat karena inisialisasi).

    Yang termasuk dan tidak termasuk dalam tiap waktu:

    * ``selector_ms``: hanya satu pemanggilan ``method.predict(state, mask)``. Itu
      termasuk lintas maju jaringan dan, untuk PPO pasangan, pembuatan kandidat
      pasangan. TIDAK termasuk penghitungan keadaan 9 angka, penghitungan mask, dan
      penyalinan mask, yang dikerjakan lingkungan.
    * ``operator_ms``: hanya pemanggilan operator perbaikan (``env._apply_repair``).
      TIDAK termasuk pemeriksa kelayakan jadwal, perhitungan stabilitas, reward,
      penerapan gangguan berikutnya, dan pencatatan log.
    * ``total_ms``: ``selector_ms + operator_ms``, yaitu waktu keputusan total.

    Memori tidak diukur di sini.

    Parameters
    ----------
    method : objek pemilih
        Protokol di ``dfjsp.evaluation.run_episode``.
    instances : dict
        Nama instans ke ``ProjectInstance`` (semua 10 MK untuk tesis).
    n_decisions : int, default=200
        Keputusan terukur per instans, sesudah warmup.
    warmup : int, default=20
        Keputusan awal per instans yang dibuang.
    random_state : int, default=0
        Seed yang memilih seed episode. Menentukan keputusan apa yang diukur, bukan
        angka waktunya.
    intensity, capacity, k_tight, scenario :
        Kondisi episode pengukuran. Bawaan: intensitas tertinggi (paling banyak
        keputusan per episode) dan kapasitas tengah.
    method_name : str, optional
        Nama metode untuk seed kebijakan. Bawaan: nama kelas.

    Returns
    -------
    pandas.DataFrame
        Satu baris per keputusan terukur, kolom: ``method``, ``instance``,
        ``n_tasks``, ``n_devs``, ``decision`` (0 sampai ``n_decisions - 1``),
        ``selector_ms``, ``operator_ms``, ``total_ms``. Ringkasan p50 dan p95 dengan
        ``summarize_latency``.

    Examples
    --------
    >>> from dfjsp.datasets import load_brandimarte
    >>> from dfjsp.evaluation import measure_latency, summarize_latency
    >>> from dfjsp.reschedulers import RuleRescheduler
    >>> inst = load_brandimarte(["MK01"])
    >>> frame = measure_latency(RuleRescheduler("edd").fit(), inst, n_decisions=5, warmup=1)
    >>> len(frame)
    5
    >>> summarize_latency(frame)["n_decisions"].tolist()
    [5]
    """
    if n_decisions < 1:
        raise ValueError(f"Parameter n_decisions={n_decisions!r} harus >= 1")
    if warmup < 0:
        raise ValueError(f"Parameter warmup={warmup!r} tidak boleh negatif")
    if not instances:
        raise ValueError("Parameter instances tidak boleh kosong")
    name = method_name or type(method).__name__
    rng = check_random_state(random_state)

    rows = []
    with _single_thread_torch():
        for instance_name, instance in instances.items():
            decisions = _timed_decisions(
                method, name, instance, n_total=warmup + n_decisions, rng=rng,
                intensity=intensity, capacity=capacity, k_tight=k_tight,
                scenario=scenario)
            for index, (selector_ns, operator_ns) in enumerate(decisions):
                if index < warmup:
                    continue
                selector_ms, operator_ms = selector_ns / 1e6, operator_ns / 1e6
                rows.append({
                    "method": name, "instance": instance_name,
                    "n_tasks": len(instance.tasks), "n_devs": len(instance.developers),
                    "decision": index - warmup, "selector_ms": selector_ms,
                    "operator_ms": operator_ms, "total_ms": selector_ms + operator_ms})
    return pd.DataFrame(rows)


def summarize_latency(frame: pd.DataFrame) -> pd.DataFrame:
    """Ringkas hasil ``measure_latency``: p50 dan p95 per instans.

    Returns
    -------
    pandas.DataFrame
        Satu baris per instans: ``instance``, ``n_decisions``, lalu
        ``selector_ms_p50``, ``selector_ms_p95``, ``operator_ms_p50``,
        ``operator_ms_p95``, ``total_ms_p50``, ``total_ms_p95``.

    Examples
    --------
    Tabel buatan dengan tiga keputusan pada satu instans (hasil nyata dari
    ``measure_latency`` punya kolom yang sama):

    >>> import pandas as pd
    >>> from dfjsp.evaluation import summarize_latency
    >>> frame = pd.DataFrame({"instance": ["MK01"] * 3, "selector_ms": [1.0, 2.0, 3.0],
    ...                       "operator_ms": [4.0, 4.0, 4.0], "total_ms": [5.0, 6.0, 7.0]})
    >>> ringkas = summarize_latency(frame)
    >>> ringkas["instance"].tolist(), ringkas["n_decisions"].tolist()
    (['MK01'], [3])
    >>> float(ringkas.loc[0, "selector_ms_p50"]), round(float(ringkas.loc[0, "total_ms_p95"]), 2)
    (2.0, 6.9)
    """
    rows = []
    for instance, group in frame.groupby("instance", sort=False):
        row = {"instance": instance, "n_decisions": len(group)}
        for column in _LATENCY_COLUMNS:
            row[f"{column}_p50"] = float(group[column].quantile(0.50))
            row[f"{column}_p95"] = float(group[column].quantile(0.95))
        rows.append(row)
    return pd.DataFrame(rows)
