"""Run one episode with exactly the code that produced the thesis numbers.

:func:`run_episode` calls ``cek_episode.run`` from ``revisi/hasil/gerbang_batch`` (the same
function ``buka_uji.py`` used when the test data was opened) with the locked models of recipe
E15 E9c. Test data stays closed: the three Brandimarte test instances, the 35 benchmark test
instances, and the test episode seeds are refused here, and ``penjaga_uji`` (imported by
``cek_episode``) blocks the instance files for the whole process as well.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

from . import LIBRARY
from .models import RUN_DIR, verify_models

#: The eight methods reported in the thesis (``docs/KUNCI_UJI.md`` section 3), code -> name
#: used in the manuscript.
METHODS: dict[str, str] = {
    "ppo_bc_ppo_g": "ARLS-PPO",
    "ppo_bc_ppo": "ARLS-PPO tanpa penjaga",
    "ppo_bc_g": "ARLS-BC",
    "aor": "AOR",
    "seq": "Sequential Lookahead",
    "fix3_g": "Fixed-L3 + penjaga",
    "acak_g": "Random + penjaga",
    "il_g": "Boosting-IL + penjaga",
}

#: Methods where the user is the selector (:mod:`arls_ppo.manual`); not thesis methods.
MANUAL_METHODS: dict[str, str] = {"manual": "Manual", "manual_g": "Manual + penjaga"}

#: The five disruption types of the TAWOS model, code -> name used in the manuscript.
DISRUPTION_TYPES: dict[str, str] = {
    "urgent_task": "Pekerjaan tidak terencana",
    "effort_revision": "Revisi estimasi",
    "rework_review_fail": "Rework (review gagal)",
    "priority_change": "Perubahan prioritas",
    "developer_unavailable": "Pengembang tidak tersedia",
}

#: The 38 numbers the selectors read, in the order of ``cek_b3.features``: the 9 numbers of
#: ``dfjsp.env.state.extract_state`` (C, Tard, O, U, B, N, H, Q, I) and 29 numbers about the
#: current disruption and its affected set A_j.
FEATURE_NAMES: tuple[str, ...] = (
    "C: makespan / horizon", "Tard: keterlambatan total ternormalisasi",
    "O: porsi pengembang kelebihan beban", "U: utilisasi pengembang",
    "B: ketidakseimbangan beban", "N: nervousness kumulatif", "H: alih konteks kumulatif",
    "Q: porsi tugas siap dikerjakan", "I: kepadatan kejadian terbaru",
    *(f"jenis gangguan: {name}" for name in DISRUPTION_TYPES.values()),
    "log(1 + |A_j|)", "log(1 + tugas A_j yang belum mulai)",
    "tugas A_j belum mulai / tugas belum selesai", "porsi A_j yang terlambat",
    "log(1 + keterlambatan A_j / E)", "rata-rata slack A_j / E", "slack terkecil A_j / E",
    "rata-rata banyak pengembang layak", "log(1 + pengembang penerima)",
    "log(1 + pasangan pindah yang layak)",
    "mask tingkat 1", "mask tingkat 2", "mask tingkat 3",
    "rata-rata beban / kapasitas", "beban / kapasitas terbesar", "simpangan beban / kapasitas",
    "log(1 + keterlambatan total / E)",
    "k: banyak gangguan dalam kejadian", "j / k", "k - j - 1",
    "porsi A_j yang juga terkena gangguan lain", "log(1 + tugas gangguan berikutnya)",
    "log(1 + perubahan sejauh ini)", "log λ",
)

#: Episode seeds of the test data (``docs/KUNCI_UJI.md`` section 4).
TEST_SEEDS = frozenset(range(10000, 10010)) | frozenset(range(11000, 11005))
#: Brandimarte test instances (``configs/split.yaml``).
TEST_INSTANCES = frozenset({"MK03", "MK04", "MK09"})

_state: dict = {"verified": False, "model": None}


def _refuse_test_data(instance: str, seed: int) -> None:
    from dfjsp.datasets import BENCHMARK_UJI

    if instance.upper() in TEST_INSTANCES or instance in BENCHMARK_UJI:
        raise PermissionError(f"{instance} adalah data uji (test data) dan tidak dapat "
                              "disimulasikan; hasil ujinya hanya dibaca dari berkas hasil.")
    if int(seed) in TEST_SEEDS:
        raise PermissionError(f"seed {seed} adalah seed episode uji (test data); "
                              "pakai seed lain, misalnya seed validasi 5000 atau 5001.")


def _cek_episode():
    """Import ``cek_episode`` once, the same way ``buka_uji.py`` does."""
    if "cek_episode" not in sys.modules:
        saved = sys.argv
        sys.argv = ["cek_episode.py", str(LIBRARY), str(RUN_DIR / "b3_model.pkl"), os.devnull]
        try:
            import cek_episode  # noqa: F401
        finally:
            sys.argv = saved
    return sys.modules["cek_episode"]


def _prepare(ce, method: str, model_seed: int) -> None:
    """Load the model file the method needs, as ``buka_uji.rencana_lengkap`` does."""
    if not _state["verified"]:
        verify_models()
        _state["verified"] = True
    model = RUN_DIR / ("hgb7d.pkl" if method == "il_g" else "b3_model.pkl")
    if _state["model"] != model:
        ce._init(str(model))
        _state["model"] = model
    ppo_dir = str(RUN_DIR / f"e9c_s{int(model_seed)}")
    if method.startswith("ppo_") and ce.PPO_DIR != ppo_dir:
        ce.PPO_DIR = ppo_dir
        ce._PPO.clear()


def run_episode(instance, capacity: float, intensity: float, seed: int, method: str,
                lam: float, *, model_seed: int = 0, check: bool = True) -> dict:
    """One episode; returns the result record of ``cek_episode.run`` unchanged.

    Parameters
    ----------
    instance : str or arls_ppo.custom.CustomProject
        Training or validation instance, e.g. ``"MK01"`` or ``"hurink/edata/la02"``, or a
        project typed in by a user (see :mod:`arls_ppo.custom`).
    capacity, intensity : float
        Capacity multiplier (1.0, 1.5, 2.0 in the thesis) and disruption intensity
        (0.05, 0.15, 0.30).
    seed : int
        Episode seed; the test seeds are refused.
    method : str
        One of :data:`METHODS`.
    lam : float
        Price of one change (λ).
    model_seed : int, default=0
        Seed 0, 1 or 2 of the PPO models (only for the three ``ppo_`` methods).
    check : bool, default=True
        Run the schedule verifier on every decision, as during the test opening (``PERIKSA``).
    """
    ce, name = _start(instance, seed, method, model_seed)
    return _run(ce, (name, float(capacity), float(intensity), int(seed), method, float(lam)),
                check, _project(instance))


@dataclass
class EpisodeTrace:
    """One traced episode.

    Attributes
    ----------
    result : dict
        The record of ``cek_episode.run``, identical to :func:`run_episode`.
    events : list of dict
        One entry per decision: time, disruptions, affected sets, schedule snapshots
        (``before`` the disruptions, ``aor`` after right-shift, ``after`` the repair),
        tardiness / E at each point, changes, and ``subs`` with one entry per disruption
        (requested and used level, outcome, J before and after; for learned selectors the 38
        state numbers, the mask and the probability of each level; for the teacher the J of
        every level it tried).
    initial : dict
        Snapshot at the first event (tasks, developers, schedule S0).
    meta : dict
        Episode settings, method name, mean effort E and the model hashes that were checked.
    """

    result: dict
    events: list = field(default_factory=list)
    initial: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def trace_episode(instance, capacity: float, intensity: float, seed: int, method: str,
                  lam: float, *, model_seed: int = 0, check: bool = True,
                  options: bool = False, _allow_manual: bool = False) -> EpisodeTrace:
    """Like :func:`run_episode`, but also record every event for display.

    The run itself is ``cek_episode.run``; the trace only watches it (see ``_trace``). In trace
    mode ``ms`` and ``ms_keputusan`` include the time of taking snapshots, while
    ``ms_pemilih`` (selector time, indicator I-1) stays as in the plain run.

    With ``options=True`` every sub-decision also records the J of each level 0-3 at that
    point (computed after the batch, for display only).
    """
    from . import _trace

    ce, name = _start(instance, seed, method, model_seed, allow_manual=_allow_manual)
    job = (name, float(capacity), float(intensity), int(seed), method, float(lam))
    rec = _trace.Recorder(lam, options=options)
    with _trace.hooks(ce, rec):
        result = _run(ce, job, check, _project(instance))
    files = verify_models()
    meta = {"instance": name, "capacity": float(capacity), "intensity": float(intensity),
            "seed": int(seed), "method": method,
            "method_name": {**METHODS, **MANUAL_METHODS}[method],
            "lam": float(lam), "model_seed": int(model_seed) if method.startswith("ppo_")
            else None, "mean_effort": rec.mean_effort, "feature_names": list(FEATURE_NAMES),
            "model_files": [{"path": f.path, "sha256": f.actual} for f in files]}
    return EpisodeTrace(result=result, events=rec.events, initial=rec.initial or {},
                        meta=meta)


def _project(instance):
    from .custom import CustomProject

    return instance if isinstance(instance, CustomProject) else None


def _start(instance, seed, method, model_seed, allow_manual=False):
    """Check the request and load what the method needs; return (cek_episode, name)."""
    from .custom import CustomProject

    if method not in METHODS and not (allow_manual and method in MANUAL_METHODS):
        raise ValueError(f"metode {method!r} tidak dilaporkan di tesis; pilih salah satu dari "
                         f"{', '.join(METHODS)}")
    if isinstance(instance, CustomProject):
        _refuse_test_data(instance.instance_name, seed)
        name = instance.instance_name
    else:
        _refuse_test_data(instance, seed)
        name = instance
    ce = _cek_episode()
    _prepare(ce, method, model_seed)
    return ce, name


def _run(ce, job, check, project=None):
    """``cek_episode.run(job)``; a user project is served to its ``instans_acak.load_any``."""
    if project is None:
        return _run_plain(ce, job, check)
    import instans_acak

    from .custom import build_instance

    inst = build_instance(project)
    original = instans_acak.load_any
    instans_acak.load_any = lambda name: inst if name == project.instance_name else original(name)
    try:
        return _run_plain(ce, job, check)
    finally:
        instans_acak.load_any = original


def _run_plain(ce, job, check):
    old = os.environ.get("PERIKSA")
    if check:
        os.environ["PERIKSA"] = "1"
    else:
        os.environ.pop("PERIKSA", None)
    try:
        return ce.run(job)
    finally:
        if old is None:
            os.environ.pop("PERIKSA", None)
        else:
            os.environ["PERIKSA"] = old
