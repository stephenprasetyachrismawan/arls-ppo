"""RepairEnv: simulator perbaikan jadwal yang digerakkan oleh gangguan."""

from __future__ import annotations

import copy
import time
import warnings
from dataclasses import dataclass

import numpy as np

from ..core.datatypes import FJSPInstance
from ..core.seeding import derive_episode_seed
from ..repair.actions import MAINTAIN
from ..repair.feasibility import inspect_all_operators
from ..repair.masks import high_level_mask
from ..repair.registry import default_operators
from ..reward.software_aware import SoftwareAwareReward
from ..utils import ParamsMixin
from .disruption_effects import apply_disruption
from .disruptions import SCENARIOS, DisruptionGenerator, Event, resolve_scenario
from .disruptions_tawos import TawosDisruptionModel
from .initial_schedule import build_initial_schedule
from .repair_step import apply_repair
from .schedule import developer_load
from .schedule_check import ringkas_kelebihan_beban
from .stability import stability_step
from .state import compute_refs, extract_state, makespan, total_tardiness
from .step_metrics import build_step_metrics
from .task_status import advance_statuses


@dataclass(frozen=True)
class DecisionContext:
    """Potret sebelum aksi: apa yang benar-benar dilihat pemilih strategi.

    Atribut: ``t_now`` (waktu keputusan), ``event_type`` (gangguan yang sedang
    ditangani), ``state`` (9 angka), ``policy_mask`` (mask yang dilihat pemilih),
    ``validity_mask`` (mask tingkat atas yang sebenarnya, untuk menilai strategi tidak
    valid), dan ``operator_feasibility`` (``{aksi: OperatorFeasibility}``).

    Examples
    --------
    >>> from dfjsp.datasets import load_brandimarte
    >>> from dfjsp.env import RepairEnv
    >>> env = RepairEnv()
    >>> state, info = env.reset(load_brandimarte(["MK01"])["MK01"], intensity=0.15, seed=0)
    >>> ctx = env.decision_context()
    >>> ctx.event_type
    'developer_unavailable'
    >>> ctx.policy_mask.tolist(), sorted(ctx.operator_feasibility)
    ([1, 1, 1, 1], [0, 1, 2, 3])
    """

    t_now: float
    event_type: str | None
    state: np.ndarray
    policy_mask: np.ndarray
    validity_mask: np.ndarray
    operator_feasibility: dict   # {aksi: OperatorFeasibility}


class RepairEnv(ParamsMixin):
    """Lingkungan perbaikan jadwal: satu langkah = satu gangguan = satu keputusan.

    Urutan episode: ``reset`` membangun jadwal awal S0, membangkitkan seluruh gangguan
    episode, lalu menerapkan gangguan pertama. Tiap ``step(action)`` menerapkan strategi
    ``action``, mengukur akibatnya, memajukan waktu ke gangguan berikutnya, dan
    menerapkan gangguan itu sebelum keadaan baru diberikan.

    Parameters
    ----------
    i_max : float, default=5.0
        Penyebut normalisasi angka I (jumlah gangguan terbaru).
    operators : dict, optional
        Peta aksi -> operator. Bawaan: ``default_operators(enforce_capacity)``.
    reward : callable, optional
        ``reward(step_metrics, action) -> float``. Bawaan: ``SoftwareAwareReward()``.
    mask_fn : callable, optional
        ``mask_fn(inst, sched, t_now) -> array (4,)``, mask yang dilihat pemilih.
        Bawaan: ``high_level_mask``.
    eval_mask_fn : callable, optional
        Mask yang menjadi "kebenaran" untuk menilai strategi tidak valid, apa pun
        yang dilihat pemilih. Bawaan: ``high_level_mask``.
    base_seed : int, default=0
        Seed dasar untuk menurunkan seed gangguan.
    enforce_capacity : bool, default=True
        Dipakai bila ``operators`` tidak diberikan.
    due_mode : {"twk", "s0"}, default="twk"
        ``"twk"``: tenggat bawaan instans (TWK dari upaya pekerjaan sendiri). ``"s0"``:
        tenggat tiap tugas = selesai di jadwal awal S0 + ``due_slack`` x upayanya, jadi S0
        tepat waktu dan keterlambatan hanya berasal dari gangguan dan perbaikan
        (rancangan v2 bagian 5.2).
    due_slack : float, default=0.5
        Kelonggaran tenggat pada ``due_mode="s0"``, kelipatan upaya tugas.
    disruption_model : {"legacy", "tawos"}, default="legacy"
        ``"legacy"``: proses Poisson lama (``DisruptionGenerator``). ``"tawos"``: model
        yang dikalibrasi dari data Jira nyata (``TawosDisruptionModel``), termasuk
        gangguan bersamaan yang digabung menjadi satu keputusan.

    Attributes
    ----------
    inst, sched, refs :
        Salinan instans, jadwal berjalan, dan referensi normalisasi episode ini.
    t_now : float
        Waktu keputusan sekarang.
    done : bool
        True bila semua gangguan sudah ditangani.

    Examples
    --------
    >>> from dfjsp.datasets import load_brandimarte
    >>> env = RepairEnv()
    >>> state, info = env.reset(load_brandimarte(["MK01"])["MK01"], intensity=0.15, seed=0)
    >>> state.shape
    (9,)
    >>> state, reward, done, info = env.step(0)        # 0 = Maintain
    >>> done
    False
    """

    def __init__(self, i_max: float = 5.0, operators: dict | None = None,
                 reward=None, mask_fn=None, eval_mask_fn=None, base_seed: int = 0,
                 enforce_capacity: bool = True, due_mode: str = "twk",
                 due_slack: float = 0.5, disruption_model: str = "legacy"):
        self.i_max = i_max
        self.operators = operators
        self.reward = reward
        self.mask_fn = mask_fn
        self.eval_mask_fn = eval_mask_fn
        self.base_seed = base_seed
        self.enforce_capacity = enforce_capacity
        self.due_mode = due_mode
        self.due_slack = due_slack
        self.disruption_model = disruption_model

        self.inst: FJSPInstance | None = None
        self.sched = None
        self.refs = None
        self.t_now = 0.0
        self.done = True
        self._events: list[Event] = []
        self._event_idx = 0

    # ------------------------------------------------------------------- reset
    def reset(self, instance: FJSPInstance, *, intensity: float,
              scenario: str = "mixed", k_tight: float = 1.5, seed: int):
        """Mulai episode baru; kembalikan ``(state, info)``.

        Parameters
        ----------
        instance : FJSPInstance
            Instans proyek. Disalin, jadi aslinya tidak berubah.
        intensity : float
            Laju gangguan per satuan waktu (0,05 / 0,15 / 0,30).
        scenario : str, default="mixed"
            Jenis gangguan. ``"mixed"`` memakai lima jenis; skenario satu jenis hanya
            untuk tes.
        k_tight : float, default=1.5
            Keketatan tenggat (ikut menentukan seed gangguan).
        seed : int
            Seed episode. Urutan gangguan hanya bergantung pada instans, intensitas,
            ``k_tight``, ``seed``, dan skenario, jadi sama untuk semua metode.
        """
        self.operators_ = self.operators if self.operators is not None \
            else default_operators(self.enforce_capacity)
        self.reward_ = self.reward if self.reward is not None else SoftwareAwareReward()
        self.mask_fn_ = self.mask_fn if self.mask_fn is not None else high_level_mask
        self.eval_mask_fn_ = self.eval_mask_fn if self.eval_mask_fn is not None \
            else high_level_mask

        self.intensity = float(intensity)
        self.scenario = scenario
        full_scenario = resolve_scenario(scenario)
        self.k_tight = float(k_tight)
        self.seed = int(seed)
        self.inst = copy.deepcopy(instance)
        self.sched = build_initial_schedule(self.inst)
        if self.due_mode == "s0":
            # S0 tetap disusun menurut tenggat TWK; sesudah itu tenggat diganti dengan
            # komitmen jadwal awal. Jadwal tidak dibangun ulang, jadi urutannya sama.
            for task in self.inst.tasks:
                entry = self.sched.entries[task.task_id]
                task.due_date = entry.end + self.due_slack * (entry.end - entry.start)
        elif self.due_mode != "twk":
            raise ValueError(f"due_mode {self.due_mode!r} tidak dikenal (twk atau s0)")
        self.refs = compute_refs(self.inst, self.sched, self.i_max,
                                 tard_floor_effort=(self.due_mode == "s0"))

        # Seed gangguan tidak bergantung pada metode, supaya semua metode menghadapi
        # urutan gangguan yang sama pada satu kombinasi (uji berpasangan).
        dis_seed = derive_episode_seed(
            self.base_seed, "__shared__", self.inst.name, self.intensity,
            self.k_tight, self.seed, full_scenario, stream="disruption",
        )
        if self.disruption_model == "tawos":
            generator = TawosDisruptionModel(self.intensity, np.random.default_rng(dis_seed))
        elif self.disruption_model == "legacy":
            generator = DisruptionGenerator(
                self.intensity, SCENARIOS[full_scenario], np.random.default_rng(dis_seed))
        else:
            raise ValueError(f"disruption_model {self.disruption_model!r} tidak dikenal")
        # Sasaran gangguan dipilih menurut jadwal awal pada waktu kejadian.
        self._events = generator.sample_timeline(
            makespan(self.inst, self.sched), self.inst, self.sched)
        self._events_no_target = generator.n_dropped_no_target
        self._event_idx = 0
        self.nervousness_cum = 0.0
        self.context_switch_cum = 0.0
        self._event_errors = 0

        self.t_now = self._events[0].time if self._events else self.refs.horizon
        advance_statuses(self.inst, self.sched, self.t_now)
        self.prev_tardiness = total_tardiness(self.inst, self.sched)
        self.prev_makespan = makespan(self.inst, self.sched)
        self.done = len(self._events) == 0

        # Keputusan pertama sudah memuat dampak gangguan pertama. Acuan selisih tetap
        # jadwal sebelum gangguan, jadi dampak gangguan dan respons operator tercatat
        # bersama, tepat satu kali.
        if not self.done:
            self._apply_event(self._events[0])

        info = {"reset": True, "n_events": len(self._events), "t_now": self.t_now,
                "scenario": scenario, "done": self.done,
                "n_events_tanpa_sasaran": self._events_no_target}
        return self._state(), info

    # -------------------------------------------------------------------- step
    def step(self, action: int):
        """Satu transisi; kembalikan ``(state, reward, done, info)``."""
        if self.done:
            raise RuntimeError("step() dipanggil setelah episode selesai.")
        event = self._events[self._event_idx]

        # Nilai validitas strategi terhadap mask tingkat atas yang sebenarnya, pada
        # keadaan keputusan (sebelum perbaikan). Maintain selalu valid.
        eval_mask = self.eval_mask_fn_(self.inst, self.sched, self.t_now)
        invalid_strategy = bool(int(action) != MAINTAIN and eval_mask[int(action)] == 0)

        load_before = developer_load(self.inst, self.sched, self.t_now)
        started = time.perf_counter()
        invalid_repair, feasible_pairs, moved = self._apply_repair(action)
        repair_time_ms = (time.perf_counter() - started) * 1000.0
        load_after = developer_load(self.inst, self.sched, self.t_now)
        overload = ringkas_kelebihan_beban(self.inst, load_before, load_after)

        stability = stability_step(self.inst, self.sched, self.t_now, self.refs.horizon)
        self.nervousness_cum += stability.nervousness
        self.context_switch_cum += stability.context_switch

        # Majukan waktu tanpa menerapkan gangguan berikutnya: reward langkah ini tidak
        # boleh memuat dampak gangguan berikutnya.
        self._event_idx += 1
        self.done = self._event_idx >= len(self._events)
        next_t = self._events[self._event_idx].time if not self.done else self.refs.horizon
        self.t_now = max(self.t_now, next_t)
        advance_statuses(self.inst, self.sched, self.t_now)

        step_metrics = build_step_metrics(
            self.inst, self.sched, self.t_now, refs=self.refs, step_index=self._event_idx,
            action=action, event_type=event.type, prev_tardiness=self.prev_tardiness,
            prev_makespan=self.prev_makespan, stability=stability,
            invalid_strategy=invalid_strategy, invalid_repair=invalid_repair,
            feasible_pairs=feasible_pairs, n_moved=len(moved), overload=overload)
        reward = float(self.reward_(step_metrics, action))
        self.prev_tardiness = step_metrics["tardiness_total"]
        self.prev_makespan = step_metrics["makespan"]

        # Gangguan berikutnya diterapkan SESUDAH hasil langkah ini tercatat.
        if not self.done:
            self._apply_event(self._events[self._event_idx])

        info = dict(step_metrics)
        info.update({
            "high_mask": self.action_mask().tolist(),
            "moved_tasks": moved,
            "reward": reward,
            "done": self.done,
            "nervousness_cum": self.nervousness_cum,
            "context_switch_cum": self.context_switch_cum,
            "repair_time_ms": repair_time_ms,
            "event_errors": self._event_errors,
        })
        return self._state(), reward, self.done, info

    # --------------------------------------------------------------- pengamatan
    def action_mask(self) -> np.ndarray:
        """Mask strategi (4,) yang dilihat pemilih; dipanggil SEBELUM ``step``."""
        return self.mask_fn_(self.inst, self.sched, self.t_now)

    def decision_context(self) -> DecisionContext:
        """Potret sebelum aksi: keadaan, kedua mask, dan kelayakan tiap operator."""
        event_type = (self._events[self._event_idx].type
                      if self._event_idx < len(self._events) else None)
        return DecisionContext(
            t_now=float(self.t_now),
            event_type=event_type,
            state=self._state().copy(),
            policy_mask=self.action_mask().copy(),
            validity_mask=np.asarray(
                self.eval_mask_fn_(self.inst, self.sched, self.t_now)).copy(),
            operator_feasibility=inspect_all_operators(
                self.operators_, self.inst, self.sched, self.t_now),
        )

    @property
    def events(self) -> list[Event]:
        """Seluruh gangguan episode ini, terurut menurut waktu."""
        return self._events

    @property
    def n_events(self) -> int:
        return len(self._events)

    @property
    def n_events_no_target(self) -> int:
        """Jumlah gangguan yang dibuang karena tidak ada sasaran yang sah."""
        return self._events_no_target

    # ----------------------------------------------------------------- internal
    def _state(self) -> np.ndarray:
        return extract_state(
            self.inst, self.sched, self.t_now, refs=self.refs,
            nervousness_cum=self.nervousness_cum,
            context_switch_cum=self.context_switch_cum,
            n_recent_events=self._recent_event_count(),
        )

    def _recent_event_count(self) -> int:
        """Gangguan yang SUDAH diterapkan pada selang (t_now - i_window, t_now]."""
        applied = min(self._event_idx + 1, len(self._events))
        lower = self.t_now - self.refs.i_window
        return sum(1 for e in self._events[:applied] if lower < e.time <= self.t_now)

    def _apply_repair(self, action: int):
        """Jalankan operator; kembalikan ``(invalid_attempt, feasible_pairs, moved_tasks)``."""
        result = apply_repair(self.operators_, action, self.inst, self.sched, self.t_now)
        self.sched = result.new_sched
        return result.invalid_attempt, result.feasible_pairs, result.moved_tasks

    def _apply_event(self, event: Event) -> None:
        """Terapkan gangguan. Galat isi gangguan dihitung dan diperingatkan, tidak dilempar.

        Sesudahnya ``sched.affected`` berisi himpunan tugas terdampak A: tugas yang
        langsung dikenai gangguan (tugas baru, direvisi, rework, atau milik pengembang
        yang tidak tersedia) ditambah tugas yang tergeser lebih lambat dan karena itu
        menjadi terlambat.
        """
        before = {tid: e.end for tid, e in self.sched.entries.items()}
        try:
            apply_disruption(self.inst, self.sched, event, self.t_now, self.refs.horizon)
        except Exception as exc:  # noqa: BLE001 (gangguan tidak boleh menghentikan episode)
            self._event_errors += 1
            warnings.warn(
                f"{self.inst.name}/{self.scenario}: payload event '{event.type}' gagal "
                f"pada t={event.time:.3f}: {exc!r}",
                RuntimeWarning,
                stacklevel=2,
            )
        self.sched.affected = affected_tasks(self.inst, self.sched, event, before,
                                             self.t_now)


def _direct_targets(event: Event, before: dict, sched, t_now: float) -> set[int]:
    """Tugas yang langsung dikenai gangguan."""
    if event.type == "batch":
        out: set[int] = set()
        for sub in event.payload.get("events", []):
            out |= _direct_targets(sub, before, sched, t_now)
        return out
    p = event.payload
    if event.type == "urgent_task":
        return {tid for tid in sched.entries if tid not in before}
    if event.type in ("effort_revision", "rework_review_fail", "priority_change"):
        return {p["task_id"]} if p.get("task_id") is not None else set()
    if event.type == "developer_unavailable":
        return {tid for tid, e in sched.entries.items()
                if e.dev_id == p.get("dev_id") and e.end > t_now}
    return set()


def affected_tasks(inst, sched, event: Event, before: dict, t_now: float) -> set[int]:
    """Himpunan tugas terdampak A sesudah ``event`` diterapkan (lihat ``_apply_event``)."""
    by_id = {t.task_id: t for t in inst.tasks}
    direct = _direct_targets(event, before, sched, t_now)
    late_shift = {
        tid for tid, e in sched.entries.items()
        if tid in before and e.end > before[tid] + 1e-9 and e.end > by_id[tid].due_date + 1e-9
    }
    return {tid for tid in direct | late_shift if tid in sched.entries}
