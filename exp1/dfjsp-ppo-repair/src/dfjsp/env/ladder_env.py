"""Lingkungan agen tangga: uji-coba tiap tingkat di keadaan, reward relatif, harga lambda.

Rancangan v2 bagian 4.5 dan 4.6. Sesudah setiap gangguan, tingkat 1 sampai 3 dijalankan
pada salinan jadwal (uji-coba). Hasilnya dipakai dua kali:

* sebagai masukan agen: berapa keterlambatan dipulihkan dan berapa perubahan dibuat tiap
  tingkat, dibanding *right-shift* (tingkat 0);
* sebagai aksi: bila agen memilih tingkat a, jadwal hasil uji-coba tingkat a yang dipasang,
  jadi tidak ada hitungan dua kali.

Reward satu keputusan::

    r = (Tard_RS - Tard_a) / E  -  lam * Perubahan_a

``E`` = rata-rata upaya tugas di S0, jadi keterlambatan diukur dalam "banyak tugas
rata-rata". ``Perubahan_a`` = tugas yang pindah pengembang + tugas yang pindah posisi
antrean, dibanding jadwal *right-shift*. Dampak gangguan sendiri tidak masuk reward, dan
perubahan masa lalu tidak ditagih lagi (memperbaiki D5 di diagnosis). ``lam`` adalah harga
satu perubahan dalam satuan keterlambatan, diacak per episode dan ikut menjadi masukan
agen (kebijakan bersyarat preferensi), sehingga satu kebijakan bisa dievaluasi pada banyak
nilai ``lam`` dan menghasilkan Pareto front.
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..core.seeding import draw_training_episode_seed
from ..repair.base import RepairOperator, RepairResult
from ..repair.ladder import (
    LADDER_REASSIGN,
    PARTIAL_RESCHEDULE,
    RESEQUENCE,
    RIGHT_SHIFT,
    _changes,
    _light_copy,
    ladder_operators,
)
from .repair_env import RepairEnv
from .schedule import partition, recompute_timeline
from .state import STATE_DIM, total_tardiness

LEVELS = (RESEQUENCE, LADDER_REASSIGN, PARTIAL_RESCHEDULE)
#: Tambahan keadaan: 3 angka per tingkat (dipulihkan, perubahan, sah), 2 angka keparahan,
#: 1 angka lambda.
LADDER_STATE_DIM = STATE_DIM + 3 * len(LEVELS) + 2 + 1
# Rentang lambda dipilih dari besaran data: satu keputusan memulihkan belasan sampai
# puluhan satuan upaya tugas dengan sekitar dua perubahan, jadi harga perubahan baru terasa
# di sekitar 1 sampai 30.
LAM_RANGE = (0.1, 30.0)


class _CachedOperator(RepairOperator):
    """Operator yang memasang jadwal hasil uji-coba yang sudah dihitung."""

    def __init__(self, env: "LadderRepairEnv", action: int):
        self.env = env
        self.action = action
        self.name = f"cached_{action}"

    def apply(self, inst, sched, t_now):
        probe = self.env.probes.get(self.action)
        if probe is None or probe["entries"] is None:
            return RepairResult(sched, [], invalid_attempt=True)
        sched.entries = dict(probe["entries"])
        recompute_timeline(inst, sched, t_now)
        return RepairResult(sched, probe["moves"], invalid_attempt=False)


class LadderRepairEnv(RepairEnv):
    """``RepairEnv`` dengan data baru, tangga, uji-coba di keadaan, dan reward relatif.

    Parameters
    ----------
    lam : float, optional
        Harga satu perubahan. None berarti diacak log-seragam dari ``lam_range`` tiap
        ``reset`` (pelatihan); angka tetap dipakai saat evaluasi.
    lam_range : tuple of float, default=(0.1, 30.0)
    reward_scale : float, default=10.0
        Pembagi reward supaya besarnya sekitar satu.
    with_objective : bool, default=False
        Versi 2: tambahkan tiga angka, nilai satu langkah tiap tingkat untuk lambda saat
        itu (dipulihkan - lambda x perubahan). Pada versi 1 lambda hanya satu angka di
        antara 21 dan agen tidak belajar memakainya; versi 2 memberi skor serakah
        langsung, seperti masukan tujuan pada UVFA, sehingga agen tinggal belajar kapan
        menyimpang darinya.
    kwargs :
        Diteruskan ke ``RepairEnv``. Bawaan di sini: ``due_mode="s0"``,
        ``disruption_model="tawos"``.

    Examples
    --------
    >>> from dfjsp.datasets import load_brandimarte
    >>> from dfjsp.env.ladder_env import LadderRepairEnv, LADDER_STATE_DIM
    >>> env = LadderRepairEnv(lam=1.0)
    >>> state, info = env.reset(load_brandimarte(["MK01"])["MK01"], intensity=0.15, seed=900)
    >>> state.shape == (LADDER_STATE_DIM,)
    True
    """

    def __init__(self, lam: float | None = None, lam_range: tuple = LAM_RANGE,
                 reward_scale: float = 10.0, with_objective: bool = False, **kwargs):
        kwargs.setdefault("due_mode", "s0")
        kwargs.setdefault("disruption_model", "tawos")
        super().__init__(**kwargs)
        self.lam = lam
        self.lam_range = lam_range
        self.reward_scale = reward_scale
        self.with_objective = with_objective
        self.state_dim = LADDER_STATE_DIM + (len(LEVELS) if with_objective else 0)
        self.probes: dict = {}
        self._lam_rng = np.random.default_rng(0)

    # ---------------------------------------------------------------- episode
    def reset(self, instance, *, intensity, scenario="mixed", k_tight=1.5, seed):
        self._ladder = ladder_operators(self.enforce_capacity)
        self.operators = {RIGHT_SHIFT: self._ladder[RIGHT_SHIFT],
                          **{a: _CachedOperator(self, a) for a in LEVELS}}
        self.mask_fn = self._mask
        self.eval_mask_fn = self._mask
        self.reward = self._reward
        self._lam_rng = np.random.default_rng([int(seed), 7])
        if self.lam is None:
            lo, hi = self.lam_range
            self.lam_ = float(math.exp(self._lam_rng.uniform(math.log(lo), math.log(hi))))
        else:
            self.lam_ = float(self.lam)
        self.probes = {}
        self._event_added = 0.0
        self.mean_effort = None
        state, info = super().reset(instance, intensity=intensity, scenario=scenario,
                                    k_tight=k_tight, seed=seed)
        self._ensure_mean_effort()
        info["lam"] = self.lam_
        return self._state(), info

    def _apply_event(self, event):
        tard_before = total_tardiness(self.inst, self.sched)
        super()._apply_event(event)
        self._event_added = total_tardiness(self.inst, self.sched) - tard_before
        self._probe()

    def _ensure_mean_effort(self) -> None:
        """Rata-rata upaya tugas di S0 (``baseline_ref``), satuan keterlambatan reward."""
        if self.mean_effort is None:
            ref = self.sched.baseline_ref or self.sched.entries
            efforts = [e.end - e.start for e in ref.values()]
            self.mean_effort = max(1e-6, float(np.mean(efforts)) if efforts else 1.0)

    def _probe(self) -> None:
        """Jalankan tingkat 1 sampai 3 pada salinan; simpan hasil untuk keadaan dan aksi."""
        self._ensure_mean_effort()
        base_tard = total_tardiness(self.inst, self.sched)
        self.probes = {RIGHT_SHIFT: {"tard": base_tard, "changes": 0, "entries": None,
                                     "moves": []}}
        for a in LEVELS:
            op = self._ladder[a]
            trial = _light_copy(self.sched)
            res = op.apply(self.inst, trial, self.t_now)
            if res.invalid_attempt:
                self.probes[a] = None
                continue
            moved_dev, moved_pos = _changes(self.sched, trial, self.t_now, self.inst)
            self.probes[a] = {"tard": total_tardiness(self.inst, trial),
                              "changes": moved_dev + moved_pos,
                              "entries": dict(trial.entries), "moves": res.moved_tasks}
        # Uji-coba mengubah waktu mulai/selesai yang tercatat di objek tugas; kembalikan.
        recompute_timeline(self.inst, self.sched, self.t_now)

    def _mask(self, inst, sched, t_now) -> np.ndarray:
        return np.array([1] + [int(self.probes.get(a) is not None) for a in LEVELS],
                        dtype=np.int64)

    def objective(self, action: int) -> float:
        """Nilai satu langkah sebelum dibagi skala: dipulihkan - lam * perubahan."""
        base = self.probes[RIGHT_SHIFT]["tard"]
        p = self.probes.get(action)
        if p is None:
            return 0.0
        return (base - p["tard"]) / self.mean_effort - self.lam_ * p["changes"]

    def step(self, action: int):
        self._last_objective = self.objective(int(action))
        p = self.probes.get(int(action)) or self.probes[RIGHT_SHIFT]
        self._last_changes = p["changes"]
        self._last_recovered = (self.probes[RIGHT_SHIFT]["tard"] - p["tard"]) / self.mean_effort
        state, reward, done, info = super().step(action)
        info.update({"lam": self.lam_, "objective": self._last_objective,
                     "changes": self._last_changes, "recovered": self._last_recovered})
        return state, reward, done, info

    def _reward(self, step_metrics, action) -> float:
        return float(self._last_objective / self.reward_scale)

    # ---------------------------------------------------------------- keadaan
    def _state(self) -> np.ndarray:
        base = super()._state()
        if not self.probes:
            return np.concatenate([base, np.zeros(self.state_dim - STATE_DIM,
                                                  dtype=np.float32)])
        extra = []
        rs = self.probes[RIGHT_SHIFT]["tard"]
        for a in LEVELS:
            p = self.probes.get(a)
            if p is None:
                extra += [0.0, 0.0, 0.0]
            else:
                rec = (rs - p["tard"]) / self.mean_effort
                extra += [math.tanh(rec / 5.0), math.tanh(p["changes"] / 10.0), 1.0]
        _, in_prog, not_started = partition(self.inst, self.sched, self.t_now)
        unfinished = max(1, len(in_prog) + len(not_started))
        extra.append(min(1.0, len(self.sched.affected) / unfinished))
        extra.append(math.tanh(self._event_added / self.mean_effort / 5.0))
        lo, hi = self.lam_range
        extra.append((math.log(self.lam_) - math.log(lo)) / (math.log(hi) - math.log(lo)))
        if self.with_objective:
            for a in LEVELS:
                extra.append(math.tanh(self.objective(a) / 10.0)
                             if self.probes.get(a) is not None else -1.0)
        return np.concatenate([base, np.asarray(extra, dtype=np.float32)])


def greedy_action(env: LadderRepairEnv) -> int:
    """Aturan serakah: tingkat dengan nilai satu langkah terbaik (seri: tingkat terendah)."""
    best, best_val = RIGHT_SHIFT, 0.0
    for a in LEVELS:
        if env.probes.get(a) is not None:
            v = env.objective(a)
            if v > best_val + 1e-9:
                best, best_val = a, v
    return best


class GymLadderEnv(gym.Env):
    """Pembungkus Gymnasium untuk ``LadderRepairEnv`` (MaskablePPO).

    Tiap ``reset`` mengambil instans, kapasitas, intensitas, dan seed episode secara acak
    berseed. Lambda diacak oleh ``LadderRepairEnv``.
    """

    metadata = {"render_modes": []}

    def __init__(self, instances: dict, intensities=(0.05, 0.15, 0.30),
                 capacities=(1.0, 1.5, 2.0), worker_seed: int = 0, lam=None,
                 with_objective: bool = False):
        super().__init__()
        from ..core.domain import set_capacity
        import copy
        self.env = LadderRepairEnv(lam=lam, base_seed=worker_seed,
                                   with_objective=with_objective)
        self.observation_space = spaces.Box(-1.0, 1.0, (self.env.state_dim,),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self._pool = []
        for cap in capacities:
            for name, inst in instances.items():
                scaled = copy.deepcopy(inst)
                set_capacity(scaled, float(cap))
                self._pool.append((name, cap, scaled))
        self._intensities = list(intensities)
        self._rng = np.random.default_rng(worker_seed)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        for _ in range(32):
            name, cap, inst = self._pool[int(self._rng.integers(len(self._pool)))]
            intensity = self._intensities[int(self._rng.integers(len(self._intensities)))]
            ep_seed = draw_training_episode_seed(self._rng)
            state, info = self.env.reset(inst, intensity=intensity, seed=ep_seed)
            if not self.env.done:
                break
        return np.clip(state, -1.0, 1.0).astype(np.float32), info

    def step(self, action):
        state, reward, done, info = self.env.step(int(action))
        return np.clip(state, -1.0, 1.0).astype(np.float32), float(reward), bool(done), False, info

    def action_masks(self) -> np.ndarray:
        return self.env.action_mask().astype(bool)
