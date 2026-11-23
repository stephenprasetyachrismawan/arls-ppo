"""Bagian pendukung PPO pasangan: mask strategi, operator PairAssign, dan pencatat kurva."""

from __future__ import annotations

import numpy as np

from ..core.datatypes import ScheduleEntry
from ..env.schedule import recompute_timeline
from ..repair.base import RepairOperator, RepairResult
from ..repair.masks import count_feasible_pairs, is_pair_feasible
from ._capacity import CapacityExposureLedger


def assign_mask(inst, sched, t_now) -> np.ndarray:
    """Mask strategi untuk PPO pasangan: hanya Maintain dan Reassign (pasangan langsung)."""
    return np.array([1, 0, 1, 0], dtype=np.int64)


class PairAssign(RepairOperator):
    """Operator yang menerapkan pasangan (tugas, pengembang) pilihan ``owner``.

    Parameters
    ----------
    owner : objek dengan atribut ``pending``
        Pemilih (atau kebijakan) yang menyimpan pasangan terpilih di ``pending``,
        berupa ``(id_tugas, id_pengembang)`` atau ``None``.
    """

    name = "PairAssign"

    def __init__(self, owner):
        self.owner = owner

    def apply(self, inst, sched, t_now: float) -> RepairResult:
        feasible = count_feasible_pairs(inst, sched, t_now)
        pair = self.owner.pending
        if pair is None:
            return RepairResult(sched, [], invalid_attempt=True, feasible_pairs=feasible)
        i, j = pair
        if not is_pair_feasible(i, j, inst, sched, t_now):
            return RepairResult(sched, [], invalid_attempt=True, feasible_pairs=feasible)
        entry = sched.entries[i]
        old = entry.dev_id
        if j == old:
            # Mempertahankan penugasan lama: tidak ada tugas yang pindah, jadi tidak
            # dicatat sebagai perpindahan.
            return RepairResult(sched, [], invalid_attempt=False, feasible_pairs=feasible)
        sched.entries[i] = ScheduleEntry(i, j, entry.start, entry.end)
        recompute_timeline(inst, sched, t_now)
        return RepairResult(sched, [{"task_id": i, "from_dev": old, "to_dev": j}],
                            invalid_attempt=False, feasible_pairs=feasible)


class Tracker:
    """Mencatat kurva latih dan paparan kapasitas selama pelatihan."""

    def __init__(self, ledger: CapacityExposureLedger):
        self.ledger = ledger
        self.rows: list[dict] = []
        self.level = None
        self.timesteps = 0
        self._run_return = 0.0
        self._run_length = 0
        self._reset_window()

    def _reset_window(self) -> None:
        self._step_rewards, self._levels = [], []
        self._ep_returns, self._ep_lengths = [], []

    def on_step(self, reward: float, ended: bool) -> None:
        self.timesteps += 1
        self._run_return += reward
        self._run_length += 1
        self._step_rewards.append(reward)
        if self.level is not None:
            self._levels.append(self.level)
            self.ledger.record(self.level, 1)
        if ended:
            self._ep_returns.append(self._run_return)
            self._ep_lengths.append(self._run_length)
            self._run_return, self._run_length = 0.0, 0

    def on_update(self, stats: dict) -> None:
        self.rows.append({
            "update": len(self.rows) + 1,
            "timesteps": self.timesteps,
            "ep_rew_mean": float(np.mean(self._ep_returns)) if self._ep_returns
            else float("nan"),
            "ep_len_mean": float(np.mean(self._ep_lengths)) if self._ep_lengths
            else float("nan"),
            "step_rew_mean": float(np.mean(self._step_rewards)),
            "entropy": float(stats["entropy_mean"]),
            "capacity_mean": float(np.mean(self._levels)) if self._levels else float("nan"),
            "capacity_assignments": " ".join(f"{m:g}" for m in sorted(set(self._levels))),
        })
        self._reset_window()
