"""Callback SB3 untuk PPOSelector: kapasitas per rollout dan kurva latih per update."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._capacity import CapacityExposureLedger, CapacityScheduler

CURVE_COLUMNS = ("update", "timesteps", "ep_rew_mean", "ep_len_mean", "step_rew_mean",
                 "entropy", "capacity_mean", "capacity_assignments")


def new_rollout_callback(scheduler: CapacityScheduler, ledger: CapacityExposureLedger,
                         rows: list):
    """Buat callback SB3 yang mengatur kapasitas dan mencatat satu baris per update.

    Di awal tiap rollout callback menetapkan pengali kapasitas tiap worker lewat
    ``set_capacity_mult``. Di akhir rollout ia memeriksa bahwa jumlah langkah utuh
    lalu mencatatnya di ``ledger``. Satu baris kurva (dict) ditambahkan ke ``rows``
    setelah update bobot selesai, karena entropi baru tersedia sesudahnya.

    Parameters
    ----------
    scheduler : CapacityScheduler
    ledger : CapacityExposureLedger
        Diisi langkah aktual tiap rollout.
    rows : list
        Diisi dict satu per update. ``curve_frame(rows)`` mengubahnya jadi tabel.
    """
    from stable_baselines3.common.callbacks import BaseCallback

    class RolloutCallback(BaseCallback):
        """Atur kapasitas per rollout dan catat kurva latih."""

        def __init__(self):
            super().__init__()
            self._update = 0
            self._last_total = 0
            self._open = None

        def _on_training_start(self) -> None:
            n_envs = scheduler.n_envs
            self._run_return = np.zeros(n_envs)
            self._run_length = np.zeros(n_envs)

        def _on_rollout_start(self) -> None:
            # Bobot hasil update sebelumnya sudah masuk; rinci hasilnya dicatat sekarang.
            self._close_update()
            self._update += 1
            assignments = scheduler.assignments(self._update, ledger)
            for index, multiplier in enumerate(assignments):
                self.training_env.env_method("set_capacity_mult", multiplier, indices=[index])
            self._open = {"assignments": assignments, "step_rewards": [],
                          "ep_returns": [], "ep_lengths": []}

        def _on_step(self) -> bool:
            rewards = np.asarray(self.locals["rewards"], dtype=float)
            dones = np.asarray(self.locals["dones"], dtype=bool)
            self._run_return += rewards
            self._run_length += 1
            self._open["step_rewards"].append(float(rewards.mean()))
            for index in np.flatnonzero(dones):
                self._open["ep_returns"].append(float(self._run_return[index]))
                self._open["ep_lengths"].append(float(self._run_length[index]))
                self._run_return[index] = 0.0
                self._run_length[index] = 0.0
            return True

        def _on_rollout_end(self) -> None:
            quantum = scheduler.n_envs * scheduler.n_steps
            delta = self.model.num_timesteps - self._last_total
            if delta != quantum:
                raise ValueError(
                    f"rollout tidak utuh: {delta} langkah, seharusnya {quantum}")
            self._last_total = self.model.num_timesteps
            for multiplier in self._open["assignments"]:
                ledger.record(multiplier, scheduler.n_steps)

        def _on_training_end(self) -> None:
            self._close_update()

        def _close_update(self) -> None:
            if self._open is None:
                return
            info = self._open
            self._open = None
            entropy_loss = self.model.logger.name_to_value.get("train/entropy_loss")
            assignments = info["assignments"]
            rows.append({
                "update": self._update,
                "timesteps": self._update * scheduler.n_envs * scheduler.n_steps,
                "ep_rew_mean": float(np.mean(info["ep_returns"])) if info["ep_returns"]
                else float("nan"),
                "ep_len_mean": float(np.mean(info["ep_lengths"])) if info["ep_lengths"]
                else float("nan"),
                "step_rew_mean": float(np.mean(info["step_rewards"])),
                "entropy": -float(entropy_loss) if entropy_loss is not None
                else float("nan"),
                "capacity_mean": float(np.mean(assignments)),
                "capacity_assignments": " ".join(f"{m:g}" for m in assignments),
            })

    return RolloutCallback()


def curve_frame(rows: list) -> pd.DataFrame:
    """Ubah baris callback menjadi tabel dengan kolom ``CURVE_COLUMNS``."""
    return pd.DataFrame(rows, columns=list(CURVE_COLUMNS))
