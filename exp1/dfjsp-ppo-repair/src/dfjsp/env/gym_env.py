"""Adapter Gymnasium untuk RepairEnv (khusus pelatihan SB3-contrib MaskablePPO)."""

from __future__ import annotations

import copy

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..core.seeding import draw_training_episode_seed
from ..repair.actions import N_ACTIONS
from .repair_env import RepairEnv
from .state import STATE_DIM


class GymRepairEnv(gym.Env):
    """Lingkungan Gymnasium ber-mask di atas ``RepairEnv``.

    Tiap ``reset`` mengambil satu episode secara acak (berseed) dari kolam: instans,
    skenario, intensitas, keketatan tenggat, dan tingkat kapasitas.

    Parameters
    ----------
    instances : dict
        Peta nama -> instans siap simulasi.
    instance_names : list of str
        Nama instans yang boleh diambil.
    scenarios : list of str, default=("mixed",)
        Skenario gangguan yang boleh diambil.
    k_tight : float, default=1.5
        Keketatan tenggat bila ``k_tights`` tidak diberikan.
    intensities : list of float, optional
        Intensitas yang diacak per episode. Bawaan: ``[0.15]``.
    k_tights : list of float, optional
        Keketatan tenggat yang diacak per episode. Bawaan: ``[k_tight]``.
    reward : callable, optional
        Lihat ``RepairEnv``.
    mask_fn : callable, optional
        Lihat ``RepairEnv``.
    operators : dict, optional
        Lihat ``RepairEnv``. Bawaan: ``default_operators(enforce_capacity)``.
    enforce_capacity : bool, default=True
        Dipakai bila ``operators`` tidak diberikan.
    capacity_curriculum : list of float, optional
        Pengali kapasitas yang diacak per episode (kurikulum). Bawaan: kapasitas dasar.
    base_seed : int, default=0
    worker_seed : int, optional
        Bila diberikan, menggantikan ``base_seed`` (satu seed per pekerja VecEnv).
    i_max : float, default=5.0
    max_reset_tries : int, default=32
        Batas pengambilan ulang bila episode terambil tanpa gangguan sama sekali.

    Examples
    --------
    >>> from dfjsp.datasets import load_brandimarte
    >>> from dfjsp.env import GymRepairEnv
    >>> env = GymRepairEnv(load_brandimarte(["MK01"]), ["MK01"])
    >>> obs, info = env.reset(seed=0)
    >>> obs.shape, env.action_masks().tolist()
    ((9,), [True, True, False, True])
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        instances: dict,
        instance_names: list[str],
        scenarios: list[str] = ("mixed",),
        k_tight: float = 1.5,
        *,
        intensities: list[float] | None = None,
        k_tights: list[float] | None = None,
        reward=None,
        mask_fn=None,
        operators: dict | None = None,
        enforce_capacity: bool = True,
        capacity_curriculum: list[float] | None = None,
        base_seed: int = 0,
        worker_seed: int | None = None,
        i_max: float = 5.0,
        max_reset_tries: int = 32,
    ):
        super().__init__()
        self.observation_space = spaces.Box(0.0, 1.0, (STATE_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)

        self._names = list(instance_names)
        self._scenarios = list(scenarios)
        self._intensities = [float(x) for x in (intensities or [0.15])]
        self._k_tights = [float(x) for x in (k_tights or [k_tight])]
        self.k_tight = float(k_tight)
        self.enforce_capacity = bool(enforce_capacity)
        self.worker_seed = None if worker_seed is None else int(worker_seed)
        self.base_seed = int(base_seed if worker_seed is None else worker_seed)
        self.max_reset_tries = int(max_reset_tries)

        # Satu RepairEnv dipakai ulang di semua episode; ``reset`` memulai semuanya dari awal.
        self._env = RepairEnv(
            i_max=i_max, operators=operators, reward=reward, mask_fn=mask_fn,
            base_seed=self.base_seed, enforce_capacity=self.enforce_capacity)

        # Kolam instans per tingkat kapasitas. Kunci None = kapasitas dasar.
        self._caps = list(capacity_curriculum) if capacity_curriculum else [None]
        self._pool = self._build_pool(instances, self._caps)
        # Kapasitas yang ditetapkan dari luar (penjadwal kurikulum lewat VecEnv.env_method).
        # None berarti kapasitas diacak dari ``_caps``.
        self._capacity_override: float | None = None

        self._rng = np.random.default_rng(self.base_seed)

    @staticmethod
    def _build_pool(instances: dict, caps: list) -> dict:
        """Peta {kapasitas -> {nama -> instans}}; kapasitas diatur dari upaya tugas."""
        if caps == [None]:
            return {None: instances}
        from ..core.domain import set_capacity

        out: dict = {}
        for cap in caps:
            by_name = {}
            for name, inst in instances.items():
                scaled = copy.deepcopy(inst)
                set_capacity(scaled, float(cap))
                by_name[name] = scaled
            out[cap] = by_name
        return out

    # ----------------------------------------------------------------- API gym
    def sample_episode(self):
        """Ambil satu episode dari kolam; kembalikan ``(env_dalam, state, info)``."""
        # Ambil ulang sampai dapat episode dengan minimal satu gangguan: episode yang
        # langsung selesai tidak bisa dijalankan MaskablePPO.
        for attempt in range(1, self.max_reset_tries + 1):
            cap = (
                self._capacity_override
                if self._capacity_override is not None
                else self._caps[int(self._rng.integers(len(self._caps)))]
            )
            name = self._names[int(self._rng.integers(len(self._names)))]
            scenario = self._scenarios[int(self._rng.integers(len(self._scenarios)))]
            intensity = self._intensities[int(self._rng.integers(len(self._intensities)))]
            k_tight = self._k_tights[int(self._rng.integers(len(self._k_tights)))]
            episode_seed = draw_training_episode_seed(self._rng)
            state, info = self._env.reset(
                self._pool[cap][name], intensity=intensity, scenario=scenario,
                k_tight=k_tight, seed=episode_seed)
            self.last_episode_condition = {
                "instance": name, "scenario": scenario, "intensity": intensity,
                "k_tight": k_tight, "episode_seed": episode_seed, "capacity": cap,
                "tries": attempt,
            }
            if not self._env.done:
                break
        return self._env, state, info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        _env, state, _info = self.sample_episode()
        return np.asarray(state, dtype=np.float32), {}

    def step(self, action):
        state, reward, done, info = self._env.step(int(action))
        return np.asarray(state, dtype=np.float32), float(reward), bool(done), False, info

    def action_masks(self) -> np.ndarray:
        """Mask strategi bertipe bool untuk MaskablePPO."""
        return self._env.action_mask().astype(bool)

    @property
    def current_capacity_mult(self) -> float | None:
        """Pengali kapasitas aktif: nilai yang ditetapkan, atau tingkat pertama kolam."""
        return self._capacity_override if self._capacity_override is not None else self._caps[0]

    def set_capacity_mult(self, multiplier: float | None) -> None:
        """Tetapkan kapasitas pekerja ini secara tetap (None = kembali diacak)."""
        if multiplier is not None and multiplier not in self._pool:
            raise ValueError(
                f"capacity multiplier belum disiapkan: {multiplier!r} "
                f"(tersedia: {sorted(k for k in self._pool if k is not None)})"
            )
        self._capacity_override = multiplier
