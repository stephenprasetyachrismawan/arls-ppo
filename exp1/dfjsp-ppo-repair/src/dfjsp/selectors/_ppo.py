"""PPOSelector: pemilih strategi yang dilatih dengan MaskablePPO (SB3-contrib)."""

from __future__ import annotations

import io
import json
import math
import zipfile

import numpy as np
import pandas as pd

from ..env.state import STATE_DIM, STATE_SEMANTICS_VERSION
from ..repair.actions import MAINTAIN, N_ACTIONS
from ..repair.masks import allow_all_mask, high_level_mask
from ..repair.registry import default_operators
from ..reward.software_aware import SoftwareAwareReward, TardinessReward
from ..utils import BaseSelector, check_is_fitted
from ._capacity import CapacityExposureLedger, CapacityScheduler
from ._common import (
    base_seed, check_instances, is_number, is_positive_list, is_whole, need as check_param,
    reject_test_instances,
)
from ._ppo_callback import curve_frame, new_rollout_callback

MASKS = ("none", "high", "two_level")
REWARDS = ("software_aware", "tardiness")
CAPACITY_SCHEDULES = ("fixed", "balanced")
FIXED_CAPACITY = 1.2          # jadwal "fixed": satu pengali, 1,2 kali beban rata-rata

_TUPLE_PARAMS = ("capacity_levels", "net_arch", "intensities")
_FILES = ("model.zip", "selector.json", "training_curve.csv")


def effective_mask(mask_level: str, mask) -> np.ndarray:
    """Mask (bool, 4 angka) yang dipakai ``predict``.

    Pada ``mask_level="none"`` agen sengaja tidak boleh melihat mask apa pun, jadi mask
    yang diberikan pemanggil diabaikan dan semua strategi dianggap boleh.
    """
    if mask_level == "none":
        return np.ones(N_ACTIONS, dtype=bool)
    return np.asarray(mask).astype(bool)


def _new_model(env, selector, seed: int):
    """Bangun MaskablePPO dengan aktor dan kritik terpisah (net_arch x net_arch, ReLU)."""
    import torch.nn as nn
    from sb3_contrib import MaskablePPO

    arch = [int(width) for width in selector.net_arch]
    return MaskablePPO(
        "MlpPolicy", env, device="cpu", seed=seed, verbose=selector.verbose,
        policy_kwargs={"net_arch": {"pi": arch, "vf": list(arch)},
                       "activation_fn": nn.ReLU, "share_features_extractor": False},
        n_steps=int(selector.n_steps), batch_size=int(selector.batch_size),
        n_epochs=int(selector.n_epochs), gamma=float(selector.gamma),
        gae_lambda=float(selector.gae_lambda), clip_range=float(selector.clip_range),
        ent_coef=float(selector.ent_coef), vf_coef=float(selector.vf_coef),
        max_grad_norm=float(selector.max_grad_norm),
        learning_rate=float(selector.learning_rate))


def _versions() -> dict:
    import sb3_contrib
    import stable_baselines3

    import dfjsp

    return {"dfjsp": dfjsp.__version__, "stable_baselines3": stable_baselines3.__version__,
            "sb3_contrib": sb3_contrib.__version__}


class PPOSelector(BaseSelector):
    """Pemilih strategi (Maintain, Delay, Reassign, Escalate) yang dilatih dengan PPO.

    Memakai ``MaskablePPO`` dengan aktor dan kritik terpisah (``net_arch`` x ``net_arch``,
    ReLU). Lingkungan latih diambil acak dari instans latih: intensitas, kapasitas
    (menurut jadwal), dan seed episode berganti tiap episode. Anggaran langkah dibulatkan
    ke atas ke kelipatan satu rollout (``n_envs * n_steps``), sama untuk semua varian.

    Parameters
    ----------
    mask : {"none", "high", "two_level"}, default="two_level"
        ``"none"``: agen tidak melihat mask apa pun dan operator tidak memeriksa
        kapasitas. ``"high"``: agen melihat mask strategi, operator tidak memeriksa
        kapasitas. ``"two_level"``: mask strategi dan operator memeriksa kapasitas.
        Strategi tidak valid selalu dinilai dengan mask strategi yang asli.
    reward : {"software_aware", "tardiness"}, default="software_aware"
        ``"tardiness"`` hanya melihat perubahan keterlambatan.
    capacity_schedule : {"fixed", "balanced"}, default="balanced"
        ``"fixed"`` memakai kapasitas 1,2. ``"balanced"`` menyeimbangkan langkah latih
        di antara ``capacity_levels``.
    capacity_levels : tuple of float, default=(1.0, 1.5, 2.0)
        Pengali kapasitas untuk jadwal seimbang.
    total_timesteps : int, default=1_540_096
        Anggaran langkah lingkungan.
    n_envs, n_steps, batch_size, n_epochs, learning_rate, gamma, gae_lambda, clip_range, \
ent_coef, vf_coef, max_grad_norm :
        Hiperparameter PPO (nilai bawaan dibekukan dari Bab III).
    net_arch : tuple of int, default=(128, 128)
        Lebar lapisan tersembunyi aktor dan kritik.
    intensities : tuple of float, default=(0.05, 0.15, 0.30)
        Intensitas gangguan yang diacak per episode latih.
    k_tight : float, default=1.5
        Keketatan tenggat.
    random_state : None, int, atau numpy.random.Generator, default=None
        Sumber semua keacakan pelatihan. Seed pekerja diturunkan darinya.
    verbose : int, default=0
        Tingkat cetakan SB3.

    Attributes
    ----------
    policy_ : sb3_contrib.MaskablePPO
        Model hasil latihan.
    n_timesteps_ : int
        Langkah lingkungan yang benar-benar terkumpul.
    training_curve_ : pandas.DataFrame
        Satu baris per update: langkah, reward rata-rata episode, panjang episode,
        reward rata-rata per langkah, entropi, dan pengali kapasitas.
    capacity_exposure_ : dict
        Pengali kapasitas -> jumlah langkah latih yang dialaminya.

    Examples
    --------
    >>> from dfjsp.selectors import PPOSelector
    >>> PPOSelector(mask="high", random_state=0)
    PPOSelector(mask='high', random_state=0)
    >>> PPOSelector().operators[2].enforce_capacity
    True
    """

    def __init__(self, mask="two_level", reward="software_aware",
                 capacity_schedule="balanced", capacity_levels=(1.0, 1.5, 2.0),
                 total_timesteps=1_540_096, n_envs=8, n_steps=1024, batch_size=256,
                 n_epochs=10, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_range=0.2, ent_coef=0.03, vf_coef=0.5, max_grad_norm=0.5,
                 net_arch=(128, 128), intensities=(0.05, 0.15, 0.30), k_tight=1.5,
                 random_state=None, verbose=0):
        self.mask = mask
        self.reward = reward
        self.capacity_schedule = capacity_schedule
        self.capacity_levels = capacity_levels
        self.total_timesteps = total_timesteps
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.net_arch = net_arch
        self.intensities = intensities
        self.k_tight = k_tight
        self.random_state = random_state
        self.verbose = verbose

    # ---------------------------------------------------- komponen lingkungan
    @property
    def mask_fn(self):
        """Mask strategi yang dilihat agen: semua-satu untuk ``mask="none"``."""
        return allow_all_mask if self.mask == "none" else high_level_mask

    @property
    def operators(self) -> dict:
        """Operator perbaikan; batas kapasitas diperiksa hanya pada ``"two_level"``."""
        return default_operators(enforce_capacity=self.mask == "two_level")

    def build_reward(self):
        """Objek reward menurut parameter ``reward``."""
        return TardinessReward() if self.reward == "tardiness" else SoftwareAwareReward()

    def _levels(self) -> tuple:
        if self.capacity_schedule == "fixed":
            return (FIXED_CAPACITY,)
        return tuple(float(x) for x in self.capacity_levels)

    def _make_env(self, instances: dict, *, worker_index: int, base_seed: int):
        """Satu lingkungan latih (satu worker) untuk ``instances``."""
        from ..core.seeding import derive_worker_seed
        from ..env.gym_env import GymRepairEnv

        return GymRepairEnv(
            instances, list(instances), scenarios=["mixed"], k_tight=float(self.k_tight),
            intensities=[float(x) for x in self.intensities], reward=self.build_reward(),
            mask_fn=self.mask_fn, operators=self.operators,
            enforce_capacity=self.mask == "two_level",
            capacity_curriculum=list(self._levels()),
            worker_seed=derive_worker_seed(base_seed, worker_index))

    # ----------------------------------------------------------- validasi
    def _validate_params(self) -> None:
        def need(ok: bool, name: str, rule: str) -> None:
            check_param(self, ok, name, rule)

        need(self.mask in MASKS, "mask", f"harus salah satu dari {MASKS}")
        need(self.reward in REWARDS, "reward", f"harus salah satu dari {REWARDS}")
        need(self.capacity_schedule in CAPACITY_SCHEDULES, "capacity_schedule",
             f"harus salah satu dari {CAPACITY_SCHEDULES}")
        need(is_positive_list(self.capacity_levels)
             and len(set(self.capacity_levels)) == len(tuple(self.capacity_levels)),
             "capacity_levels", "harus daftar bilangan positif tanpa pengulangan")
        need(is_whole(self.total_timesteps, 1), "total_timesteps", "harus bilangan bulat >= 1")
        need(is_whole(self.n_envs, 1), "n_envs", "harus bilangan bulat >= 1")
        need(is_whole(self.n_steps, 1), "n_steps", "harus bilangan bulat >= 1")
        need(is_whole(self.batch_size, 1) and self.batch_size <= self.n_envs * self.n_steps,
             "batch_size", "harus bilangan bulat di antara 1 dan n_envs * n_steps")
        need(is_whole(self.n_epochs, 1), "n_epochs", "harus bilangan bulat >= 1")
        need(is_number(self.learning_rate) and self.learning_rate > 0,
             "learning_rate", "harus > 0")
        need(is_number(self.gamma) and 0 < self.gamma <= 1, "gamma", "harus di (0, 1]")
        need(is_number(self.gae_lambda) and 0 <= self.gae_lambda <= 1,
             "gae_lambda", "harus di [0, 1]")
        need(is_number(self.clip_range) and self.clip_range > 0, "clip_range", "harus > 0")
        need(is_number(self.ent_coef) and self.ent_coef >= 0, "ent_coef", "harus >= 0")
        need(is_number(self.vf_coef) and self.vf_coef >= 0, "vf_coef", "harus >= 0")
        need(is_number(self.max_grad_norm) and self.max_grad_norm > 0,
             "max_grad_norm", "harus > 0")
        need(is_positive_list(self.net_arch), "net_arch", "harus daftar lebar lapisan > 0")
        need(is_positive_list(self.intensities), "intensities",
             "harus daftar bilangan positif")
        need(is_number(self.k_tight) and self.k_tight > 0, "k_tight", "harus > 0")

    # ---------------------------------------------------------------- fit
    def fit(self, instances, *, allow_test_instances: bool = False):
        """Latih pemilih pada instans latih.

        Parameters
        ----------
        instances : dict
            Nama -> instans latih (misalnya dari ``split_instances``).
        allow_test_instances : bool, default=False
            Hanya untuk tes unit. Tanpa ini, instans uji Brandimarte (MK03, MK04, MK09)
            ditolak dengan ``ValueError``. Pipeline tidak memakainya.

        Returns
        -------
        self
        """
        pool = check_instances(instances)
        reject_test_instances(pool, allow_test_instances)
        self._validate_params()
        if not pool:
            raise ValueError("instances kosong: tidak ada instans latih")

        import torch
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

        # Satu thread torch: adil antar proses dan aman untuk SubprocVecEnv (fork).
        torch.set_num_threads(1)
        seed = base_seed(self.random_state)
        if not 0 <= seed < 2**32:
            raise ValueError(f"random_state harus di [0, 2**32), diterima {self.random_state!r}")

        from ..core.seeding import derive_seed

        n_envs, n_steps = int(self.n_envs), int(self.n_steps)
        quantum = n_envs * n_steps
        n_updates = math.ceil(int(self.total_timesteps) / quantum)
        scheduler = CapacityScheduler(
            self._levels(), n_envs=n_envs, n_steps=n_steps,
            random_state=derive_seed(seed, "domain"))
        ledger, rows = CapacityExposureLedger(), []

        thunks = [(lambda i=i: self._make_env(pool, worker_index=i, base_seed=seed))
                  for i in range(n_envs)]
        vec_env = SubprocVecEnv(thunks, start_method="fork") if n_envs > 1 \
            else DummyVecEnv(thunks)
        try:
            model = _new_model(vec_env, self, seed)
            model.learn(total_timesteps=n_updates * quantum,
                        callback=new_rollout_callback(scheduler, ledger, rows),
                        reset_num_timesteps=True, progress_bar=False)
        finally:
            vec_env.close()

        scheduler.validate_exposure(ledger, expected_total_steps=n_updates * quantum)
        self.policy_ = model
        self.n_timesteps_ = int(model.num_timesteps)
        self.training_curve_ = curve_frame(rows)
        self.capacity_exposure_ = dict(ledger.actual_steps)
        return self

    # ------------------------------------------------------------- keputusan
    def _use_mask(self, mask) -> np.ndarray:
        return effective_mask(self.mask, mask)

    def predict(self, state, mask) -> int:
        """Strategi pilihan terbaik (argmax) di antara yang diizinkan mask.

        Parameters
        ----------
        state : array (9,)
        mask : array (4,)
            Mask strategi. Diabaikan bila ``mask="none"``.
        """
        check_is_fitted(self, "policy_")
        allowed = self._use_mask(mask)
        if not allowed.any():
            return MAINTAIN
        obs = np.asarray(state, dtype=np.float32)
        action, _ = self.policy_.predict(obs, action_masks=allowed, deterministic=True)
        action = int(action)
        return action if allowed[action] else MAINTAIN

    def predict_proba(self, state, mask) -> np.ndarray:
        """Peluang tiap strategi dari kebijakan, 0 untuk yang dilarang mask."""
        import torch

        check_is_fitted(self, "policy_")
        allowed = self._use_mask(mask)
        if not allowed.any():
            proba = np.zeros(N_ACTIONS)
            proba[MAINTAIN] = 1.0
            return proba
        obs = np.asarray(state, dtype=np.float32)
        with torch.no_grad():
            obs_t, _ = self.policy_.policy.obs_to_tensor(obs)
            dist = self.policy_.policy.get_distribution(obs_t, action_masks=allowed)
            return dist.distribution.probs.detach().cpu().numpy().reshape(-1).astype(float)

    # ------------------------------------------------------- simpan dan muat
    def save(self, path) -> None:
        """Simpan model, parameter, dan versi paket dalam satu berkas zip.

        Parameters
        ----------
        path : str, Path, atau berkas terbuka
        """
        check_is_fitted(self, "policy_")
        model_bytes = io.BytesIO()
        self.policy_.save(model_bytes)
        params = {k: (list(v) if k in _TUPLE_PARAMS else v)
                  for k, v in self.get_params(deep=False).items()}
        if not isinstance(params["random_state"], (int, type(None))):
            params["random_state"] = None      # Generator tidak bisa disimpan
        meta = {
            "class": "PPOSelector", "params": params, "versions": _versions(),
            "state_semantics_version": STATE_SEMANTICS_VERSION, "state_dim": STATE_DIM,
            "n_timesteps": self.n_timesteps_,
            "capacity_exposure": {repr(k): v for k, v in self.capacity_exposure_.items()},
        }
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("model.zip", model_bytes.getvalue())
            zf.writestr("selector.json", json.dumps(meta, indent=2, sort_keys=True))
            zf.writestr("training_curve.csv", self.training_curve_.to_csv(index=False))

    @classmethod
    def load(cls, path) -> "PPOSelector":
        """Muat pemilih yang disimpan ``save``.

        Raises
        ------
        ValueError
            Bila berkas bukan hasil ``save`` atau arti keadaannya (``state_semantics_version``)
            berbeda dari kode sekarang.
        """
        from sb3_contrib import MaskablePPO

        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            if "selector.json" not in names:
                raise ValueError("berkas bukan hasil PPOSelector.save: selector.json tidak ada")
            meta = json.loads(zf.read("selector.json"))
            missing = [f for f in _FILES if f not in names]
            if missing:
                raise ValueError(f"berkas pemilih tidak lengkap, hilang: {missing}")
            if meta.get("state_semantics_version") != STATE_SEMANTICS_VERSION:
                raise ValueError(
                    f"state_semantics_version berkas {meta.get('state_semantics_version')!r} "
                    f"berbeda dari kode ({STATE_SEMANTICS_VERSION}); model dilatih untuk "
                    "keadaan lain")
            model = MaskablePPO.load(io.BytesIO(zf.read("model.zip")), device="cpu")
            curve = pd.read_csv(io.StringIO(zf.read("training_curve.csv").decode()))
        params = {k: (tuple(v) if k in _TUPLE_PARAMS else v)
                  for k, v in meta["params"].items()}
        selector = cls(**params)
        selector.policy_ = model
        selector.n_timesteps_ = int(meta["n_timesteps"])
        selector.training_curve_ = curve
        selector.capacity_exposure_ = {float(k): int(v)
                                       for k, v in meta["capacity_exposure"].items()}
        return selector
