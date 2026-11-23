"""PairPPOSelector: PPO yang memilih pasangan (tugas, pengembang) langsung.

Berbeda dari ``PPOSelector`` yang memilih satu dari empat strategi, pemilih ini memilih
satu pasangan dari semua kandidat (bisa ratusan sampai ribuan). Karena itu antarmukanya
jujur berbeda: ``predict(state, mask, env)`` butuh ``env`` untuk melihat jadwal, dan
mengembalikan pasangan, bukan angka strategi. Operator khususnya (``PairAssign``)
menerapkan pasangan itu.
"""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pandas as pd

from ..env.state import STATE_DIM, STATE_SEMANTICS_VERSION
from ..repair.actions import MAINTAIN, N_ACTIONS, REASSIGN
from ..repair.maintain import Maintain
from ..reward.software_aware import SoftwareAwareReward
from ..utils import BaseSelector, check_is_fitted
from ._capacity import CapacityExposureLedger
from ._common import (
    base_seed, check_instances, is_number, is_positive_list, is_whole, need,
    reject_test_instances,
)
from ._pair_support import PairAssign, Tracker, assign_mask
from ._pair_ppo import (
    PAIR_INPUT_DIM, PairHyper, PairPolicy, candidate_pairs, pair_features,
)
from ._ppo_callback import curve_frame

_TUPLE_PARAMS = ("capacity_levels", "intensities")
_FILES = ("weights.npz", "selector.json", "training_curve.csv")
IDENTITY = "pair-ppo-clip-numpy-9state-8pair-1"


class PairPPOSelector(BaseSelector):
    """PPO yang memilih pasangan (tugas, pengembang) langsung, tanpa lapisan strategi.

    Kandidat dan fitur: tiap pasangan punya 17 angka (9 angka keadaan, 4 angka tugas,
    4 angka pengembang). Jaringan ``scorer`` memberi skor tiap kandidat, pasangan dipilih
    menurut softmax skor saat melatih dan menurut skor tertinggi saat memutuskan.
    Anggaran langkah sama persis dengan ``PPOSelector``: ``total_timesteps`` langkah
    lingkungan terkumpul, tidak dibulatkan.

    Kapasitas latih: bila ``capacity_levels`` diisi, tiap episode baru memakai pengali
    yang langkahnya paling sedikit sejauh ini, jadi langkah latih seimbang antar
    pengali seperti pada ``PPOSelector``.

    Parameters
    ----------
    use_mask : bool, default=True
        Kandidat hanya pasangan yang lolos mask tingkat bawah. Tanpa mask semua
        pasangan jadi kandidat dan pasangan yang tidak layak ditolak operator.
    hidden : int, default=64
        Lebar lapisan tersembunyi scorer dan critic.
    total_timesteps : int, default=1_540_096
        Anggaran langkah lingkungan.
    n_steps : int, default=1024
        Langkah yang dikumpulkan sebelum satu update.
    batch_size : int, default=256
        Keputusan per minibatch.
    n_epochs, learning_rate, gamma, gae_lambda, clip_range, ent_coef, vf_coef, \
max_grad_norm :
        Hiperparameter PPO (nilai bawaan sama dengan ``PPOSelector``).
    capacity_levels : tuple of float atau None, default=(1.0, 1.5, 2.0)
        Pengali kapasitas latih. ``None`` memakai kapasitas dasar.
    intensities : tuple of float, default=(0.05, 0.15, 0.30)
    k_tight : float, default=1.5
    random_state : None, int, atau numpy.random.Generator, default=None
    verbose : int, default=0
        Bila > 0, cetak kemajuan tiap sekitar 5% anggaran.

    Attributes
    ----------
    policy_ : PairPolicy
        Jaringan scorer dan critic hasil latihan.
    n_timesteps_ : int
    training_curve_ : pandas.DataFrame
        Kolom sama dengan ``PPOSelector.training_curve_``.
    capacity_exposure_ : dict
        Pengali kapasitas -> langkah latih (kosong bila ``capacity_levels=None``).
    pending : tuple atau None
        Pasangan hasil ``predict`` terakhir, dibaca ``PairAssign``.

    Examples
    --------
    >>> from dfjsp.selectors import PairPPOSelector
    >>> sel = PairPPOSelector(use_mask=False, hidden=32)
    >>> sel
    PairPPOSelector(hidden=32, use_mask=False)
    >>> sorted(sel.operators)
    [0, 2]
    """

    def __init__(self, use_mask=True, hidden=64, total_timesteps=1_540_096, n_steps=1024,
                 batch_size=256, n_epochs=10, learning_rate=3e-4, gamma=0.99,
                 gae_lambda=0.95, clip_range=0.2, ent_coef=0.03, vf_coef=0.5,
                 max_grad_norm=0.5, capacity_levels=(1.0, 1.5, 2.0),
                 intensities=(0.05, 0.15, 0.30), k_tight=1.5, random_state=None, verbose=0):
        self.use_mask = use_mask
        self.hidden = hidden
        self.total_timesteps = total_timesteps
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
        self.capacity_levels = capacity_levels
        self.intensities = intensities
        self.k_tight = k_tight
        self.random_state = random_state
        self.verbose = verbose

    # ---------------------------------------------------- komponen lingkungan
    @property
    def pending(self):
        """Pasangan ``(tugas, pengembang)`` hasil ``predict`` terakhir, atau ``None``."""
        return getattr(self, "_pending", None)

    @property
    def mask_fn(self):
        """Mask strategi untuk lingkungan: hanya Maintain dan Reassign yang terbuka."""
        return assign_mask

    @property
    def operators(self) -> dict:
        """Maintain dan ``PairAssign`` (menerapkan pasangan hasil ``predict``)."""
        return {MAINTAIN: Maintain(), REASSIGN: PairAssign(self)}

    def candidates(self, env) -> list:
        """Pasangan kandidat pada keadaan ``env`` sekarang, urut id tugas lalu pengembang."""
        return candidate_pairs(env, bool(self.use_mask))

    def pair_features(self, env, pairs, state) -> np.ndarray:
        """Matriks ``(jumlah pasangan, 17)``: 9 angka keadaan lalu 8 angka pasangan."""
        return pair_features(env, pairs, state)

    # ----------------------------------------------------------- validasi
    def _validate_params(self) -> None:
        def check(ok, name, rule):
            need(self, ok, name, rule)

        check(isinstance(self.use_mask, (bool, np.bool_)), "use_mask", "harus True atau False")
        check(is_whole(self.hidden, 1), "hidden", "harus bilangan bulat >= 1")
        check(is_whole(self.total_timesteps, 1), "total_timesteps", "harus bilangan bulat >= 1")
        check(is_whole(self.n_steps, 1), "n_steps", "harus bilangan bulat >= 1")
        check(is_whole(self.batch_size, 1), "batch_size", "harus bilangan bulat >= 1")
        check(is_whole(self.n_epochs, 1), "n_epochs", "harus bilangan bulat >= 1")
        check(is_number(self.learning_rate) and self.learning_rate > 0,
              "learning_rate", "harus > 0")
        check(is_number(self.gamma) and 0 < self.gamma <= 1, "gamma", "harus di (0, 1]")
        check(is_number(self.gae_lambda) and 0 <= self.gae_lambda <= 1,
              "gae_lambda", "harus di [0, 1]")
        check(is_number(self.clip_range) and self.clip_range > 0, "clip_range", "harus > 0")
        check(is_number(self.ent_coef) and self.ent_coef >= 0, "ent_coef", "harus >= 0")
        check(is_number(self.vf_coef) and self.vf_coef >= 0, "vf_coef", "harus >= 0")
        check(is_number(self.max_grad_norm) and self.max_grad_norm > 0,
              "max_grad_norm", "harus > 0")
        check(self.capacity_levels is None or (
            is_positive_list(self.capacity_levels)
            and len(set(self.capacity_levels)) == len(tuple(self.capacity_levels))),
            "capacity_levels", "harus None atau daftar bilangan positif tanpa pengulangan")
        check(is_positive_list(self.intensities), "intensities", "harus daftar bilangan positif")
        check(is_number(self.k_tight) and self.k_tight > 0, "k_tight", "harus > 0")

    def _hyper(self) -> PairHyper:
        return PairHyper(
            gamma=float(self.gamma), gae_lambda=float(self.gae_lambda),
            clip_range=float(self.clip_range), learning_rate=float(self.learning_rate),
            n_epochs=int(self.n_epochs), n_steps=int(self.n_steps),
            batch_size=int(self.batch_size), ent_coef=float(self.ent_coef),
            vf_coef=float(self.vf_coef), max_grad_norm=float(self.max_grad_norm))

    # ---------------------------------------------------------------- fit
    def fit(self, instances, *, allow_test_instances: bool = False):
        """Latih pada instans latih.

        Parameters
        ----------
        instances : dict
            Nama -> instans latih.
        allow_test_instances : bool, default=False
            Hanya untuk tes unit. Tanpa ini, MK03, MK04, dan MK09 ditolak.

        Returns
        -------
        self
        """
        pool = check_instances(instances)
        reject_test_instances(pool, allow_test_instances)
        self._validate_params()
        if not pool:
            raise ValueError("instances kosong: tidak ada instans latih")
        seed = base_seed(self.random_state)
        if not 0 <= seed < 2**31 - 1:
            raise ValueError(f"random_state harus di [0, 2**31 - 1), diterima {self.random_state!r}")

        from ..core.seeding import derive_seed, derive_worker_seed
        from ..env.gym_env import GymRepairEnv

        policy = PairPolicy(bool(self.use_mask), int(self.hidden), self._hyper(), seed)
        levels = None if self.capacity_levels is None else [float(x) for x in self.capacity_levels]
        sampler = GymRepairEnv(
            pool, list(pool), scenarios=["mixed"], k_tight=float(self.k_tight),
            intensities=[float(x) for x in self.intensities], reward=SoftwareAwareReward(),
            mask_fn=assign_mask, operators={MAINTAIN: Maintain(), REASSIGN: PairAssign(policy)},
            capacity_curriculum=levels, worker_seed=derive_worker_seed(seed, 0))
        ledger = CapacityExposureLedger()
        tracker = Tracker(ledger)
        tie_rng = np.random.default_rng(derive_seed(seed, "domain"))
        budget = int(self.total_timesteps)

        def new_episode():
            if levels is not None:
                tracker.level = _least_exposed(levels, ledger, tie_rng)
                sampler.set_capacity_mult(tracker.level)
            env, state, _info = sampler.sample_episode()
            if env.done:
                raise RuntimeError(
                    "tidak ada episode bergangguan setelah beberapa percobaan reset; "
                    "episode tanpa titik keputusan tidak dapat dilatih")
            return env, state

        def on_update(stats):
            tracker.on_update(stats)
            if self.verbose:
                print(f"[pair-ppo] {tracker.timesteps:,}/{budget:,} langkah", flush=True)

        policy.train(new_episode, max_env_steps=budget, on_step=tracker.on_step,
                     on_update=on_update)
        self.policy_ = policy
        self.n_timesteps_ = int(tracker.timesteps)
        self.training_curve_ = curve_frame(tracker.rows)
        self.capacity_exposure_ = dict(ledger.actual_steps)
        return self

    # ------------------------------------------------------------- keputusan
    def predict(self, state, mask, env):
        """Pasangan terbaik ``(id_tugas, id_pengembang)``, atau ``None`` bila tidak ada kandidat.

        Pasangan juga disimpan di ``pending`` supaya ``PairAssign`` bisa menerapkannya.

        Parameters
        ----------
        state : array (9,)
        mask : array (4,)
            Tidak dipakai (ada demi keseragaman dengan pemilih lain).
        env : RepairEnv
            Lingkungan yang jadwalnya dilihat pemilih.
        """
        check_is_fitted(self, "policy_")
        if env is None:
            raise ValueError("PairPPOSelector.predict butuh env untuk melihat jadwal")
        self.policy_.choose(state, env, sample=False)
        self._pending = self.policy_.pending
        self.policy_.pending = None
        return self._pending

    def act(self, state, mask, env=None) -> int:
        """Strategi untuk ``env.step``: Reassign bila ada pasangan, Maintain bila tidak."""
        return MAINTAIN if self.predict(state, mask, env) is None else REASSIGN

    def predict_proba(self, state, mask, env=None) -> np.ndarray:
        """Satu-panas atas empat strategi (Reassign bila ada pasangan, Maintain bila tidak)."""
        proba = np.zeros(N_ACTIONS)
        proba[self.act(state, mask, env)] = 1.0
        return proba

    def reset(self, seed=None) -> None:
        """Hapus pasangan terpilih sebelum episode baru."""
        self._pending = None

    # ------------------------------------------------------- simpan dan muat
    def save(self, path) -> None:
        """Simpan bobot, parameter, dan versi dalam satu berkas zip."""
        check_is_fitted(self, "policy_")
        weights = io.BytesIO()
        np.savez(weights, **self.policy_.weights())
        params = {k: (list(v) if k in _TUPLE_PARAMS and v is not None else v)
                  for k, v in self.get_params(deep=False).items()}
        if not isinstance(params["random_state"], (int, type(None))):
            params["random_state"] = None      # Generator tidak bisa disimpan
        meta = {
            "class": "PairPPOSelector", "identity": IDENTITY, "params": params,
            "versions": _versions(), "state_semantics_version": STATE_SEMANTICS_VERSION,
            "state_dim": STATE_DIM, "input_dim": PAIR_INPUT_DIM,
            "n_timesteps": self.n_timesteps_,
            "capacity_exposure": {repr(k): v for k, v in self.capacity_exposure_.items()},
        }
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("weights.npz", weights.getvalue())
            zf.writestr("selector.json", json.dumps(meta, indent=2, sort_keys=True))
            zf.writestr("training_curve.csv", self.training_curve_.to_csv(index=False))

    @classmethod
    def load(cls, path) -> "PairPPOSelector":
        """Muat pemilih yang disimpan ``save``.

        Raises
        ------
        ValueError
            Bila berkas bukan hasil ``save``, arti keadaannya berbeda, atau skema fitur
            pasangan berbeda dari kode sekarang.
        """
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            if "selector.json" not in names:
                raise ValueError(
                    "berkas bukan hasil PairPPOSelector.save: selector.json tidak ada")
            meta = json.loads(zf.read("selector.json"))
            missing = [f for f in _FILES if f not in names]
            if missing:
                raise ValueError(f"berkas pemilih tidak lengkap, hilang: {missing}")
            if meta.get("state_semantics_version") != STATE_SEMANTICS_VERSION:
                raise ValueError(
                    f"state_semantics_version berkas {meta.get('state_semantics_version')!r} "
                    f"berbeda dari kode ({STATE_SEMANTICS_VERSION})")
            if meta.get("identity") != IDENTITY:
                raise ValueError(
                    f"identity berkas {meta.get('identity')!r} berbeda dari kode ({IDENTITY!r})")
            with np.load(io.BytesIO(zf.read("weights.npz"))) as npz:
                weights = {name: npz[name] for name in npz.files}
            curve = pd.read_csv(io.StringIO(zf.read("training_curve.csv").decode()))
        params = {k: (tuple(v) if k in _TUPLE_PARAMS and v is not None else v)
                  for k, v in meta["params"].items()}
        selector = cls(**params)
        policy = PairPolicy(bool(selector.use_mask), int(selector.hidden), selector._hyper(),
                            seed=0)
        policy.load_weights(weights)
        selector.policy_ = policy
        selector.n_timesteps_ = int(meta["n_timesteps"])
        selector.training_curve_ = curve
        selector.capacity_exposure_ = {float(k): int(v)
                                       for k, v in meta["capacity_exposure"].items()}
        return selector


def _least_exposed(levels, ledger: CapacityExposureLedger, rng) -> float:
    """Pengali kapasitas dengan langkah latih paling sedikit; seri dipecah acak."""
    steps = [ledger.actual_steps.get(level, 0) for level in levels]
    low = min(steps)
    tied = [level for level, s in zip(levels, steps) if s == low]
    return tied[0] if len(tied) == 1 else float(rng.choice(tied))


def _versions() -> dict:
    import dfjsp

    return {"dfjsp": dfjsp.__version__}
