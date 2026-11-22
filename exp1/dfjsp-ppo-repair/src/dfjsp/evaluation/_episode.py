"""Satu episode evaluasi: pemilih memutuskan, lingkungan menjalankan, log mencatat.

Protokol pemilih
----------------
``run_episode`` menerima objek apa pun yang memenuhi protokol ini. ``BaseSelector``
(``dfjsp.utils``) cukup, tetapi tidak wajib.

========================  =======  ==============================================
anggota                   wajib?   arti
========================  =======  ==============================================
``predict(state, mask)``  ya       mengembalikan strategi 0 sampai 3 (bilangan
                                   bulat). ``mask`` adalah mask tingkat atas yang
                                   dilihat pemilih, empat angka 0 atau 1
``reset(seed)``           tidak    dipanggil sekali di awal episode dengan seed
                                   kebijakan (lihat ``policy_seed``)
``operators``             tidak    kamus aksi ke operator, dibaca SESUDAH ``reset``.
                                   Bila tidak ada atau ``None``: operator kanonis
``mask_fn``               tidak    ``mask_fn(inst, sched, t_now)``: mask yang
                                   dilihat pemilih. Bila tidak ada: mask tingkat
                                   atas. Penilaian "strategi tidak valid" tetap
                                   memakai mask tingkat atas yang sebenarnya
``bind_env(env)``         tidak    dipanggil sekali sesudah ``env.reset``, dengan
                                   lingkungan yang hidup
========================  =======  ==============================================

Titik sambung untuk pemilih pasangan (``PairPPOSelector``)
----------------------------------------------------------
Pemilih yang memilih pasangan (tugas, pengembang) tetap mengembalikan SATU strategi dari
``predict``, yaitu Reassign (2). Pasangan yang dipilih disimpan di pemilih itu sendiri,
dan ``operators[2]`` adalah operator yang menerapkan pasangan tersebut pada jadwal.

* ``bind_env(env)`` memberi pemilih akses ke ``env.inst``, ``env.sched``, dan
  ``env.t_now``, sehingga ia bisa membangun kandidat pasangan di dalam ``predict``.
* ``operators`` dibaca sekali per episode, jadi operator pasangan boleh dibuat baru
  tiap episode di ``reset``.
* Pemilihan pasangan terjadi di dalam ``predict``, jadi waktunya ikut tercatat sebagai
  waktu pemilih. Itu memang yang dimaksud: PPO pasangan lebih lambat karena ruang
  aksinya besar.
"""

from __future__ import annotations

import hashlib
import json
import time

from ..core.seeding import derive_episode_seed
from ..datasets import with_capacity
from ..env.disruptions import resolve_scenario
from ..env.repair_env import RepairEnv
from ..env.schedule_check import check_schedule
from ..metrics import EpisodeLog, StepRecord, episode_summary


class EvalEnv(RepairEnv):
    """``RepairEnv`` yang juga mengukur operator dan memeriksa kelayakan jadwal.

    Sesudah tiap perbaikan, pemeriksa jadwal yang berdiri sendiri
    (``check_schedule``) dijalankan pada keadaan saat itu, sebelum jam maju. Hasilnya
    ada di ``last_hard_feasible``. Waktu operator (tanpa pemeriksa) ada di
    ``last_repair_ns``, diukur dengan ``time.perf_counter_ns``.

    Examples
    --------
    >>> from dfjsp.datasets import load_brandimarte
    >>> from dfjsp.evaluation import EvalEnv
    >>> env = EvalEnv()
    >>> state, info = env.reset(load_brandimarte(["MK01"])["MK01"], intensity=0.05, seed=0)
    >>> state, reward, done, info = env.step(3)          # 3 = Escalate
    >>> env.last_hard_feasible, env.last_repair_ns > 0
    (True, True)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_hard_feasible = True
        self.last_repair_ns = 0

    def _apply_repair(self, action):
        started = time.perf_counter_ns()
        result = super()._apply_repair(action)
        self.last_repair_ns = time.perf_counter_ns() - started
        self.last_hard_feasible = check_schedule(
            self.inst, self.sched, self.t_now).layak_keras
        return result


def policy_seed(method_name: str, instance_name: str, *, intensity: float,
                k_tight: float, seed: int, scenario: str) -> int:
    """Seed yang diberikan ke ``reset`` pemilih untuk satu episode.

    Bergantung pada nama metode dan kombinasi episode, bukan pada kapasitas atau urutan
    jalan, jadi hasil sama pada ``n_jobs`` berapa pun.

    Examples
    --------
    >>> from dfjsp.evaluation import policy_seed
    >>> kunci = dict(intensity=0.15, k_tight=1.5, seed=0, scenario="mixed")
    >>> policy_seed("Random", "MK01", **kunci) == policy_seed("Random", "MK01", **kunci)
    True
    >>> policy_seed("Random", "MK01", **kunci) == policy_seed("Threshold", "MK01", **kunci)
    False
    """
    return derive_episode_seed(0, method_name, instance_name, intensity, k_tight, seed,
                               resolve_scenario(scenario), stream="policy")


def _plain(value):
    """Ubah tipe numpy menjadi tipe Python supaya bisa masuk JSON."""
    return value.item() if hasattr(value, "item") else str(value)


def events_digest(events) -> str:
    """Sidik 16 karakter dari urutan gangguan (waktu, jenis, isi).

    Dua episode dengan sidik sama menghadapi gangguan yang sama persis.
    """
    rows = [[round(float(e.time), 9), e.type, e.payload] for e in events]
    text = json.dumps(rows, sort_keys=True, separators=(",", ":"), default=_plain)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def open_episode(method, method_name: str, instance, *, intensity: float,
                 capacity: float, seed: int, k_tight: float, scenario: str):
    """Siapkan satu episode; kembalikan ``(env, state)``.

    Urutan: salin instans dengan kapasitas, ``method.reset(seed_kebijakan)``, baca
    ``method.operators`` dan ``method.mask_fn``, ``env.reset``, lalu ``method.bind_env``.
    """
    inst = with_capacity(instance, capacity)
    if hasattr(method, "reset"):
        method.reset(policy_seed(method_name, inst.name, intensity=intensity,
                                 k_tight=k_tight, seed=seed, scenario=scenario))
    env = EvalEnv(operators=getattr(method, "operators", None),
                  mask_fn=getattr(method, "mask_fn", None), base_seed=0)
    state, _ = env.reset(inst, intensity=intensity, scenario=scenario, k_tight=k_tight,
                         seed=seed)
    if hasattr(method, "bind_env"):
        method.bind_env(env)
    return env, state


def decide_and_step(method, env: EvalEnv, state):
    """Satu keputusan: pemilih memilih, lingkungan menjalankan.

    Returns
    -------
    tuple
        ``(state_baru, reward, info, mask, predict_ns)``. ``mask`` adalah mask yang
        dilihat pemilih SEBELUM aksi. ``predict_ns`` hanya mencakup pemanggilan
        pemilih: ``method.act(state, mask, env=env)`` bila pemilih punya ``act``
        (semua turunan ``BaseSelector``; ``PairPPOSelector`` butuh ``env``), kalau
        tidak ``method.predict(state, mask)``.
    """
    mask = env.action_mask().copy()
    started = time.perf_counter_ns()
    if hasattr(method, "act"):
        action = int(method.act(state, mask, env=env))
    else:
        action = int(method.predict(state, mask))
    predict_ns = time.perf_counter_ns() - started
    state, reward, _, info = env.step(action)
    return state, reward, info, mask, predict_ns


def run_episode(method, instance, *, intensity: float, capacity: float, seed: int,
                k_tight: float = 1.5, scenario: str = "mixed",
                record_steps: bool = False, method_name: str | None = None) -> EpisodeLog:
    """Jalankan satu episode dan kembalikan ``EpisodeLog``.

    Parameters
    ----------
    method : objek pemilih
        Memenuhi protokol di docstring modul ini.
    instance : ProjectInstance
        Instans dasar. Disalin, dengan kapasitas diatur ke ``capacity``.
    intensity : float
        Laju gangguan.
    capacity : float
        Pengali kapasitas (1,0 / 1,5 / 2,0).
    seed : int
        Seed episode. Menentukan gangguan, sama untuk semua metode.
    k_tight : float, default=1.5
        Keketatan tenggat.
    scenario : str, default="mixed"
        Skenario gangguan.
    record_steps : bool, default=False
        Bila True, ``log.steps`` berisi satu ``StepRecord`` per keputusan. Bila False,
        ``log.steps`` kosong (hemat memori) dan ukuran tetap lengkap di
        ``log.final["summary"]``.
    method_name : str, optional
        Nama metode untuk pengenal dan seed kebijakan. Bawaan: nama kelas.

    Returns
    -------
    EpisodeLog
        ``identity``: method, instance, intensity, capacity, k_tight, seed, scenario.
        ``final``: horizon, n_events, n_events_no_target, total_reward, events_digest,
        summary (hasil ``episode_summary``).

    Examples
    --------
    >>> from dfjsp.datasets import load_brandimarte
    >>> from dfjsp.evaluation import run_episode
    >>> from dfjsp.reschedulers import RuleRescheduler
    >>> inst = load_brandimarte(["MK01"])["MK01"]
    >>> log = run_episode(RuleRescheduler("edd").fit(), inst, intensity=0.05, capacity=1.5,
    ...                   seed=0)
    >>> log.identity["instance"], log.steps
    ('MK01', [])
    >>> sorted(log.final["summary"])[:3]
    ['action_share_delay', 'action_share_escalate', 'action_share_maintain']
    """
    name = method_name or type(method).__name__
    env, state = open_episode(method, name, instance, intensity=intensity,
                              capacity=capacity, seed=seed, k_tight=k_tight,
                              scenario=scenario)
    steps, total_reward = [], 0.0
    while not env.done:
        state, reward, info, mask, predict_ns = decide_and_step(method, env, state)
        total_reward += reward
        steps.append(StepRecord.from_info(
            info, mask, hard_feasible=env.last_hard_feasible,
            inference_time_ms=predict_ns / 1e6))
    log = EpisodeLog(
        steps=steps,
        identity={"method": name, "instance": instance.name, "intensity": intensity,
                  "capacity": capacity, "k_tight": k_tight, "seed": seed,
                  "scenario": scenario},
        final={"horizon": float(env.refs.horizon), "n_events": env.n_events,
               "n_events_no_target": env.n_events_no_target,
               "total_reward": total_reward, "events_digest": events_digest(env.events)})
    log.final["summary"] = episode_summary(log)
    if not record_steps:
        log.steps = []
    return log
