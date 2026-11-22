"""Keadaan 9 angka yang dilihat pemilih strategi (Tabel 3.3 Bab III)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.datatypes import FJSPInstance, Schedule
from .schedule import developer_load, partition
from .work_calendar import working_time

EPS = 1e-8

# Sembilan angka keadaan, urut: C, Tard, O, U, B, N, H, Q, I (semua di [0, 1]).
STATE_FEATURES = ("makespan", "tardiness", "overload", "utilization", "imbalance",
                  "nervousness", "context_switch", "queue", "intensity")
STATE_DIM = len(STATE_FEATURES)

# Letak tiap angka pada vektor keadaan, supaya tidak ada angka indeks yang tersebar.
(MAKESPAN, TARDINESS, OVERLOAD, UTILIZATION, IMBALANCE,
 NERVOUSNESS, CONTEXT_SWITCH, QUEUE, INTENSITY) = range(STATE_DIM)

# Nama lama untuk angka I_t; dipakai pemilih Threshold.
I_T_INDEX = INTENSITY

# Naik dari 3 ke 4: keadaan berubah dari 14 angka menjadi 9 (tanpa jenis gangguan).
STATE_SEMANTICS_VERSION = 4


@dataclass
class EpisodeRefs:
    """Referensi normalisasi tetap per episode (dihitung saat reset)."""

    horizon: float
    tard0_denom: float
    n_ref: float
    h_ref: float
    i_max: float
    i_window: float


def total_tardiness(inst: FJSPInstance, sched: Schedule) -> float:
    """Total keterlambatan kumulatif: sum max(0, finish_i - due_i)."""
    tot = 0.0
    for t in inst.tasks:
        e = sched.entries.get(t.task_id)
        if e is None:
            continue
        tot += max(0.0, e.end - t.due_date)
    return tot


def makespan(inst: FJSPInstance, sched: Schedule) -> float:
    """Estimasi makespan saat ini: finish maksimum seluruh task terjadwal."""
    ends = [e.end for e in sched.entries.values()]
    return max(ends) if ends else 0.0


def compute_refs(inst: FJSPInstance, sched: Schedule, i_max: float,
                 tard_floor_effort: bool = False) -> EpisodeRefs:
    """Hitung referensi normalisasi sekali saat reset (Subbab 5.3, catatan 2).

    ``tard_floor_effort``: penyebut keterlambatan paling kecil sama dengan total upaya
    S0. Dipakai bila tenggat diturunkan dari S0, karena keterlambatan S0 lalu nol dan
    tidak bisa menjadi penyebut.
    """
    from ..core.due_date import e_nominal

    horizon = max(EPS, inst.horizon)
    tard0 = total_tardiness(inst, sched)
    if tard_floor_effort:
        tard0 = max(tard0, sum(e.end - e.start for e in sched.entries.values()))
    n_tasks = max(1.0, float(len(inst.tasks)))
    # Panjang selang pengamatan I_t: rata-rata upaya nominal satu tugas. Satuannya
    # waktu simulasi, diambil dari data instans, sehingga tidak ada angka tetap yang
    # dipilih tanpa dasar. Artinya "berapa gangguan datang selama kira-kira satu
    # tugas dikerjakan".
    upaya = [e_nominal(t) for t in inst.tasks]
    i_window = max(EPS, float(sum(upaya) / len(upaya))) if upaya else EPS
    return EpisodeRefs(
        horizon=horizon,
        tard0_denom=max(EPS, tard0),
        n_ref=n_tasks,          # N_ref ~ nervousness "Escalate penuh" (per langkah <= 1)
        h_ref=n_tasks,          # H_ref ~ total progress (per task <= 1)
        i_max=max(1.0, float(i_max)),
        i_window=i_window,
    )


def _clip01(x: float) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(min(1.0, max(0.0, x)))


def extract_state(
    inst: FJSPInstance,
    sched: Schedule,
    t_now: float,
    *,
    refs: EpisodeRefs,
    nervousness_cum: float,
    context_switch_cum: float,
    n_recent_events: int,
) -> np.ndarray:
    """Vektor keadaan (9,) float32 dengan semua angka di [0, 1].

    Urutannya ``STATE_FEATURES``: C, Tard, O, U, B, N, H, Q, I.
    """
    n_dev = max(1, len(inst.developers))
    completed, _, not_started = partition(inst, sched, t_now)
    w = developer_load(inst, sched, t_now)
    w_arr = np.array([w[d.dev_id] for d in inst.developers], dtype=float)
    cap_arr = np.array([d.capacity for d in inst.developers], dtype=float)

    cur_makespan = makespan(inst, sched)
    cur_tard = total_tardiness(inst, sched)

    c_t = _clip01(cur_makespan / refs.horizon)
    tard_t = _clip01(cur_tard / refs.tard0_denom)
    overload = int(np.sum(w_arr > cap_arr + EPS))
    o_t = _clip01(overload / n_dev)
    # U_t = jumlah waktu kerja / (jumlah pengembang * makespan)
    busy = 0.0
    for e in sched.entries.values():
        busy += working_time(e.start, e.end, sched.pauses.get(e.dev_id, []))
    u_t = _clip01(busy / (n_dev * max(EPS, cur_makespan)))
    mean_w = float(np.mean(w_arr)) if w_arr.size else 0.0
    b_t = _clip01(float(np.std(w_arr)) / max(EPS, mean_w)) if mean_w > EPS else 0.0
    n_t = _clip01(nervousness_cum / max(EPS, refs.n_ref))
    h_t = _clip01(context_switch_cum / max(EPS, refs.h_ref))
    # Q_t = tugas siap dikerjakan / tugas yang belum mulai
    by_id = {t.task_id: t for t in inst.tasks}
    ready = {
        tid for tid in not_started
        if all(dep in completed for dep in by_id[tid].deps)
    }
    q_t = _clip01(len(ready) / max(1, len(not_started)))
    i_t = _clip01(n_recent_events / refs.i_max)

    return np.array([c_t, tard_t, o_t, u_t, b_t, n_t, h_t, q_t, i_t], dtype=np.float32)
