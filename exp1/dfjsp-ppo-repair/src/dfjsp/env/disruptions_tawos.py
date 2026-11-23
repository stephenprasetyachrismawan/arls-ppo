"""Model gangguan yang dikalibrasi dari data Jira nyata (TAWOS).

Angka bawaan di ``TawosParams`` berasal dari ``configs/gangguan_tawos.json``, keluaran
``scripts/tawos_kalibrasi.py`` atas TAWOS v1.1 (Tawosi dkk., MSR 2022, 458.232 issue dari 39
proyek Jira terbuka). Semua angka itu berupa peluang, rasio, atau bentuk distribusi, jadi
tidak bergantung satuan waktu simulator. Ringkasannya ada di
``revisi/hasil/tawos_kalibrasi.md``.

Perbedaan utama dengan ``DisruptionGenerator`` (model lama):

* Banyak gangguan sebanding ukuran proyek (banyak tugas), bukan laju tetap per satuan
  waktu. Model lama memberi MK08 sekitar 270 gangguan dan MK01 sekitar 22 pada laju sama.
* Pekerjaan tidak terencana datang bergerombol: jarak antar-kedatangan Gamma dengan bentuk
  0,62 (TAWOS, median per sprint), bukan eksponensial (bentuk 1).
* Revisi upaya dan rework melekat pada tugas tertentu, pada saat yang masuk akal: revisi
  terungkap ketika tugas mulai dikerjakan, rework muncul ketika tugas selesai.
* Gangguan yang jatuh dalam satu jendela waktu digabung menjadi satu kejadian ``batch``
  sehingga ditangani dalam satu keputusan (catatan penguji R12).

Yang tidak ada di TAWOS, dan karena itu masih asumsi yang ditulis terang: ketidakhadiran
pengembang dan perubahan prioritas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.datatypes import FJSPInstance, Schedule
from .disruptions import Event


@dataclass
class TawosParams:
    """Parameter model gangguan. Nilai bawaan dari TAWOS kecuali yang ditandai asumsi."""

    #: Pekerjaan tidak terencana per pekerjaan terencana dalam satu sprint (median 2.123 sprint).
    unplanned_per_task: float = 0.5
    #: Bentuk Gamma jarak antar-kedatangan pekerjaan tidak terencana (median per sprint).
    arrival_gamma_shape: float = 0.62
    #: Porsi pekerjaan tidak terencana yang berprioritas tinggi.
    unplanned_high_priority: float = 0.383
    #: Ukuran pekerjaan tidak terencana dibanding pekerjaan terencana (rasio median menit
    #: dikerjakan, 1.193 / 4.332).
    unplanned_size_ratio: float = 0.275
    #: Sebaran ukuran pekerjaan tidak terencana (sigma log). ASUMSI: TAWOS tidak memberi
    #: sebaran ukuran yang bisa dipercaya, jadi dipakai sigma revisi estimasi.
    unplanned_size_sigma: float = 0.82
    #: Peluang estimasi sebuah tugas direvisi.
    revision_prob: float = 0.123
    #: Rasio estimasi baru / lama: lognormal (mu, sigma log).
    revision_mu: float = 0.0896
    revision_sigma: float = 0.818
    #: Peluang tugas selesai dibuka lagi (rework).
    rework_prob: float = 0.076
    #: Upaya tambahan rework dibanding upaya semula (median; sebaran TAWOS terlalu lebar
    #: untuk dipakai, lihat laporan kalibrasi).
    rework_ratio: float = 0.47
    #: ASUMSI: rata-rata masa tidak tersedia per pengembang sepanjang proyek.
    unavailable_per_dev: float = 0.5
    #: ASUMSI: lama tidak tersedia, kelipatan upaya tugas rata-rata (seragam).
    unavailable_lo: float = 1.0
    unavailable_hi: float = 3.0
    #: ASUMSI: peluang prioritas sebuah tugas berubah.
    priority_change_prob: float = 0.05
    #: Kelonggaran tenggat pekerjaan tidak terencana, kelipatan upayanya.
    due_slack: float = 0.5


def intensity_multiplier(intensity: float) -> float:
    """Pengali banyak gangguan dari nilai intensitas lama: 0,15 = median TAWOS.

    Jadi 0,05 / 0,15 / 0,30 menjadi 1/3, 1, dan 2 kali median TAWOS.
    """
    return float(intensity) / 0.15


class TawosDisruptionModel:
    """Bangkitkan gangguan satu episode menurut ``TawosParams``.

    Parameters
    ----------
    intensity : float
        Nilai intensitas lama (0,05 / 0,15 / 0,30); lihat ``intensity_multiplier``.
    rng : numpy.random.Generator
        Generator berseed aliran "disruption".
    params : TawosParams, optional
    batch_window : float, optional
        Gangguan yang jatuh kurang dari jarak ini dari gangguan pertama sebuah kelompok
        digabung menjadi satu kejadian ``batch``. None berarti rata-rata upaya tugas.

    Examples
    --------
    >>> import numpy as np
    >>> from dfjsp.datasets import load_brandimarte
    >>> from dfjsp.env.initial_schedule import build_initial_schedule
    >>> from dfjsp.env.disruptions_tawos import TawosDisruptionModel
    >>> inst = load_brandimarte(["MK01"])["MK01"]
    >>> sched = build_initial_schedule(inst)
    >>> events = TawosDisruptionModel(0.15, np.random.default_rng(0)).sample_timeline(
    ...     75.0, inst, sched)
    >>> all(a.time <= b.time for a, b in zip(events, events[1:]))
    True
    """

    def __init__(self, intensity: float, rng: np.random.Generator,
                 params: TawosParams | None = None, batch_window: float | None = None):
        self.intensity = float(intensity)
        self.rng = rng
        self.params = params or TawosParams()
        self.batch_window = batch_window
        self.n_dropped_no_target = 0

    # ------------------------------------------------------------------ utama
    def sample_timeline(self, horizon: float, inst: FJSPInstance,
                        sched: Schedule) -> list[Event]:
        """Semua gangguan episode, terurut waktu, sudah digabung per jendela."""
        m = intensity_multiplier(self.intensity)
        if m <= 0:
            return []
        p = self.params
        rng = self.rng
        tasks = [t for t in inst.tasks if t.task_id in sched.entries]
        efforts = np.array([sched.entries[t.task_id].end - sched.entries[t.task_id].start
                            for t in tasks], dtype=float)
        mean_eff = float(efforts.mean()) if len(efforts) else 1.0
        events: list[Event] = []

        # 1. Pekerjaan tidak terencana: proses pembaruan Gamma.
        expected = m * p.unplanned_per_task * len(tasks)
        if expected > 0:
            k = p.arrival_gamma_shape
            scale = horizon / expected / k
            t = 0.0
            while True:
                t += float(rng.gamma(k, scale))
                if t > horizon:
                    break
                template = tasks[int(rng.integers(len(tasks)))]
                owner = sched.entries[template.task_id].dev_id
                size = p.unplanned_size_ratio * float(
                    rng.lognormal(0.0, p.unplanned_size_sigma))
                eligible = {j: max(0.25, e * size) for j, e in template.eligible_devs.items()}
                high = bool(rng.random() < p.unplanned_high_priority)
                events.append(Event(t, "urgent_task", {
                    "dev_id": int(owner), "eligible": eligible,
                    "effort": float(eligible.get(owner, next(iter(eligible.values())))),
                    "priority": 1.0 if high else 0.5,
                    "due_rule": "segera" if high else "antrean",
                    "due_slack": p.due_slack,
                }))

        # 2. Revisi estimasi: terungkap seperempat jalan pengerjaan menurut S0.
        prob_rev = min(1.0, m * p.revision_prob)
        for t_obj in tasks:
            if rng.random() < prob_rev:
                e = sched.entries[t_obj.task_id]
                factor = float(np.clip(rng.lognormal(p.revision_mu, p.revision_sigma),
                                       0.25, 4.0))
                when = e.start + 0.25 * (e.end - e.start)
                if 0.0 < when < horizon:
                    events.append(Event(when, "effort_revision",
                                        {"task_id": t_obj.task_id, "factor": factor}))

        # 3. Rework: muncul tepat saat tugas selesai menurut S0.
        prob_rw = min(1.0, m * p.rework_prob)
        for t_obj in tasks:
            if rng.random() < prob_rw:
                e = sched.entries[t_obj.task_id]
                when = float(np.nextafter(e.end, np.inf))
                if when < horizon:
                    extra = p.rework_ratio * (e.end - e.start)
                    events.append(Event(when, "rework_review_fail",
                                        {"task_id": t_obj.task_id, "extra_effort": extra}))

        # 4. Pengembang tidak tersedia (ASUMSI, tidak ada di TAWOS).
        for d in inst.developers:
            for _ in range(int(rng.poisson(m * p.unavailable_per_dev))):
                when = float(rng.uniform(0.0, horizon))
                dur = float(rng.uniform(p.unavailable_lo, p.unavailable_hi)) * mean_eff
                events.append(Event(when, "developer_unavailable",
                                    {"dev_id": int(d.dev_id), "duration": dur}))

        # 5. Perubahan prioritas (ASUMSI): hanya prioritas, tenggat tetap.
        prob_pc = min(1.0, m * p.priority_change_prob)
        for t_obj in tasks:
            if rng.random() < prob_pc:
                e = sched.entries[t_obj.task_id]
                when = float(rng.uniform(0.0, max(1e-6, e.start)))
                if 0.0 < when < horizon:
                    events.append(Event(when, "priority_change", {
                        "task_id": t_obj.task_id,
                        "new_priority": float(rng.uniform(0.0, 1.0)), "due_delta": 0.0}))

        events.sort(key=lambda ev: (ev.time, ev.type))
        window = self.batch_window if self.batch_window is not None else mean_eff
        return batch_events(events, window)


def batch_events(events: list[Event], window: float) -> list[Event]:
    """Gabungkan gangguan yang jatuh dalam ``window`` dari gangguan pertama kelompoknya.

    Kelompok berisi satu gangguan dikembalikan apa adanya. Kelompok berisi lebih dari
    satu menjadi ``Event(t_terakhir, "batch", {"events": [...]})``. Semua anggota diterapkan
    pada waktu keputusan kelompok itu.
    """
    if window <= 0 or not events:
        return list(events)
    out: list[Event] = []
    group: list[Event] = []
    for ev in events:
        if group and ev.time - group[0].time > window:
            out.append(_close(group))
            group = []
        group.append(ev)
    if group:
        out.append(_close(group))
    return out


def _close(group: list[Event]) -> Event:
    if len(group) == 1:
        return group[0]
    # Keputusan diambil sesudah anggota terakhir terjadi (manajer menanggapi semuanya
    # sekaligus), jadi waktu kelompok = waktu anggota terakhir.
    return Event(group[-1].time, "batch", {"events": list(group)})
