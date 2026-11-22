"""Generator gangguan (Subbab 5.4 rancangan; Tabel 3.7, 3.13)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.datatypes import FJSPInstance

# Skenario gangguan: nama -> jenis gangguan yang boleh muncul. Hanya "mixed" (lima jenis)
# yang dipakai tesis. Skenario satu jenis tetap ada untuk tes dan rekaman emas.
SCENARIOS: dict[str, tuple[str, ...]] = {
    "no_disruption": (),
    "developer_unavailability": ("developer_unavailable",),
    "urgent_task_arrival": ("urgent_task",),
    "effort_estimate_revision": ("effort_revision",),
    "priority_change": ("priority_change",),
    "review_failure_rework": ("rework_review_fail",),
    "mixed_disruptions": ("developer_unavailable", "urgent_task", "effort_revision",
                          "priority_change", "rework_review_fail"),
}

# "mixed" hanyalah nama pendek. Seed gangguan diturunkan dari nama lengkapnya, supaya
# gangguan yang sudah direkam tetap bisa dibangkitkan ulang persis.
SCENARIO_ALIASES = {"mixed": "mixed_disruptions"}


def resolve_scenario(name: str) -> str:
    """Nama lengkap skenario dari nama atau alias; ``ValueError`` bila tidak dikenal."""
    full = SCENARIO_ALIASES.get(name, name)
    if full not in SCENARIOS:
        raise ValueError(
            f"scenario {name!r} tidak dikenal (yang sah: "
            f"{sorted(set(SCENARIOS) | set(SCENARIO_ALIASES))})"
        )
    return full


@dataclass
class Event:
    """Satu event gangguan pada waktu `time` dengan `type` dan `payload` terselesaikan."""

    time: float
    type: str
    payload: dict = field(default_factory=dict)


@dataclass
class PayloadParams:
    """Parameter besaran payload gangguan (sebelumnya konstanta tertanam di kode)."""

    urgent_effort_lo: float = 1.0
    urgent_effort_hi: float = 5.0
    urgent_priority: float = 1.0
    urgent_due_factor: float = 0.3
    revision_factor_lo: float = 0.8
    revision_factor_hi: float = 1.5
    priority_new_lo: float = 0.0
    priority_new_hi: float = 1.0
    priority_due_delta_lo: float = -0.2
    priority_due_delta_hi: float = 0.2
    rework_extra_lo: float = 0.5
    rework_extra_hi: float = 2.0
    unavailable_duration: float = 5.0

    @classmethod
    def from_config(cls, cfg) -> "PayloadParams":
        """Bangun dari namespace/dict config; field absen memakai bawaan (perilaku lama)."""
        get = (cfg.get if isinstance(cfg, dict) else lambda k, d: getattr(cfg, k, d))
        d = cls()
        return cls(**{f: float(get(f, getattr(d, f))) for f in d.__dataclass_fields__})


class DisruptionGenerator:
    """Bangkitkan timeline gangguan sebagai proses Poisson homogen.

    Parameters
    ----------
    intensity : float
        Laju Poisson per satuan waktu (Tabel 3.7: 0.05/0.15/0.30).
    types : list of str
        Subset tipe yang diizinkan (dari ``SCENARIOS[nama_lengkap]``).
    rng : numpy.random.Generator
        Generator berseed aliran "disruption".
    payload : PayloadParams, optional
        Besaran payload; None berarti bawaan (perilaku lama identik).

    Examples
    --------
    >>> import numpy as np
    >>> from dfjsp.datasets import load_brandimarte
    >>> from dfjsp.env import DisruptionGenerator, RepairEnv
    >>> env = RepairEnv()
    >>> state, info = env.reset(load_brandimarte(["MK01"])["MK01"], intensity=0.05, seed=0)
    >>> gen = DisruptionGenerator(0.15, ["urgent_task", "effort_revision"],
    ...                           np.random.default_rng(0))
    >>> events = gen.sample_timeline(40.0, env.inst, env.sched)
    >>> [e.type for e in events]
    ['effort_revision', 'urgent_task', 'effort_revision']
    >>> DisruptionGenerator(0.0, ["urgent_task"], np.random.default_rng(0)).sample_timeline(40.0)
    []
    """

    def __init__(self, intensity: float, types: list[str], rng: np.random.Generator,
                 payload: PayloadParams | None = None):
        """Simpan parameter pembangkit; lihat parameter di docstring kelas."""
        self.intensity = float(intensity)
        self.types = list(types)
        self.rng = rng
        self.payload = payload or PayloadParams()
        self.n_dropped_no_target = 0
        self.n_retargeted = 0

    def sample_timeline(self, horizon: float, inst: FJSPInstance | None = None,
                        sched=None) -> list[Event]:
        """Bangkitkan event sampai melewati horizon; payload diisi & divalidasi.

        Returns
        -------
        list of Event
            Daftar Event terurut menaik berdasarkan waktu. Kosong bila tidak ada tipe
            (S0) atau intensitas <= 0.
        """
        if not self.types or self.intensity <= 0:
            return []
        events: list[Event] = []
        t = 0.0
        while True:
            t += float(self.rng.exponential(1.0 / self.intensity))
            if t > horizon:
                break
            typ = self.types[int(self.rng.integers(len(self.types)))]
            payload = self._payload(typ, inst, sched, t)
            if payload is None:
                lain = [
                    kandidat for kandidat in self.types
                    if kandidat != typ and self._ada_sasaran(kandidat, inst, sched, t)
                ]
                if not lain:
                    self.n_dropped_no_target += 1
                    continue
                typ = lain[int(self.rng.integers(len(lain)))]
                payload = self._payload(typ, inst, sched, t)
                self.n_retargeted += 1
            events.append(Event(time=t, type=typ, payload=payload))
        events.sort(key=lambda e: e.time)
        return events

    def _ada_sasaran(self, typ: str, inst: FJSPInstance | None, sched, t: float) -> bool:
        """Apakah tipe ini punya sasaran sah pada waktu t. Tidak memakai angka acak."""
        if inst is None or sched is None:
            return True
        if typ == "urgent_task":
            return True
        if typ == "developer_unavailable":
            return any(entry.end > t for entry in sched.entries.values())
        return bool(
            self._kandidat_tugas(
                inst, sched, t, harus_selesai=(typ == "rework_review_fail")
            )
        )

    def _kandidat_tugas(self, inst: FJSPInstance, sched, t: float, harus_selesai: bool):
        """Tugas yang sah menjadi sasaran pada waktu t menurut jadwal awal."""
        keluar = []
        for task in inst.tasks:
            entry = sched.entries.get(task.task_id)
            if entry is None:
                continue
            sudah_selesai = entry.end <= t
            if sudah_selesai == harus_selesai:
                keluar.append(task.task_id)
        return keluar

    def _payload(self, typ: str, inst: FJSPInstance | None,
                 sched=None, t: float = 0.0) -> dict | None:
        """Isi payload deterministik sesuai tipe; None bila tidak ada sasaran sah."""
        if inst is None:
            return {}
        pp = self.payload
        if sched is None:
            # Pemakaian tanpa jadwal (uji laju Poisson) mempertahankan perilaku lama.
            task_ids = [task.task_id for task in inst.tasks]
            dev_ids = list(range(inst.n_devs))
        else:
            dev_ids = sorted({
                entry.dev_id for entry in sched.entries.values() if entry.end > t
            })
            task_ids = self._kandidat_tugas(
                inst, sched, t, harus_selesai=(typ == "rework_review_fail")
            )
        if typ == "developer_unavailable":
            if not dev_ids:
                return None
            return {"dev_id": int(self.rng.choice(dev_ids)),
                    "duration": pp.unavailable_duration}
        if typ == "urgent_task":
            # task mendesak baru: eligible dari satu dev acak; prioritas tinggi, tenggat ketat.
            dev = int(self.rng.integers(inst.n_devs))  # tugas baru, bukan sasaran lama
            eff = float(self.rng.uniform(pp.urgent_effort_lo, pp.urgent_effort_hi))
            return {"dev_id": dev, "effort": eff, "priority": pp.urgent_priority,
                    "due_factor": pp.urgent_due_factor}
        if typ == "effort_revision":
            if not task_ids:
                return None
            return {
                "task_id": int(self.rng.choice(task_ids)),
                "factor": float(self.rng.uniform(pp.revision_factor_lo, pp.revision_factor_hi)),
            }
        if typ == "priority_change":
            if not task_ids:
                return None
            return {
                "task_id": int(self.rng.choice(task_ids)),
                "new_priority": float(self.rng.uniform(pp.priority_new_lo, pp.priority_new_hi)),
                "due_delta": float(self.rng.uniform(pp.priority_due_delta_lo, pp.priority_due_delta_hi)),
            }
        if typ == "rework_review_fail":
            if not task_ids:
                return None
            return {
                "task_id": int(self.rng.choice(task_ids)),
                "extra_effort": float(self.rng.uniform(pp.rework_extra_lo, pp.rework_extra_hi)),
            }
        return {}
