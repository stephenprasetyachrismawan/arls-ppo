"""Catatan episode (``EpisodeLog``): bentuk data yang dibaca semua fungsi ukuran.

Bentuk ``EpisodeLog``
---------------------
Satu episode = satu jadwal yang diperbaiki berulang kali sampai gangguan habis.
Lingkungan menulis satu ``StepRecord`` per keputusan, urut waktu, ke dalam
``EpisodeLog.steps``. Evaluasi membaca log ini, tidak membaca lingkungan.

``EpisodeLog``

=============  ==================================================================
``identity``   kamus bebas pengenal episode (misal ``method``, ``instance``,
               ``scenario``, ``intensity``, ``capacity``, ``k_tight``, ``seed``)
``steps``      daftar ``StepRecord``, satu per keputusan
``final``      kamus bebas ringkasan akhir dari lingkungan (misal ``horizon``,
               ``n_events``, ``n_events_no_target``, ``total_reward``)
=============  ==================================================================

``StepRecord`` (nilai diambil SESUDAH aksi dijalankan, kecuali ``mask``)

=========================  ======================================================
``action``                 strategi yang diminta: 0 Maintain, 1 Delay, 2 Reassign,
                           3 Escalate
``mask``                   mask tingkat atas SEBELUM aksi, empat angka 0 atau 1
``t_now``                  waktu keputusan berikutnya (jam)
``event_type``             jenis gangguan yang ditangani pada keputusan ini
``tardiness_total``        total keterlambatan jadwal saat ini
``makespan``               waktu selesai terakhir
``n_late_tasks``           jumlah tugas terlambat
``nervousness_step``       nervousness langkah ini terhadap jadwal awal
``context_switch_step``    biaya alih konteks langkah ini terhadap jadwal awal
``start_deviation_step``   jumlah selisih waktu mulai terhadap jadwal awal (jam)
``workload_imbalance``     simpangan baku beban antar pengembang
``utilization``            pemakaian pengembang, 0 sampai 1
``invalid_strategy``       True bila aksi diminta saat mask melarangnya
``invalid_repair``         True bila operator gagal membuat perbaikan
``hard_feasible``          hasil pemeriksa jadwal terpisah: True bila jadwal
                           sesudah perbaikan tidak melanggar batasan keras
``feasible_pairs``         jumlah pasangan tugas-pengembang yang layak
``n_moved``                jumlah tugas yang disentuh operator
``overload_devs_after``    jumlah pengembang melebihi kapasitas
``overload_excess_after``  total kelebihan beban
``overload_excess_added``  kelebihan beban yang ditambahkan aksi ini
``inference_time_ms``      waktu pemilih memutuskan (boleh 0 bila tidak diukur)
``repair_time_ms``         waktu operator bekerja (boleh 0 bila tidak diukur)
=========================  ======================================================

Empat kolom pertama (``action``, ``tardiness_total``, ``nervousness_step``,
``context_switch_step``) wajib; sisanya punya nilai bawaan supaya log kecil mudah
ditulis di tes. Lingkungan sebaiknya memakai ``StepRecord.from_info``, yang menolak
(``KeyError``) bila ada kolom ukuran yang hilang dari ``info``.

Contoh berkas ada di ``tests/library/fixtures/episode_log_example.json``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields


@dataclass
class StepRecord:
    """Satu keputusan dalam episode. Arti tiap kolom ada di docstring modul.

    Examples
    --------
    >>> from dfjsp.metrics import StepRecord
    >>> StepRecord(action=1, tardiness_total=4.0, nervousness_step=0.1,
    ...            context_switch_step=0.0).action
    1
    """

    action: int
    tardiness_total: float
    nervousness_step: float
    context_switch_step: float
    mask: tuple = (1, 1, 1, 1)
    t_now: float = 0.0
    event_type: str | None = None
    makespan: float = 0.0
    n_late_tasks: int = 0
    start_deviation_step: float = 0.0
    workload_imbalance: float = 0.0
    utilization: float = 0.0
    invalid_strategy: bool = False
    invalid_repair: bool = False
    hard_feasible: bool = True
    feasible_pairs: int = 0
    n_moved: int = 0
    overload_devs_after: float = 0.0
    overload_excess_after: float = 0.0
    overload_excess_added: float = 0.0
    inference_time_ms: float = 0.0
    repair_time_ms: float = 0.0

    @classmethod
    def from_info(cls, info: dict, mask, hard_feasible: bool = True,
                  inference_time_ms: float = 0.0) -> "StepRecord":
        """Bangun dari kamus ``info`` yang dikeluarkan ``env.step``.

        Parameters
        ----------
        info : dict
            Kamus ``info`` lingkungan. Semua kunci ukuran wajib ada, kalau tidak
            ``KeyError``. ``repair_time_ms`` boleh tidak ada.
        mask : daftar 4 angka
            Mask tingkat atas SEBELUM aksi (``info["high_mask"]`` adalah mask
            sesudah aksi, jadi bukan itu).
        hard_feasible : bool, default=True
            Hasil pemeriksa jadwal terpisah sesudah perbaikan.
        inference_time_ms : float, default=0.0
            Waktu pemilih memutuskan, bila diukur.

        Returns
        -------
        StepRecord
        """
        return cls(
            action=int(info["action"]),
            tardiness_total=float(info["tardiness_total"]),
            nervousness_step=float(info["nervousness_step"]),
            context_switch_step=float(info["context_switch_step"]),
            mask=tuple(int(m) for m in mask),
            t_now=float(info["t_now"]),
            event_type=info["event_type"],
            makespan=float(info["makespan"]),
            n_late_tasks=int(info["n_late_tasks"]),
            start_deviation_step=float(info["start_deviation_step"]),
            workload_imbalance=float(info["workload_imbalance"]),
            utilization=float(info["utilization"]),
            invalid_strategy=bool(info["invalid_strategy"]),
            invalid_repair=bool(info["invalid_repair"]),
            hard_feasible=bool(hard_feasible),
            feasible_pairs=int(info["feasible_pairs"]),
            n_moved=int(info["n_moved"]),
            overload_devs_after=float(info["overload_devs_after"]),
            overload_excess_after=float(info["overload_excess_after"]),
            overload_excess_added=float(info["overload_excess_added"]),
            inference_time_ms=float(inference_time_ms),
            repair_time_ms=float(info.get("repair_time_ms", 0.0)),
        )


@dataclass
class EpisodeLog:
    """Catatan satu episode: pengenal, daftar langkah, dan ringkasan akhir.

    Parameters
    ----------
    steps : daftar StepRecord
        Satu per keputusan, urut waktu. Boleh kosong (episode tanpa gangguan).
    identity : dict, default={}
        Pengenal episode, bebas isinya.
    final : dict, default={}
        Ringkasan akhir dari lingkungan, bebas isinya.

    Examples
    --------
    >>> from dfjsp.metrics import EpisodeLog, StepRecord, total_tardiness
    >>> log = EpisodeLog(steps=[StepRecord(0, 5.0, 0.0, 0.0),
    ...                         StepRecord(1, 3.0, 0.1, 0.0)])
    >>> total_tardiness(log)
    3.0
    >>> EpisodeLog.from_dict(log.to_dict()) == log
    True
    """

    steps: list = field(default_factory=list)
    identity: dict = field(default_factory=dict)
    final: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Ubah ke kamus biasa (siap ``json.dump``)."""
        return {
            "identity": dict(self.identity),
            "steps": [asdict(s) | {"mask": list(s.mask)} for s in self.steps],
            "final": dict(self.final),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EpisodeLog":
        """Kebalikan ``to_dict``. Kolom yang tidak dikenal di langkah menimbulkan galat."""
        known = {f.name for f in fields(StepRecord)}
        steps = []
        for row in data["steps"]:
            extra = set(row) - known
            if extra:
                raise ValueError(f"kolom langkah tidak dikenal: {sorted(extra)}")
            steps.append(StepRecord(**(row | {"mask": tuple(row.get("mask", (1, 1, 1, 1)))})))
        return cls(steps=steps, identity=dict(data.get("identity", {})),
                   final=dict(data.get("final", {})))
