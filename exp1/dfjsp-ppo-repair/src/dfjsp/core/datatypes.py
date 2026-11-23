"""Struktur data inti DFJSP-PPO (Bagian 3 rancangan; Tabel 3.1 pemetaan domain)."""

from __future__ import annotations

from dataclasses import dataclass, field

# Status task yang sah (dipakai partisi jadwal, Pers. 3.1).
STATUS_NOT_STARTED = "not_started"
STATUS_IN_PROGRESS = "in_progress"
STATUS_COMPLETED = "completed"
VALID_STATUSES = (STATUS_NOT_STARTED, STATUS_IN_PROGRESS, STATUS_COMPLETED)


@dataclass
class Task:
    """Sebuah "operasi" pada FJSP, ditafsirkan sebagai tugas/subtask proyek SW.

    Attributes
    ----------
    task_id : int
        Identitas global unik task.
    job_id : int
        Job/fitur induk pembawa prioritas bisnis dan tenggat.
    op_index : int
        Urutan operasi dalam job (untuk precedence intra-job).
    eligible_devs : dict
        ``{dev_id: effort_jam}``: machine eligibility + waktu proses. Pada FJSP satu
        operasi dapat dikerjakan beberapa mesin dengan waktu berbeda; di domain SW ini
        adalah pengembang yang kompatibel beserta estimasi upaya masing-masing (jam).
    deps : list of int
        Daftar task_id prasyarat (precedence; diisi pada Tahap E domain).
    due_date : float
        Tenggat dari Total Work Content (TWK, Tahap D / Pers. Subbab 4.4).
    priority : float
        Prioritas bisnis di [0, 1] dari kuantil waktu proses (ECDF).
    status : str
        ``not_started``, ``in_progress``, atau ``completed``.
    assigned_dev : int or None
        Pengembang yang ditugaskan (None bila belum).
    start_time : float or None
        Waktu mulai terjadwal (None bila belum).
    finish_time : float or None
        Waktu selesai terjadwal (None bila belum).
    progress : float
        Di [0, 1]; dipakai context-switching cost (Pers. 2.4) saat task yang sudah
        berprogres dipindah.
    """

    task_id: int
    job_id: int
    op_index: int
    eligible_devs: dict[int, float]
    deps: list[int] = field(default_factory=list)
    due_date: float = 0.0
    priority: float = 0.0
    status: str = STATUS_NOT_STARTED
    assigned_dev: int | None = None
    start_time: float | None = None
    finish_time: float | None = None
    progress: float = 0.0


@dataclass
class Developer:
    """Sebuah "mesin" pada FJSP, ditafsirkan sebagai pengembang/QA/reviewer.

    Attributes
    ----------
    dev_id : int
        Identitas pengembang (= machine_id basis-0).
    skills : set of int
        Himpunan job/skill yang kompatibel (diturunkan dari eligibility).
    capacity : float
        Kapasitas beban kerja berkelanjutan (jam); kelipatan terkendali dari rata-rata
        beban (Tahap E). Pelanggaran ``w_j + e_i <= cap_j`` ditolak low-level mask
        (Pers. 3.10).
    available : bool
        Berubah False saat gangguan ``developer_unavailable``.
    workload : float
        ``w_j`` berjalan (akumulasi efort yang dipikul).
    """

    dev_id: int
    skills: set[int] = field(default_factory=set)
    capacity: float = 0.0
    available: bool = True
    workload: float = 0.0


@dataclass
class FJSPInstance:
    """Instans DFJSP lengkap siap-simulasi.

    Attributes
    ----------
    name : str
        "MK01" ... "MK10".
    n_jobs : int
        Jumlah job.
    n_devs : int
        Jumlah pengembang (= jumlah mesin).
    tasks : list of Task
        Daftar seluruh Task.
    developers : list of Developer
        Daftar Developer (kosong sebelum Tahap E).
    horizon : float
        Horizon perencanaan untuk normalisasi state (Subbab 5.3).
    meta : dict
        Metadata audit (avg_eligible, sumber, checksum, domain_hash, dst.).

    Examples
    --------
    ``dfjsp.datasets.ProjectInstance`` adalah nama lain kelas ini.

    >>> from dfjsp.datasets import ProjectInstance, load_brandimarte
    >>> inst = load_brandimarte(["MK01"])["MK01"]
    >>> isinstance(inst, ProjectInstance)
    True
    >>> inst.name, inst.n_jobs, inst.n_devs, len(inst.tasks)
    ('MK01', 10, 6, 55)
    >>> inst.tasks[0].eligible_devs        # {pengembang: upaya jam}
    {0: 5.0, 2: 4.0}
    """

    name: str
    n_jobs: int
    n_devs: int
    tasks: list[Task] = field(default_factory=list)
    developers: list[Developer] = field(default_factory=list)
    horizon: float = 0.0
    meta: dict = field(default_factory=dict)


@dataclass
class ScheduleEntry:
    """Satu penugasan terjadwal: task `task_id` ke dev `dev_id` pada [start, end]."""

    task_id: int
    dev_id: int
    start: float
    end: float


@dataclass
class Schedule:
    """Jadwal sebagai pemetaan task_id -> ScheduleEntry.

    Attributes
    ----------
    entries : dict
        ``task_id -> ScheduleEntry`` untuk jadwal berjalan S_t.
    baseline_ref : dict or None
        Salinan S0 (jadwal awal prediktif) sebagai referensi tetap untuk menghitung
        nervousness (Pers. 2.3) dan context-switch (Pers. 2.4); None bila belum
        ditetapkan.
    """

    entries: dict[int, ScheduleEntry] = field(default_factory=dict)
    baseline_ref: dict[int, ScheduleEntry] | None = None
    # Waktu rilis (earliest start) per task. Keputusan implementasi (di luar Bagian 3):
    # diperlukan agar operator Delay (Pers. 3.5) memberi penundaan yang BERTAHAN saat
    # recompute_timeline menjadwal ulang T_not_started. Default kosong = tanpa kendala.
    release: dict[int, float] = field(default_factory=dict)
    # Hanya jeda yang sudah terungkap; tidak memuat gangguan masa depan.
    pauses: dict[int, list[tuple[float, float]]] = field(default_factory=dict)
    # True: hitung ulang mempertahankan urutan antrean tiap pengembang (right-shift
    # sebenarnya). False: perilaku lama, antrean disusun ulang menurut nomor tugas.
    # Perilaku lama hanya untuk memutar ulang rekaman emas dan hasil resmi1.
    keep_order: bool = True
    # Tugas terdampak gangguan terakhir (himpunan A, rancangan v2 bagian 4.3). Diisi
    # lingkungan setiap kali gangguan diterapkan; dipakai operator tangga.
    affected: set[int] = field(default_factory=set)
