"""Pustaka ``dfjsp``: memilih strategi perbaikan jadwal proyek perangkat lunak.

Gayanya meniru scikit-learn: kelas kecil, parameter di ``__init__``, belajar lewat
``fit``, memutuskan lewat ``predict``. Subpaket publik:

============== =====================================================================
``datasets``    muat instans Brandimarte MK01 sampai MK10, pisah latih dan uji
``env``         lingkungan perbaikan jadwal (``RepairEnv``) dan adaptor Gymnasium
``repair``      empat operator strategi (Maintain, Delay, Reassign, Escalate) dan mask
``reward``      fungsi imbalan (``SoftwareAwareReward``, ``TardinessReward``)
``selectors``   pemilih strategi: aturan, ``PPOSelector``, ``PairPPOSelector``
``reschedulers`` heuristik penjadwalan ulang (FIFO, SPT, LPT, EDD, ..., GA)
``metrics``     ukuran episode: keterlambatan, nervousness, alih konteks, dan lain-lain
``evaluation``  jalankan banyak metode pada grid yang sama, ukur waktu keputusan
``stats``       IQM, bootstrap bertingkat, peluang perbaikan
``utils``       kelas dasar, ``check_random_state``, ``NotFittedError``
``verify``      pemeriksa jadwal mandiri: kelayakan jadwal dan keputusan perbaikan
============== =====================================================================

Subpaket tidak diimpor di sini; impor yang dibutuhkan saja, misalnya
``from dfjsp.selectors import ThresholdSelector``. Modul ``dfjsp.core`` (pembaca data,
tipe data, tenggat, seed) adalah bagian dalam pustaka dan bukan API publik.
"""

__version__ = "0.1.0"

__all__ = [
    "datasets",
    "env",
    "evaluation",
    "metrics",
    "repair",
    "reschedulers",
    "reward",
    "selectors",
    "stats",
    "utils",
    "verify",
]
