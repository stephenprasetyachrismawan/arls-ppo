"""Pemilih yang selalu meminta penyusunan ulang penuh (Escalate dengan operator heuristik)."""

from __future__ import annotations

from ..repair.actions import ESCALATE, MAINTAIN
from ..repair.registry import default_operators
from ..utils import BaseSelector, check_is_fitted
from ._operators import RULES, GARepair, RuleRepair


class _Rescheduler(BaseSelector):
    """Dasar bersama: pada tiap gangguan minta ``Escalate``, yang kini berarti
    "susun ulang semua tugas belum mulai".

    Titik sambung dengan ``dfjsp.evaluation.run_episode``: atribut ``operators`` berisi
    empat operator kanonis, tetapi slot Escalate diganti operator heuristik. Slot lain
    tetap kanonis supaya semua metode memakai satu standar operator.
    """

    def fit(self, instances=None):
        """Memeriksa parameter, tidak belajar apa pun. Mengembalikan ``self``.

        Tidak ada yang dipelajari dari ``instances``. ``fit`` tetap wajib dipanggil
        sebelum ``predict``, supaya pemakaiannya sama dengan pemilih lain. Hasilnya
        hanya tanda ``is_fitted_``.

        Parameters
        ----------
        instances : dict, optional
            Diabaikan. Ada supaya tanda tangan sama dengan pemilih lain.

        Returns
        -------
        self
        """
        self._check_params()
        self.is_fitted_ = True
        return self

    def reset(self, seed=None) -> None:
        """Tidak ada keadaan antar-episode; ada supaya sesuai protokol pemilih."""

    def predict(self, state, mask) -> int:
        """Escalate bila mask mengizinkan, kalau tidak Maintain (seperti ``respect_mask``)."""
        check_is_fitted(self)
        return ESCALATE if mask[ESCALATE] == 1 else MAINTAIN

    @property
    def operators(self) -> dict:
        """Peta aksi ke operator: kanonis, kecuali Escalate yang diganti heuristik."""
        self._check_params()
        ops = default_operators()
        ops[ESCALATE] = self._make_operator()
        return ops


class RuleRescheduler(_Rescheduler):
    """Penjadwalan ulang parsial menurut aturan urutan.

    Parameters
    ----------
    rule : {"fifo", "spt", "lpt", "edd", "mof", "skillfit"}, default="edd"
        Aturan urutan tugas. FIFO menurut nomor tugas, SPT dan LPT menurut upaya
        nominal terpendek dan terpanjang, EDD menurut tenggat terdekat, MOF
        (Most-Overloaded-First) mendahulukan tugas pada pengembang paling berat,
        SkillFit memilih pengembang paling spesialis.

    Attributes
    ----------
    is_fitted_ : bool
        Benar sesudah ``fit``.

    Examples
    --------
    >>> from dfjsp.reschedulers import RuleRescheduler
    >>> sel = RuleRescheduler(rule="edd").fit()
    >>> sel.predict(None, [1, 1, 1, 1])      # selalu meminta Escalate
    3
    >>> sel.predict(None, [1, 1, 1, 0])      # mask melarang, jatuh ke Maintain
    0
    >>> sel.operators[3].name
    'H_edd'
    >>> RuleRescheduler().predict(None, [1, 1, 1, 1])      # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    dfjsp.utils._validation.NotFittedError: RuleRescheduler belum dilatih. ...
    """

    def __init__(self, rule: str = "edd"):
        self.rule = rule

    def _check_params(self) -> None:
        if self.rule not in RULES:
            raise ValueError(
                f"Parameter rule={self.rule!r} tidak dikenal. Pilihan: {list(RULES)}")

    def _make_operator(self):
        return RuleRepair(self.rule)


class GARescheduler(_Rescheduler):
    """Penjadwalan ulang parsial dengan algoritma genetika ringan.

    Parameters
    ----------
    population : int, default=12
        Ukuran populasi, minimal 2.
    generations : int, default=8
        Jumlah generasi, minimal 1.
    mutation : float, default=0.2
        Peluang mutasi, di antara 0 dan 1.
    random_state : int, default=0
        Seed keacakan algoritma genetika.

    Attributes
    ----------
    is_fitted_ : bool
        Benar sesudah ``fit``.

    Examples
    --------
    >>> from dfjsp.reschedulers import GARescheduler
    >>> sel = GARescheduler(population=6, generations=2).fit()
    >>> sel.operators[3].name
    'H_GA'
    """

    def __init__(self, population: int = 12, generations: int = 8,
                 mutation: float = 0.2, random_state=0):
        self.population = population
        self.generations = generations
        self.mutation = mutation
        self.random_state = random_state

    def _check_params(self) -> None:
        if self.population < 2:
            raise ValueError(f"Parameter population={self.population!r} harus >= 2")
        if self.generations < 1:
            raise ValueError(f"Parameter generations={self.generations!r} harus >= 1")
        if not 0.0 <= self.mutation <= 1.0:
            raise ValueError(
                f"Parameter mutation={self.mutation!r} harus di antara 0 dan 1")

    def _make_operator(self):
        return GARepair(self.population, self.generations, self.mutation,
                        self.random_state)
