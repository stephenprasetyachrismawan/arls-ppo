"""Kelas dasar dengan ``get_params`` dan kelas dasar pemilih strategi.

Gayanya meniru ``sklearn.base.BaseEstimator``.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod

import numpy as np


class ParamsMixin:
    """Memberi ``get_params``, ``set_params``, dan ``__repr__`` seperti scikit-learn.

    Aturannya:

    * ``__init__`` hanya menyimpan parameter ke atribut dengan nama yang sama.
    * ``get_params`` membaca daftar parameter dari tanda tangan ``__init__``.
    * ``set_params`` mengubah parameter, termasuk objek bersarang (``a__b``).

    Dipakai oleh pemilih, reward, dan lingkungan, jadi semuanya bisa dibuat ulang
    dari ``get_params`` dan ditampilkan dengan ringkas.

    Examples
    --------
    >>> from dfjsp.utils import ParamsMixin
    >>> class Kecil(ParamsMixin):
    ...     def __init__(self, a=1, b="x"):
    ...         self.a = a
    ...         self.b = b
    >>> Kecil(a=3)
    Kecil(a=3)
    >>> Kecil().set_params(b="y").get_params()
    {'a': 1, 'b': 'y'}
    """

    @classmethod
    def _get_param_names(cls) -> list[str]:
        """Nama parameter ``__init__``, terurut abjad, tanpa ``self``."""
        init = cls.__init__
        if init is object.__init__:
            return []
        signature = inspect.signature(init)
        params = [p for p in signature.parameters.values() if p.name != "self"]
        for p in params:
            if p.kind is p.VAR_POSITIONAL:
                raise RuntimeError(
                    f"{cls.__name__} memakai *{p.name} di __init__. Tulis semua "
                    "parameter dengan nama jelas supaya get_params bisa bekerja."
                )
        return sorted(p.name for p in params if p.kind is not p.VAR_KEYWORD)

    def get_params(self, deep: bool = True) -> dict:
        """Ambil parameter ``__init__`` sebagai kamus.

        Parameters
        ----------
        deep : bool, default=True
            Bila benar, parameter objek bersarang ikut diambil dengan nama
            ``induk__anak``.

        Returns
        -------
        dict
            Nama parameter ke nilainya.
        """
        out = {}
        for name in self._get_param_names():
            value = getattr(self, name)
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                for sub_name, sub_value in value.get_params().items():
                    out[f"{name}__{sub_name}"] = sub_value
            out[name] = value
        return out

    def set_params(self, **params) -> "ParamsMixin":
        """Ubah parameter, termasuk objek bersarang lewat ``induk__anak=nilai``.

        Returns
        -------
        self
        """
        if not params:
            return self
        valid = self.get_params(deep=True)
        nested: dict[str, dict] = {}
        for key, value in params.items():
            name, sep, sub_name = key.partition("__")
            if name not in valid:
                raise ValueError(
                    f"{key!r} bukan parameter {type(self).__name__}. "
                    f"Parameter yang ada: {sorted(self._get_param_names())}"
                )
            if sep:
                nested.setdefault(name, {})[sub_name] = value
            else:
                setattr(self, name, value)
        for name, sub_params in nested.items():
            getattr(self, name).set_params(**sub_params)
        return self

    def __repr__(self) -> str:
        """Tampilkan hanya parameter yang berbeda dari nilai bawaan."""
        signature = inspect.signature(type(self).__init__)
        changed = []
        for name in self._get_param_names():
            default = signature.parameters[name].default
            value = getattr(self, name)
            if default is inspect.Parameter.empty or not _same(value, default):
                changed.append(f"{name}={value!r}")
        return f"{type(self).__name__}({', '.join(changed)})"


class BaseSelector(ParamsMixin, ABC):
    """Kelas dasar pemilih strategi perbaikan jadwal.

    Hasil belajar disimpan di atribut berakhiran ``_`` oleh ``fit``. Subkelas wajib
    mengisi ``fit`` dan ``predict``; sisanya punya isi bawaan yang boleh diganti.

    Attributes
    ----------
    operators : dict atau None
        Operator perbaikan khusus pemilih ini (aksi -> operator). ``None`` berarti
        pakai ``repair.default_operators()``.
    mask_fn : callable atau None
        Mask tingkat atas yang harus dipakai lingkungan untuk pemilih ini. ``None``
        berarti ``repair.high_level_mask``.

    Examples
    --------
    >>> from dfjsp.utils import BaseSelector
    >>> class Tetap(BaseSelector):
    ...     def __init__(self, aksi=0):
    ...         self.aksi = aksi
    ...     def fit(self, instances):
    ...         return self
    ...     def predict(self, state, mask):
    ...         return self.aksi
    >>> Tetap(aksi=2)
    Tetap(aksi=2)
    >>> Tetap().set_params(aksi=1).get_params()
    {'aksi': 1}
    >>> Tetap(aksi=2).act(state=None, mask=None)
    2
    """

    operators = None
    mask_fn = None

    @abstractmethod
    def fit(self, instances):
        """Belajar dari ``instances`` (instans latih) dan kembalikan ``self``."""

    @abstractmethod
    def predict(self, state, mask):
        """Pilih satu strategi dari ``state`` dengan ``mask`` aksi yang diizinkan."""

    def predict_proba(self, state, mask) -> np.ndarray:
        """Peluang tiap strategi, larik berisi 4 angka. Bawaan: satu-panas dari ``predict``."""
        proba = np.zeros(4, dtype=float)
        proba[int(self.predict(state, mask))] = 1.0
        return proba

    def reset(self, seed=None) -> None:
        """Siapkan keadaan dalam episode (misalnya keacakan) sebelum episode baru."""

    def act(self, state, mask, env=None) -> int:
        """Strategi (0 sampai 3) yang diberikan ke ``env.step``.

        Pintu tunggal bagi penjalan episode. Pemilih strategi biasa meneruskan ke
        ``predict``. Pemilih yang butuh ``env`` (PPO pasangan) mengganti metode ini.
        """
        return int(self.predict(state, mask))


def _same(a, b) -> bool:
    """Bandingkan dua nilai parameter tanpa galat untuk larik atau objek aneh."""
    if a is b:
        return True
    try:
        return bool(a == b)
    except (ValueError, TypeError):
        return False
