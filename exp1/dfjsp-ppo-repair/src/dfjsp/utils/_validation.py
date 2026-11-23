"""Pemeriksaan bahwa sebuah objek sudah dilatih (fit)."""

from __future__ import annotations


class NotFittedError(ValueError, AttributeError):
    """Dilempar bila objek dipakai sebelum ``fit`` dipanggil.

    Turunan ``ValueError`` dan ``AttributeError`` supaya perilakunya sama dengan
    ``sklearn.exceptions.NotFittedError``.

    Examples
    --------
    >>> from dfjsp.utils import NotFittedError
    >>> issubclass(NotFittedError, ValueError)
    True
    """


def _fitted_names(obj) -> list[str]:
    """Nama atribut hasil belajar: berakhiran ``_`` dan tidak berawalan ``__``."""
    return [name for name in vars(obj)
            if name.endswith("_") and not name.startswith("__")]


def check_is_fitted(obj, attributes=None) -> None:
    """Pastikan ``obj`` sudah dilatih, kalau belum lempar ``NotFittedError``.

    Aturannya sama dengan scikit-learn: objek dianggap sudah dilatih bila punya
    atribut yang namanya berakhiran ``_`` (misalnya ``policy_``), karena ``fit``
    yang menciptakan atribut itu.

    Parameters
    ----------
    obj : object
        Objek yang diperiksa.
    attributes : str atau daftar str, default=None
        Bila diisi, yang diperiksa hanya atribut bernama itu. Bila ``None``,
        cukup ada satu atribut berakhiran ``_``.

    Raises
    ------
    NotFittedError
        Bila belum dilatih.

    Examples
    --------
    >>> from dfjsp.utils import NotFittedError, check_is_fitted
    >>> class Model:
    ...     def fit(self):
    ...         self.coef_ = 1.0
    ...         return self
    >>> check_is_fitted(Model().fit())
    >>> try:
    ...     check_is_fitted(Model())
    ... except NotFittedError as err:
    ...     print("belum dilatih")
    belum dilatih
    """
    if attributes is None:
        fitted = bool(_fitted_names(obj))
    else:
        names = [attributes] if isinstance(attributes, str) else list(attributes)
        fitted = all(hasattr(obj, name) for name in names)
    if not fitted:
        raise NotFittedError(
            f"{type(obj).__name__} belum dilatih. Panggil 'fit' dulu sebelum "
            "memakai metode ini."
        )
