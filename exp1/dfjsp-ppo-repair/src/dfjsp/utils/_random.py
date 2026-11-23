"""Pemeriksaan keacakan: semua keacakan di pustaka lewat satu pintu."""

from __future__ import annotations

import numbers

import numpy as np


def check_random_state(seed=None) -> np.random.Generator:
    """Ubah ``seed`` menjadi ``numpy.random.Generator``.

    Semua keacakan di pustaka ``dfjsp`` lewat fungsi ini, sama seperti
    ``sklearn.utils.check_random_state``, tetapi hasilnya ``Generator`` modern.

    Parameters
    ----------
    seed : None, int, atau numpy.random.Generator
        * ``None``: generator baru dengan seed acak dari sistem.
        * ``int``: generator baru dengan seed itu (hasil bisa diulang).
        * ``Generator``: dikembalikan apa adanya (tidak disalin).

    Returns
    -------
    numpy.random.Generator

    Raises
    ------
    ValueError
        Bila ``seed`` bukan salah satu jenis di atas, atau bilangan bulat negatif.

    Examples
    --------
    >>> from dfjsp.utils import check_random_state
    >>> a = check_random_state(0).integers(0, 100, 3)
    >>> b = check_random_state(0).integers(0, 100, 3)
    >>> bool((a == b).all())
    True
    """
    if seed is None:
        return np.random.default_rng()
    if isinstance(seed, np.random.Generator):
        return seed
    if isinstance(seed, numbers.Integral) and not isinstance(seed, (bool, np.bool_)):
        if seed < 0:
            raise ValueError(f"random_state tidak boleh negatif, diterima {seed!r}")
        return np.random.default_rng(int(seed))
    raise ValueError(
        "random_state harus None, bilangan bulat, atau numpy.random.Generator, "
        f"diterima {seed!r}"
    )
