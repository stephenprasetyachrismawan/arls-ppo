"""Agregat titik: interquartile mean (IQM)."""

from __future__ import annotations

import numpy as np
from scipy import stats as _sps

#: Bagian yang dibuang di tiap ekor (25%), sehingga yang tersisa separuh tengah.
IQM_TRIM = 0.25


def iqm(scores) -> float:
    """Interquartile mean: rata-rata separuh tengah nilai, tahan pencilan.

    Semua nilai diratakan jadi satu deret, nilai ``NaN`` dibuang, 25% terkecil dan
    25% terbesar dibuang, sisanya dirata-rata (Agarwal dkk. 2021).

    Parameters
    ----------
    scores : array-like
        Nilai apa saja, satu dimensi atau lebih.

    Returns
    -------
    float
        ``nan`` bila tidak ada nilai.

    Examples
    --------
    >>> from dfjsp.stats import iqm
    >>> iqm([1, 2, 3, 4, 5, 6, 7, 8])
    4.5
    >>> iqm([1, 2, 3, 4, 5, 6, 7, 1_000_000])
    4.5
    """
    values = np.asarray(scores, dtype=float).ravel()
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan")
    return float(_sps.trim_mean(values, IQM_TRIM))
