"""Peluang perbaikan (probability of improvement), gaya rliable."""

from __future__ import annotations

import numpy as np

from ..utils import check_random_state
from ._bootstrap import (
    DEFAULT_ALPHA,
    DEFAULT_N_RESAMPLES,
    BootstrapResult,
    as_score_matrix,
    check_bootstrap_params,
    percentile_interval,
)


def _check_same_tasks(x: np.ndarray, y: np.ndarray) -> None:
    if x.shape[1] != y.shape[1]:
        raise ValueError(
            f"jumlah task x ({x.shape[1]}) dan y ({y.shape[1]}) harus sama")


def _poi(x: np.ndarray, y: np.ndarray, lower_is_better: bool) -> float:
    """P(X lebih baik dari Y) per task, dirata-rata lintas task; NaN dibuang per task."""
    diff = x[:, None, :] - y[None, :, :]                      # (nx, ny, task)
    win = (diff < 0) if lower_is_better else (diff > 0)
    ties = diff == 0
    n_pairs = (~np.isnan(x)).sum(axis=0) * (~np.isnan(y)).sum(axis=0)
    ok = n_pairs > 0
    if not ok.any():
        return float("nan")
    score = win.sum(axis=(0, 1)) + 0.5 * ties.sum(axis=(0, 1))
    return float((score[ok] / n_pairs[ok]).mean())


def probability_of_improvement(x, y, *, lower_is_better: bool = True) -> float:
    """Peluang ``x`` lebih baik dari ``y``, dirata-rata lintas task.

    Untuk tiap task, semua pasangan (satu nilai ``x``, satu nilai ``y``) dibandingkan.
    Menang dihitung 1, seri 0,5, kalah 0. Nilai 0,5 berarti tidak ada beda, di atas
    0,5 berarti ``x`` lebih baik (Agarwal dkk. 2021).

    Parameters
    ----------
    x, y : array-like, bentuk (n_run, n_task)
        Jumlah run boleh berbeda, jumlah task harus sama. Deret 1D = satu task.
        Nilai ``NaN`` dibuang per task.
    lower_is_better : bool, default=True
        Benar bila ukuran makin kecil makin baik (keterlambatan, nervousness).

    Returns
    -------
    float
        Di antara 0 dan 1; ``nan`` bila tidak ada pasangan yang bisa dibandingkan.

    Raises
    ------
    ValueError
        Bila jumlah task ``x`` dan ``y`` berbeda.

    Examples
    --------
    >>> from dfjsp.stats import probability_of_improvement
    >>> probability_of_improvement([1, 2], [2, 3])
    0.875
    >>> probability_of_improvement([1, 2], [5, 6], lower_is_better=False)
    0.0
    """
    a, b = as_score_matrix(x, "x"), as_score_matrix(y, "y")
    _check_same_tasks(a, b)
    return _poi(a, b, lower_is_better)


def probability_of_improvement_ci(x, y, *, lower_is_better: bool = True,
                                  n_resamples: int = DEFAULT_N_RESAMPLES,
                                  alpha: float = DEFAULT_ALPHA,
                                  random_state=None) -> BootstrapResult:
    """Selang kepercayaan bootstrap bertingkat untuk peluang perbaikan.

    Run ``x`` dan run ``y`` diambil ulang terpisah dan bebas untuk tiap task, karena
    keduanya hasil pelatihan yang berbeda (seperti pada rliable).

    Parameters
    ----------
    x, y, lower_is_better
        Sama seperti pada :func:`probability_of_improvement`.
    n_resamples, alpha, random_state
        Sama seperti pada ``bootstrap_ci``.

    Returns
    -------
    BootstrapResult
        ``(estimate, low, high)``; ``estimate`` sama dengan
        ``probability_of_improvement(x, y)``.

    Examples
    --------
    >>> import numpy as np
    >>> from dfjsp.stats import probability_of_improvement_ci
    >>> x = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    >>> r = probability_of_improvement_ci(x, x + 10.0, n_resamples=50, random_state=0)
    >>> (r.estimate, r.low, r.high)
    (1.0, 1.0, 1.0)
    """
    a, b = as_score_matrix(x, "x"), as_score_matrix(y, "y")
    _check_same_tasks(a, b)
    check_bootstrap_params(n_resamples, alpha)
    rng = check_random_state(random_state)
    point = _poi(a, b, lower_is_better)
    boot = np.empty(int(n_resamples), dtype=float)
    for r in range(boot.size):
        ia = rng.integers(0, a.shape[0], size=a.shape)
        ib = rng.integers(0, b.shape[0], size=b.shape)
        boot[r] = _poi(np.take_along_axis(a, ia, axis=0),
                       np.take_along_axis(b, ib, axis=0), lower_is_better)
    low, high = percentile_interval(boot, alpha)
    return BootstrapResult(point, low, high)
