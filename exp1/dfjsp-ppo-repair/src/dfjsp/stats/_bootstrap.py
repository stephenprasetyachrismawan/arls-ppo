"""Bootstrap bertingkat (stratified) untuk selang kepercayaan, gaya rliable."""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np

from ..utils import check_random_state
from ._aggregate import iqm

#: Jumlah resample bawaan.
DEFAULT_N_RESAMPLES = 10_000
#: 1 - ALPHA = 95% selang kepercayaan.
DEFAULT_ALPHA = 0.05


class BootstrapResult(NamedTuple):
    """Hasil bootstrap: nilai titik dan batas bawah serta atas selang kepercayaan.

    Examples
    --------
    >>> from dfjsp.stats import BootstrapResult
    >>> hasil = BootstrapResult(estimate=1.0, low=0.5, high=1.5)
    >>> estimate, low, high = hasil
    >>> low < estimate < high
    True
    """

    estimate: float
    low: float
    high: float


def as_score_matrix(scores, name: str = "scores") -> np.ndarray:
    """Ubah ``scores`` jadi matriks (jumlah run, jumlah task); deret 1D = satu task."""
    matrix = np.asarray(scores, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        raise ValueError(
            f"{name} harus berdimensi 1 atau 2 (run x task), diterima {matrix.ndim}")
    return matrix


def check_bootstrap_params(n_resamples: int, alpha: float) -> None:
    if int(n_resamples) < 1:
        raise ValueError(f"n_resamples harus >= 1, diterima {n_resamples}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha harus di antara 0 dan 1, diterima {alpha}")


def percentile_interval(boot: np.ndarray, alpha: float) -> tuple[float, float]:
    """Batas bawah dan atas selang persentil dari sampel bootstrap."""
    return (float(np.nanquantile(boot, alpha / 2)),
            float(np.nanquantile(boot, 1.0 - alpha / 2)))


def _label_array(labels, n_tasks: int, name: str) -> np.ndarray:
    """Label per kolom sebagai larik 1D sepanjang ``n_tasks``."""
    array = np.asarray(labels)
    if array.ndim != 1 or array.shape[0] != n_tasks:
        raise ValueError(
            f"{name} harus punya satu label per kolom ({n_tasks}), bentuk diterima {array.shape}")
    return array


def _stack(blocks: list[np.ndarray]) -> np.ndarray:
    """Susun gugus (larik indeks kolom) jadi satu 2D bila ukurannya sama, kalau tidak daftar."""
    if len({len(b) for b in blocks}) == 1:
        return np.stack(blocks)
    return blocks


def make_column_sampler(n_tasks: int, groups=None, strata=None):
    """Fungsi ``rng -> indeks kolom`` yang mengambil ulang gugus utuh dengan pengembalian.

    Tanpa ``groups`` dan ``strata``, tiap kolom adalah gugusnya sendiri (hitungan bootstrap
    task biasa, urutan angka acak sama persis dengan versi tanpa gugus). Dengan ``strata``,
    gugus diambil ulang di dalam strata-nya sendiri dan hasilnya disusun strata demi strata
    (urutan label terurut), jadi ``statistic`` bisa memisahkan blok per strata.

    Raises
    ------
    ValueError
        Panjang label salah, atau satu gugus tersebar di dua strata.
    """
    if groups is None and strata is None:
        return lambda rng: rng.integers(0, n_tasks, size=n_tasks)
    group_labels = np.arange(n_tasks) if groups is None else _label_array(groups, n_tasks, "groups")
    strata_labels = np.zeros(n_tasks, dtype=int) if strata is None \
        else _label_array(strata, n_tasks, "strata")
    layers = []
    for layer in np.unique(strata_labels):
        in_layer = strata_labels == layer
        names = np.unique(group_labels[in_layer])
        layers.append(_stack([np.flatnonzero(in_layer & (group_labels == g)) for g in names]))
    if sum(len(members) for members in layers) != len(np.unique(group_labels)):
        raise ValueError("satu gugus tersebar di lebih dari satu strata; "
                         "gugus harus berada di dalam satu strata saja")

    def sample(rng):
        picked = []
        for members in layers:
            draw = rng.integers(0, len(members), size=len(members))
            if isinstance(members, np.ndarray):
                picked.append(members[draw].ravel())
            else:
                picked.append(np.concatenate([members[i] for i in draw]))
        return np.concatenate(picked)

    return sample


def bootstrap_ci(scores, statistic=iqm, *, n_resamples: int = DEFAULT_N_RESAMPLES,
                 alpha: float = DEFAULT_ALPHA, random_state=None,
                 task_bootstrap: bool = False, groups=None,
                 strata=None) -> BootstrapResult:
    """Selang kepercayaan bootstrap bertingkat untuk ``statistic(scores)``.

    ``scores`` berbentuk (run, task): baris = satu run (misalnya satu seed
    pelatihan), kolom = satu task (misalnya satu kombinasi instans, intensitas, dan
    kapasitas). Tiap resample mengambil ulang run DENGAN pengembalian, terpisah untuk
    tiap task, sehingga struktur per task terjaga (stratified bootstrap, Agarwal dkk.
    2021). Selang diambil dari persentil ``alpha/2`` dan ``1 - alpha/2``.

    Dengan ``task_bootstrap=True`` ada satu tingkat lagi di atasnya: tiap resample
    lebih dulu mengambil ulang task (kolom) DENGAN pengembalian, lalu run diambil
    ulang di dalam tiap task terpilih. Ini bootstrap dua tingkat seperti opsi
    rliable, dan membuat metode dengan 1 run tetap punya selang yang bermakna
    karena ketidakpastiannya datang dari variasi antartask.

    Parameters
    ----------
    scores : array-like, bentuk (n_run, n_task)
        Deret 1D dianggap satu task.
    statistic : fungsi, default=iqm
        Menerima matriks (n_run, n_task) dan mengembalikan satu angka.
    n_resamples : int, default=10000
        Jumlah resample.
    alpha : float, default=0.05
        Selang kepercayaan = ``1 - alpha`` (bawaan 95%).
    random_state : None, int, atau Generator
        Sumber keacakan; lihat ``dfjsp.utils.check_random_state``. Dengan seed dan
        masukan yang sama, hasilnya sama dengan hitungan lama (``rliable_agg``, sudah
        dihapus).
    task_bootstrap : bool, default=False
        Bila True, task juga diambil ulang (lihat atas). Bawaan False memberi angka
        yang sama persis dengan versi sebelumnya.
    groups : array-like dengan satu label per kolom, default=None
        Gugus task yang tidak bebas, misalnya kolom yang melihat urutan gangguan yang
        sama. Dengan ``groups``, tingkat pertama mengambil ulang GUGUS utuh (bukan kolom
        satu-satu), lalu run diambil ulang di dalam tiap kolom terpilih. Selang jadi
        lebih lebar atau sama dibanding tanpa gugus bila kolom segugus saling mirip.
        Butuh ``task_bootstrap=True``. Bawaan ``None``: tiap kolom gugusnya sendiri.
    strata : array-like dengan satu label per kolom, default=None
        Kelompok kolom yang harus tetap terpisah, misalnya instans. Gugus diambil ulang
        di dalam strata-nya sendiri, dan kolom hasil disusun strata demi strata (label
        terurut), jadi ``statistic`` dapat menghitung nilai per strata dari blok kolom.
        Satu gugus tidak boleh tersebar di dua strata. Butuh ``task_bootstrap=True``.

    Returns
    -------
    BootstrapResult
        ``(estimate, low, high)``. Bila hanya ada 1 run dan ``task_bootstrap`` False,
        bootstrap tidak bisa dilakukan: ketiganya sama dan ada ``UserWarning``.

    Raises
    ------
    ValueError
        Bila ``scores`` bukan 1D atau 2D, ``n_resamples`` < 1, ``alpha`` di
        luar (0, 1), ``groups`` atau ``strata`` dipakai tanpa ``task_bootstrap=True``,
        panjang labelnya salah, atau satu gugus tersebar di dua strata.

    Examples
    --------
    >>> import numpy as np
    >>> from dfjsp.stats import bootstrap_ci
    >>> scores = np.array([[10.0, 20.0], [12.0, 22.0], [11.0, 21.0]])
    >>> r = bootstrap_ci(scores, n_resamples=200, random_state=0)
    >>> bool(r.low <= r.estimate <= r.high)
    True

    Dua kolom pertama sama-sama dari satu gugus, jadi diambil ulang bersama:

    >>> r = bootstrap_ci(np.array([[1.0, 1.0, 9.0]]), np.mean, n_resamples=200,
    ...                  random_state=0, task_bootstrap=True, groups=[0, 0, 1])
    >>> r.estimate
    3.6666666666666665
    """
    matrix = as_score_matrix(scores)
    check_bootstrap_params(n_resamples, alpha)
    rng = check_random_state(random_state)
    n_runs, n_tasks = matrix.shape
    if (groups is not None or strata is not None) and not task_bootstrap:
        raise ValueError("groups dan strata hanya bekerja dengan task_bootstrap=True")
    pick_columns = make_column_sampler(n_tasks, groups, strata)
    point = float(statistic(matrix))
    if n_runs < 2 and not task_bootstrap:
        warnings.warn(
            "bootstrap_ci menerima 1 run saja, selang kepercayaan tidak bisa dihitung; "
            "batas bawah dan atas disamakan dengan nilai titik.",
            UserWarning, stacklevel=2)
        return BootstrapResult(point, point, point)
    boot = np.empty(int(n_resamples), dtype=float)
    for r in range(boot.size):
        sample = matrix
        if task_bootstrap:
            # Tingkat pertama: ambil ulang gugus (bawaan: tiap kolom) dengan pengembalian.
            sample = matrix[:, pick_columns(rng)]
        # Tingkat kedua: run dipilih ulang BEBAS untuk tiap task (kolom).
        idx = rng.integers(0, n_runs, size=sample.shape)
        boot[r] = statistic(np.take_along_axis(sample, idx, axis=0))
    low, high = percentile_interval(boot, alpha)
    return BootstrapResult(point, low, high)


def paired_difference_ci(x, y, statistic=iqm, *, n_resamples: int = DEFAULT_N_RESAMPLES,
                         alpha: float = DEFAULT_ALPHA, random_state=None,
                         task_bootstrap: bool = False, groups=None,
                         strata=None) -> BootstrapResult:
    """Selang kepercayaan untuk selisih berpasangan ``x - y``.

    Selisih dihitung per elemen (run dan task yang sama), lalu ``statistic`` dari
    matriks selisih itu diberi selang bootstrap bertingkat. Jadi yang dihitung
    adalah statistik dari selisih (misalnya IQM selisih), BUKAN selisih dua IQM.
    Pasangan harus bermakna: kolom yang sama berarti kondisi evaluasi yang sama
    (instans, intensitas, kapasitas, urutan gangguan yang sama).

    Parameters
    ----------
    x, y : array-like, bentuk sama (n_run, n_task)
    statistic : fungsi, default=iqm
    n_resamples, alpha, random_state, task_bootstrap, groups, strata
        Sama seperti pada :func:`bootstrap_ci`. Gugus yang sama dipakai untuk ``x`` dan
        ``y``, jadi pasangan tidak terpisah.

    Returns
    -------
    BootstrapResult
        Negatif berarti ``x`` lebih kecil dari ``y`` (lebih baik bila ukurannya
        makin kecil makin baik).

    Raises
    ------
    ValueError
        Bila bentuk ``x`` dan ``y`` berbeda.

    Examples
    --------
    >>> import numpy as np
    >>> from dfjsp.stats import paired_difference_ci
    >>> x = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    >>> paired_difference_ci(x, x + 1.0, n_resamples=50, random_state=0)
    BootstrapResult(estimate=-1.0, low=-1.0, high=-1.0)
    """
    a, b = as_score_matrix(x, "x"), as_score_matrix(y, "y")
    if a.shape != b.shape:
        raise ValueError(f"bentuk x {a.shape} dan y {b.shape} harus sama untuk pasangan")
    return bootstrap_ci(a - b, statistic, n_resamples=n_resamples, alpha=alpha,
                        random_state=random_state, task_bootstrap=task_bootstrap,
                        groups=groups, strata=strata)
