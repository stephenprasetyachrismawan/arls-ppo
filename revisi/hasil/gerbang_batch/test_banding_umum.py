"""Perbandingan berpasangan dua metode dengan bootstrap bergugus (``banding_umum.bandingkan``).

Jalankan dari folder ini::

    PYTHONPATH=../../../exp1/dfjsp-ppo-repair/src $PY -m pytest -p no:cacheprovider -q test_banding_umum.py
"""
from __future__ import annotations

import pandas as pd
import pytest

from banding_umum import bandingkan


def _baris(metode, J, instans="MK01", kapasitas=1.0, intensitas=0.05, seed=5000, lam=3.0):
    return dict(metode=metode, instans=instans, kapasitas=kapasitas, intensitas=intensitas,
                seed=seed, lam=lam, J=J)


def test_method_always_ten_percent_worse_gives_plus_ten_with_both_measures():
    baris = []
    for k, jb in enumerate([10.0, 50.0, 200.0, 7.5]):
        baris += [_baris("B", jb, seed=5000 + k), _baris("A", jb * 1.1, seed=5000 + k)]
    hasil = bandingkan(pd.DataFrame(baris), "A", "B", n_resamples=200)[3.0]
    for ukuran in (hasil.rasio_rata, hasil.rasio_total):
        assert ukuran.estimate == pytest.approx(10.0)
        assert ukuran.low == pytest.approx(10.0)
        assert ukuran.high == pytest.approx(10.0)


def _lebar(r):
    return r.high - r.low


def test_capacities_sharing_a_disruption_sequence_are_resampled_together():
    # 12 gugus (seed berbeda), tiap gugus 3 kapasitas dengan selisih yang sama persis.
    # Versi bebas: 36 kondisi dengan nilai yang sama, tetapi tiap kondisi seed-nya sendiri.
    # Gugus 3 kembar hanya berisi 12 informasi bebas, jadi selangnya kira-kira akar 3 kali
    # lebih lebar daripada 36 kondisi bebas.
    bergugus, bebas = [], []
    for k in range(12):
        for c, cap in enumerate((1.0, 1.5, 2.0)):
            ja = 100.0 * (1 + 0.02 * k)
            bergugus += [_baris("B", 100.0, kapasitas=cap, seed=5000 + k),
                         _baris("A", ja, kapasitas=cap, seed=5000 + k)]
            bebas += [_baris("B", 100.0, kapasitas=cap, seed=6000 + 3 * k + c),
                      _baris("A", ja, kapasitas=cap, seed=6000 + 3 * k + c)]
    g = bandingkan(pd.DataFrame(bergugus), "A", "B", n_resamples=4000)[3.0]
    s = bandingkan(pd.DataFrame(bebas), "A", "B", n_resamples=4000)[3.0]
    assert g.rasio_rata.estimate == pytest.approx(s.rasio_rata.estimate)
    assert _lebar(g.rasio_rata) > 1.4 * _lebar(s.rasio_rata)
    assert _lebar(g.rasio_total) > 1.4 * _lebar(s.rasio_total)


def test_condition_missing_for_one_method_is_an_error_not_silently_dropped():
    df = pd.DataFrame([_baris("A", 11.0, seed=5000), _baris("B", 10.0, seed=5000),
                       _baris("A", 12.0, seed=5001)])
    with pytest.raises(ValueError, match="kondisi"):
        bandingkan(df, "A", "B", n_resamples=10)


def test_zero_reference_J_is_an_error_because_the_ratio_is_undefined():
    df = pd.DataFrame([_baris("A", 1.0, seed=5000), _baris("B", 0.0, seed=5000)])
    with pytest.raises(ValueError, match="J = 0"):
        bandingkan(df, "A", "B", n_resamples=10)


def test_the_two_measures_differ_as_computed_by_hand():
    # Kondisi kecil: A dua kali B (+100%). Kondisi besar: sama (0%).
    # Rata-rata rasio = (100 + 0) / 2 = +50%. Rasio total = 102 / 101 - 1 = +0,990...%.
    df = pd.DataFrame([_baris("B", 1.0, seed=5000), _baris("A", 2.0, seed=5000),
                       _baris("B", 100.0, seed=5001), _baris("A", 100.0, seed=5001)])
    hasil = bandingkan(df, "A", "B", n_resamples=10)[3.0]
    assert hasil.rasio_rata.estimate == pytest.approx(50.0)
    assert hasil.rasio_total.estimate == pytest.approx(100 / 101)


def test_several_training_seeds_of_one_method_are_averaged_per_condition_first():
    # Dua model PPO (seed latih berbeda) di kondisi yang sama: J 9 dan 13, rata-rata 11.
    df = pd.DataFrame([_baris("A", 9.0), _baris("A", 13.0), _baris("B", 10.0)])
    assert bandingkan(df, "A", "B", n_resamples=10)[3.0].rasio_rata.estimate == pytest.approx(10.0)


def test_each_lambda_is_compared_separately():
    df = pd.DataFrame([_baris("A", 11.0, lam=0.3), _baris("B", 10.0, lam=0.3),
                       _baris("A", 15.0, lam=30.0), _baris("B", 10.0, lam=30.0)])
    hasil = bandingkan(df, "A", "B", n_resamples=10)
    assert hasil[0.3].rasio_rata.estimate == pytest.approx(10.0)
    assert hasil[30.0].rasio_rata.estimate == pytest.approx(50.0)


# --- bandingkan_celah: celah a ke b di λ sasaran dikurangi celah di λ acuan --------------

from banding_umum import bandingkan_celah  # noqa: E402


def test_gap_twenty_percent_at_target_and_ten_at_reference_gives_plus_ten_points():
    baris = []
    for k, jb in enumerate([10.0, 50.0, 200.0, 7.5]):
        for lam, celah in ((1.0, 0.20), (3.0, 0.10)):
            baris += [_baris("B", jb, seed=5000 + k, lam=lam),
                      _baris("A", jb * (1 + celah), seed=5000 + k, lam=lam)]
    hasil = bandingkan_celah(pd.DataFrame(baris), "A", "B", {1.0: 1.0}, {3.0: 1.0},
                             n_resamples=200)
    for ukuran in (hasil.rasio_rata, hasil.rasio_total):
        assert ukuran.estimate == pytest.approx(10.0)
        assert ukuran.low == pytest.approx(10.0)
        assert ukuran.high == pytest.approx(10.0)


def _dua_lam(celah_sasaran, celah_acuan):
    # J pembanding 100 di semua kondisi, jadi kedua ukuran sama dengan rata-rata celah.
    baris = []
    for k, (cs, ca) in enumerate(zip(celah_sasaran, celah_acuan)):
        for lam, celah in ((1.0, cs), (3.0, ca)):
            baris += [_baris("B", 100.0, seed=5000 + k, lam=lam),
                      _baris("A", 100.0 * (1 + celah), seed=5000 + k, lam=lam)]
    return pd.DataFrame(baris)


def test_interval_widens_when_the_gap_difference_varies_between_clusters():
    # Selisih celah per gugus: 0, 2, 4, ..., 22 poin; rata-rata 11.
    df = _dua_lam([0.02 * k for k in range(12)], [0.0] * 12)
    hasil = bandingkan_celah(df, "A", "B", {1.0: 1.0}, {3.0: 1.0}, n_resamples=2000)
    for ukuran in (hasil.rasio_rata, hasil.rasio_total):
        assert ukuran.estimate == pytest.approx(11.0)
        assert 5.0 < ukuran.low < 11.0 < ukuran.high < 17.0


def test_all_lambdas_are_resampled_with_the_same_clusters():
    # Celah naik-turun antar gugus, tetapi di tiap gugus celah sasaran = celah acuan + 5 poin.
    # Kalau tiap λ diambil ulang sendiri-sendiri, selangnya lebar; kalau bersama, tepat 5.
    acuan = [0.03 * k for k in range(12)]
    df = _dua_lam([c + 0.05 for c in acuan], acuan)
    hasil = bandingkan_celah(df, "A", "B", {1.0: 1.0}, {3.0: 1.0}, n_resamples=2000)
    for ukuran in (hasil.rasio_rata, hasil.rasio_total):
        assert ukuran.low == pytest.approx(5.0)
        assert ukuran.high == pytest.approx(5.0)


def test_conditions_that_differ_between_lambdas_are_an_error_not_silently_paired():
    # Jumlah kondisi sama (2), tetapi λ acuan memakai seed lain.
    baris = []
    for lam, seeds in ((1.0, (5000, 5001)), (3.0, (5000, 5002))):
        for s in seeds:
            baris += [_baris("B", 10.0, seed=s, lam=lam), _baris("A", 11.0, seed=s, lam=lam)]
    with pytest.raises(ValueError, match="kondisi"):
        bandingkan_celah(pd.DataFrame(baris), "A", "B", {1.0: 1.0}, {3.0: 1.0}, n_resamples=10)


def test_condition_missing_for_one_method_inside_a_lambda_is_an_error_for_gaps_too():
    baris = [_baris("A", 11.0, seed=5000, lam=1.0), _baris("B", 10.0, seed=5000, lam=1.0),
             _baris("A", 12.0, seed=5001, lam=1.0),
             _baris("A", 11.0, seed=5000, lam=3.0), _baris("B", 10.0, seed=5000, lam=3.0),
             _baris("A", 12.0, seed=5001, lam=3.0), _baris("B", 10.0, seed=5001, lam=3.0)]
    with pytest.raises(ValueError, match="kondisi"):
        bandingkan_celah(pd.DataFrame(baris), "A", "B", {1.0: 1.0}, {3.0: 1.0}, n_resamples=10)


def test_reference_lambdas_are_combined_with_their_weights():
    baris = []
    for k in range(4):
        for lam, celah in ((10.0, 0.25), (3.0, 0.10), (30.0, 0.30)):
            baris += [_baris("B", 100.0, seed=5000 + k, lam=lam),
                      _baris("A", 100.0 * (1 + celah), seed=5000 + k, lam=lam)]
    hasil = bandingkan_celah(pd.DataFrame(baris), "A", "B", {10.0: 1.0}, {3.0: 0.5, 30.0: 0.5},
                             n_resamples=50)
    assert hasil.rasio_rata.estimate == pytest.approx(5.0)
    assert hasil.rasio_total.estimate == pytest.approx(5.0)
