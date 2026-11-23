"""Seam A2 (full grid tool): ``arls_ppo.parity`` compares stored validation records."""
from __future__ import annotations

from arls_ppo import parity


def test_stored_records_cover_the_whole_validation_grid():
    records = parity.stored_records()
    # 7 MK x 3 capacities x 3 intensities x 2 seeds x 6 λ = 756 jobs per method
    per_method = {}
    for r in records:
        per_method[(r["metode"], r["model_seed"])] = per_method.get(
            (r["metode"], r["model_seed"]), 0) + 1
    assert set(per_method.values()) == {756}
    assert len(per_method) == 3 + 1 + 3 * 3   # aturan, boosting, 3 PPO methods x 3 seeds


def test_check_reports_no_difference_on_a_small_sample():
    sample = [r for r in parity.stored_records()
              if r["instans"] == "MK01" and r["kapasitas"] == 1.0 and r["intensitas"] == 0.05
              and r["seed"] == 5001 and r["lam"] == 10.0]
    assert len(sample) == 13
    report = parity.check(sample, processes=1)
    assert report["episodes"] == 13
    assert report["mismatches"] == []
    assert report["max_abs_diff_J"] == 0.0
