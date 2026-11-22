"""Seam A3: the episode API cannot open test data (instances or episode seeds)."""
from __future__ import annotations

import pytest

from arls_ppo import episode


@pytest.mark.parametrize("instance", ["MK03", "MK04", "MK09"])
def test_test_instances_of_brandimarte_are_refused(instance):
    with pytest.raises(PermissionError, match="data uji|test data"):
        episode.run_episode(instance, 1.5, 0.15, 5000, "aor", 3.0)


def test_benchmark_test_instances_are_refused():
    with pytest.raises(PermissionError):
        episode.run_episode("brandimarte/mk11", 1.5, 0.15, 5000, "aor", 3.0)


@pytest.mark.parametrize("seed", [10000, 10009, 11000, 11004])
def test_test_episode_seeds_are_refused(seed):
    with pytest.raises(PermissionError, match="seed"):
        episode.run_episode("MK01", 1.5, 0.15, seed, "aor", 3.0)


def test_only_the_eight_reported_methods_are_offered():
    assert set(episode.METHODS) == {"ppo_bc_ppo_g", "ppo_bc_ppo", "ppo_bc_g", "aor", "seq",
                                    "fix3_g", "acak_g", "il_g"}
    with pytest.raises(ValueError, match="b1"):
        episode.run_episode("MK01", 1.5, 0.15, 5000, "b1", 3.0)
