"""Seam A2: in this repository layout the episodes reproduce the stored E15 validation results.

Same requirement as section 10 of the test lock document: on the validation seeds the result
must equal ``e15/ep_*.json`` exactly (difference 0.0). CI runs a sample on MK01; the full grid
runs locally with ``python -m arls_ppo.parity``.
"""
from __future__ import annotations

import json

import pytest

from arls_ppo import SCRIPTS, episode

STORED = {"seq": "ep_aturan", "fix3_g": "ep_aturan", "acak_g": "ep_aturan", "il_g": "ep_hgb7d",
          "ppo_bc_ppo_g": "ep_e9c_s{k}", "ppo_bc_ppo": "ep_e9c_s{k}", "ppo_bc_g": "ep_e9c_s{k}"}
JOB = ("MK01", 1.5, 0.15, 5000)


def _stored(method, lam, model_seed=0):
    name = STORED[method].format(k=model_seed)
    rows = json.loads((SCRIPTS / "e15" / f"{name}.json").read_text())
    key = (*JOB, method, lam)
    found = [r for r in rows if (r["instans"], r["kapasitas"], r["intensitas"], r["seed"],
                                 r["metode"], r["lam"]) == key]
    assert len(found) == 1, key
    return found[0]


def _same(result, stored):
    for field in ("telat", "ubah", "J", "keputusan"):
        assert result[field] == stored[field], field


@pytest.mark.parametrize("lam", [0.3, 3.0])
@pytest.mark.parametrize("method", sorted(STORED))
def test_episode_equals_stored_validation_result(method, lam):
    _same(episode.run_episode(*JOB, method, lam), _stored(method, lam))


@pytest.mark.parametrize("model_seed", [1, 2])
def test_other_ppo_model_seeds_equal_their_stored_results(model_seed):
    result = episode.run_episode(*JOB, "ppo_bc_ppo_g", 3.0, model_seed=model_seed)
    _same(result, _stored("ppo_bc_ppo_g", 3.0, model_seed))
