"""Seam B1: the trace of an episode observes the run without changing it.

``trace_episode`` runs ``cek_episode.run`` itself and only watches it, so its result must equal
``run_episode`` for every reported method, and the trace must add up to that result.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from arls_ppo import episode

JOB = ("MK01", 1.5, 0.15, 5000)
FIELDS = ("telat", "ubah", "J", "keputusan", "tingkat")
LEARNED = ("ppo_bc_ppo_g", "ppo_bc_ppo", "ppo_bc_g", "il_g")


@pytest.fixture(scope="module", params=[0.3, 3.0], ids=["lam0.3", "lam3"])
def lam(request):
    return request.param


@pytest.fixture(scope="module")
def runs(lam):
    return {m: (episode.run_episode(*JOB, m, lam), episode.trace_episode(*JOB, m, lam))
            for m in episode.METHODS}


@pytest.mark.parametrize("method", list(episode.METHODS))
def test_trace_gives_the_same_result_as_the_plain_run(runs, method):
    plain, traced = runs[method]
    for field in FIELDS:
        assert traced.result[field] == plain[field], field


@pytest.mark.parametrize("method", list(episode.METHODS))
def test_changes_and_final_tardiness_add_up_to_the_result(runs, method):
    _, tr = runs[method]
    assert len(tr.events) == tr.result["keputusan"]
    assert sum(ev["changes"] for ev in tr.events) == tr.result["ubah"]
    assert tr.events[-1]["tard_after"] == pytest.approx(tr.result["telat"], abs=1e-9)


@pytest.mark.parametrize("method", [m for m in episode.METHODS if m != "aor"])
def test_levels_in_the_trace_match_the_level_counts_of_the_run(runs, method):
    _, tr = runs[method]
    subs = [s for ev in tr.events for s in ev["subs"]]
    used = Counter(s["used"] for s in subs)
    requested = Counter(s["requested"] for s in subs)
    outcome = Counter(s["outcome"] for s in subs)
    counts = tr.result["tingkat"]
    assert [used[lv] for lv in range(4)] == counts["dipakai"]
    assert [requested[lv] for lv in range(4)] == counts["diminta"]
    assert outcome["ditolak_penjaga"] == counts["ditolak_penjaga"]
    assert outcome["gagal"] == counts["gagal"]


def test_aor_never_changes_the_order(runs):
    _, tr = runs["aor"]
    assert all(s["used"] == 0 for ev in tr.events for s in ev["subs"])
    assert tr.result["ubah"] == 0


@pytest.mark.parametrize("method", [m for m in episode.METHODS if m != "aor"])
def test_replayed_value_path_ends_at_the_value_of_each_decision(runs, method, lam):
    _, tr = runs[method]
    for ev in tr.events:
        assert ev["subs"][-1]["J_after"] == ev["J_after_batch"]
        assert ev["J_after_batch"] == pytest.approx(
            ev["tard_after"] + lam * ev["changes"], abs=1e-9)


@pytest.mark.parametrize("method", list(episode.METHODS))
def test_listed_changes_count_exactly_the_changes_of_the_thesis(runs, method):
    """U = tasks moved to another developer + (n - LIS) per developer, listed one by one."""
    _, tr = runs[method]
    for ev in tr.events:
        items = ev["change_items"]
        assert len(items) == ev["changes"]
        assert {i["kind"] for i in items} <= {"assignee", "rank"}
        for i in items:
            if i["kind"] == "assignee":
                assert i["from_dev"] != i["to_dev"]
                assert ev["aor"]["entries"][i["task"]]["dev"] == i["from_dev"]
                assert ev["after"]["entries"][i["task"]]["dev"] == i["to_dev"]
            else:
                assert ev["aor"]["entries"][i["task"]]["dev"] == i["dev"]
                assert ev["after"]["entries"][i["task"]]["dev"] == i["dev"]


@pytest.mark.parametrize("method", LEARNED)
def test_learned_selectors_record_state_mask_and_probabilities(runs, method):
    _, tr = runs[method]
    subs = [s for ev in tr.events for s in ev["subs"]]
    assert subs
    for s in subs:
        assert len(s["features"]) == len(episode.FEATURE_NAMES) == 38
        assert len(s["mask"]) == 3
        p = np.asarray(s["probs"])
        assert p.shape == (4,) and p.sum() == pytest.approx(1.0, abs=1e-5)
        allowed = [1, *s["mask"]]
        best = max((lv for lv in range(4) if allowed[lv]), key=lambda lv: p[lv])
        assert s["requested"] == best


def test_teacher_records_the_value_of_every_level_it_tried(runs):
    _, tr = runs["seq"]
    for ev in tr.events:
        for s in ev["subs"]:
            options = {int(k): v for k, v in s["options"].items()}
            assert 0 in options
            assert s["used"] == min(options, key=lambda lv: (options[lv] - (1e-9 if lv == 0
                                                                           else 0), lv))


def test_snapshots_show_the_schedule_before_after_aor_and_after_repair(runs):
    _, tr = runs["ppo_bc_ppo_g"]
    times = [ev["time"] for ev in tr.events]
    assert times == sorted(times)
    for ev in tr.events:
        assert ev["disruptions"] and all(d["type"] in episode.DISRUPTION_TYPES
                                         for d in ev["disruptions"])
        for key in ("before", "aor", "after"):
            snap = ev[key]
            assert snap["entries"], key
        assert set(ev["after"]["entries"]) == set(ev["aor"]["entries"])
        ids = {t["id"] for t in ev["aor"]["tasks"]}
        assert set(ev["aor"]["entries"]) <= ids
    assert tr.initial["tasks"] and tr.initial["developers"]
