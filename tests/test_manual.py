"""Seam B3: the user as the selector, judged by the same J and the same schedule checks."""
from __future__ import annotations

import pytest

from arls_ppo import episode, manual

JOB = ("MK01", 1.5, 0.15, 5000)
LAM = 3.0


@pytest.fixture(scope="module")
def ppo():
    return episode.trace_episode(*JOB, "ppo_bc_ppo_g", LAM)


def test_choosing_level_zero_everywhere_is_aor():
    plain = episode.run_episode(*JOB, "aor", LAM)
    tr = manual.run(*JOB, LAM, levels={})
    assert (tr.result["J"], tr.result["ubah"], tr.result["keputusan"]) == \
        (plain["J"], plain["ubah"], plain["keputusan"])


def test_choosing_what_arls_ppo_chose_with_the_guard_gives_its_result(ppo):
    levels = {(ev["index"], s["j"]): s["requested"] for ev in ppo.events for s in ev["subs"]}
    tr = manual.run(*JOB, LAM, levels=levels, guard=True)
    for field in ("telat", "ubah", "J", "keputusan"):
        assert tr.result[field] == ppo.result[field], field


def test_every_sub_decision_shows_the_value_of_each_level(ppo):
    tr = manual.run(*JOB, LAM, levels={}, options=True)
    for ev in tr.events:
        for s in ev["subs"]:
            options = {int(k): v for k, v in s["options"].items()}
            assert options[0] == pytest.approx(s["J_before"])
            assert set(options) <= {0, 1, 2, 3}
            assert options[s["used"]] == pytest.approx(s["J_after"])


def _a_waiting_task_with_two_developers(tr):
    for ev in tr.events:
        statuses = {t["id"]: t for t in ev["aor"]["tasks"]}
        for tid, e in ev["aor"]["entries"].items():
            t = statuses[tid]
            if t["status"] == "not_started" and len(t["eligible"]) > 1:
                others = [d for d in t["eligible"] if d != e["dev"]]
                yield ev["index"], tid, others, t


def test_a_valid_card_move_is_counted_and_valued_like_the_thesis():
    aor = manual.run(*JOB, LAM, levels={})
    for k, tid, others, _ in _a_waiting_task_with_two_developers(aor):
        for dev in others:
            try:
                tr = manual.run(*JOB, LAM, moves={k: [{"task": tid, "to_dev": dev,
                                                       "to_pos": 1}]})
            except manual.MoveRejected:
                continue
            ev = tr.events[k]
            assert ev["after"]["entries"][tid]["dev"] == dev
            assert ev["changes"] >= 1
            assert len(ev["change_items"]) == ev["changes"]
            assert ev["J_after_batch"] == pytest.approx(ev["tard_after"] + LAM * ev["changes"])
            assert tr.result["ubah"] == sum(e["changes"] for e in tr.events)
            assert tr.result["periksa"]["hitung"].get("sesudah:A3:keras", 0) == 0
            return
    pytest.fail("no feasible move found")


def test_a_move_to_a_developer_without_the_skill_is_refused():
    aor = manual.run(*JOB, LAM, levels={})
    k, tid, _, task = next(_a_waiting_task_with_two_developers(aor))
    unskilled = next(d["id"] for d in aor.initial["developers"] if d["id"] not in task["eligible"])
    with pytest.raises(manual.MoveRejected, match="keahlian") as err:
        manual.run(*JOB, LAM, moves={k: [{"task": tid, "to_dev": unskilled}]})
    assert err.value.rule == "A3"


def test_a_started_task_cannot_be_moved():
    aor = manual.run(*JOB, LAM, levels={})
    ev = aor.events[3]
    started = next(t["id"] for t in ev["aor"]["tasks"]
                   if t["status"] in ("in_progress", "completed") and t["id"] in ev["aor"]["entries"])
    with pytest.raises(manual.MoveRejected, match="To Do"):
        manual.run(*JOB, LAM, moves={3: [{"task": started, "to_dev": 0}]})
