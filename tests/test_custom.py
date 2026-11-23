"""Seam B4: a project typed in by a user goes through the same path as the thesis instances."""
from __future__ import annotations

import pytest

from arls_ppo import custom, episode

SPEC = {
    "key": "TOKO", "name": "Toko daring",
    "developers": [{"name": "Ayu (backend)"}, {"name": "Budi (backend)"},
                   {"name": "Citra (frontend)"}],
    "epics": [
        {"name": "Login", "issues": [
            {"summary": "Rancang API", "efforts": {"0": 3, "1": 4}},
            {"summary": "Halaman masuk", "efforts": {"2": 3}},
        ]},
        {"name": "Pembayaran", "issues": [
            {"summary": "Integrasi gateway", "efforts": {"0": 5, "1": 6}},
            {"summary": "Uji pembayaran", "efforts": {"1": 2, "2": 2.5}},
        ]},
    ],
    # "Rancang API" blocks "Integrasi gateway"
    "links": [{"from": [0, 0], "to": [1, 0]}],
}


def test_the_project_becomes_a_validated_instance():
    project = custom.parse(SPEC)
    inst = custom.build_instance(project)
    assert inst.name == "custom:TOKO"
    assert [t.task_id for t in inst.tasks] == [0, 1, 2, 3]
    assert [t.job_id for t in inst.tasks] == [0, 0, 1, 1]
    assert inst.tasks[3].eligible_devs == {1: 2.0, 2: 2.5}
    assert len(inst.developers) == 3
    assert 0 in inst.tasks[1].deps            # issues inside one epic follow each other
    assert 2 in inst.tasks[3].deps
    assert 0 in inst.tasks[2].deps            # the explicit link
    assert inst.tasks[2].deps.count(0) == 1


def test_labels_name_every_task_developer_and_epic():
    labels = custom.labels(custom.parse(SPEC))
    assert labels["tasks"] == {0: "Rancang API", 1: "Halaman masuk", 2: "Integrasi gateway",
                               3: "Uji pembayaran"}
    assert labels["devs"] == {0: "Ayu (backend)", 1: "Budi (backend)", 2: "Citra (frontend)"}
    assert labels["epics"] == {0: "Login", 1: "Pembayaran"}


def test_due_dates_follow_the_thesis_rule_on_the_initial_schedule():
    """Due date = completion in S0 + half of the effort (Chapter III)."""
    trace = episode.trace_episode(custom.parse(SPEC), 1.5, 0.30, 5000, "aor", 3.0)
    entries = trace.initial["entries"]
    for t in trace.initial["tasks"]:
        e = entries[t["id"]]
        assert t["due"] == pytest.approx(e["end"] + 0.5 * (e["end"] - e["start"]))


def test_a_learned_method_runs_on_the_project_and_repeats_exactly():
    project = custom.parse(SPEC)
    a = episode.run_episode(project, 1.5, 0.30, 5000, "ppo_bc_ppo_g", 3.0)
    b = episode.run_episode(project, 1.5, 0.30, 5000, "ppo_bc_ppo_g", 3.0)
    assert (a["J"], a["ubah"], a["keputusan"]) == (b["J"], b["ubah"], b["keputusan"])


@pytest.mark.parametrize("change, message", [
    (lambda s: s["links"].append({"from": [1, 0], "to": [0, 0]}), "siklus"),
    (lambda s: s["epics"][0]["issues"][0].update(efforts={}), "pengembang"),
    (lambda s: s["epics"][0]["issues"][0].update(efforts={"0": 0}), "upaya"),
    (lambda s: s["epics"][0]["issues"][0].update(efforts={"7": 2}), "pengembang"),
    (lambda s: s.update(key="ab c"), "key"),
])
def test_invalid_projects_are_refused_with_a_clear_reason(change, message):
    import copy

    spec = copy.deepcopy(SPEC)
    change(spec)
    with pytest.raises(ValueError, match=message):
        custom.build_instance(custom.parse(spec))
