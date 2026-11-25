"""Seam A4: the number log of a traced episode (``trace_episode(..., explain=True)``).

The log explains where every number of an episode comes from: the input (S0, due dates, E),
the disruptions drawn from the episode seed, and for every sub-decision the 38 state
numbers, the network of the learned selector, the value J of every level tried, and the
guard. These tests check the log against independent sources of truth: the plain run, the
numbers the selector really read, the thesis functions ``score`` and ``_lis_length``, and
the worked example of Bab IV 4.6 (``fixtures/kejadian_contoh.json``, generated for the
thesis by a separate script).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from arls_ppo import episode

JOB = ("MK01", 1.5, 0.15, 5000)
FIELDS = ("telat", "ubah", "J", "keputusan", "tingkat")


@pytest.fixture(scope="module", params=[0.3, 3.0], ids=["lam0.3", "lam3"])
def lam(request):
    return request.param


@pytest.fixture(scope="module")
def explained(lam):
    return {m: (episode.run_episode(*JOB, m, lam),
                episode.trace_episode(*JOB, m, lam, explain=True))
            for m in episode.METHODS}


@pytest.mark.parametrize("method", list(episode.METHODS))
def test_the_number_log_does_not_change_the_result(explained, method):
    plain, traced = explained[method]
    for field in FIELDS:
        assert traced.result[field] == plain[field], field


#: Methods whose selector reads the 38 numbers (AOR has no selector; the teacher tries every
#: level instead of reading them, so its numbers are explained but not recorded as ``x``).
READERS = [m for m in episode.METHODS if m not in ("aor", "seq")]


@pytest.mark.parametrize("method", READERS)
def test_every_explained_state_number_is_the_number_the_selector_read(explained, method):
    _, tr = explained[method]
    n = 0
    for ev in tr.events:
        for sub in ev["subs"]:
            items = sub["explain"]["features"]
            assert [it["name"] for it in items] == list(episode.FEATURE_NAMES)
            assert all(it["formula"] for it in items)
            assert ([np.float32(it["value"]) for it in items]
                    == [np.float32(v) for v in sub["features"]])
            n += 1
    assert n > 0


SELECTORS = [m for m in episode.METHODS if m != "aor"]


def _breakdowns(tr):
    for ev in tr.events:
        for sub in ev["subs"]:
            for lv, b in sub["explain"]["levels"].items():
                yield sub, int(lv), b


@pytest.mark.parametrize("method", SELECTORS)
def test_explained_value_of_every_level_adds_up_to_its_J(explained, method, lam):
    from dfjsp.repair.ladder import _lis_length

    _, tr = explained[method]
    n = 0
    for sub, lv, b in _breakdowns(tr):
        if not b["feasible"]:
            continue
        assert b["tard"] == pytest.approx(sum(r["telat"] for r in b["late"]), abs=1e-9)
        assert b["md"] == len(b["moved_dev"])
        assert b["mp"] == sum(q["n"] - q["lis"] for q in b["queues"])
        for q in b["queues"]:
            assert q["lis"] == _lis_length(q["posisi_lama"])
            assert q["n"] == len(q["posisi_lama"])
        assert b["J"] == b["tard"] / b["E"] + lam * (b["md"] + b["mp"])
        n += 1
    assert n > 0


@pytest.mark.parametrize("method", SELECTORS)
def test_explained_values_are_the_values_the_episode_used(explained, method):
    _, tr = explained[method]
    for ev in tr.events:
        for sub in ev["subs"]:
            levels = {int(k): v for k, v in sub["explain"]["levels"].items()}
            assert levels[0]["J"] == sub["J_before"]
            if sub.get("J_candidate") is not None:
                assert levels[sub["requested"]]["J"] == sub["J_candidate"]
            for lv, value in (sub.get("options") or {}).items():
                assert levels[int(lv)]["J"] == value
            assert levels[sub["used"]]["J"] == sub["J_after"]


PPO = ("ppo_bc_ppo_g", "ppo_bc_ppo", "ppo_bc_g")


@pytest.mark.parametrize("method", PPO)
def test_network_steps_reproduce_the_probabilities_and_the_requested_level(explained, method):
    _, tr = explained[method]
    n = 0
    for ev in tr.events:
        for sub in ev["subs"]:
            s = sub["explain"]["selector"]
            assert s["kind"] == "ppo"
            x = np.asarray(sub["features"], dtype=np.float32)
            obs = np.clip((x - np.asarray(s["mu"], np.float32)) / np.asarray(s["sd"], np.float32),
                          -5, 5).astype(np.float32)
            assert np.array_equal(obs, np.asarray(s["obs"], dtype=np.float32))
            assert len(s["h1"]) == len(s["h2"]) == 128
            allowed = [1] + list(sub["mask"])
            z = np.asarray(s["logits"], dtype=float)
            zmax = max(z[lv] for lv in range(4) if allowed[lv])
            e = [math.exp(z[lv] - zmax) if allowed[lv] else 0.0 for lv in range(4)]
            p = [v / sum(e) for v in e]
            assert p == pytest.approx(sub["probs"], abs=1e-6)
            assert p == pytest.approx(s["probs"], abs=1e-12)
            assert int(np.argmax(p)) == sub["requested"]
            n += 1
    assert n > 0


def test_boosting_il_explains_its_class_probabilities(explained):
    _, tr = explained["il_g"]
    for ev in tr.events:
        for sub in ev["subs"]:
            s = sub["explain"]["selector"]
            assert s["kind"] == "boosting"
            assert s["probs"] == pytest.approx(sub["probs"], abs=1e-12)
            allowed = [1] + list(sub["mask"])
            best = max(range(4), key=lambda lv: s["probs"][lv] if allowed[lv] else -1.0)
            assert best == sub["requested"]


@pytest.mark.parametrize("method", ["fix3_g", "acak_g"])
def test_rule_selectors_explain_the_allowed_levels(explained, method):
    _, tr = explained[method]
    for ev in tr.events:
        for sub in ev["subs"]:
            s = sub["explain"]["selector"]
            assert s["kind"] == "rule"
            assert s["allowed"] == [lv for lv in range(4) if ([1] + list(sub["mask"]))[lv]]
            assert sub["requested"] in s["allowed"]
            if method == "fix3_g":
                assert sub["requested"] == (3 if 3 in s["allowed"] else 0)


def test_the_worked_example_of_bab_iv_is_reproduced_number_by_number():
    """Bab IV 4.6: MK01, capacity 1.5, intensity 0.15, validation seed 5000, λ = 3, ARLS-PPO
    model seed 0. The example event is the first event (four unplanned tasks)."""
    import json
    from pathlib import Path

    ref = json.loads((Path(__file__).parent / "fixtures" / "kejadian_contoh.json").read_text())
    tr = episode.trace_episode(ref["instans"], ref["kapasitas"], ref["intensitas"], ref["seed"],
                               "ppo_bc_ppo_g", ref["lam"], model_seed=0, explain=True)
    assert len(tr.events) == ref["n_kejadian_episode"]
    ev = tr.events[ref["nomor_kejadian"] - 1]
    assert ev["time"] == pytest.approx(ref["t"], abs=1e-12)
    assert tr.meta["mean_effort"] == pytest.approx(ref["E"], abs=1e-12)
    assert ev["tard_aor"] == pytest.approx(ref["j_aor"], abs=1e-12)
    assert len(ev["subs"]) == ref["k"]
    for sub, rs in zip(ev["subs"], ref["sub"]):
        assert sub["type"] == rs["jenis"]
        assert len(sub["affected"]) == rs["nA"]
        assert [1] + sub["mask"] == rs["mask"]      # the example lists level 0 too
        assert [np.float32(v) for v in sub["features"]] == [np.float32(v) for v in rs["x"]]
        s = sub["explain"]["selector"]
        assert [np.float32(v) for v in s["obs"]] == [np.float32(v) for v in rs["obs"]]
        assert s["logits"] == pytest.approx(rs["logit"], abs=1e-6)
        assert s["probs"] == pytest.approx(rs["p"], abs=1e-7)
        assert sub["requested"] == rs["diminta"] and sub["used"] == rs["dipakai"]
        levels = sub["explain"]["levels"]
        # the example lists the levels the mask allows; None = the operator found no repair
        for k, b in rs["semua_tingkat"].items():
            mine = levels[int(k)]
            assert mine["feasible"] == (b is not None)
            if b is None:
                continue
            assert mine["J"] == pytest.approx(b["J"], abs=1e-12)
            if int(k) > 0:          # the example writes no change counts for level 0
                assert (mine["md"], mine["mp"]) == (b["md"], b["mp"])
        assert sub["J_after"] == pytest.approx(rs["j_sesudah"], abs=1e-12)
        assert sub["changes_after"] == rs["ubah_kumulatif"]
    # the queue of the first sub-decision: new task 55 moves from position 10 to 1 on dev 5
    used = ev["subs"][0]["explain"]["levels"][ev["subs"][0]["used"]]
    (q,) = [q for q in used["queues"] if q["dev"] == ref["sub"][0]["antrean"][0]["dev"]]
    rq = ref["sub"][0]["antrean"][0]
    assert (q["lama"], q["baru"], q["posisi_lama"], q["lis"]) == \
        (rq["lama"], rq["baru"], rq["posisi_lama"], rq["lis"])
    assert q["pindah_posisi"] == [55]
    assert ev["J_after_batch"] == pytest.approx(ref["j_akhir"], abs=1e-12)
    assert ev["changes"] == ref["ubah"]


@pytest.mark.parametrize("method", SELECTORS)
def test_the_decision_of_every_sub_decision_is_written_out(explained, method):
    _, tr = explained[method]
    for ev in tr.events:
        for sub in ev["subs"]:
            d = sub["explain"]["decision"]
            levels = {int(k): v for k, v in sub["explain"]["levels"].items()}
            assert (d["requested"], d["used"]) == (sub["requested"], sub["used"])
            if method == "seq":
                ok = {lv: b["J"] for lv, b in levels.items() if b["feasible"]}
                best = min(ok, key=lambda lv: (ok[lv] - (1e-9 if lv == 0 else 0), lv))
                assert d["rule"] == "teacher" and sub["used"] == best
            elif sub["requested"] == 0:
                assert d["rule"] == "tingkat_0" and sub["used"] == 0
            elif not levels[sub["requested"]]["feasible"]:
                assert d["rule"] == "operator_gagal" and sub["used"] == 0
            elif method.endswith("_g"):
                assert d["rule"] == "guard"
                assert d["limit"] == d["J_now"] + 1e-9
                assert d["accepted"] == (d["J_candidate"] <= d["limit"])
                assert sub["used"] == (sub["requested"] if d["accepted"] else 0)
            else:
                assert d["rule"] == "tanpa_guard" and sub["used"] == sub["requested"]


@pytest.mark.parametrize("method", ["ppo_bc_ppo_g", "aor"])
def test_every_event_explains_its_disruptions_shifts_and_affected_sets(explained, method):
    _, tr = explained[method]
    for ev in tr.events:
        effects = ev["explain"]["disruptions"]
        assert [e["type"] for e in effects] == [d["type"] for d in ev["disruptions"]]
        for e, a in zip(effects, ev["affected"]):
            assert sorted(e["affected"]) == a["tasks"]
            assert set(e["affected"]) <= set(e["direct"]) | {r["tugas"] for r in e["late_shift"]}
            for r in e["late_shift"]:
                assert r["C_sesudah"] > r["C_sebelum"] + 1e-9 and r["C_sesudah"] > r["d"] + 1e-9
        before, aor = ev["before"]["entries"], ev["aor"]["entries"]
        moved = {tid for tid, b in before.items() if tid in aor and aor[tid]["end"] != b["end"]}
        assert moved <= {s["tugas"] for e in effects for s in e["shifts"]}


def test_the_input_explains_s0_due_dates_capacity_and_mean_effort(explained):
    _, tr = explained["ppo_bc_ppo_g"]
    inp = tr.explain["input"]
    assert inp["s0_rebuilt_equal"] is True
    for t in inp["tasks"]:
        assert t["d"] == t["C"] + 0.5 * (t["C"] - t["S"])
        assert t["upaya"] == t["C"] - t["S"]
    E = inp["E"]
    assert E["n"] == len(inp["tasks"]) and E["value"] == tr.meta["mean_effort"]
    assert E["sum"] / E["n"] == pytest.approx(E["value"], rel=1e-15)
    cap = inp["capacity"]
    assert cap["m"] == JOB[1]
    assert cap["value"] == cap["m"] * (cap["total_nominal"] / cap["n_devs"])


def test_the_disruptions_are_redrawn_from_the_episode_seed_and_shared_by_all_methods(explained):
    first = None
    for method in ("ppo_bc_ppo_g", "aor", "seq"):
        _, tr = explained[method]
        dis = tr.explain["disruptions"]
        assert dis["regenerated_equal"] is True
        assert len(dis["events"]) == len(tr.events)
        for group in dis["events"]:
            times = [m["time"] for m in group["members"]]
            assert times == sorted(times) and times[-1] - times[0] <= dis["window"]
            assert group["time"] == times[-1]
        if first is None:
            first = dis
        assert (dis["seed"], dis["events"]) == (first["seed"], first["events"])


def test_the_output_adds_up_to_the_result(explained, lam):
    for method in episode.METHODS:
        _, tr = explained[method]
        out = tr.explain["output"]
        assert out["tard"] == pytest.approx(sum(r["telat"] for r in out["late"]), abs=1e-9)
        assert out["tard_over_E"] == tr.result["telat"]
        assert out["changes"] == tr.result["ubah"] == sum(ev["changes"] for ev in tr.events)
        assert out["J"] == tr.result["J"] == out["tard_over_E"] + lam * out["changes"]


def test_manual_levels_and_card_moves_are_explained_too():
    from arls_ppo import manual

    lam = 3.0
    tr = manual.run(*JOB, lam, levels={"0:0": 3}, guard=True, explain=True)
    sub = tr.events[0]["subs"][0]
    assert sub["explain"]["selector"]["kind"] == "manual"
    assert sub["explain"]["decision"]["requested"] == 3
    assert "input" in tr.explain and "output" in tr.explain

    aor = manual.run(*JOB, lam, levels={})
    for k, ev in enumerate(aor.events):
        waiting = [t for t in ev["aor"]["tasks"] if t["status"] == "not_started"
                   and len(t["eligible"]) > 1]
        for task in waiting:
            for dev in task["eligible"]:
                if dev == ev["aor"]["entries"][task["id"]]["dev"]:
                    continue
                try:
                    tr = manual.run(*JOB, lam, moves={k: [{"task": task["id"], "to_dev": dev,
                                                           "to_pos": 1}]}, explain=True)
                except manual.MoveRejected:
                    continue
                b = tr.events[k]["explain"]["manual"]
                assert b["J"] == tr.events[k]["J_after_batch"]
                assert b["U"] == tr.events[k]["changes"]
                assert {"tugas": task["id"], "dari": ev["aor"]["entries"][task["id"]]["dev"],
                        "ke": dev} in b["moved_dev"]
                return
    pytest.fail("no feasible move found")
