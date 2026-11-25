"""Explain where the numbers of one episode come from (the number log of ScheduleSim).

Every explanation here is computed at the moment the thesis code computes the same number,
on the same objects, and is then compared with the number the thesis code produced. A
difference stops the run with :class:`ExplainMismatch`, so the log cannot quietly drift
from what the selector really read. Nothing here feeds back into the episode.

The formulas mirror, in the same order of operations:

* the 9 schedule numbers: ``dfjsp.env.state.extract_state``;
* the 29 numbers about the disruption and A_j: ``cek_b3.features`` (``FITUR=v1``);
* the value of a schedule: ``cek_b1_b2.score`` with ``dfjsp.repair.ladder.count_changes``;
* the network of a learned selector: ``cek_episode.PPOPol`` (MaskablePPO actor).
"""
from __future__ import annotations

import copy
import math

import numpy as np


class ExplainMismatch(AssertionError):
    """An explained number differs from the number the thesis code computed."""


def _check(name, explained, actual):
    if np.float32(explained) != np.float32(actual):
        raise ExplainMismatch(f"uraian angka {name!r} = {explained!r}, tetapi kode penelitian "
                              f"memberi {actual!r}")


def _r(x, nd=6):
    return None if x is None else float(round(float(x), nd))


def _item(group, name, formula, inputs, value, note=None):
    out = {"group": group, "name": name, "formula": formula, "inputs": inputs,
           "value": float(value)}
    if note:
        out["note"] = note
    return out


# --------------------------------------------------------------------------- 9 numbers
def state9(env, base_state):
    """The 9 schedule numbers C, Tard, O, U, B, N, H, Q, I, as ``extract_state`` computes them.

    ``env.sched`` must be the schedule ``base_state`` was computed on (the schedule after AOR
    at the start of the event); the run_batch hook calls this before the batch runs.
    """
    from dfjsp.env.schedule import developer_load, partition
    from dfjsp.env.state import EPS, _clip01, makespan, total_tardiness
    from dfjsp.env.work_calendar import working_time

    inst, sched, t, refs = env.inst, env.sched, env.t_now, env.refs
    devs = inst.developers
    n_dev = max(1, len(devs))
    completed, _, not_started = partition(inst, sched, t)
    w = developer_load(inst, sched, t)
    w_arr = np.array([w[d.dev_id] for d in devs], dtype=float)
    cap_arr = np.array([d.capacity for d in devs], dtype=float)
    ms = makespan(inst, sched)
    tard = total_tardiness(inst, sched)
    overload = [int(d.dev_id) for d, wi, ci in zip(devs, w_arr, cap_arr) if wi > ci + EPS]
    busy = 0.0
    for e in sched.entries.values():
        busy += working_time(e.start, e.end, sched.pauses.get(e.dev_id, []))
    mean_w = float(np.mean(w_arr)) if w_arr.size else 0.0
    std_w = float(np.std(w_arr))
    by_id = {x.task_id: x for x in inst.tasks}
    ready = sorted(int(tid) for tid in not_started
                   if all(dep in completed for dep in by_id[tid].deps))
    recent = env._recent_event_count()
    loads = [{"dev": int(d.dev_id), "w": _r(wi), "cap": _r(ci)}
             for d, wi, ci in zip(devs, w_arr, cap_arr)]
    S = "jadwal"
    items = [
        _item(S, "C: makespan / horizon", "min(1, makespan / horizon)",
              {"makespan": _r(ms), "horizon": _r(refs.horizon)}, _clip01(ms / refs.horizon)),
        _item(S, "Tard: keterlambatan total ternormalisasi",
              "min(1, Tard / penyebut Tard S0)",
              {"Tard": _r(tard), "penyebut Tard S0": _r(refs.tard0_denom)},
              _clip01(tard / refs.tard0_denom),
              "penyebut = max(Tard S0, total upaya S0); Tard S0 = 0 karena tenggat dari S0"),
        _item(S, "O: porsi pengembang kelebihan beban",
              "min(1, banyak pengembang dengan beban > kapasitas / banyak pengembang)",
              {"kelebihan beban": overload, "banyak pengembang": n_dev, "beban": loads},
              _clip01(len(overload) / n_dev)),
        _item(S, "U: utilisasi pengembang",
              "min(1, total waktu kerja / (banyak pengembang × makespan))",
              {"total waktu kerja": _r(busy), "banyak pengembang": n_dev,
               "makespan": _r(ms)}, _clip01(busy / (n_dev * max(EPS, ms)))),
        _item(S, "B: ketidakseimbangan beban", "min(1, simpangan baku beban / rata-rata beban)",
              {"rata-rata beban": _r(mean_w), "simpangan baku beban": _r(std_w),
               "beban": loads},
              _clip01(std_w / max(EPS, mean_w)) if mean_w > EPS else 0.0),
        _item(S, "N: nervousness kumulatif", "min(1, nervousness kumulatif / banyak tugas)",
              {"nervousness kumulatif": _r(env.nervousness_cum), "banyak tugas": refs.n_ref},
              _clip01(env.nervousness_cum / max(EPS, refs.n_ref))),
        _item(S, "H: alih konteks kumulatif", "min(1, alih konteks kumulatif / banyak tugas)",
              {"alih konteks kumulatif": _r(env.context_switch_cum),
               "banyak tugas": refs.h_ref},
              _clip01(env.context_switch_cum / max(EPS, refs.h_ref))),
        _item(S, "Q: porsi tugas siap dikerjakan",
              "min(1, tugas siap / tugas belum mulai)",
              {"tugas siap": ready, "banyak tugas belum mulai": len(not_started)},
              _clip01(len(ready) / max(1, len(not_started))),
              "siap = belum mulai dan semua pendahulunya sudah selesai"),
        _item(S, "I: kepadatan kejadian terbaru",
              "min(1, gangguan dalam jendela terakhir / i_max)",
              {"gangguan dalam (t - jendela, t]": recent, "jendela": _r(refs.i_window),
               "i_max": refs.i_max}, _clip01(recent / refs.i_max)),
    ]
    for it, actual in zip(items, base_state):
        _check(it["name"], it["value"], actual)
    return items


# --------------------------------------------------------------------------- 38 numbers
def features(env, cur, base_items, j, A, types, lam, changes_so_far, x, mask, ops, type_names):
    """All 38 numbers of sub-decision ``j``, mirroring ``cek_b3.features`` (v1).

    ``x`` and ``mask`` are what ``cek_b3.features`` returned for the same call; every
    explained number is checked against them.
    """
    from dfjsp.env.schedule import developer_load, partition
    from dfjsp.env.state import total_tardiness
    from dfjsp.repair.masks import is_pair_feasible

    inst, t, E = env.inst, env.t_now, env.mean_effort
    _, in_prog, not_started = partition(inst, cur, t)
    ns = set(not_started)
    by_id = {q.task_id: q for q in inst.tasks}
    dev_by_id = {d.dev_id: d for d in inst.developers}
    loads = developer_load(inst, cur, t)
    a_j = A[j]
    a_ns = [q for q in a_j if q in ns]
    late = [q for q in a_j if cur.entries[q].end > by_id[q].due_date + 1e-9]
    tard_a = sum(max(0.0, cur.entries[q].end - by_id[q].due_date) for q in a_j) / E
    slack = [(by_id[q].due_date - cur.entries[q].end) / E for q in a_ns] or [0.0]
    n_elig = [len(by_id[q].eligible_devs) for q in a_ns] or [0]
    recv, pairs = set(), []
    for q in a_ns:
        curdev = cur.entries[q].dev_id
        for d in by_id[q].eligible_devs:
            if d != curdev and is_pair_feasible(q, d, inst, cur, t, loads=loads,
                                                by_id=by_id, dev_by_id=dev_by_id):
                recv.add(d)
                pairs.append([int(q), int(d)])
    trial = copy.copy(cur)
    trial.affected = set(a_j)
    own_mask = [int(ops[lv].inspect(inst, trial, t).feasible) for lv in (1, 2, 3)]
    util = [loads.get(d.dev_id, 0.0) / max(1e-9, d.capacity) for d in inst.developers]
    others = set().union(*[A[i] for i in range(len(A)) if i != j]) if len(A) > 1 else set()
    rest = set().union(*A[j + 1:]) if j + 1 < len(A) else set()
    unfinished = max(1, len(in_prog) + len(not_started))
    k = len(A)
    tard_all = total_tardiness(inst, cur)

    def tasks(ids):
        return sorted(int(q) for q in ids)

    def row(q):
        e = cur.entries[q]
        return {"tugas": int(q), "pengembang": int(e.dev_id), "C": _r(e.end),
                "d": _r(by_id[q].due_date)}

    G, A_, D, K = "gangguan", "A_j", "pengembang", "kejadian"
    items = list(base_items)
    for ty in ("urgent_task", "effort_revision", "rework_review_fail", "priority_change",
               "developer_unavailable"):
        items.append(_item(G, f"jenis gangguan: {type_names[ty]}",
                           f"1 bila jenis gangguan ke-j = {type_names[ty]}, selain itu 0",
                           {"jenis gangguan ke-j": type_names.get(types[j], types[j])},
                           1.0 if types[j] == ty else 0.0))
    items += [
        _item(A_, "log(1 + |A_j|)", "log(1 + |A_j|)",
              {"|A_j|": len(a_j), "A_j": tasks(a_j)}, math.log1p(len(a_j))),
        _item(A_, "log(1 + tugas A_j yang belum mulai)", "log(1 + banyak tugas A_j belum mulai)",
              {"tugas A_j belum mulai": tasks(a_ns)}, math.log1p(len(a_ns))),
        _item(A_, "tugas A_j belum mulai / tugas belum selesai",
              "banyak tugas A_j belum mulai / (tugas sedang dikerjakan + belum mulai)",
              {"tugas A_j belum mulai": len(a_ns), "sedang dikerjakan": len(in_prog),
               "belum mulai": len(not_started)}, len(a_ns) / unfinished),
        _item(A_, "porsi A_j yang terlambat", "banyak tugas A_j dengan C > d / |A_j|",
              {"terlambat": [row(q) for q in sorted(late)], "|A_j|": len(a_j)},
              len(late) / max(1, len(a_j))),
        _item(A_, "log(1 + keterlambatan A_j / E)", "log(1 + Σ max(0, C − d) atas A_j / E)",
              {"Σ max(0, C − d) atas A_j": _r(tard_a * E), "E": _r(E)}, math.log1p(tard_a)),
        _item(A_, "rata-rata slack A_j / E", "rata-rata (d − C) / E atas tugas A_j belum mulai",
              {"slack / E per tugas": [{**row(q), "slack/E": _r((by_id[q].due_date
                                                                 - cur.entries[q].end) / E)}
                                       for q in a_ns], "E": _r(E)},
              float(np.mean(slack)), None if a_ns else "tidak ada tugas belum mulai, jadi 0"),
        _item(A_, "slack terkecil A_j / E", "min (d − C) / E atas tugas A_j belum mulai",
              {"slack / E terkecil": _r(float(np.min(slack)))}, float(np.min(slack))),
        _item(A_, "rata-rata banyak pengembang layak",
              "rata-rata banyak pengembang yang boleh mengerjakan tugas A_j belum mulai",
              {"pengembang layak per tugas": [{"tugas": int(q), "pengembang layak":
                                               sorted(int(d) for d in by_id[q].eligible_devs)}
                                              for q in a_ns]}, float(np.mean(n_elig))),
        _item(D, "log(1 + pengembang penerima)",
              "log(1 + banyak pengembang lain yang lolos is_pair_feasible untuk tugas A_j)",
              {"pengembang penerima": sorted(int(d) for d in recv)}, math.log1p(len(recv)),
              "lolos = keahlian, ketersediaan, pendahulu, dan sisa kapasitas terpenuhi"),
        _item(D, "log(1 + pasangan pindah yang layak)",
              "log(1 + banyak pasangan (tugas, pengembang lain) yang lolos is_pair_feasible)",
              {"pasangan (tugas, pengembang)": pairs}, math.log1p(len(pairs))),
    ]
    for lv, name in zip((1, 2, 3), ("Resequence", "Reassign", "Partial rescheduling")):
        items.append(_item("mask", f"mask tingkat {lv}",
                           f"1 bila operator {name} punya calon perbaikan untuk A_j",
                           {"operator": name}, own_mask[lv - 1]))
    util_rows = [{"dev": int(d.dev_id), "beban": _r(loads.get(d.dev_id, 0.0)),
                  "kapasitas": _r(d.capacity), "beban/kapasitas": _r(u)}
                 for d, u in zip(inst.developers, util)]
    items += [
        _item(D, "rata-rata beban / kapasitas", "rata-rata (sisa beban / kapasitas)",
              {"per pengembang": util_rows}, float(np.mean(util))),
        _item(D, "beban / kapasitas terbesar", "max (sisa beban / kapasitas)",
              {"terbesar": _r(float(np.max(util)))}, float(np.max(util))),
        _item(D, "simpangan beban / kapasitas", "simpangan baku (sisa beban / kapasitas)",
              {"simpangan baku": _r(float(np.std(util)))}, float(np.std(util))),
        _item("jadwal", "log(1 + keterlambatan total / E)", "log(1 + Tard / E)",
              {"Tard": _r(tard_all), "E": _r(E)}, math.log1p(tard_all / E)),
        _item(K, "k: banyak gangguan dalam kejadian", "k",
              {"k": k}, k),
        _item(K, "j / k", "j / k, dengan j mulai dari 0",
              {"j": j, "k": k}, j / k),
        _item(K, "k - j - 1", "banyak gangguan yang masih menunggu sesudah gangguan ini",
              {"j": j, "k": k}, (k - j - 1)),
        _item(K, "porsi A_j yang juga terkena gangguan lain",
              "|A_j ∩ A gangguan lain| / |A_j|",
              {"irisan": tasks(set(a_j) & others), "|A_j|": len(a_j)},
              len(set(a_j) & others) / max(1, len(a_j))),
        _item(K, "log(1 + tugas gangguan berikutnya)",
              "log(1 + |gabungan A gangguan sesudah j|)",
              {"tugas": tasks(rest)}, math.log1p(len(rest))),
        _item(K, "log(1 + perubahan sejauh ini)",
              "log(1 + perubahan yang sudah dipakai pada kejadian ini)",
              {"perubahan sejauh ini": int(changes_so_far)}, math.log1p(changes_so_far)),
        _item("λ", "log λ", "log λ", {"λ": float(lam)}, math.log(lam)),
    ]
    if list(own_mask) != list(mask):
        raise ExplainMismatch(f"uraian mask {own_mask} berbeda dari mask {list(mask)}")
    if len(items) != len(x):
        raise ExplainMismatch(f"{len(items)} angka diuraikan, tetapi x berisi {len(x)}")
    for it, actual in zip(items, x):
        _check(it["name"], it["value"], actual)
    return items


# --------------------------------------------------------------------------- value J
def value(inst, base, trial, t, E, lam, level, actual=None):
    """Value J of ``trial`` against ``base`` (the schedule after AOR), as ``score`` computes it.

    J = Tard / E + λ · U with U = tasks moved to another developer + Σ (n − LIS) over the
    waiting queues of the developers. ``actual`` (J, U) from ``cek_b1_b2.score`` is checked.
    """
    from dfjsp.env.schedule import queue_order
    from dfjsp.env.state import total_tardiness
    from dfjsp.repair.ladder import _lis_length

    from ._trace import _lis_members

    late = []
    for task in inst.tasks:
        e = trial.entries.get(task.task_id)
        if e is not None and e.end - task.due_date > 0.0:
            late.append({"tugas": int(task.task_id), "pengembang": int(e.dev_id),
                         "C": float(e.end), "d": float(task.due_date),
                         "telat": float(e.end - task.due_date)})
    moved_dev = [{"tugas": int(tid), "dari": int(base.entries[tid].dev_id), "ke": int(e.dev_id)}
                 for tid, e in trial.entries.items()
                 if tid in base.entries and base.entries[tid].dev_id != e.dev_id]
    qb = queue_order(inst, base, t)
    waiting = {q for qq in qb.values() for q in qq}
    qa: dict = {}
    for tid in waiting:
        e = trial.entries.get(tid)
        if e is not None:
            qa.setdefault(e.dev_id, []).append(tid)
    queues, mp = [], 0
    for dev, new in qa.items():
        new.sort(key=lambda tid: (trial.entries[tid].start, tid))
        old_pos = {q: k for k, q in enumerate(qb.get(dev, []))}
        common_ids = [q for q in new if q in old_pos]
        common = [old_pos[q] for q in common_ids]
        lis = _lis_length(common)
        mp += len(common) - lis
        if len(common) - lis > 0 or len(common) != len(qb.get(dev, [])) or len(new) != len(common):
            keep = _lis_members(common)
            queues.append({"dev": int(dev), "lama": [int(q) for q in qb.get(dev, [])],
                           "baru": [int(q) for q in new], "tetap": [int(q) for q in common_ids],
                           "posisi_lama": [int(p) for p in common], "lis": int(lis),
                           "n": len(common),
                           "pindah_posisi": [int(common_ids[i]) for i in range(len(common))
                                             if i not in keep]})
    queues.sort(key=lambda q: q["dev"])
    md = len(moved_dev)
    tard = total_tardiness(inst, trial)
    J = tard / E + lam * (md + mp)
    if actual is not None:
        if J != actual[0] or md + mp != actual[1]:
            raise ExplainMismatch(f"uraian J tingkat {level} = {J!r} (U = {md + mp}), tetapi "
                                  f"score memberi {actual!r}")
    return {"level": level if isinstance(level, str) else int(level), "feasible": True, "tard": float(tard), "E": float(E),
            "tard_over_E": float(tard / E), "late": late, "moved_dev": moved_dev,
            "queues": queues, "md": int(md), "mp": int(mp), "U": int(md + mp),
            "lam": float(lam), "lamU": float(lam * (md + mp)), "J": float(J)}


def infeasible(level, reason):
    return {"level": int(level), "feasible": False, "reason": reason}


# --------------------------------------------------------------------------- selector
def _softmax_steps(z, allowed):
    """Masked softmax written out: z_max over allowed levels, e^(z − z_max), sum, p."""
    zmax = max(z[lv] for lv in range(4) if allowed[lv])
    e = [math.exp(z[lv] - zmax) if allowed[lv] else 0.0 for lv in range(4)]
    total = sum(e)
    return zmax, e, total, [v / total for v in e]


def selector(policy, x, mask, requested):
    """How the selector of one sub-decision turned the 38 numbers into a level.

    * learned PPO selector (``PPOPol``): normalisation, the two Tanh layers of the actor,
      the four logits, and the masked softmax; the level with the largest probability is
      checked against the level the selector returned;
    * Boosting-IL (``cek_b3.Pol``): raw class scores and class probabilities;
    * Fixed-L3 and Random (``RulePol``): the rule and the allowed levels.
    """
    allowed = [1] + [int(m) for m in mask]
    if hasattr(policy, "nrm") and hasattr(policy, "model"):
        import torch

        nomask = bool(getattr(policy, "nomask", False))
        if nomask:
            allowed = [1, 1, 1, 1]
        mu = np.asarray(policy.nrm["mu"], dtype=np.float32)
        sd = np.asarray(policy.nrm["sd"], dtype=np.float32)
        raw = (x - policy.nrm["mu"]) / policy.nrm["sd"]
        obs = np.clip(raw, -5, 5).astype("float32")
        pol = policy.model.policy
        with torch.no_grad():
            obs_t, _ = pol.obs_to_tensor(obs)
            h = pol.extract_features(obs_t, pol.pi_features_extractor)
            hidden = []
            for layer in pol.mlp_extractor.policy_net:
                h = layer(h)
                if isinstance(layer, torch.nn.Tanh):
                    hidden.append(h[0].cpu().numpy().astype(float))
            logits = pol.action_net(h)[0].cpu().numpy().astype(float)
        z = [float(v) for v in logits]
        zmax, e, total, p = _softmax_steps(z, allowed)
        best = int(np.argmax(p))
        if best != int(requested):
            raise ExplainMismatch(f"argmax peluang = tingkat {best}, tetapi selector memilih "
                                  f"tingkat {requested}")
        layers = [int(pol.mlp_extractor.policy_net[0].in_features)]
        layers += [int(m.out_features) for m in pol.mlp_extractor.policy_net
                   if isinstance(m, torch.nn.Linear)]
        layers.append(int(pol.action_net.out_features))
        return {"kind": "ppo", "guard": bool(policy.guard), "nomask": nomask,
                "x": [float(v) for v in x], "mu": mu.astype(float).tolist(),
                "sd": sd.astype(float).tolist(), "raw": [float(v) for v in raw],
                "obs": obs.astype(float).tolist(),
                "clipped": [int(i) for i in np.flatnonzero(np.abs(raw) > 5)],
                "h1": [float(v) for v in hidden[0]], "h2": [float(v) for v in hidden[1]],
                "layers": layers, "activation": "Tanh", "logits": z, "allowed": allowed,
                "z_max": zmax, "exp": e, "sum_exp": total, "probs": p, "chosen": best}
    if hasattr(policy, "model") and hasattr(policy.model, "predict_proba"):
        model = policy.model
        cls = [int(c) for c in model.classes_]
        proba = model.predict_proba(x[None, :])[0]
        probs = [float(proba[cls.index(lv)]) if lv in cls else 0.0 for lv in range(4)]
        scores = None
        if hasattr(model, "decision_function"):
            d = np.atleast_1d(model.decision_function(x[None, :])[0]).astype(float)
            if len(d) == len(cls):
                scores = [float(d[cls.index(lv)]) if lv in cls else None for lv in range(4)]
        best = max(range(4), key=lambda lv: probs[lv] if lv in cls and allowed[lv] else -1.0)
        if best != int(requested):
            raise ExplainMismatch(f"peluang terbesar yang sah = tingkat {best}, tetapi selector "
                                  f"memilih tingkat {requested}")
        return {"kind": "boosting", "guard": bool(policy.guard), "classes": cls,
                "model": type(model).__name__,
                "n_trees": int(getattr(model, "n_iter_", 0)) or None,
                "scores": scores, "probs": probs, "allowed": allowed, "chosen": best}
    if hasattr(policy, "levels") and hasattr(policy, "_j"):        # manual._Scripted
        return {"kind": "manual", "guard": bool(policy.guard),
                "rule": "tingkat dipilih pengguna; sub-keputusan yang tidak dipilih memakai 0",
                "allowed": [lv for lv in range(4) if allowed[lv]], "chosen": int(requested)}
    kind = getattr(policy, "kind", None)
    if kind is not None:
        levels = [lv for lv in range(4) if allowed[lv]]
        if kind == "acak":
            rule = ("pilih acak seragam di antara tingkat yang sah; generator numpy "
                    "default_rng(seed episode), sama untuk setiap λ")
        else:
            lv = int(kind[3:])
            rule = (f"selalu tingkat {lv}; bila tingkat {lv} tidak sah menurut mask, "
                    "pakai tingkat 0")
        return {"kind": "rule", "rule": rule, "name": kind, "guard": bool(policy.guard),
                "allowed": levels, "chosen": int(requested)}
    return {"kind": "unknown", "chosen": int(requested)}


# --------------------------------------------------------------------------- decision
def decision(policy, sub, levels):
    """Why the level used at a sub-decision is the one it is.

    ``sub`` carries the requested and used level and ``J_before`` (the value J of the
    schedule so far); ``levels`` is the output of :func:`value` for levels 0-3.
    """
    req, used, j_now = int(sub["requested"]), int(sub["used"]), float(sub["J_before"])
    base = {"requested": req, "used": used, "J_now": j_now}
    if policy is None:
        ok = {lv: b["J"] for lv, b in levels.items() if b["feasible"]}
        return {**base, "rule": "teacher", "candidates": {int(lv): v for lv, v in ok.items()},
                "text": "Sequential Lookahead mencoba ketiga tingkat lalu memakai J terkecil; "
                        "bila sama, tingkat 0 diutamakan (selisih 1e-9), lalu tingkat yang lebih "
                        "rendah."}
    if req == 0:
        return {**base, "rule": "tingkat_0",
                "text": "Selector meminta tingkat 0, jadi jadwal sesudah AOR dipertahankan "
                        "untuk gangguan ini."}
    cand = levels[req]
    if not cand["feasible"]:
        return {**base, "rule": "operator_gagal",
                "text": f"Operator tingkat {req} tidak menemukan perbaikan yang sah, jadi "
                        "tingkat 0 dipakai."}
    jc = float(cand["J"])
    if getattr(policy, "guard", False):
        limit = j_now + 1e-9
        accepted = jc <= limit
        return {**base, "rule": "guard", "J_candidate": jc, "limit": limit,
                "accepted": accepted,
                "text": (f"Guard: J calon {jc:.4f} ≤ J sekarang {j_now:.4f} (+1e-9), jadi "
                         f"tingkat {req} dipakai." if accepted else
                         f"Guard: J calon {jc:.4f} > J sekarang {j_now:.4f} (+1e-9), jadi "
                         "ditolak dan kembali ke tingkat 0.")}
    return {**base, "rule": "tanpa_guard", "J_candidate": jc,
            "text": f"Tanpa guard: tingkat {req} langsung dipakai, walaupun J bisa naik."}


# --------------------------------------------------------------------------- disruptions
def task_state(inst):
    """What a disruption can change about the tasks, for a before/after comparison."""
    return {int(q.task_id): {"eligible": {int(d): float(e) for d, e in q.eligible_devs.items()},
                             "due": float(q.due_date), "priority": float(q.priority),
                             "status": q.status} for q in inst.tasks}


def effect(inst, sched, event, t_now, before_entries, before_tasks):
    """One disruption applied with AOR (tingkat 0): what it changed and which tasks form A_j.

    ``before_entries`` maps task -> (dev, start, end) and ``before_tasks`` is
    :func:`task_state`, both taken just before ``apply_disruption`` ran.
    """
    from dfjsp.env.repair_env import _direct_targets

    by_id = {q.task_id: q for q in inst.tasks}
    end_before = {tid: v[2] for tid, v in before_entries.items()}
    direct = _direct_targets(event, end_before, sched, t_now)
    late = {tid for tid, e in sched.entries.items()
            if tid in end_before and e.end > end_before[tid] + 1e-9
            and e.end > by_id[tid].due_date + 1e-9}
    affected = sorted(int(q) for q in direct | late if q in sched.entries)
    p = dict(event.payload)
    out = {"type": event.type, "time": float(event.time), "direct": sorted(int(q) for q in direct),
           "affected": affected,
           "late_shift": [{"tugas": int(tid), "C_sebelum": float(end_before[tid]),
                           "C_sesudah": float(sched.entries[tid].end),
                           "d": float(by_id[tid].due_date)} for tid in sorted(late)],
           "shifts": [{"tugas": int(tid), "pengembang": int(e.dev_id),
                       "mulai_sebelum": float(before_entries[tid][1]), "mulai_sesudah": float(e.start),
                       "C_sebelum": float(before_entries[tid][2]), "C_sesudah": float(e.end)}
                      for tid, e in sorted(sched.entries.items())
                      if tid in before_entries and (abs(e.end - before_entries[tid][2]) > 0
                                                    or abs(e.start - before_entries[tid][1]) > 0)]}
    after = task_state(inst)
    if event.type == "urgent_task":
        new = [tid for tid in sched.entries if tid not in before_entries]
        if new:
            tid = new[0]
            q, e = by_id[tid], sched.entries[tid]
            eff = float(q.eligible_devs[e.dev_id])
            slack = float(p.get("due_slack", 0.5))
            rule = p.get("due_rule")
            out["detail"] = {
                "tugas_baru": int(tid), "pengembang": int(e.dev_id), "upaya": eff,
                "pengembang_layak": {int(d): float(v) for d, v in q.eligible_devs.items()},
                "prioritas": float(q.priority), "aturan_tenggat": rule, "tenggat": float(q.due_date),
                "rumus_tenggat": (f"t + (1 + {slack:g}) × upaya = {t_now:.4f} + {1 + slack:g} × "
                                  f"{eff:.4f}" if rule == "segera" else
                                  f"selesai di akhir antrean + {slack:g} × upaya = {e.end:.4f} + "
                                  f"{slack:g} × {eff:.4f}"),
                "mulai": float(e.start), "selesai": float(e.end)}
    elif event.type in ("effort_revision", "rework_review_fail", "priority_change"):
        tid = p.get("task_id")
        b, a = before_tasks.get(tid), after.get(tid)
        if b is not None and a is not None:
            e = sched.entries.get(tid)
            dev = None if e is None else int(e.dev_id)
            d = {"tugas": int(tid), "status_sebelum": b["status"], "status_sesudah": a["status"]}
            if event.type == "effort_revision":
                d.update(faktor=float(p.get("factor", 1.0)),
                         upaya_sebelum=b["eligible"].get(dev), upaya_sesudah=a["eligible"].get(dev),
                         rumus="upaya baru = upaya lama × faktor (untuk setiap pengembang layak)")
            elif event.type == "rework_review_fail":
                d.update(tambahan=float(p.get("extra_effort", 0.0)),
                         upaya_sebelum=b["eligible"].get(dev), upaya_sesudah=a["eligible"].get(dev),
                         rumus="upaya baru = upaya lama + 0,47 × upaya semula (rework)")
            else:
                d.update(prioritas_sebelum=b["priority"], prioritas_sesudah=a["priority"],
                         tenggat_sebelum=b["due"], tenggat_sesudah=a["due"])
            if tid in before_entries and e is not None:
                d.update(C_sebelum=float(before_entries[tid][2]), C_sesudah=float(e.end))
            out["detail"] = d
    elif event.type == "developer_unavailable":
        dur = float(p.get("duration", 0.0))
        out["detail"] = {"pengembang": int(p.get("dev_id")), "lama": dur,
                         "jeda": [float(t_now), float(t_now + dur)],
                         "tugas_pengembang_ini": sorted(int(q) for q in direct)}
    return out


# --------------------------------------------------------------------------- episode
def episode_input(env, capacity):
    """Input of the episode: tasks, capacity, S0, due dates, E, normalising references.

    Called at the first event, before it is applied, so ``env`` is the state right after
    ``reset``. S0 is rebuilt from the instance file with ``build_initial_schedule`` and
    compared with the S0 of the run.
    """
    import instans_acak
    from dfjsp.core.domain import set_capacity
    from dfjsp.core.due_date import e_nominal
    from dfjsp.env.initial_schedule import build_initial_schedule

    inst0 = copy.deepcopy(instans_acak.load_any(env.inst.name))
    set_capacity(inst0, capacity)
    twk = {q.task_id: float(q.due_date) for q in inst0.tasks}
    total_nominal = sum(e_nominal(q) for q in inst0.tasks)
    rebuilt = build_initial_schedule(inst0).entries
    ref = env.sched.baseline_ref or env.sched.entries
    equal = (set(rebuilt) == set(ref) and all(
        rebuilt[q].dev_id == ref[q].dev_id and rebuilt[q].start == ref[q].start
        and rebuilt[q].end == ref[q].end for q in ref))
    rows = []
    for q in env.inst.tasks:
        e = ref.get(q.task_id)
        if e is None:
            continue
        d = e.end + env.due_slack * (e.end - e.start)
        _check(f"tenggat tugas {q.task_id}", d, q.due_date)
        rows.append({"tugas": int(q.task_id), "job": int(q.job_id), "op": int(q.op_index),
                     "layak": {int(k): float(v) for k, v in q.eligible_devs.items()},
                     "deps": [int(x) for x in q.deps], "d_twk": twk.get(q.task_id),
                     "pengembang": int(e.dev_id), "S": float(e.start), "C": float(e.end),
                     "upaya": float(e.end - e.start), "d": float(q.due_date)})
    efforts = [e.end - e.start for e in ref.values()]
    # the same formula as LadderRepairEnv._ensure_mean_effort; at the first event it may not
    # have run yet, so the recorder checks this value against env.mean_effort afterwards
    mean = max(1e-6, float(np.mean(efforts)) if efforts else 1.0)
    refs = env.refs
    n_devs = inst0.n_devs
    return {
        "instance": env.inst.name, "n_tasks": len(rows), "n_jobs": len({r["job"] for r in rows}),
        "n_devs": len(env.inst.developers),
        "capacity": {"m": float(capacity), "total_nominal": float(total_nominal),
                     "n_devs": int(n_devs), "value": float(env.inst.developers[0].capacity),
                     "rumus": "cap_j = m × (Σ upaya nominal / banyak pengembang); upaya nominal "
                              "= rata-rata upaya di pengembang yang layak"},
        "s0_rule": "list scheduling: tugas yang pendahulunya sudah terjadwal diurutkan menurut "
                   "(tenggat TWK, id); tiap tugas ke pengembang layak dengan beban terkecil di "
                   "antara yang masih muat kapasitas (bila tidak ada yang muat, semua yang layak); "
                   "mulai = max(pendahulu selesai, pengembang bebas)",
        "s0_rebuilt_equal": bool(equal), "due_slack": float(env.due_slack),
        "due_rule": f"d_i = C_i + {env.due_slack:g} × (C_i − S_i), dari S0",
        "tasks": rows,
        "E": {"sum": float(sum(efforts)), "n": len(efforts), "value": mean,
              "rumus": "E = Σ (C_i − S_i) di S0 / banyak tugas"},
        "refs": {"horizon": float(refs.horizon), "tard0_denom": float(refs.tard0_denom),
                 "n_ref": float(refs.n_ref), "i_window": float(refs.i_window),
                 "i_max": float(refs.i_max)},
    }


def _plain_event(ev):
    from ._trace import plain

    return {"type": ev.type, "time": float(ev.time), "payload": plain(dict(ev.payload))}


def disruption_draw(env):
    """How the episode seed becomes the list of disruptions (the same for every method)."""
    from dfjsp.core.seeding import derive_episode_seed
    from dfjsp.env.disruptions import resolve_scenario
    from dfjsp.env.disruptions_tawos import (TawosDisruptionModel, TawosParams,
                                             intensity_multiplier)
    from dfjsp.env.state import makespan

    full = resolve_scenario(env.scenario)
    seed = derive_episode_seed(env.base_seed, "__shared__", env.inst.name, env.intensity,
                               env.k_tight, env.seed, full, stream="disruption")
    inst_c, sched_c = copy.deepcopy(env.inst), copy.deepcopy(env.sched)
    horizon = makespan(inst_c, sched_c)
    gen = TawosDisruptionModel(env.intensity, np.random.default_rng(seed))
    redrawn = gen.sample_timeline(horizon, inst_c, sched_c)

    def groups(events):
        out = []
        for ev in events:
            members = ev.payload.get("events", []) if ev.type == "batch" else [ev]
            out.append({"time": float(ev.time), "members": [_plain_event(m) for m in members]})
        return out

    actual = groups(env._events)
    p = TawosParams()
    m = intensity_multiplier(env.intensity)
    n = sum(1 for q in env.inst.tasks if q.task_id in env.sched.entries)
    efforts = [e.end - e.start for e in env.sched.entries.values()]
    counts: dict = {}
    for g in actual:
        for mem in g["members"]:
            counts[mem["type"]] = counts.get(mem["type"], 0) + 1
    return {
        "seed": int(seed),
        "seed_rule": "derive_episode_seed(base_seed, \"__shared__\", instans, intensitas, "
                     "k_tight, seed episode, skenario, stream=\"disruption\"); tidak memuat nama "
                     "metode, jadi semua metode mendapat gangguan yang sama",
        "seed_inputs": {"base_seed": int(env.base_seed), "instans": env.inst.name,
                        "intensitas": float(env.intensity), "k_tight": float(env.k_tight),
                        "seed episode": int(env.seed), "skenario": full},
        "generator": "numpy.random.default_rng(seed) (PCG64)",
        "horizon": float(horizon), "m": float(m),
        "params": {
            "m = intensitas / 0,15": float(m),
            "harapan banyak pekerjaan tidak terencana = m × 0,5 × n": float(m * p.unplanned_per_task * n),
            "jarak antar-kedatangan: Gamma, bentuk": float(p.arrival_gamma_shape),
            "peluang revisi estimasi per tugas = min(1, m × 0,123)": float(min(1.0, m * p.revision_prob)),
            "peluang rework per tugas = min(1, m × 0,076)": float(min(1.0, m * p.rework_prob)),
            "rata-rata masa tidak tersedia per pengembang = m × 0,5": float(m * p.unavailable_per_dev),
            "peluang perubahan prioritas per tugas = min(1, m × 0,05)": float(min(1.0, m * p.priority_change_prob)),
        },
        "n": int(n), "window": float(np.mean(efforts)) if efforts else 1.0,
        "window_rule": "gangguan yang jatuh dalam satu E dari gangguan pertama kelompoknya digabung "
                       "menjadi satu kejadian; keputusan diambil pada waktu anggota terakhir",
        "counts": counts, "events": actual, "regenerated_equal": groups(redrawn) == actual,
    }


def episode_output(env, lam, changes):
    """Output of the episode: final tardiness, total changes, and J."""
    from dfjsp.env.state import total_tardiness

    late = []
    for q in env.inst.tasks:
        e = env.sched.entries.get(q.task_id)
        if e is not None and e.end - q.due_date > 0.0:
            late.append({"tugas": int(q.task_id), "pengembang": int(e.dev_id), "C": float(e.end),
                         "d": float(q.due_date), "telat": float(e.end - q.due_date)})
    tard = total_tardiness(env.inst, env.sched)
    telat = tard / env.mean_effort
    return {"tard": float(tard), "E": float(env.mean_effort), "tard_over_E": float(telat),
            "late": late, "changes": int(changes), "lam": float(lam),
            "J": float(telat + lam * changes),
            "rumus": "J episode = Tard akhir / E + λ × Σ U semua kejadian"}


__all__ = ["disruption_draw", "episode_input", "episode_output", "effect", "task_state", "decision", "ExplainMismatch", "features", "infeasible", "selector", "state9", "value"]
