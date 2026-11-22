"""Watch ``cek_episode.run`` without changing it: snapshots and per-decision details.

Two hooks are installed only while one traced episode runs:

* the episode environment class ``cek_episode.EpProbe`` is replaced by a subclass that copies
  the schedule before each event, after AOR (the event applied by right-shift), and after the
  repair chosen by the method, just before ``step``;
* ``cek_b3.run_batch`` is wrapped so the selector call of every sub-decision is recorded
  (state, mask, requested level). After the batch has finished, the value path is replayed
  with the same ``try_level`` and ``score`` functions to show J before and after each
  sub-decision, and the action probabilities of the learned selector are computed.

Nothing here feeds back into the episode: the selector returns its own action, the replay
works on the schedules the batch already produced, and seam B1 (``tests/test_trace.py``)
checks that the result equals the untraced run.
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np


def plain(value):
    """Turn numpy numbers, sets and tuples into plain Python values."""
    if isinstance(value, dict):
        return {(int(k) if isinstance(k, (np.integer,)) else k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(plain(v) for v in value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _tasks(inst):
    return [{"id": t.task_id, "job": t.job_id, "op": t.op_index,
             "eligible": {int(d): float(e) for d, e in t.eligible_devs.items()},
             "deps": [int(d) for d in t.deps], "due": float(t.due_date),
             "priority": float(t.priority), "status": t.status,
             "assigned_dev": t.assigned_dev, "start": t.start_time, "finish": t.finish_time,
             "progress": float(t.progress)} for t in inst.tasks]


def _entries(entries):
    return {int(tid): {"dev": int(e.dev_id), "start": float(e.start), "end": float(e.end)}
            for tid, e in entries.items()}


def _snapshot(env, entries=None, with_tasks=True):
    sched = env.sched
    snap = {"t": float(env.t_now),
            "entries": _entries(sched.entries if entries is None else entries),
            "pauses": {int(d): [[float(a), float(b)] for a, b in p]
                       for d, p in sched.pauses.items()}}
    if with_tasks:
        snap["tasks"] = _tasks(env.inst)
        snap["developers"] = [{"id": d.dev_id, "capacity": float(d.capacity),
                               "available": bool(d.available)} for d in env.inst.developers]
    return snap


def _disruptions(event):
    subs = event.payload.get("events", []) if event.type == "batch" else [event]
    return [{"type": s.type, "time": float(s.time), "payload": plain(dict(s.payload))}
            for s in subs]


def _lis_members(seq):
    """Indices of one longest strictly increasing subsequence of ``seq``.

    Same length as ``dfjsp.repair.ladder._lis_length`` (patience sorting with
    ``bisect_left``); the predecessor links pick one such subsequence.
    """
    import bisect

    tails, tail_idx, parent = [], [], [None] * len(seq)
    for i, x in enumerate(seq):
        k = bisect.bisect_left(tails, x)
        if k == len(tails):
            tails.append(x)
            tail_idx.append(i)
        else:
            tails[k] = x
            tail_idx[k] = i
        parent[i] = tail_idx[k - 1] if k > 0 else None
    out, i = set(), tail_idx[-1] if tail_idx else None
    while i is not None:
        out.add(i)
        i = parent[i]
    return out


def change_items(inst, before, after_entries, t_now):
    """The changes of ``dfjsp.repair.ladder.count_changes``, one item per counted change.

    Tasks moved to another developer are ``assignee`` items. For tasks that stay with the same
    developer and had not started in ``before``, the tasks outside one longest increasing
    subsequence of their old positions are ``rank`` items, so their number is n - LIS.
    """
    from dfjsp.env.schedule import queue_order

    items = []
    for tid, e in after_entries.items():
        b = before.entries.get(tid)
        if b is not None and b.dev_id != e.dev_id:
            items.append({"task": int(tid), "kind": "assignee", "from_dev": int(b.dev_id),
                          "to_dev": int(e.dev_id)})
    qb = queue_order(inst, before, t_now)
    waiting = {t for q in qb.values() for t in q}
    qa: dict = {}
    for tid in waiting:
        e = after_entries.get(tid)
        if e is not None:
            qa.setdefault(e.dev_id, []).append(tid)
    for dev, new in sorted(qa.items()):
        new.sort(key=lambda tid: (after_entries[tid].start, tid))
        old_pos = {t: k for k, t in enumerate(qb.get(dev, []))}
        common = [t for t in new if t in old_pos]
        keep = _lis_members([old_pos[t] for t in common])
        for idx, t in enumerate(common):
            if idx not in keep:
                items.append({"task": int(t), "kind": "rank", "dev": int(dev),
                              "from_pos": old_pos[t] + 1, "to_pos": new.index(t) + 1})
    return items


class _RecordingPolicy:
    """Pass every call to the real selector; remember state, mask and the level it chose."""

    def __init__(self, inner, calls):
        self._inner = inner
        self._calls = calls

    @property
    def guard(self):
        return self._inner.guard

    def __call__(self, x, mask):
        level = self._inner(x, mask)
        self._calls.append((np.array(x, copy=True), list(mask), int(level)))
        return level


def _probabilities(policy, x, mask):
    """Probability of each level 0-3 the selector saw, or None for rule-based selectors."""
    if hasattr(policy, "nrm") and hasattr(policy, "model"):          # PPOPol (MaskablePPO)
        import torch
        nrm = policy.nrm
        obs = np.clip((x - nrm["mu"]) / nrm["sd"], -5, 5).astype("float32")
        allowed = (np.ones(4, dtype=bool) if getattr(policy, "nomask", False)
                   else np.array([1] + list(mask), dtype=bool))
        with torch.no_grad():
            obs_t, _ = policy.model.policy.obs_to_tensor(obs)
            dist = policy.model.policy.get_distribution(obs_t, action_masks=allowed[None, :])
            return dist.distribution.probs[0].cpu().numpy().astype(float).tolist()
    if hasattr(policy, "model") and hasattr(policy.model, "predict_proba"):   # cek_b3.Pol
        p = policy.model.predict_proba(x[None, :])[0]
        cls = list(policy.model.classes_)
        return [float(p[cls.index(lv)]) if lv in cls else 0.0 for lv in range(4)]
    return None


class Recorder:
    """Collects the events of one traced episode."""

    def __init__(self, lam):
        self.lam = float(lam)
        self.events: list[dict] = []
        self.initial: dict | None = None
        self.mean_effort: float | None = None
        self._current: dict | None = None

    # ------------------------------------------------------------------ events
    def open_event(self, env, event, before, tard_before):
        from dfjsp.env.state import total_tardiness

        E = env.mean_effort
        self.mean_effort = float(E)
        if self.initial is None:
            self.initial = before
        self._current = {
            "index": len(self.events), "time": float(env.t_now), "type": event.type,
            "disruptions": _disruptions(event),
            "affected": [{"type": ty, "tasks": sorted(int(x) for x in a)}
                         for ty, a in env.per_sub],
            "before": before, "aor": _snapshot(env),
            "tard_before": tard_before / E,
            "tard_aor": total_tardiness(env.inst, env.sched) / E,
            "subs": None,
        }

    def close_event(self, env, action):
        from dfjsp.env.state import total_tardiness

        ev, E = self._current, env.mean_effort
        if action == 3:
            probe = env.probes[3]
            ev["after"] = _snapshot(env, entries=probe["entries"], with_tasks=False)
            ev["tard_after"] = probe["tard"] / E
            ev["changes"] = int(probe["changes"])
        else:
            ev["after"] = _snapshot(env, with_tasks=False)
            ev["tard_after"] = total_tardiness(env.inst, env.sched) / E
            ev["changes"] = 0
        if ev["subs"] is None:      # AOR: no selector, every sub-decision stays at level 0
            ev["subs"] = [{"j": j, "type": a["type"], "affected": a["tasks"], "requested": 0,
                           "used": 0, "outcome": "aor", "J_before": ev["tard_aor"],
                           "J_after": ev["tard_aor"], "changes_after": 0}
                          for j, a in enumerate(ev["affected"])]
            ev["J_after_batch"] = ev["tard_aor"]
        ev.setdefault("change_items", [])
        self.events.append(ev)
        self._current = None

    # ------------------------------------------------------------------ batch
    def batch_done(self, env, lam, policy, calls, out):
        import cek_b3
        from dfjsp.env.state import total_tardiness

        inst, base, t, E = env.inst, env.sched, env.t_now, env.mean_effort
        A = [a for _, a in env.per_sub]
        types = [ty for ty, _ in env.per_sub]
        choice, final_sched = out[1], out[3]
        j_aor = total_tardiness(inst, base) / E
        cur, cur_j, chg = base, j_aor, 0
        subs = []
        for j, used in enumerate(choice):
            sub = {"j": j, "type": types[j], "affected": sorted(int(x) for x in A[j]),
                   "J_before": cur_j, "changes_before": chg, "used": int(used)}
            if policy is None:                      # Sequential Lookahead (teacher)
                trials = {0: (cur_j, cur, chg)}
                for lv in (1, 2, 3):
                    tr = cek_b3.try_level(inst, cur, t, A[j], cek_b3.OPS[lv])
                    if tr is not None:
                        jj, c = cek_b3.score(inst, base, tr, t, E, lam)
                        trials[lv] = (jj, tr, c)
                sub["options"] = {lv: v[0] for lv, v in trials.items()}
                sub["requested"] = int(used)
                sub["outcome"] = "dipakai" if used > 0 else "tingkat_0"
                cur_j, cur, chg = trials[used]
            else:
                x, mask, req = calls[j]
                sub.update(features=x.astype(float).tolist(), mask=mask, requested=req,
                           probs=_probabilities(policy, x, mask))
                if req > 0:
                    tr = cek_b3.try_level(inst, cur, t, A[j], cek_b3.OPS[req])
                    if tr is None:
                        sub["outcome"] = "gagal"
                    else:
                        jj, c = cek_b3.score(inst, base, tr, t, E, lam)
                        sub["J_candidate"], sub["changes_candidate"] = jj, c
                        if used == 0:
                            sub["outcome"] = "ditolak_penjaga"
                        else:
                            sub["outcome"] = "dipakai"
                            cur, cur_j, chg = tr, jj, c
                else:
                    sub["outcome"] = "tingkat_0"
            sub["J_after"], sub["changes_after"] = cur_j, chg
            subs.append(sub)
        self._current["subs"] = subs
        self._current["J_after_batch"] = (
            j_aor if final_sched is base
            else cek_b3.score(inst, base, final_sched, t, E, lam)[0])
        self._current["change_items"] = (
            [] if final_sched is base else change_items(inst, base, final_sched.entries, t))


def _probe_class(base, rec):
    from dfjsp.env.state import total_tardiness

    class TracingProbe(base):
        def _apply_event(self, event):
            before = _snapshot(self)
            tard_before = total_tardiness(self.inst, self.sched)
            super()._apply_event(event)
            rec.open_event(self, event, before, tard_before)

        def step(self, action):
            rec.close_event(self, action)
            return super().step(action)

    TracingProbe.__name__ = base.__name__
    return TracingProbe


def _run_batch(original, rec):
    def run_batch(env, lam, policy, base_state, record, *args, **kwargs):
        calls: list = []
        wrapped = _RecordingPolicy(policy, calls) if policy is not None else None
        out = original(env, lam, wrapped, base_state, record, *args, **kwargs)
        rec.batch_done(env, lam, policy, calls, out)
        return out
    return run_batch


@contextmanager
def hooks(ce, rec):
    """Install the two observers on ``cek_episode`` (``ce``) and ``cek_b3`` for one run."""
    import cek_b3

    probe, run_batch = ce.EpProbe, cek_b3.run_batch
    ce.EpProbe = _probe_class(probe, rec)
    cek_b3.run_batch = _run_batch(run_batch, rec)
    try:
        yield rec
    finally:
        ce.EpProbe, cek_b3.run_batch = probe, run_batch


__all__ = ["Recorder", "hooks", "plain"]
