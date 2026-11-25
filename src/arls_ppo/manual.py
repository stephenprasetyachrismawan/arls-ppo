"""The user as the selector (seam B3), judged by the same J and the same schedule checks.

Two ways to decide an event, both inside ``cek_episode.run`` so the episode, the disruptions,
the change count and the verifier are those of the thesis:

* **levels**: the user picks a repair level 0-3 for each sub-decision. The choice goes
  through ``cek_b3.run_batch`` exactly like the fixed-level selector: an unavailable level
  falls back to 0, and with ``guard=True`` a level that makes J worse is refused;
* **moves**: the user drags cards on the board (``{"task", "to_dev", "to_pos"}``) on the
  schedule after AOR. A move is accepted only for a waiting task, a developer with the
  skill, and a pair that passes ``dfjsp.repair.masks.is_pair_feasible`` (the check of the
  repair operators). The queues are rebuilt with ``set_queue`` and ``recompute_timeline``,
  and the result must have no hard violation of ``dfjsp.verify``. J is ``cek_b3.score``.

Events without a choice stay at AOR (level 0).
"""
from __future__ import annotations

from .episode import EpisodeTrace, trace_episode


class MoveRejected(ValueError):
    """A card move that breaks a scheduling rule; ``rule`` is the verifier code."""

    def __init__(self, message: str, rule: str, task: int | None = None,
                 dev: int | None = None):
        super().__init__(message)
        self.rule, self.task, self.dev = rule, task, dev


class _Scripted:
    """Selector that returns the levels chosen by the user, in sub-decision order."""

    def __init__(self, levels, guard):
        self.levels, self.guard, self._j = list(levels), guard, 0

    def __call__(self, x, mask):
        level = self.levels[self._j] if self._j < len(self.levels) else 0
        self._j += 1
        return int(level)


def apply_moves(inst, base, t_now, moves, *, rework=()):
    """Schedule ``base`` with the card ``moves`` applied, or :class:`MoveRejected`."""
    from dfjsp.env.schedule import partition, queue_order, recompute_timeline
    from dfjsp.repair.ladder import _light_copy, set_queue
    from dfjsp.repair.masks import is_pair_feasible
    from dfjsp.verify import periksa_jadwal, periksa_peralihan

    by_id = {t.task_id: t for t in inst.tasks}
    cur = _light_copy(base)
    for move in moves:
        tid, dev = int(move["task"]), int(move["to_dev"])
        if tid not in by_id or tid not in cur.entries:
            raise MoveRejected(f"issue {tid + 1} tidak ada di jadwal", "A1", tid, dev)
        _, _, waiting = partition(inst, cur, t_now)
        if tid not in waiting:
            raise MoveRejected("hanya issue berstatus To Do yang bisa dipindah; issue ini "
                               "sudah dikerjakan atau selesai", "status", tid, dev)
        if dev not in by_id[tid].eligible_devs:
            raise MoveRejected(f"pengembang {dev + 1} tidak punya keahlian untuk issue ini",
                               "A3", tid, dev)
        src = cur.entries[tid].dev_id
        if dev != src and not is_pair_feasible(tid, dev, inst, cur, t_now):
            raise MoveRejected(f"pengembang {dev + 1} tidak bisa menerima issue ini sekarang: "
                               "beban melewati kapasitas, pengembang sedang tidak tersedia, "
                               "atau pendahulunya belum terjadwal", "K1", tid, dev)
        queues = queue_order(inst, cur, t_now)
        target = [x for x in queues.get(dev, []) if x != tid]
        pos = int(move.get("to_pos") or len(target) + 1)
        target.insert(min(max(pos, 1), len(target) + 1) - 1, tid)
        if dev != src:
            set_queue(cur, src, [x for x in queues.get(src, []) if x != tid], t_now)
        set_queue(cur, dev, target, t_now)
        recompute_timeline(inst, cur, t_now)
    hard = [p for p in periksa_jadwal(inst, cur, t_now, rework=rework)
            + periksa_peralihan(inst, base, cur, t_now) if p.keras]
    if hard:
        p = hard[0]
        raise MoveRejected(f"jadwal hasil pemindahan melanggar aturan {p.kode}: {p.pesan}",
                           p.kode, p.tugas, p.pengembang)
    return cur


def _decide(original, state):
    def decide(env, method, lam):
        if not method.startswith("manual"):
            return original(env, method, lam)
        import cek_b3
        from dfjsp.env.repair_env import RepairEnv

        k = state["k"]
        state["k"] += 1
        if k in state["moves"]:
            from ._trace import change_items

            inst, base, t = env.inst, env.sched, env.t_now
            rework = getattr(getattr(env, "pengamat", None), "rework", set())
            cur = apply_moves(inst, base, t, state["moves"][k], rework=rework)
            jj, c = cek_b3.score(inst, base, cur, t, env.mean_effort, lam)
            env._manual_move = {"items": change_items(inst, base, cur.entries, t), "J": jj,
                                "moves": list(state["moves"][k])}
            if state.get("explain"):
                from . import _explain
                env._manual_move["explain"] = _explain.value(
                    inst, base, cur, t, env.mean_effort, lam, "manual", (jj, c))
            return cur, c
        pol = _Scripted([state["levels"].get((k, j), 0) for j in range(len(env.per_sub))],
                        method.endswith("_g"))
        _, _, _, cur, chg = cek_b3.run_batch(
            env, lam, pol, RepairEnv._state(env), record=False, want_sched=True,
            catat=getattr(env, "catat_tingkat", None),
            waktu_pemilih=getattr(env, "waktu_pemilih", None))
        return (None if cur is env.sched else cur), chg
    return decide


def _levels(levels):
    out = {}
    for key, level in (levels or {}).items():
        if isinstance(key, str):
            k, j = key.split(":")
            key = (int(k), int(j))
        out[(int(key[0]), int(key[1]))] = int(level)
    return out


def run(instance, capacity: float, intensity: float, seed: int, lam: float, *,
        levels: dict | None = None, moves: dict | None = None, guard: bool = False,
        options: bool = True, check: bool = True, explain: bool = False) -> EpisodeTrace:
    """One traced episode decided by the user.

    Parameters
    ----------
    levels : dict, optional
        ``{(event, sub): level}`` (or ``"event:sub"`` keys); missing sub-decisions use 0.
    moves : dict, optional
        ``{event: [{"task": id, "to_dev": dev, "to_pos": position}, ...]}``; an event with
        moves ignores ``levels``.
    guard : bool, default=False
        Refuse a chosen level that makes J worse, as the guard of ARLS-PPO does.
    options : bool, default=True
        Record the J of every level at every sub-decision, to help the user choose.
    explain : bool, default=False
        Add the number log, as ``trace_episode(..., explain=True)``; a card move is explained
        by the value J of the moved schedule.
    """
    state = {"k": 0, "levels": _levels(levels), "explain": bool(explain),
             "moves": {int(k): v for k, v in (moves or {}).items()}}
    from . import episode as ep

    ce, _ = ep._start(instance, seed, "manual", 0, allow_manual=True)
    original = ce.decide
    ce.decide = _decide(original, state)
    try:
        return trace_episode(instance, capacity, intensity, seed,
                             "manual_g" if guard else "manual", lam, check=check,
                             options=options, explain=explain, _allow_manual=True)
    finally:
        ce.decide = original
