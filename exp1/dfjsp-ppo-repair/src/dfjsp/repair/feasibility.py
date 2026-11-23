"""Probe kelayakan seluruh operator sekaligus."""

from __future__ import annotations

from .base import OperatorFeasibility, RepairOperator


def inspect_all_operators(
    operators: dict[int, RepairOperator], inst, sched, t_now: float
) -> dict[int, OperatorFeasibility]:
    """Probe kelayakan semua operator (terurut menurut aksi) tanpa mengubah jadwal.

    Examples
    --------
    >>> from dfjsp.datasets import load_brandimarte
    >>> from dfjsp.env import RepairEnv
    >>> env = RepairEnv()
    >>> state, info = env.reset(load_brandimarte(["MK01"])["MK01"], intensity=0.05, seed=0)
    >>> from dfjsp.repair import default_operators, inspect_all_operators
    >>> probe = inspect_all_operators(default_operators(), env.inst, env.sched, env.t_now)
    >>> {aksi: p.choice_count for aksi, p in probe.items()}
    {0: 1, 1: 1, 2: 10, 3: 6}
    """
    return {
        int(action): operator.inspect(inst, sched, t_now)
        for action, operator in sorted(operators.items())
    }
