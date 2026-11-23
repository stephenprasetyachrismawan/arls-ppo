"""Nomor dan nama empat strategi perbaikan (dipakai bersama oleh repair, env, dan reward)."""

from __future__ import annotations

MAINTAIN, DELAY, REASSIGN, ESCALATE = 0, 1, 2, 3
N_ACTIONS = 4
ACTION_NAMES = ["Maintain", "Delay", "Reassign", "Escalate"]
