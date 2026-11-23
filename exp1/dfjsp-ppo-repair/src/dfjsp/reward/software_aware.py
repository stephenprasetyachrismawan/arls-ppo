"""Reward per langkah: SoftwareAwareReward (6 suku) dan TardinessReward (1 suku)."""

from __future__ import annotations

from ..repair.actions import ESCALATE
from ..utils import ParamsMixin


def _scale(step_metrics: dict) -> float:
    """Skala normalisasi (horizon episode); nilai nol atau negatif dianggap 1."""
    scale = float(step_metrics.get("reward_scale", 1.0))
    return scale if scale > 1e-8 else 1.0


class SoftwareAwareReward(ParamsMixin):
    """Reward sadar-proyek-perangkat-lunak dengan enam suku.

    ``R_t`` adalah jumlah suku berikut. Suku waktu dibagi ``reward_scale`` (horizon
    episode) supaya sebanding antar instans; suku lain tanpa satuan.

    ====================  ==========================================================
    ``tard``              ``-alpha * perubahan_keterlambatan / skala``
    ``imbalance``         ``-beta * ketimpangan_beban / skala``
    ``nervous``           ``-gamma * nervousness_langkah``
    ``ctx``               ``-delta * alih_konteks_langkah``
    ``escalate``          ``-eta * beban_escalate`` (hanya bila aksi Escalate)
    ``makespan_bonus``    ``lam * penurunan_makespan / skala``
    ====================  ==========================================================

    Parameters
    ----------
    alpha, beta, gamma, delta, eta, lam : float
        Bobot suku (Tabel 3.5 Bab III).

    Examples
    --------
    >>> from dfjsp.reward import SoftwareAwareReward
    >>> step = {"d_tardiness": 2.0, "imbalance": 0.0, "nervousness_step": 0.0,
    ...         "context_switch_step": 0.0, "makespan_decrease": 0.0}
    >>> SoftwareAwareReward()(step, action=0)
    -2.0
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.5, gamma: float = 1.0,
                 delta: float = 1.5, eta: float = 0.3, lam: float = 0.2):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.eta = eta
        self.lam = lam

    def terms(self, step_metrics: dict, action: int) -> dict[str, float]:
        """Enam suku reward untuk satu langkah, masing-masing terpisah."""
        scale = _scale(step_metrics)
        # Beban Escalate = tugas dipindah / jumlah pengembang. Bila tidak dicatat,
        # dianggap 1. Escalate yang tidak memindahkan apa pun tidak dihukum.
        escalate_load = float(step_metrics.get("escalate_load", 1.0))
        is_escalate = 1.0 if int(action) == ESCALATE else 0.0
        return {
            "tard": -self.alpha * float(step_metrics["d_tardiness"]) / scale,
            "imbalance": -self.beta * float(step_metrics["imbalance"]) / scale,
            "nervous": -self.gamma * float(step_metrics["nervousness_step"]),
            "ctx": -self.delta * float(step_metrics["context_switch_step"]),
            "escalate": -self.eta * escalate_load * is_escalate,
            "makespan_bonus": (self.lam * max(0.0, float(step_metrics["makespan_decrease"]))
                               / scale),
        }

    def __call__(self, step_metrics: dict, action: int) -> float:
        """Reward satu langkah = jumlah semua suku."""
        return float(sum(self.terms(step_metrics, action).values()))


class TardinessReward(ParamsMixin):
    """Reward yang hanya melihat perubahan keterlambatan (varian PPO, PPO+HM, PPO+HM+LM).

    Parameters
    ----------
    alpha : float, default=1.0
        Bobot suku keterlambatan.

    Examples
    --------
    Keterlambatan naik 2 jam pada episode berskala 10 jam, jadi reward -0,2:

    >>> from dfjsp.reward import TardinessReward
    >>> reward = TardinessReward(alpha=1.0)
    >>> reward({"d_tardiness": 2.0, "reward_scale": 10.0}, 0)
    -0.2
    >>> reward.terms({"d_tardiness": -3.0, "reward_scale": 10.0}, 1)
    {'tard': 0.3}
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def terms(self, step_metrics: dict, action: int) -> dict[str, float]:
        """Satu suku: ``tard``."""
        scale = _scale(step_metrics)
        return {"tard": -self.alpha * float(step_metrics["d_tardiness"]) / scale}

    def __call__(self, step_metrics: dict, action: int) -> float:
        return float(sum(self.terms(step_metrics, action).values()))
