"""Pemilih strategi perbaikan jadwal: aturan, PPO, dan PPO pasangan.

Semua pemilih mewarisi ``BaseSelector`` (gaya scikit-learn): parameter di ``__init__``,
belajar lewat ``fit``, memutuskan lewat ``predict``.

Pemilih PPO dimuat malas (PEP 562) supaya ``import dfjsp.selectors`` untuk pemilih
aturan tidak ikut memuat torch.

Examples
--------
>>> from dfjsp.selectors import ThresholdSelector
>>> ThresholdSelector(tau1=0.1)
ThresholdSelector(tau1=0.1)
"""

from ._rules import MaintainSelector, RandomSelector, ThresholdSelector

_LAZY = {
    "PPOSelector": "_ppo",
    "PairPPOSelector": "_pair",
}

__all__ = sorted(["MaintainSelector", "RandomSelector", "ThresholdSelector", *_LAZY])


def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f".{_LAZY[name]}", __name__)
    return getattr(module, name)
