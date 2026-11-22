"""Lingkungan simulasi perbaikan jadwal.

Nama publik dimuat malas (PEP 562). Paket ``dfjsp.repair`` mengimpor modul kecil di sini,
misalnya ``dfjsp.env.schedule``, sedangkan ``RepairEnv`` mengimpor ``dfjsp.repair``.
Tanpa pemuatan malas, urutan impor itu melingkar.

Examples
--------
>>> from dfjsp.env import STATE_DIM, STATE_FEATURES
>>> STATE_DIM
9
"""

_EXPORTS = {
    "RepairEnv": "repair_env",
    "DecisionContext": "repair_env",
    "GymRepairEnv": "gym_env",
    "DisruptionGenerator": "disruptions",
    "STATE_DIM": "state",
    "STATE_FEATURES": "state",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f".{_EXPORTS[name]}", __name__)
    return getattr(module, name)
