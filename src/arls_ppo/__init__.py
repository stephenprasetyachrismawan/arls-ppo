"""ARLS-PPO: thin public API over the code that produced the thesis numbers.

The experiment scripts (``revisi/hasil/gerbang_batch``) and the ``dfjsp`` library are kept
byte-for-byte as they were when the test data was opened, in the same folder layout as the
thesis repository. This package only adds two things on top of them:

* :mod:`arls_ppo.models`: check the locked model files against the test lock document;
* :mod:`arls_ppo.episode`: run one episode with the same code, optionally with a trace of
  every event for display (used by ScheduleSim).
"""
from __future__ import annotations

import sys
from pathlib import Path

#: Repository root (this file is ``src/arls_ppo/__init__.py``).
ROOT = Path(__file__).resolve().parents[2]
#: ``dfjsp`` library, exactly as in the thesis repository.
LIBRARY = ROOT / "exp1" / "dfjsp-ppo-repair" / "src"
#: Experiment scripts of recipe E15 E9c, exactly as in the thesis repository.
SCRIPTS = ROOT / "revisi" / "hasil" / "gerbang_batch"

if not LIBRARY.is_dir() or not SCRIPTS.is_dir():
    raise ImportError("arls_ppo must run from a checkout of the arls-ppo repository "
                      "(pip install -e . or PYTHONPATH=src); the dfjsp library and the "
                      f"experiment scripts were not found under {ROOT}")

for _path in (SCRIPTS, LIBRARY):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

__all__ = ["ROOT", "LIBRARY", "SCRIPTS"]
