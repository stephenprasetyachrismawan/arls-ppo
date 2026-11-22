"""Full-grid check: rerun every stored E15 validation episode and compare (difference 0.0).

Usage (from the repository root, about an hour with 8 processes)::

    OMP_NUM_THREADS=1 PYTHONPATH=src python -m arls_ppo.parity --processes 8 --out parity.json
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import get_context

from . import SCRIPTS

#: Stored E15 validation files of the reported methods and the PPO model seed they used.
FILES = {"ep_aturan": None, "ep_hgb7d": None,
         "ep_e9c_s0": 0, "ep_e9c_s1": 1, "ep_e9c_s2": 2}
FIELDS = ("telat", "ubah", "J", "keputusan")


def stored_records() -> list[dict]:
    """Stored validation records of the reported methods, each with its PPO ``model_seed``.

    ``ep_hgb7d`` also holds ``il`` (boosting without guard), which the thesis does not report.
    """
    from .episode import METHODS

    out = []
    for name, model_seed in FILES.items():
        for r in json.loads((SCRIPTS / "e15" / f"{name}.json").read_text()):
            if r["metode"] in METHODS:
                out.append({**r, "model_seed": model_seed})
    return out


def _rerun(r: dict) -> dict:
    from .episode import run_episode

    res = run_episode(r["instans"], r["kapasitas"], r["intensitas"], r["seed"], r["metode"],
                      r["lam"], model_seed=r["model_seed"] or 0)
    return {"stored": {f: r[f] for f in FIELDS}, "rerun": {f: res[f] for f in FIELDS},
            "job": [r["instans"], r["kapasitas"], r["intensitas"], r["seed"], r["metode"],
                    r["lam"], r["model_seed"]]}


def check(records: list[dict], processes: int = 8) -> dict:
    """Rerun ``records`` and report every field that differs."""
    if processes == 1:
        pairs = [_rerun(r) for r in records]
    else:
        with get_context("fork").Pool(processes) as pool:
            pairs = pool.map(_rerun, records, chunksize=1)
    mismatches = [p for p in pairs if p["stored"] != p["rerun"]]
    max_diff = max((abs(p["stored"]["J"] - p["rerun"]["J"]) for p in pairs), default=0.0)
    return {"episodes": len(pairs), "mismatches": mismatches, "max_abs_diff_J": max_diff}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--processes", type=int, default=8)
    p.add_argument("--out", default=None, help="write the report as JSON")
    a = p.parse_args(argv)
    report = check(stored_records(), a.processes)
    print(f"{report['episodes']} episodes, {len(report['mismatches'])} mismatches, "
          f"max |ΔJ| = {report['max_abs_diff_J']}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(report, f, indent=1)
    return 0 if not report["mismatches"] else 1


if __name__ == "__main__":
    sys.exit(main())
