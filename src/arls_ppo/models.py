"""Locked model files of recipe E15 E9c and their check against the test lock document.

The hashes below are copied from section 2 of ``docs/KUNCI_UJI.md`` (approved by the thesis
author on 19 November 2026, before the test data was opened). A model is used only when every
file has exactly this hash.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from . import ROOT

#: Folder with the locked models, same path as in the thesis repository.
RUN_DIR = ROOT / "exp1" / "dfjsp-ppo-repair" / "runs" / "e15_validasi"

_NORM = "28353feeea419bec41bbbfb05cdabfad91af88e75dfe491f54a3993a0dcb5ee8"

#: Relative path inside :data:`RUN_DIR` -> SHA-256 from ``docs/KUNCI_UJI.md`` section 2.
LOCKED_HASHES: dict[str, str] = {
    "e9c_s0/bc_ppo.zip": "58b40fed9b61faa75906dd855ae9f5e81d8076e7d30e42cfb178f70e5dc1563f",
    "e9c_s0/bc.zip": "b5bc3d0db83520f21dbda6ef7e7f01539006a18dd0e4fc4691ee2348df8a6744",
    "e9c_s1/bc_ppo.zip": "047dc19be6f268d9411f3f1fa8dc4abc0a6a9bb7196aa1aca6521d6efe51cdb1",
    "e9c_s1/bc.zip": "8dbe62a1a06b6096c6d5a5b24238cded1e5f648754929e055e613d15b28cb72f",
    "e9c_s2/bc_ppo.zip": "6a41d5a1fa8567b2581278a1fab4e7886ea10622294cf09f5e8aacc74e1d9bc3",
    "e9c_s2/bc.zip": "b2148ba981b5859b5dea2d2f1b3d0f4ea8210285288aaeec2570a9a8d81be6b0",
    "e9c_s0/norm.pkl": _NORM,
    "e9c_s1/norm.pkl": _NORM,
    "e9c_s2/norm.pkl": _NORM,
    "hgb7d.pkl": "c43890937fac922332ccd3652e2561c617e37e4f74311fc8f69bb48628dd44f7",
    "b3_model.pkl": "367de178ddd76c7d24554d784a0d83c4f4e6062e5872522a485d13e2b5effd9e",
}


class ModelMismatch(RuntimeError):
    """A locked model file is missing or differs from the test lock document."""


@dataclass(frozen=True)
class ModelFile:
    """Result of checking one locked file."""

    path: str
    expected: str
    actual: str | None

    @property
    def ok(self) -> bool:
        return self.actual == self.expected


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_models(run_dir: str | Path = RUN_DIR) -> list[ModelFile]:
    """Check every locked file in ``run_dir``; raise :class:`ModelMismatch` on any difference.

    Returns the list of checked files (all ``ok``) so callers can show the hashes.
    """
    run_dir = Path(run_dir)
    checked = [ModelFile(rel, digest, _sha256(run_dir / rel))
               for rel, digest in LOCKED_HASHES.items()]
    bad = [f for f in checked if not f.ok]
    if bad:
        lines = [f"{f.path}: {'missing' if f.actual is None else 'sha256 ' + f.actual}, "
                 f"expected {f.expected}" for f in bad]
        raise ModelMismatch("model files differ from docs/KUNCI_UJI.md:\n  " + "\n  ".join(lines))
    return checked
