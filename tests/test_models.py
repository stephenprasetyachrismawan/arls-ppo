"""Seam A1: the locked model files are verified against docs/KUNCI_UJI.md before use."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

from arls_ppo import models

ROOT = Path(__file__).resolve().parents[1]


def _hashes_in_test_lock_document():
    """SHA-256 lines from section 2 of the test lock document, approved 19 November 2026."""
    text = (ROOT / "docs" / "KUNCI_UJI.md").read_text(encoding="utf-8")
    return dict((path, digest) for digest, path in
                re.findall(r"^([0-9a-f]{64})  (\S+)$", text, flags=re.MULTILINE))


def _expand(lock):
    """``e9c_s{0,1,2}/norm.pkl`` in the document stands for three files with one hash."""
    out = {}
    for path, digest in lock.items():
        m = re.fullmatch(r"(.*)\{(.*)\}(.*)", path)
        for part in (m.group(2).split(",") if m else [None]):
            out[f"{m.group(1)}{part}{m.group(3)}" if m else path] = digest
    return out


def test_every_locked_file_matches_the_test_lock_document():
    lock = _expand(_hashes_in_test_lock_document())
    assert len(lock) == 11
    checked = models.verify_models()
    assert {f.path: f.expected for f in checked} == lock
    assert all(f.ok for f in checked)


def test_one_changed_byte_rejects_the_model(tmp_path):
    copy = tmp_path / "e15_validasi"
    shutil.copytree(models.RUN_DIR, copy)
    target = copy / "e9c_s1" / "bc_ppo.zip"
    data = bytearray(target.read_bytes())
    data[100] ^= 0xFF
    target.write_bytes(bytes(data))

    with pytest.raises(models.ModelMismatch, match=r"e9c_s1/bc_ppo\.zip"):
        models.verify_models(copy)


def test_a_missing_file_rejects_the_model(tmp_path):
    copy = tmp_path / "e15_validasi"
    shutil.copytree(models.RUN_DIR, copy)
    (copy / "hgb7d.pkl").unlink()

    with pytest.raises(models.ModelMismatch, match=r"hgb7d\.pkl"):
        models.verify_models(copy)
