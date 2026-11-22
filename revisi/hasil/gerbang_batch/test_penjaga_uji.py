"""Penjaga data uji (``penjaga_uji``): skrip percobaan yang mengimpornya tidak bisa membuka
berkas instans uji MK03, MK04, MK09.

Jalankan dari folder ini::

    PYTHONPATH=../../../exp1/dfjsp-ppo-repair/src $PY -m pytest -p no:cacheprovider -q test_penjaga_uji.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

DI_SINI = Path(__file__).resolve().parent
SRC = DI_SINI.parents[2] / "exp1" / "dfjsp-ppo-repair" / "src"


def _jalankan(kode: str) -> subprocess.CompletedProcess:
    # Proses baru, supaya penjaga yang terpasang tidak ikut ke proses pytest ini.
    return subprocess.run([sys.executable, "-c", kode], cwd=DI_SINI, capture_output=True,
                          text=True, env={"PYTHONPATH": f"{DI_SINI}:{SRC}"}, timeout=120)


@pytest.mark.parametrize("nama", ["MK03", "MK04", "MK09"])
def test_loading_a_test_instance_after_importing_the_guard_fails(nama):
    hasil = _jalankan("import penjaga_uji\n"
                      "from dfjsp.datasets import load_brandimarte\n"
                      f"load_brandimarte(['{nama}'])\n")
    assert hasil.returncode != 0
    assert "data uji" in hasil.stderr
    assert nama in hasil.stderr


def test_training_instances_still_load_after_importing_the_guard():
    hasil = _jalankan("import penjaga_uji\n"
                      "from dfjsp.datasets import BRANDIMARTE_TRAIN, load_brandimarte\n"
                      "print(sorted(load_brandimarte(list(BRANDIMARTE_TRAIN))))\n")
    assert hasil.returncode == 0, hasil.stderr
    assert hasil.stdout.strip() == ("['MK01', 'MK02', 'MK05', 'MK06', 'MK07', 'MK08', 'MK10']")


def test_pool_workers_of_a_guarded_script_are_guarded_too(tmp_path):
    # Skrip percobaan memakai Pool; pekerja forkserver memuat ulang skrip, jadi impor di
    # bagian atas skrip harus memasang penjaga di pekerja juga.
    skrip = tmp_path / "skrip_pool.py"
    skrip.write_text("import penjaga_uji\n"
                     "from multiprocessing import get_context\n"
                     "from dfjsp.datasets import load_brandimarte\n"
                     "def muat(nama):\n"
                     "    return sorted(load_brandimarte([nama]))\n"
                     "if __name__ == '__main__':\n"
                     "    with get_context('forkserver').Pool(1) as pool:\n"
                     "        print(pool.map(muat, ['MK01']))\n"
                     "        print(pool.map(muat, ['MK03']))\n")
    hasil = subprocess.run([sys.executable, str(skrip)], cwd=DI_SINI, capture_output=True,
                           text=True, env={"PYTHONPATH": f"{DI_SINI}:{SRC}"}, timeout=120)
    assert hasil.stdout.splitlines() == ["[['MK01']]"]
    assert hasil.returncode != 0
    assert "data uji" in hasil.stderr


def _skrip_pemuat_data():
    return sorted(p for p in DI_SINI.glob("*.py")
                  if not p.name.startswith("test_") and p.name != "penjaga_uji.py"
                  and "load_brandimarte" in p.read_text(encoding="utf-8"))


def test_every_script_that_loads_instances_imports_the_guard_before_dfjsp():
    skrip = _skrip_pemuat_data()
    assert len(skrip) >= 10           # kalau 0, glob salah dan tes ini lolos tanpa arti
    tanpa = []
    for p in skrip:
        baris = p.read_text(encoding="utf-8").splitlines()
        penjaga = [i for i, b in enumerate(baris) if b.split("#")[0].strip() == "import penjaga_uji"]
        dfjsp = [i for i, b in enumerate(baris) if b.lstrip().startswith(("from dfjsp", "import dfjsp"))]
        if not penjaga or (dfjsp and penjaga[0] > dfjsp[0]):
            tanpa.append(p.name)
    assert tanpa == []
