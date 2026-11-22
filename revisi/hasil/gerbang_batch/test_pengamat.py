"""Pengamat pemeriksa jadwal di loop episode ``cek_episode.run`` (gerbang 2.5).

Seam disetujui pemilik 17 November 2026: pengamat hidup bila ``PERIKSA=1``, memeriksa setiap
keputusan dengan ``dfjsp.verify``, dan tidak boleh mengubah hasil episode. Hanya data latih
(MK01) dan seed validasi. ``cek_episode`` membaca ``sys.argv`` saat diimpor, jadi episode
dijalankan di proses terpisah.

Jalankan dari folder ini::

    PYTHONPATH=../../../exp1/dfjsp-ppo-repair/src $PY -m pytest -p no:cacheprovider -q test_pengamat.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

DI_SINI = Path(__file__).resolve().parent
SRC = (DI_SINI / "../../../exp1/dfjsp-ppo-repair/src").resolve()
JOB = ["MK01", 1.0, 0.15, 5000, "acak_g", 3.0]

DRIVER = r"""
import copy, json, os, sys
sys.argv = ["cek_episode.py", os.environ["SRC_DFJSP"], "/dev/null", "/dev/null"]
import cek_episode as c
if os.environ.get("RUSAK"):
    asli = c.decide
    def rusak(env, method, lam):
        # Sengaja melanggar B1: tugas yang sedang jalan dimundurkan selesainya 1 jam.
        s = copy.deepcopy(env.sched)
        jalan = sorted(t for t, e in s.entries.items() if e.start <= env.t_now < e.end)
        if jalan:
            s.entries[jalan[0]].end += 1.0
            return s, 0
        return asli(env, method, lam)
    c.decide = rusak
print(json.dumps(c.run(tuple(json.loads(os.environ["JOB"])))))
"""


def _jalan(periksa=False, rusak=False, job=JOB):
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": f"{DI_SINI}:{SRC}",
           "SRC_DFJSP": str(SRC), "JOB": json.dumps(job), "OMP_NUM_THREADS": "1"}
    if periksa:
        env["PERIKSA"] = "1"
    if rusak:
        env["RUSAK"] = "1"
    keluar = subprocess.run([sys.executable, "-c", DRIVER], cwd=DI_SINI, env=env,
                            capture_output=True, text=True, timeout=600)
    assert keluar.returncode == 0, keluar.stderr[-3000:]
    return json.loads(keluar.stdout.strip().splitlines()[-1])


def test_pengamat_tidak_mengubah_hasil_episode_dan_menulis_ringkasan():
    tanpa = _jalan()
    dengan = _jalan(periksa=True)
    for kolom in ("telat", "ubah", "J", "keputusan"):
        assert dengan[kolom] == tanpa[kolom], kolom
    assert "periksa" not in tanpa
    assert dengan["periksa"]["keputusan_diperiksa"] == dengan["keputusan"]


def test_pengamat_melaporkan_tugas_sedang_jalan_yang_diubah():
    hasil = _jalan(periksa=True, rusak=True)
    hitung = hasil["periksa"]["hitung"]
    assert hitung.get("peralihan:B1:keras", 0) > 0
    assert any(c["kode"] == "B1" for c in hasil["periksa"]["contoh"])
