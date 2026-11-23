"""Jalur pembukaan data uji (``buka_uji``), versi ringan pilihan B.

Seam disetujui pemilik 18 November 2026: ``buka_uji.py --buka-uji`` memanggil
``cek_episode.run`` yang sama dengan validasi; ``rencana_uji()`` memberi daftar pekerjaan
uji, ``catat_pembukaan()`` mengisi ``buka_uji.log`` dan menolak pembukaan kedua. Penjaga
data uji hanya dilepas di proses itu lewat ``penjaga_uji.izinkan()``. Tes ini tidak membuka
berkas uji sebenarnya.

    PYTHONPATH=../../../exp1/dfjsp-ppo-repair/src $PY -m pytest -p no:cacheprovider -q test_buka_uji.py
"""
from __future__ import annotations

from dfjsp.datasets import BENCHMARK_UJI

from buka_uji import rencana_uji


def test_rencana_uji_mk_dan_benchmark_dengan_seed_masing_masing():
    jobs = rencana_uji(metode=["seq"], lams=[3.0])
    mk = [j for j in jobs if j[0] in ("MK03", "MK04", "MK09")]
    bench = [j for j in jobs if j[0] in BENCHMARK_UJI]
    assert len(mk) == 3 * 3 * 3 * 10 and len(bench) == 35 * 3 * 3 * 5
    assert len(jobs) == len(mk) + len(bench)
    assert {j[3] for j in mk} == set(range(10000, 10010))
    assert {j[3] for j in bench} == set(range(11000, 11005))
    assert {j[0] for j in bench} == set(BENCHMARK_UJI)
    assert {(j[1], j[2]) for j in jobs} == {(c, i) for c in (1.0, 1.5, 2.0)
                                            for i in (0.05, 0.15, 0.30)}
    assert all(j[4:] == ("seq", 3.0) for j in jobs)


def test_pembukaan_dicatat_dan_pembukaan_kedua_ditolak(tmp_path):
    import json

    import pytest

    from buka_uji import catat_pembukaan

    log = tmp_path / "buka_uji.log"
    catat_pembukaan(log, {"resep": "p1", "hash_model": {"akhir_s0/bc_ppo.zip": "ab12"}})
    isi = [json.loads(x) for x in log.read_text().splitlines()]
    assert len(isi) == 1
    assert isi[0]["resep"] == "p1" and isi[0]["hash_model"] == {"akhir_s0/bc_ppo.zip": "ab12"}
    assert isi[0]["waktu"] and len(isi[0]["commit"]) == 40
    with pytest.raises(RuntimeError, match="sudah pernah dibuka"):
        catat_pembukaan(log, {"resep": "p1"})
    assert len(log.read_text().splitlines()) == 1


def test_penjaga_dilepas_hanya_sesudah_izinkan(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    # Berkas buatan bernama instans uji, dibuat di subproses sebelum penjaga diimpor (proses
    # pytest bisa saja sudah memasang penjaga lewat modul lain, misalnya sifat_instans).
    palsu = tmp_path / "MK03.json"
    kode = (
        f"p = {str(palsu)!r}\n"
        "open(p, 'w').write('{}')\n"
        "import penjaga_uji\n"
        "try:\n    open(p); print('terbuka-sebelum')\n"
        "except PermissionError:\n    print('ditolak-sebelum')\n"
        "print(penjaga_uji.diizinkan())\n"
        "penjaga_uji.izinkan('tes')\n"
        "open(p); print('terbuka-sesudah', penjaga_uji.diizinkan())\n")
    di_sini = Path(__file__).resolve().parent
    keluar = subprocess.run([sys.executable, "-c", kode], cwd=di_sini, capture_output=True,
                            text=True, env={"PYTHONPATH": str(di_sini)}, timeout=60)
    assert keluar.returncode == 0, keluar.stderr
    assert keluar.stdout.split("\n")[:3] == ["ditolak-sebelum", "False", "terbuka-sesudah True"]


def test_load_any_meneruskan_izin_buka_uji_hanya_bila_penjaga_dilepas(monkeypatch):
    import sys
    from types import SimpleNamespace

    import instans_acak

    dipanggil = []
    monkeypatch.setattr(instans_acak, "load_benchmark",
                        lambda names, **kw: dipanggil.append(kw) or {names[0]: "inst"})
    monkeypatch.delitem(sys.modules, "penjaga_uji", raising=False)
    instans_acak.load_any("barnes/mt10c1")
    monkeypatch.setitem(sys.modules, "penjaga_uji", SimpleNamespace(diizinkan=lambda: False))
    instans_acak.load_any("barnes/mt10c1")
    monkeypatch.setitem(sys.modules, "penjaga_uji", SimpleNamespace(diizinkan=lambda: True))
    instans_acak.load_any("barnes/mt10c1")
    assert [kw.get("buka_uji", False) for kw in dipanggil] == [False, False, True]


def test_tanpa_bendera_buka_uji_berhenti_tanpa_mencatat(tmp_path):
    from buka_uji import main

    log = tmp_path / "buka_uji.log"
    kode = main(["--resep", str(tmp_path / "akhir_s"), "--b3", "b3.pkl", "--hgb", "h.pkl",
                 "--keluar", str(tmp_path / "uji"), "--log", str(log)])
    assert kode == 2
    assert not log.exists() and not (tmp_path / "uji").exists()


def test_rencana_lengkap_tiap_seed_model_memakai_foldernya_sendiri(tmp_path):
    from buka_uji import rencana_lengkap

    r = rencana_lengkap("m/akhir_s", "b3.pkl", "h.pkl", tmp_path)
    ppo = [x for x in r if x["ppo_dir"] is not None]
    assert [x["ppo_dir"] for x in ppo] == ["m/akhir_s0", "m/akhir_s1", "m/akhir_s2"]
    assert [x["keluar"].name for x in ppo] == ["ep_ppo_s0.json", "ep_ppo_s1.json",
                                              "ep_ppo_s2.json"]
    lain = {x["keluar"].name: x["model"] for x in r if x["ppo_dir"] is None}
    assert lain == {"ep_aturan.json": "b3.pkl", "ep_hgb.json": "h.pkl"}
    assert {j[4] for j in ppo[0]["jobs"]} == {"ppo_bc_ppo_g", "ppo_bc_ppo", "ppo_bc_g"}


def test_lanjut_hanya_untuk_pembukaan_yang_sama(tmp_path):
    """Proses yang mati di tengah boleh dilanjutkan (bukan uji ulang) bila model dan data sama
    persis; model atau data lain tetap ditolak."""
    import json

    import pytest

    from buka_uji import catat_pembukaan

    log = tmp_path / "buka_uji.log"
    with pytest.raises(RuntimeError, match="belum pernah dibuka"):
        catat_pembukaan(log, {"hash_model": {"a": "1"}}, lanjut=True)
    info = {"hash_model": {"a": "1"}, "hash_checksum_data": {"c": "9"}}
    catat_pembukaan(log, info)
    with pytest.raises(RuntimeError, match="berbeda"):
        catat_pembukaan(log, {**info, "hash_model": {"a": "2"}}, lanjut=True)
    with pytest.raises(RuntimeError, match="berbeda"):
        catat_pembukaan(log, {**info, "hash_checksum_data": {"c": "8"}}, lanjut=True)
    catat_pembukaan(log, info, lanjut=True)
    isi = [json.loads(x) for x in log.read_text().splitlines()]
    assert [x.get("lanjut", False) for x in isi] == [False, True]


def test_src_dan_berkas_checksum_ada():
    """Hash checksums.yaml di log pembukaan hanya bermakna bila jalurnya benar."""
    from pathlib import Path

    import buka_uji

    assert (Path(buka_uji.SRC) / "dfjsp" / "datasets").is_dir()
    assert (Path(buka_uji.SRC).parent / "configs" / "checksums.yaml").is_file()


def test_pekerja_mewarisi_ppo_dir_yang_diisi_saat_jalan(tmp_path, monkeypatch):
    """Python 3.14 memakai forkserver sebagai bawaan; pekerja harus tetap melihat PPO_DIR (dan
    izin penjaga) yang diisi main() sesudah impor, jadi pool wajib memakai fork."""
    import importlib
    import json

    from buka_uji import _jalankan

    (tmp_path / "palsu_cek.py").write_text(
        "PPO_DIR = None\n"
        "def _init(model):\n    pass\n"
        "def run(job):\n"
        "    n, c, i, s, m, lam = job\n"
        "    return {'instans': n, 'kapasitas': c, 'intensitas': i, 'seed': s, 'metode': m,\n"
        "            'lam': lam, 'ppo_dir': PPO_DIR}\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    palsu = importlib.import_module("palsu_cek")
    palsu.PPO_DIR = "folder/akhir_s1"
    keluar = tmp_path / "ep.json"
    _jalankan(palsu, [("A", 1.0, 0.05, 1, "m", 3.0), ("B", 1.0, 0.05, 1, "m", 3.0)], "x.pkl",
              keluar)
    assert {r["ppo_dir"] for r in json.loads(keluar.read_text())} == {"folder/akhir_s1"}


def test_satu_thread_openmp_per_pekerja():
    """Delapan pekerja boosting masing-masing dengan thread OpenMP bawaan berebut core dan
    sangat lambat; skrip shell gerbang memakai OMP_NUM_THREADS=1, buka_uji juga."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    di_sini = Path(__file__).resolve().parent
    env = {k: v for k, v in os.environ.items() if k != "OMP_NUM_THREADS"}
    env["PYTHONPATH"] = os.pathsep.join([str(di_sini), *sys.path])
    keluar = subprocess.run([sys.executable, "-c", "import os, buka_uji; "
                             "print(os.environ.get('OMP_NUM_THREADS'))"],
                            cwd=di_sini, capture_output=True, text=True, env=env, timeout=120)
    assert keluar.returncode == 0, keluar.stderr
    assert keluar.stdout.strip() == "1"


def _cegah_pembukaan(monkeypatch):
    """Tes main() dengan --buka-uji tidak boleh pernah sampai ke pembukaan: bila kode lolos
    pemeriksaan, catat_pembukaan dan izinkan langsung gagal sebelum berkas uji disentuh."""
    import penjaga_uji

    import buka_uji

    def stop(*a, **k):
        raise AssertionError("main lolos ke pembukaan data uji")

    monkeypatch.setattr(buka_uji, "catat_pembukaan", stop)
    monkeypatch.setattr(penjaga_uji, "izinkan", stop)


def test_model_yang_hilang_ditolak_sebelum_pembukaan_dicatat(tmp_path, monkeypatch):
    """Path model yang salah tidak boleh menghabiskan satu-satunya pembukaan: main berhenti
    sebelum buka_uji.log ditulis."""
    from buka_uji import main

    _cegah_pembukaan(monkeypatch)
    log = tmp_path / "buka_uji.log"
    kode = main(["--buka-uji", "--resep", str(tmp_path / "tidak_ada" / "e9c_s"),
                 "--b3", str(tmp_path / "tidak_ada" / "b3.pkl"),
                 "--hgb", str(tmp_path / "tidak_ada" / "h.pkl"),
                 "--keluar", str(tmp_path / "uji"), "--log", str(log)])
    assert kode == 3
    assert not log.exists() and not (tmp_path / "uji").exists()


def _folder_model_palsu(tmp_path):
    """Folder model buatan dengan SHA256SUMS yang benar, seperti runs/e15_validasi."""
    import hashlib

    m = tmp_path / "model"
    berkas = [f"e9c_s{k}/{f}" for k in range(3) for f in ("bc_ppo.zip", "bc.zip", "norm.pkl")]
    berkas += ["b3.pkl", "h.pkl"]
    for i, b in enumerate(berkas):
        (m / b).parent.mkdir(parents=True, exist_ok=True)
        (m / b).write_bytes(f"isi {i}".encode())
    (m / "SHA256SUMS").write_text("".join(
        f"{hashlib.sha256((m / b).read_bytes()).hexdigest()}  ./{b}\n" for b in berkas))
    return m


def test_model_yang_berubah_dari_sha256sums_ditolak(tmp_path, monkeypatch):
    """Model yang isinya tidak sama dengan SHA256SUMS di folder model bukan model yang dikunci
    di dokumen kunci uji; ditolak sebelum dicatat. Model yang cocok lolos pemeriksaan."""
    import pytest

    from buka_uji import main

    _cegah_pembukaan(monkeypatch)
    m = _folder_model_palsu(tmp_path)
    argv = ["--buka-uji", "--resep", str(m / "e9c_s"), "--b3", str(m / "b3.pkl"),
            "--hgb", str(m / "h.pkl"), "--keluar", str(tmp_path / "uji"),
            "--log", str(tmp_path / "buka_uji.log")]
    with pytest.raises(AssertionError, match="lolos ke pembukaan"):
        main(argv)
    (m / "e9c_s1" / "bc_ppo.zip").write_bytes(b"model lain")
    assert main(argv) == 3
    assert not (tmp_path / "buka_uji.log").exists()
