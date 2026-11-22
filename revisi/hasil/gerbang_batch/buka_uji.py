"""Pembukaan resmi data uji (versi ringan pilihan B, draf kunci uji bagian 10).

Memanggil ``cek_episode.run`` yang sama persis dengan evaluasi validasi, pada data uji:
MK03, MK04, MK09 dengan seed 10000-10009 dan 35 benchmark uji dengan seed 11000-11004.
Hanya jalan dengan bendera ``--buka-uji``, sesudah pemilik memberi izin. Setiap pembukaan
dicatat di ``buka_uji.log``; pembukaan kedua ditolak.

Pemakaian (dari folder gerbang_batch, sesudah pemilik memberi izin tertulis)::

    R=../../../exp1/dfjsp-ppo-repair/runs/e15_validasi
    PYTHONPATH=.:../../../exp1/dfjsp-ppo-repair/src $PY buka_uji.py --buka-uji \
        --resep $R/e9c_s --b3 $R/b3_model.pkl --hgb $R/hgb7d.pkl --keluar uji

Sebelum pembukaan dicatat, semua berkas model diperiksa: harus ada dan sama dengan
``SHA256SUMS`` di folder induk model. Bila tidak, skrip berhenti dengan kode 3 tanpa mencatat.

Resep yang dikunci E15 (E16 dan E17 tidak lolos gerbang). Bila proses mati di tengah,
jalankan perintah yang sama dengan ``--lanjut``; pekerjaan yang sudah ada di jsonl dilewati.
"""
from __future__ import annotations

import os

# Satu thread OpenMP per pekerja, sama dengan skrip shell gerbang. Diisi sebelum numpy dan
# scikit-learn dimuat; tanpa ini 8 pekerja boosting berebut core dan sangat lambat.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import datetime  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from dfjsp.datasets import BENCHMARK_UJI  # noqa: E402

MK_UJI = ("MK03", "MK04", "MK09")
SEED_MK = tuple(range(10000, 10010))
SEED_BENCH = tuple(range(11000, 11005))
KAPASITAS = (1.0, 1.5, 2.0)
INTENSITAS = (0.05, 0.15, 0.30)


def rencana_uji(metode, lams):
    """Daftar pekerjaan (instans, kapasitas, intensitas, seed, metode, λ) data uji."""
    bagian = [(MK_UJI, SEED_MK), (BENCHMARK_UJI, SEED_BENCH)]
    return [(n, c, i, s, m, lam) for nama, seeds in bagian for n in nama for c in KAPASITAS
            for i in INTENSITAS for s in seeds for m in metode for lam in lams]


#: Kunci ``info`` yang harus sama persis supaya pembukaan boleh dilanjutkan.
_KUNCI_SAMA = ("hash_model", "hash_checksum_data")


def catat_pembukaan(log, info, *, lanjut=False):
    """Tambah satu baris JSON ke ``log``: waktu, commit kode, dan ``info``.

    Data uji dibuka sekali saja (draf kunci uji bagian 11), jadi bila ``log`` sudah berisi
    catatan pembukaan, pemanggilan ditolak dan tidak ada yang ditulis. Pengecualiannya
    ``lanjut=True``: proses pembukaan yang mati di tengah boleh diteruskan, asal hash model
    dan hash data sama persis dengan catatan pertama. Itu bukan uji ulang, karena pekerjaan
    yang sudah selesai di berkas jsonl tidak dijalankan lagi.
    """
    log = Path(log)
    ada = log.exists() and log.read_text().strip()
    if lanjut:
        if not ada:
            raise RuntimeError(f"data uji belum pernah dibuka (lihat {log}); tidak ada yang "
                               "bisa dilanjutkan")
        pertama = json.loads(log.read_text().splitlines()[0])
        beda = [k for k in _KUNCI_SAMA if pertama.get(k) != info.get(k)]
        if beda:
            raise RuntimeError(f"{', '.join(beda)} berbeda dengan pembukaan pertama; "
                               "melanjutkan dengan model atau data lain sama dengan uji ulang")
        info = {**info, "lanjut": True}
    elif ada:
        raise RuntimeError(f"data uji sudah pernah dibuka (lihat {log}); tidak ada uji ulang")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parent, check=True).stdout.strip()
    baris = {"waktu": datetime.datetime.now().isoformat(timespec="seconds"),
             "commit": commit, **info}
    with log.open("a") as f:
        f.write(json.dumps(baris, ensure_ascii=False) + "\n")


SRC = str(Path(__file__).resolve().parents[3] / "exp1" / "dfjsp-ppo-repair" / "src")
LAMS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
METODE_PPO = ("ppo_bc_ppo_g", "ppo_bc_ppo", "ppo_bc_g")
METODE_ATURAN = ("aor", "seq", "fix3_g", "acak_g")
METODE_BOOSTING = ("il_g",)


def _hash_berkas(*folder_atau_berkas):
    hasil = {}
    for x in folder_atau_berkas:
        x = Path(x)
        for f in sorted(x.rglob("*") if x.is_dir() else [x]):
            if f.is_file():
                hasil[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
    return hasil


#: Berkas yang harus ada di tiap folder seed model (``ppo_bc_ppo`` dan ``ppo_bc`` + normalisasi).
_BERKAS_SEED = ("bc_ppo.zip", "bc.zip", "norm.pkl")


def _masalah_model(folder_model, b3, hgb):
    """Daftar masalah berkas model; kosong bila semuanya ada.

    Dipanggil sebelum pembukaan dicatat, supaya path yang salah tidak menghabiskan
    satu-satunya pembukaan data uji.
    """
    wajib = [Path(d) / f for d in folder_model for f in _BERKAS_SEED] + [Path(b3), Path(hgb)]
    hilang = [f"tidak ada: {f}" for f in wajib if not f.is_file()]
    if hilang:
        return hilang
    # Semua berkas harus sama dengan SHA256SUMS di folder induk model (runs/e15_validasi).
    induk = Path(folder_model[0]).resolve().parent
    sums = induk / "SHA256SUMS"
    if not sums.is_file():
        return [f"tidak ada: {sums}"]
    tercatat = {}
    for baris in sums.read_text().splitlines():
        if baris.strip():
            h, nama = baris.split(maxsplit=1)
            tercatat[(induk / nama.strip()).resolve()] = h
    masalah = []
    for f in wajib:
        h = tercatat.get(f.resolve())
        if h is None:
            masalah.append(f"tidak tercatat di {sums}: {f}")
        elif hashlib.sha256(f.read_bytes()).hexdigest() != h:
            masalah.append(f"hash tidak cocok dengan {sums}: {f}")
    return masalah


def _jalankan(cek_episode, jobs, model, keluar):
    """Jalankan ``cek_episode.run`` untuk ``jobs`` dengan 8 proses; tulis jsonl lalu json.

    Pool memakai ``fork``, bukan bawaan Python 3.14 (``forkserver``): pekerja harus mewarisi
    ``cek_episode.PPO_DIR`` dan izin ``penjaga_uji`` yang diisi ``main()`` sesudah impor.
    """
    import multiprocessing

    keluar = Path(keluar)
    selesai = set()
    if keluar.with_suffix(".jsonl").exists():
        for x in keluar.with_suffix(".jsonl").read_text().splitlines():
            r = json.loads(x)
            selesai.add((r["instans"], r["kapasitas"], r["intensitas"], r["seed"],
                         r["metode"], r["lam"]))
    sisa = [j for j in jobs if j not in selesai]
    pool_fork = multiprocessing.get_context("fork").Pool
    with pool_fork(8, initializer=cek_episode._init, initargs=(model,)) as pool, \
            keluar.with_suffix(".jsonl").open("a") as f:
        for n, r in enumerate(pool.imap_unordered(cek_episode.run, sisa, chunksize=1), 1):
            f.write(json.dumps(r) + "\n")
            f.flush()
            if n % 200 == 0:
                print(f"{keluar.name}: {n}/{len(sisa)}", flush=True)
    baris = [json.loads(x) for x in keluar.with_suffix(".jsonl").read_text().splitlines()]
    keluar.write_text(json.dumps(baris))


def rencana_lengkap(resep, b3, hgb, keluar):
    """Urutan pekerjaan pembukaan: aturan, boosting, lalu PPO tiga seed model.

    Tiap butir: ``jobs``, ``model`` (untuk ``cek_episode._init``), ``ppo_dir`` (folder model
    PPO, atau None), dan ``keluar`` (berkas JSON).
    """
    keluar = Path(keluar)
    r = [{"jobs": rencana_uji(METODE_ATURAN, LAMS), "model": b3, "ppo_dir": None,
          "keluar": keluar / "ep_aturan.json"},
         {"jobs": rencana_uji(METODE_BOOSTING, LAMS), "model": hgb, "ppo_dir": None,
          "keluar": keluar / "ep_hgb.json"}]
    r += [{"jobs": rencana_uji(METODE_PPO, LAMS), "model": b3, "ppo_dir": f"{resep}{k}",
           "keluar": keluar / f"ep_ppo_s{k}.json"} for k in range(3)]
    return r


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--buka-uji", action="store_true",
                   help="izin membuka data uji; hanya sesudah pemilik setuju")
    p.add_argument("--resep", required=True, help="awalan folder model, ditambah 0/1/2")
    p.add_argument("--b3", required=True)
    p.add_argument("--hgb", required=True, help="model boosting pembanding")
    p.add_argument("--keluar", required=True)
    p.add_argument("--log", default="buka_uji.log")
    p.add_argument("--lanjut", action="store_true",
                   help="teruskan pembukaan yang mati di tengah, dengan model dan data sama")
    a = p.parse_args(argv)
    if not a.buka_uji:
        print("buka_uji: bendera --buka-uji tidak diberikan; data uji tidak dibuka.",
              file=sys.stderr)
        return 2

    folder_model = [f"{a.resep}{k}" for k in range(3)]
    masalah = _masalah_model(folder_model, a.b3, a.hgb)
    if masalah:
        print("buka_uji: berkas model bermasalah; data uji tidak dibuka dan tidak dicatat.\n  "
              + "\n  ".join(masalah), file=sys.stderr)
        return 3
    catat_pembukaan(a.log, {"argv": sys.argv, "hash_model": _hash_berkas(*folder_model,
                                                                         a.b3, a.hgb),
                            "hash_checksum_data": _hash_berkas(
                                Path(SRC).parent / "configs" / "checksums.yaml")},
                    lanjut=a.lanjut)
    import penjaga_uji
    penjaga_uji.izinkan("buka_uji.py --buka-uji")
    os.environ["PERIKSA"] = "1"
    sys.argv = ["cek_episode.py", SRC, a.b3, os.devnull]
    import cek_episode

    Path(a.keluar).mkdir(parents=True, exist_ok=True)
    # Sifat instans uji disimpan sekarang, karena sesudah proses ini penjaga memblokir berkas
    # uji lagi; analisis_uji.py membacanya lewat SIFAT=<keluar>/sifat_uji.json.
    import instans_acak
    from sifat_instans import sifat_instans
    sifat = {n: sifat_instans(instans_acak.load_any(n)) for n in (*MK_UJI, *BENCHMARK_UJI)}
    (Path(a.keluar) / "sifat_uji.json").write_text(json.dumps(sifat, default=float, indent=1))
    for butir in rencana_lengkap(a.resep, a.b3, a.hgb, a.keluar):
        # cek_episode membaca PPO_DIR sekali saat diimpor, jadi diisi langsung di modulnya;
        # pekerja dibuat dengan fork sesudah ini (_jalankan) dan ikut memakai nilai baru. Cache model
        # dikosongkan supaya pekerja tidak mewarisi model seed sebelumnya.
        cek_episode.PPO_DIR = butir["ppo_dir"]
        cek_episode._PPO.clear()
        _jalankan(cek_episode, butir["jobs"], butir["model"], butir["keluar"])
    print("UJI-SELESAI")
    return 0


if __name__ == "__main__":
    sys.exit(main())
