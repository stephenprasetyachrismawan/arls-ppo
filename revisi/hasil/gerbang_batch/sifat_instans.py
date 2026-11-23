"""Sifat instans untuk membaca perilaku metode menurut sifat (rencana benchmark bagian 7).

Semua sifat dihitung dari instans sesudah transformasi domain, sebelum episode dijalankan:

- ukuran: banyak job, tugas, pengembang, dan tugas per pengembang;
- fleksibilitas: rata-rata dan simpangan baku banyak pengembang sah per tugas;
- upaya: rata-rata upaya per pasangan tugas-pengembang, dan faktor skala benchmark;
- beban: total upaya nominal dibagi banyak pengembang (jam per pengembang);
- ketimpangan keahlian: koefisien variasi banyak tugas yang sah untuk tiap pengembang;
- keketatan: beban per pengembang dibagi rata-rata tenggat akhir job. Tenggat job = 1,5 x
  total upaya job, jadi nilai di atas 1 berarti pengembang butuh waktu lebih lama dari
  tenggat rata-rata walaupun beban terbagi rata;
- kepadatan dependensi: rata-rata banyak prasyarat per tugas.
"""
from __future__ import annotations

import penjaga_uji  # noqa: F401  (memblokir pembukaan data uji)

import numpy as np

from dfjsp.core.due_date import e_nominal


def sifat_instans(inst) -> dict:
    """Sifat satu ``ProjectInstance`` sebagai kamus angka."""
    tasks = inst.tasks
    n_dev = inst.n_devs
    fleks = np.array([len(t.eligible_devs) for t in tasks], dtype=float)
    beban = sum(e_nominal(t) for t in tasks) / n_dev
    per_dev = np.zeros(n_dev)
    for t in tasks:
        for d in t.eligible_devs:
            per_dev[d] += 1
    tenggat_job = {}
    for t in tasks:
        tenggat_job[t.job_id] = max(tenggat_job.get(t.job_id, 0.0), t.due_date)
    return {
        "n_job": len(tenggat_job),
        "n_tugas": len(tasks),
        "n_dev": n_dev,
        "tugas_per_dev": len(tasks) / n_dev,
        "fleks_rata": float(fleks.mean()),
        "fleks_sd": float(fleks.std()),
        "upaya_rata": float(np.mean([u for t in tasks for u in t.eligible_devs.values()])),
        "skala_upaya": float(inst.meta.get("skala_upaya", 1.0)),
        "beban_per_dev": float(beban),
        "ketimpangan_keahlian": float(per_dev.std() / per_dev.mean()),
        "keketatan": float(beban / np.mean(list(tenggat_job.values()))),
        "dep_rata": float(np.mean([len(t.deps) for t in tasks])),
    }


def di_rentang_latih(sifat, sifat_latih) -> dict:
    """Posisi tiap sifat terhadap rentang (terkecil-terbesar) di instans latih.

    Kembalikan {sifat: "di bawah" | "di dalam" | "di atas"}; batas rentang termasuk di dalam,
    dengan toleransi pembulatan bilangan pecahan (relatif 1e-9).
    """
    posisi = {}
    for k, v in sifat.items():
        nilai = [s[k] for s in sifat_latih]
        lo, hi = min(nilai), max(nilai)
        tol = 1e-9 * max(1.0, abs(lo), abs(hi))
        posisi[k] = "di bawah" if v < lo - tol else "di atas" if v > hi + tol else "di dalam"
    return posisi


KOLOM = ("n_job", "n_tugas", "n_dev", "tugas_per_dev", "fleks_rata", "fleks_sd", "upaya_rata",
         "skala_upaya", "beban_per_dev", "ketimpangan_keahlian", "keketatan", "dep_rata")


def main():
    """Tabel sifat 7 MK latih, 60 benchmark latih, dan 23 benchmark validasi (tanpa data uji).

    Pemakaian (dari folder gerbang_batch)::

        PYTHONPATH=../../../exp1/dfjsp-ppo-repair/src $PY sifat_instans.py > sifat/SIFAT_LATIH_VALIDASI.md
    """
    from dfjsp.datasets import (BENCHMARK_LATIH, BENCHMARK_VALIDASI, BRANDIMARTE_TRAIN,
                                load_benchmark, load_brandimarte)

    mk = load_brandimarte(list(BRANDIMARTE_TRAIN))
    bench = load_benchmark(list(BENCHMARK_LATIH) + list(BENCHMARK_VALIDASI))
    kelompok = [("latih MK", n, mk[n]) for n in BRANDIMARTE_TRAIN]
    kelompok += [("latih benchmark", n, bench[n]) for n in BENCHMARK_LATIH]
    kelompok += [("validasi", n, bench[n]) for n in BENCHMARK_VALIDASI]
    baris = [(g, n, sifat_instans(i)) for g, n, i in kelompok]
    latih = [s for g, _, s in baris if g.startswith("latih")]

    def angka(v):
        return (f"{v:d}" if isinstance(v, int) else f"{v:.2f}").replace(".", ",")

    print("# Sifat instans latih dan validasi\n")
    print("Dihitung dengan `sifat_instans.py` sesudah transformasi domain, sebelum episode. "
          "Data uji tidak dimuat. Arti tiap kolom ada di kepala `sifat_instans.py`.\n")
    print("## Rentang per kelompok\n")
    print("| sifat | " + " | ".join(("latih MK", "latih benchmark", "validasi")) + " |")
    print("|---|---|---|---|")
    for k in KOLOM:
        sel = []
        for g in ("latih MK", "latih benchmark", "validasi"):
            v = [s[k] for gg, _, s in baris if gg == g]
            sel.append(f"{angka(min(v))} sampai {angka(max(v))}")
        print(f"| {k} | " + " | ".join(sel) + " |")
    print("\n## Instans validasi di luar rentang latih (7 MK + 60 benchmark)\n")
    ada = False
    for g, n, s in baris:
        if g != "validasi":
            continue
        # Upaya sengaja disamakan lewat skala, jadi tidak dibandingkan.
        banding = {k: v for k, v in s.items() if k not in ("upaya_rata", "skala_upaya")}
        luar = {k: p for k, p in di_rentang_latih(banding, latih).items() if p != "di dalam"}
        if luar:
            ada = True
            print(f"- {n}: " + ", ".join(f"{k} {p} ({angka(s[k])})" for k, p in luar.items()))
    if not ada:
        print("Tidak ada: semua sifat instans validasi berada di dalam rentang latih.")
    print("\n## Per instans\n")
    print("| kelompok | instans | " + " | ".join(KOLOM) + " |")
    print("|---|---|" + "---|" * len(KOLOM))
    for g, n, s in baris:
        print(f"| {g} | {n} | " + " | ".join(angka(s[k]) for k in KOLOM) + " |")


if __name__ == "__main__":
    main()
