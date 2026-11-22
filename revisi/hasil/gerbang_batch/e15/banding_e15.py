"""E15: tabel J semua metode dengan model hasil latih ulang (hitungan perubahan baru, PR #164).

Kriteria ditulis sebelum menjalankan di docs/ARSITEKTUR_PIPELINE.md 5.6 (butir "E15").
Ukuran 1 = rata-rata rasio J per kondisi terhadap pembanding. Ukuran 2 = rasio J total.
Selang 95% dari bootstrap bergugus (instans, intensitas, seed), 10.000 sampel ulang
(``banding_umum.bandingkan``). PPO: J per kondisi dirata-rata atas 3 seed latih dulu.
Seed validasi 5000-5001 saja; data uji tidak dibuka.

Pemakaian (dari folder gerbang_batch)::

    PYTHONPATH=../../../exp1/dfjsp-ppo-repair/src:. $PY e15/banding_e15.py > e15/banding_e15.md
"""
import json

import numpy as np
import pandas as pd

import math

from banding_umum import bandingkan, bandingkan_celah

BERKAS = {"aturan": ["e15/ep_aturan.json"],
          "E1": [f"e15/ep_e1ppo_s{k}.json" for k in range(3)],
          "E9c": [f"e15/ep_e9c_s{k}.json" for k in range(3)],
          **{h: [f"e15/ep_{h}.json"] for h in ("hgb7", "hgb7r50", "hgb7d", "hgb7r50d")}}
NAMA = {"seq": "Sequential", "fix3_g": "tingkat 3 tetap + penjaga", "acak_g": "acak + penjaga",
        "E9c:ppo_bc_ppo_g": "PPO E9c (BC + DAgger + PPO) + penjaga",
        "E9c:ppo_bc_ppo": "PPO E9c tanpa penjaga", "E9c:ppo_bc_g": "BC + DAgger E9c + penjaga",
        "E1:ppo_bc_ppo_g": "PPO E1 (BC + PPO) + penjaga", "E1:ppo_bc_ppo": "PPO E1 tanpa penjaga",
        "E1:ppo_bc_g": "BC E1 + penjaga",
        "hgb7:il_g": "boosting 7 MK + penjaga", "hgb7:il": "boosting 7 MK",
        "hgb7r50:il_g": "boosting 7 MK + 50 acak + penjaga", "hgb7r50:il": "boosting 7 MK + 50 acak",
        "hgb7d:il_g": "boosting 7 MK + DAgger + penjaga", "hgb7d:il": "boosting 7 MK + DAgger",
        "hgb7r50d:il_g": "boosting 7 MK + 50 acak + DAgger + penjaga",
        "hgb7r50d:il": "boosting 7 MK + 50 acak + DAgger"}
PPO = "E9c:ppo_bc_ppo_g"
LAM_LABEL = (0.3, 3.0, 30.0)
LAM_TANPA = (0.1, 1.0, 10.0)


def muat():
    baris = []
    for grup, daftar in BERKAS.items():
        for f in daftar:
            for r in json.load(open(f)):
                assert r["seed"] in (5000, 5001) and r["instans"] not in ("MK03", "MK04", "MK09")
                m = r["metode"] if grup == "aturan" else f"{grup}:{r['metode']}"
                baris.append(dict(metode=m, instans=r["instans"], kapasitas=r["kapasitas"],
                                  intensitas=r["intensitas"], seed=r["seed"], lam=r["lam"],
                                  J=r["J"]))
    return pd.DataFrame(baris)


def koma(x, d=2):
    return f"{x:+.{d}f}".replace(".", ",")


def selang(b):
    tanda = "*" if b.low > 0 or b.high < 0 else ""
    return f"{koma(b.estimate)} [{koma(b.low)}; {koma(b.high)}]{tanda}"


def tabel_rasio(df):
    print("## Rasio J terhadap Sequential\n")
    print("Nilai = J metode / J Sequential. Uk. 1 = rata-rata rasio per kondisi; uk. 2 = rasio "
          "J total. 126 kondisi per λ (7 MK × 3 kapasitas × 3 intensitas × 2 seed).\n")
    for lam, g in df.groupby("lam"):
        t = g.pivot_table(index=["instans", "kapasitas", "intensitas", "seed"],
                          columns="metode", values="J")
        assert not t.isna().any().any() and len(t) == 126, lam
        r1 = (t.div(t["seq"], axis=0)).mean()
        r2 = t.sum() / t["seq"].sum()
        print(f"### λ = {lam:g}\n".replace(".", ","))
        print("| metode | uk. 1 | uk. 2 |")
        print("|---|---|---|")
        for m in r2.sort_values().index:
            print(f"| {NAMA[m]} | {r1[m]:.4f} | {r2[m]:.4f} |".replace(".", ","))
        print()


def tabel_banding(df, judul, pasangan, lams):
    print(f"## {judul}\n")
    print("Selisih J metode A terhadap B dalam persen; negatif = A lebih baik. "
          "Selang 95% bootstrap bergugus; * = selang tidak memuat nol.\n")
    print("| A | B | λ | uk. 1 (rata-rata rasio) | uk. 2 (rasio total) |")
    print("|---|---|---|---|---|")
    for a, b in pasangan:
        hasil = bandingkan(df[df.lam.isin(lams) & df.metode.isin([a, b])], a, b)
        for lam in lams:
            h = hasil[lam]
            print(f"| {NAMA[a]} | {NAMA[b]} | {str(lam).rstrip("0").rstrip(".").replace(".", ",")} | {selang(h.rasio_rata)} | "
                  f"{selang(h.rasio_total)} |")
    print()


def _log_interp(lam, kiri, kanan):
    w = math.log(lam / kiri) / math.log(kanan / kiri)
    return {kiri: 1 - w, kanan: w}


# Pasangan λ butir 1, keputusan pemilik 16 November 2026 (docs/ARSITEKTUR_PIPELINE.md 5.6).
PASANGAN_BUTIR1 = [("0,1 lawan 0,3", {0.1: 1.0}, {0.3: 1.0}),
                   ("1 lawan interpolasi log 0,3 dan 3", {1.0: 1.0}, _log_interp(1.0, 0.3, 3.0)),
                   ("10 lawan interpolasi log 3 dan 30", {10.0: 1.0}, _log_interp(10.0, 3.0, 30.0)),
                   ("pelengkap: rata-rata tanpa label lawan rata-rata berlabel",
                    {lam: 1 / 3 for lam in LAM_TANPA}, {lam: 1 / 3 for lam in LAM_LABEL})]


def tabel_butir1(df):
    print("## Butir 1: celah PPO E9c + penjaga ke Sequential, λ tanpa label dikurangi acuan\n")
    print("Celah = selisih J PPO + penjaga ke Sequential (persen). Nilai = celah sasaran dikurangi "
          "celah acuan, dalam poin persen; positif = celah di λ tanpa label lebih besar. Selang "
          "95% bootstrap bergugus, gugus diambil ulang bersama untuk semua λ. Butir 1 ditulis "
          "untuk λ yang selangnya memuat nol atau negatif dengan kedua ukuran.\n")
    print("| sasaran lawan acuan | bobot acuan | uk. 1 | uk. 2 | butir 1 |")
    print("|---|---|---|---|---|")
    d = df[df.metode.isin([PPO, "seq"])]
    for judul, sasaran, acuan in PASANGAN_BUTIR1:
        h = bandingkan_celah(d, PPO, "seq", sasaran, acuan)
        lolos = h.rasio_rata.low <= 0 and h.rasio_total.low <= 0
        bobot = " + ".join(f"{w:.3f} × λ {lam:g}" for lam, w in acuan.items()).replace(".", ",")
        hasil = "ya" if lolos else "tidak"
        if judul.startswith("pelengkap"):
            hasil += " (tidak menentukan)"
        print(f"| {judul} | {bobot} | {selang(h.rasio_rata)} | {selang(h.rasio_total)} | "
              f"{hasil} |")
    print()


def main():
    df = muat()
    semua = tuple(sorted(LAM_TANPA + LAM_LABEL))
    print("# E15: hasil latih ulang dengan hitungan perubahan baru\n")
    print("Kode: `e15/banding_e15.py`. Episode: `e15/ep_*.json`. Seed validasi 5000-5001.\n")
    tabel_rasio(df)
    tabel_banding(df, "Butir 2: PPO E9c + penjaga lawan pembanding tanpa latih",
                  [(PPO, "fix3_g"), (PPO, "acak_g")], semua)
    tabel_banding(df, "Selisih PPO E9c + penjaga ke Sequential per λ (bahan butir 1)",
                  [(PPO, "seq")], semua)
    tabel_butir1(df)
    tabel_banding(df, "PPO E9c + penjaga lawan pemilih terlatih lain (λ berlabel dan tanpa label)",
                  [(PPO, "hgb7d:il_g"), (PPO, "hgb7r50d:il_g")], semua)
    tabel_banding(df, "Peran penjaga dan tahap PPO (E9c)",
                  [(PPO, "E9c:ppo_bc_ppo"), (PPO, "E9c:ppo_bc_g")], semua)
    tabel_banding(df, "E1 lawan E9c (manfaat DAgger), λ berlabel",
                  [("E1:ppo_bc_ppo_g", PPO)], LAM_LABEL)


if __name__ == "__main__":
    np.seterr(all="raise")
    main()
