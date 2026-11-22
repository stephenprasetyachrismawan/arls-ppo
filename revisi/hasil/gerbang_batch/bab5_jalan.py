"""Membangkitkan tabel dan data gambar Bab V dari berkas yang sudah ada (tanpa latih ulang).

Fungsi pengolah ada di ``naskah_bab5`` (seam S1-S6, dites di ``test_naskah_bab5.py``). Berkas
ini hanya memuat berkas, memanggil fungsi itu, lalu menulis hasilnya ke
``latex/TemplateLatexLaporanTesis/chapters/chapter5/generated/``. Setiap ringkasan juga ditulis
sebagai JSON di folder yang sama, supaya angka di teks dapat ditelusuri.

Dijalankan dari folder ``revisi/hasil/gerbang_batch``::

    # A7: tahap pelatihan pada data validasi (berkas e15/ di repositori)
    $PY bab5_jalan.py validasi

    # B: data uji (berkas episode di mesin penelitian, sesudah pembukaan 19 November 2026)
    SIFAT=uji/sifat_uji.json PYTHONPATH=.:$SRC $PY bab5_jalan.py uji uji

    # A1-A6: log, data guru, dan model E15 (RUN = exp1/dfjsp-ppo-repair/runs/e15_validasi)
    PYTHONPATH=.:$SRC $PY bab5_jalan.py latih $RUN $SRC

    # C: satu episode validasi di tiga λ, dicocokkan dengan cek_episode.run
    PYTHONPATH=.:$SRC $PY bab5_jalan.py episode $RUN $SRC

Tidak ada langkah yang melatih model. Langkah ``uji`` hanya membaca berkas episode uji yang
sudah ada; instans uji tidak dimuat.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import naskah_bab5 as nb

KELUAR = Path(__file__).resolve().parents[3] / "latex/TemplateLatexLaporanTesis/chapters" \
    / "chapter5/generated"
LAM_LABEL = (0.3, 3.0, 30.0)


def _tulis(nama, teks):
    KELUAR.mkdir(parents=True, exist_ok=True)
    (KELUAR / nama).write_text(teks, encoding="utf-8")
    print("ditulis", KELUAR / nama)


def _json(nama, isi):
    def bersih(v):
        if hasattr(v, "tolist"):
            return v.tolist()
        if isinstance(v, dict):
            return {str(k): bersih(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [bersih(x) for x in v]
        return v
    _tulis(nama, json.dumps(bersih(isi), indent=1, ensure_ascii=False) + "\n")


def _baca(*berkas):
    return [r for f in berkas for r in json.load(open(f))]


# ------------------------------------------------------------------ A7 validasi

def validasi():
    """Tahap pelatihan pada data validasi: BC, BC + DAgger, BC + DAgger + PPO, Sequential."""
    baris = []
    for k in (0, 1, 2):
        for r in _baca(f"e15/ep_e1ppo_s{k}.json"):
            if r["metode"] == "ppo_bc_g":
                baris.append({**r, "metode": "tahap_bc"})
        for r in _baca(f"e15/ep_e9c_s{k}.json"):
            if r["lam"] in LAM_LABEL and r["metode"] in ("ppo_bc_g", "ppo_bc_ppo_g"):
                baris.append({**r, "metode": {"ppo_bc_g": "tahap_dagger",
                                              "ppo_bc_ppo_g": "tahap_ppo"}[r["metode"]]})
    baris += [r for r in _baca("e15/ep_aturan.json") if r["lam"] in LAM_LABEL]
    nb.NAMA.update({"tahap_bc": "BC", "tahap_dagger": "BC + DAgger",
                    "tahap_ppo": "BC + DAgger + PPO"})
    df = nb.uraian_j(baris)
    urutan = ["acak_g", "fix3_g", "tahap_bc", "tahap_dagger", "tahap_ppo", "seq"]
    _tulis("bab5_validasi_tahap.tex", nb.tabel_uraian(
        df, urutan, "Suku $J$ setiap tahap pelatihan pada data validasi",
        "tab:validasi-tahap",
        "Tujuh instans latih, \\emph{seed} validasi 5000 dan 5001, tiga kapasitas, tiga "
        "intensitas. Semua tahap memakai penjaga. Nilai jaringan dirata-rata atas tiga "
        "\\emph{seed} model per kondisi, lalu per instans. Keterlambatan dalam satuan $E$. "
        "AOR tidak dijalankan pada validasi E15."))
    _json("bab5_validasi_tahap.json", df.to_dict(orient="records"))


# ------------------------------------------------------------------ B data uji

def uji(folder):
    """Semua tabel data uji: S1 dari analisis, S2 uraian J, dan pemeriksaan kuartil."""
    import analisis_uji
    berkas = [f"{folder}/{n}.json" for n in ("ep_aturan", "ep_hgb", "ep_ppo_s0", "ep_ppo_s1",
                                             "ep_ppo_s2")]
    baris = _baca(*berkas)
    usulan = "ppo_bc_ppo_g"
    pembanding = sorted({r["metode"] for r in baris} - {usulan})
    sifat = json.load(open(os.environ["SIFAT"])) if os.environ.get("SIFAT") else {}
    h = analisis_uji.analisis(baris, usulan, pembanding, sifat)
    for nama, teks in nb.tabel_analisis(h).items():
        _tulis(f"bab5_{nama}.tex", teks)
    # Pelengkap RM1: ARLS-PPO dan Sequential Lookahead terhadap AOR (Tabel 3.x, selang 95%).
    import pandas as pd
    from banding_umum import bandingkan_gabungan, bandingkan_selisih_gabungan
    dfb = pd.DataFrame(baris)
    aor = {m: {"J": bandingkan_gabungan(dfb, m, "aor"),
               "telat": bandingkan_selisih_gabungan(dfb, m, "aor", "telat"),
               "ubah": bandingkan_selisih_gabungan(dfb, m, "aor", "ubah")}
           for m in (usulan, "seq")}
    _tulis("bab5_uji_aor.tex", nb.tabel_terhadap_aor(aor))
    _json("bab5_uji_aor.json", {m: {k: {lam: (list(x) if k != "J" else
                                             {"uk1": list(x.rasio_rata), "uk2": list(x.rasio_total)})
                                         for lam, x in v.items()} for k, v in d.items()}
                                 for m, d in aor.items()})
    sifat_tabel(h["per_sifat"])
    # S2: suku J semua metode di semua λ.
    df = nb.uraian_j(baris)
    urutan = ["ppo_bc_ppo_g", "aor", "acak_g", "fix3_g", "il_g", "ppo_bc_g", "ppo_bc_ppo",
              "seq"]
    _tulis("bab5_uji_uraian_label.tex", nb.tabel_uraian(
        df[df.lam.isin(LAM_LABEL)], urutan,
        "Suku $J$ setiap metode pada data uji, $\\lambda$ berlabel", "tab:uji-uraian",
        "Rata-rata per instans lalu atas 38 instans. Keterlambatan dalam satuan $E$. Nilai "
        "ARLS-PPO, ARLS-BC, dan ARLS-PPO tanpa penjaga dirata-rata atas tiga \\emph{seed} "
        "model per kondisi. $\\lambda$ tanpa label ada di Lampiran~\\ref{lamp:hasil-rinci}."))
    _tulis("bab5_uji_uraian_tanpa_label.tex", nb.tabel_uraian(
        df[~df.lam.isin(LAM_LABEL)], urutan,
        "Suku $J$ setiap metode pada data uji, $\\lambda$ tanpa label",
        "tab:uji-uraian-tanpa-label", "Sama dengan Tabel~\\ref{tab:uji-uraian}."))
    _json("bab5_uji_uraian.json", df.to_dict(orient="records"))
    # Data kurva keterlambatan lawan perubahan (pgfplots): satu berkas per metode.
    for m in urutan:
        d = df[df.metode == m].sort_values("lam")
        _tulis(f"bab5_kurva_{m}.csv", "lam,telat,ubah,J\n" + "".join(
            f"{r.lam:g},{r.telat:.4f},{r.ubah:.4f},{r.J:.4f}\n" for r in d.itertuples()))
    # Pemeriksaan sesudah hasil dibaca (HASIL_UJI.md bagian 7), dihitung ulang dari baris.
    kuartil = {}
    for lam in (0.3, 3.0):
        q = nb.kuartil_rasio(baris, usulan, "fix3_g", lam)
        kuartil[lam] = q.to_dict(orient="records")
    _json("bab5_uji_kuartil.json", kuartil)
    rows = []
    for lam, qs in kuartil.items():
        for q in qs:
            rows.append(f"{nb._lam(lam)} & Q{q['kelompok']} & {nb.angka(q['rata_rasio'], 3)} & "
                        f"{nb.angka(q['median_rasio'], 3)} & "
                        f"{nb.angka(q['porsi_a_lebih_kecil'])} & {nb.angka(q['porsi_J_b'])} & "
                        f"{nb.angka(q['rasio_total'], 3)} & {nb.angka(q['ubah_a'], 1)} & "
                        f"{nb.angka(q['ubah_b'], 1)}")
    _tulis("bab5_uji_kuartil.tex", nb._tabel(
        "Rasio $J$ ARLS-PPO terhadap Fixed-L3 + penjaga menurut besar $J$ kondisi",
        "tab:uji-kuartil", "llrrrrrrr",
        "$\\lambda$ & Kelompok & \\shortstack[r]{Rata-rata\\\\rasio} & "
        "\\shortstack[r]{Median\\\\rasio} & \\shortstack[r]{Porsi\\\\ARLS lebih kecil} & "
        "\\shortstack[r]{Porsi $J$\\\\total} & \\shortstack[r]{Rasio\\\\$J$ total} & "
        "\\shortstack[r]{Perubahan\\\\ARLS} & \\shortstack[r]{Perubahan\\\\Fixed-L3}", rows,
        "Pemeriksaan deskriptif sesudah hasil dibaca, tanpa selang. Di dalam setiap instans, "
        "kondisi diurutkan menurut $J$ Fixed-L3 + penjaga lalu dibagi empat kelompok yang "
        "hampir sama banyak (Q1 terkecil); sisa pembagian masuk Q4. Porsi $J$ total: bagian $J$ Fixed-L3 + penjaga seluruh data uji yang "
        "berada di kelompok itu."))
    # Tingkat diminta dan dipakai per λ (semua keluarga digabung), untuk teks.
    tingkat_lam = {}
    for kel, per in h["tingkat"].items():
        for lam, x in per.items():
            tingkat_lam.setdefault(lam, []).append(x)
    _json("bab5_uji_tingkat.json", h["tingkat"])
    def br(x):
        return [x.estimate, x.low, x.high]

    def dua(d):
        return {b: {lam: {"uk1": br(x.rasio_rata), "uk2": br(x.rasio_total)}
                    for lam, x in v.items()} for b, v in d.items()}

    # Satu JSON untuk artikel dan dek (angka_arls.py): selang 98,75% (utama), 95% (gabungan,
    # benchmark_saja, butir1, stabil), waktu, dan pelanggaran.
    _json("bab5_uji_ringkas.json", {
        "utama": dua(h["utama"]),
        "gabungan": dua(h["gabungan"]),
        "benchmark_saja": dua(h["benchmark_saja"]),
        "butir1": {nama: {"uk1": br(x.rasio_rata), "uk2": br(x.rasio_total)}
                   for nama, x in h["butir1"].items()},
        "stabil": {b: {lam: br(x) for lam, x in v.items()} for b, v in h["stabil"].items()},
        "waktu": h["waktu"], "pelanggaran": h["pelanggaran"]})


def sifat_tabel(per_sifat=None):
    """Korelasi Spearman selisih J dengan sifat instans (deskriptif). Tanpa argumen, dibaca
    dari ``uji/analisis_uji_sifat.csv`` keluaran analisis uji yang sudah ada di repositori."""
    import pandas as pd
    if per_sifat is None:
        per_sifat = pd.read_csv("uji/analisis_uji_sifat.csv")
    kolom = list(nb.NAMA_SIFAT)
    for b in ("fix3_g", "acak_g"):
        rho = nb.ringkas_sifat(per_sifat, b, (0.3, 3.0), kolom)
        _tulis(f"bab5_uji_sifat_{b}.tex", nb.tabel_sifat(rho, b, (0.3, 3.0)))
        _json(f"bab5_uji_sifat_{b}.json", {s: {f"{lam:g}|{u}": v for (lam, u), v in r.items()}
                                           for s, r in rho.items()})


# ------------------------------------------------------------------ A1-A6 pelatihan

def _muat_latih_ppo_seq(src):
    """Impor latih_ppo_seq tanpa menjalankan pelatihan (modul membaca sys.argv saat diimpor)."""
    simpan = sys.argv
    sys.argv = [simpan[0], src, "/dev/null", "/dev/null"]
    try:
        import latih_ppo_seq
    finally:
        sys.argv = simpan
    return latih_ppo_seq


def latih(run, src):
    import pickle

    import numpy as np
    lps = _muat_latih_ppo_seq(src)
    data = f"{run}/bc_data_7000.pkl"
    dagger = f"{run}/dagger_e1s0.pkl"
    guru = pickle.load(open(data, "rb"))["rows"]
    tambah = pickle.load(open(dagger, "rb"))["rows"]
    # S5: data guru dengan bobot BC dari fungsi asli; bobot dihitung atas data BC E9c (guru +
    # DAgger), sama dengan pelatihan.
    w = lps.regret_weights(data + "," + dagger)
    ring_guru = nb.ringkas_guru(guru, w[:len(guru)])
    ring_dagger = nb.ringkas_guru(tambah, w[len(guru):])
    _json("bab5_latih_guru.json", {"guru": ring_guru, "dagger": ring_dagger,
                                   "n_guru": len(guru), "n_dagger": len(tambah)})
    baris = []
    for nama, ring in (("Guru", ring_guru), ("DAgger", ring_dagger)):
        for lam, r in sorted(ring.items()):
            baris.append(f"{nama} & {nb._lam(lam)} & " + f"{r['n']:,}".replace(",", ".") + " & "
                         + " & ".join(nb.angka(v) for v in r["porsi_label"]) + " & "
                         + " & ".join(nb.angka(v) for v in r["rata_selisih"][1:]) + " & "
                         + nb.angka(r["porsi_seri"]) + " & " + nb.angka(r["bobot"]["median"])
                         + " & " + nb.angka(r["bobot"]["porsi_kecil"]))
        baris.append("\\addlinespace")
    _tulis("bab5_latih_guru.tex", nb._tabel(
        "Label guru, selisih $J$ setiap tingkat, dan bobot contoh BC menurut $\\lambda$",
        "tab:latih-guru", "ll" + "r" * 11,
        "Data & $\\lambda$ & $n$ & \\multicolumn{4}{c}{Porsi label 0/1/2/3} & "
        "\\multicolumn{3}{c}{Rata-rata $\\Delta J$ tingkat 1/2/3} & Seri & "
        "\\shortstack[r]{Median\\\\bobot} & \\shortstack[r]{Bobot\\\\$<0{,}1$}",
        baris[:-1],
        "$\\Delta J$: $J$ sesudah tingkat itu dikurangi $J$ sekarang, rata-rata atas "
        "keputusan yang tingkatnya sah; negatif berarti tingkat itu menurunkan $J$. "
        "\\emph{Seri}: porsi keputusan yang semua tingkat sahnya memberi $J$ sama. Bobot contoh "
        "dihitung dengan fungsi \\texttt{regret\\_weights} yang dipakai pelatihan."))
    # S3: log pelatihan.
    logs = {}
    for k in (0, 1, 2):
        for nama in (f"e1ppo_s{k}", f"e9c_s{k}"):
            p = f"{run}/{nama}.log"
            if os.path.exists(p):
                logs[nama] = nb.baca_log(open(p).read())
    _json("bab5_latih_log.json", logs)
    baris = []
    for nama, r in logs.items():
        bc = {e["epoch"]: e for e in r["bc"]}
        tb = bc.get(r["terbaik"], {})
        ppo = r["ppo"]
        awal = sum(v for _, v in ppo[:3]) / max(1, len(ppo[:3]))
        akhir = sum(v for _, v in ppo[-3:]) / max(1, len(ppo[-3:]))
        tahap = "BC" if nama.startswith("e1") else "BC + DAgger"
        baris.append(f"{tahap} & {nama[-1]} & {len(r['bc'])} & {r['terbaik']} & "
                     f"{nb.angka(tb.get('loss'), 3)} & {nb.angka(tb.get('akurasi'), 3)} & "
                     f"{nb.angka(tb.get('tahan'), 3)} & {nb.angka(tb.get('rugi'), 3)} & "
                     f"{nb.angka(awal, 4)} & {nb.angka(akhir, 4)}")
    _tulis("bab5_latih_log.tex", nb._tabel(
        "Ringkasan pelatihan BC dan PPO per \\emph{seed} dari log", "tab:latih-log",
        "llrrrrrrrr",
        "Data BC & \\emph{Seed} & Epoch & Terbaik & Loss & \\shortstack[r]{Ketepatan\\\\latih} & "
        "\\shortstack[r]{Ketepatan\\\\sisihan} & \\shortstack[r]{Rugi $J$\\\\sisihan} & "
        "\\shortstack[r]{Reward PPO\\\\awal} & \\shortstack[r]{Reward PPO\\\\akhir}", baris,
        "Epoch: banyak epoch BC yang dijalankan. Terbaik: epoch yang bobotnya dipakai (rugi "
        "$J$ contoh sisihan terkecil). Loss, ketepatan, dan rugi diambil pada epoch terbaik. "
        "Reward PPO awal dan akhir: rata-rata tiga titik pertama dan terakhir (setiap 4.096 "
        "langkah). Log hanya mencatat reward total, tidak per suku."))
    for nama, r in logs.items():
        _tulis(f"bab5_kurva_bc_{nama}.csv", "epoch,loss,akurasi,tahan,rugi\n" + "".join(
            f"{e['epoch']},{e['loss']},{e['akurasi']},{e['tahan']},{e['rugi']}\n"
            for e in r["bc"]))
        _tulis(f"bab5_kurva_ppo_{nama}.csv", "langkah,reward\n" + "".join(
            f"{a},{b}\n" for a, b in r["ppo"]))
    # S4: bobot jaringan sebelum dan sesudah PPO, dan kesamaan pilihan pada keadaan guru.
    import torch
    from sb3_contrib import MaskablePPO
    torch.set_num_threads(1)
    X = np.stack([r[0] for r in guru])
    D = np.array([[np.nan if v is None else v for v in r[2]] for r in guru], float)
    y = np.array([r[1] for r in guru])
    lams = np.array([r[3] for r in guru])
    mask = ~np.isnan(D)
    mask[:, 0] = True

    def pilih(model, nrm):
        obs = np.clip((X - nrm["mu"]) / nrm["sd"], -5, 5).astype("float32")
        with torch.no_grad():
            lg = model.policy.get_distribution(torch.as_tensor(obs)).distribution.logits.numpy()
        return np.where(mask, lg, -np.inf).argmax(1)

    hasil_bobot, hasil_pilih = {}, {}
    for k in (0, 1, 2):
        d = f"{run}/e9c_s{k}"
        nrm = pickle.load(open(f"{d}/norm.pkl", "rb"))
        bc = MaskablePPO.load(f"{d}/bc.zip", device="cpu")
        pp = MaskablePPO.load(f"{d}/bc_ppo.zip", device="cpu")
        sa = {n: v.numpy() for n, v in bc.policy.state_dict().items()}
        sb = {n: v.numpy() for n, v in pp.policy.state_dict().items()}
        hasil_bobot[k] = nb.banding_bobot(sa, sb)
        hasil_pilih[k] = nb.kesamaan_pilihan(pilih(bc, nrm), pilih(pp, nrm), y, lams)
    _json("bab5_latih_bobot.json", {"bobot": hasil_bobot, "pilih": hasil_pilih})
    baris = []
    for x in hasil_bobot[0]:
        sel = " & ".join(nb.angka(hasil_bobot[k][i]["ubah_relatif"], 4)
                         for k in (0, 1, 2)
                         for i, y_ in enumerate(hasil_bobot[k]) if y_["lapis"] == x["lapis"])
        lapis = x["lapis"].replace("_", "\\_")
        # Diawali "{" supaya _tabel memberi akhir baris; baris yang diawali "\\" dianggap
        # perintah seperti \\addlinespace.
        baris.append(f"{{\\ttfamily {lapis}}} & {x['n']} & "
                     f"{nb.angka(x['norma_awal'])} & {sel}")
    _tulis("bab5_latih_bobot.tex", nb._tabel(
        "Perubahan relatif bobot jaringan dari sebelum ke sesudah tahap PPO",
        "tab:latih-bobot", "T{5.2cm}rrrrr",
        "Lapis & Angka & \\shortstack[r]{Norma\\\\sebelum (s0)} & \\multicolumn{3}{c}{"
        "Perubahan relatif \\emph{seed} 0/1/2}", baris,
        "Perubahan relatif = norma selisih bobot dibagi norma bobot sebelum PPO. "
        "\\emph{Sebelum PPO}: jaringan hasil BC + DAgger (\\texttt{bc.zip}); \\emph{sesudah}: "
        "\\texttt{bc\\_ppo.zip}. Tanda --: norma sebelum PPO nol, karena BC hanya melatih "
        "\\emph{actor} sehingga bias \\emph{critic} masih nol; perubahan relatifnya tidak "
        "terdefinisi."))
    baris = []
    for k, r in hasil_pilih.items():
        for lam, x in sorted(r["per_lam"].items()):
            baris.append(f"{k} & {nb._lam(lam)} & {nb.angka(x['sama'], 3)} & "
                         f"{nb.angka(x['sama_guru_a'], 3)} & {nb.angka(x['sama_guru_b'], 3)}")
        baris.append(f"{k} & semua & {nb.angka(r['sama'], 3)} & "
                     f"{nb.angka(r['sama_guru_a'], 3)} & {nb.angka(r['sama_guru_b'], 3)}")
        baris.append("\\addlinespace")
    _tulis("bab5_latih_pilih.tex", nb._tabel(
        "Kesamaan pilihan jaringan sebelum dan sesudah PPO pada keadaan data guru",
        "tab:latih-pilih", "llrrr",
        "\\emph{Seed} & $\\lambda$ & \\shortstack[r]{Sebelum = sesudah} & "
        "\\shortstack[r]{Sebelum = guru} & \\shortstack[r]{Sesudah = guru}", baris[:-1],
        f"{len(guru):,}".replace(",", ".") + " keadaan data guru (\\emph{seed} 7000-7001, "
        "tujuh instans latih). Tingkat yang boleh dipilih: tingkat 0 dan tingkat yang sah "
        "menurut data guru. Pilihan deterministik (peluang terbesar)."))


# ------------------------------------------------------------------ C satu episode

def episode(run, src, model_seed=0, kondisi=("MK01", 1.5, 0.15, 5000)):
    """Catatan per kejadian untuk episode contoh Bab IV 4.6 di λ 0,3, 3, dan 30.

    J akhir setiap λ dicocokkan dengan ``cek_episode.run`` untuk kondisi yang sama; skrip
    berhenti bila berbeda.
    """
    import copy

    os.environ["PPO_DIR"] = f"{run}/e9c_s{model_seed}"
    os.environ["METODE"] = "ppo_bc_ppo_g"
    os.environ.pop("PERIKSA", None)
    simpan = sys.argv
    sys.argv = [simpan[0], src, f"{run}/b3_model.pkl", "/dev/null"]
    try:
        import cek_episode as ce
    finally:
        sys.argv = simpan
    import cek_b3
    import instans_acak
    from dfjsp.core.domain import set_capacity
    from dfjsp.env.repair_env import RepairEnv
    from dfjsp.env.schedule import recompute_timeline
    from dfjsp.env.state import total_tardiness
    ce._init(f"{run}/b3_model.pkl")
    nama, cap, inten, seed = kondisi
    semua = {}
    for lam in LAM_LABEL:
        inst = copy.deepcopy(instans_acak.load_any(nama))
        set_capacity(inst, cap)
        env = ce.EpProbe(lam=lam)
        env.reset(inst, intensity=inten, seed=seed)
        pol = ce.PPOPol("bc_ppo", True)

        class Rekam:
            guard = True

            def __init__(self):
                self.minta = []

            def __call__(self, x, m):
                a = pol(x, m)
                self.minta.append(a)
                return a

        langkah = []
        while not env.done:
            E = env.mean_effort
            rek = Rekam()
            jenis = [str(getattr(ty, "value", ty)) for ty, _ in env.per_sub]
            telat0 = total_tardiness(env.inst, env.sched) / E
            bs = RepairEnv._state(env)
            _, dipakai, _, cur, chg = cek_b3.run_batch(env, lam, rek, bs, record=False,
                                                       want_sched=True)
            sched = None if cur is env.sched else cur
            recompute_timeline(env.inst, env.sched, env.t_now)
            telat1 = total_tardiness(env.inst, env.sched if sched is None else sched) / E
            langkah.append({"t": float(env.t_now), "jenis": jenis, "diminta": rek.minta,
                            "dipakai": list(dipakai), "telat_sebelum": telat0,
                            "telat_sesudah": telat1, "ubah": int(chg if sched else 0)})
            if sched is None:
                env.step(0)
            else:
                env.probes[3] = {"tard": total_tardiness(env.inst, sched), "changes": chg,
                                 "entries": dict(sched.entries), "moves": []}
                env.step(3)
        telat_akhir = total_tardiness(env.inst, env.sched) / env.mean_effort
        rows = nb.catatan_kejadian(langkah, lam)
        acuan = ce.run((nama, cap, inten, seed, "ppo_bc_ppo_g", lam))
        j_skrip = telat_akhir + lam * rows[-1]["ubah_kumulatif"]
        if abs(j_skrip - acuan["J"]) > 1e-6 or rows[-1]["ubah_kumulatif"] != acuan["ubah"]:
            raise SystemExit(f"λ {lam}: J skrip {j_skrip} lawan cek_episode.run {acuan['J']}")
        if abs(rows[-1]["J"] - acuan["J"]) > 1e-6:
            print(f"catatan: λ {lam}, J sesudah kejadian terakhir {rows[-1]['J']} berbeda "
                  f"dengan J akhir {acuan['J']} (keterlambatan berubah sesudah kejadian "
                  "terakhir)")
        semua[lam] = {"kejadian": rows, "J_akhir": acuan["J"], "telat_akhir": telat_akhir,
                      "ubah": acuan["ubah"]}
        _tulis(f"bab5_episode_lam{lam:g}.tex", nb.tabel_kejadian(
            rows, lam, f"Perkembangan jadwal per kejadian pada episode contoh, "
            f"$\\lambda = {nb._lam(lam)}$", f"tab:episode-lam{lam:g}".replace(".", "")))
    _json("bab5_episode.json", {"kondisi": list(kondisi), "model_seed": model_seed,
                                "lam": semua})


if __name__ == "__main__":
    perintah, *arg = sys.argv[1:]
    {"validasi": validasi, "uji": uji, "sifat": sifat_tabel, "latih": latih, "episode": episode}[perintah](*arg)
