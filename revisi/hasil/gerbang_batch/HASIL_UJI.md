# Hasil uji E15 E9c (ARLS-PPO) pada data uji

Berkas ini mencatat angka dari satu-satunya pembukaan data uji untuk resep akhir, sesuai
`docs/KUNCI_UJI.md`. Kunci uji tidak diubah. Berkas ini hanya mencatat angka dan pemeriksaannya;
cara menafsirkannya di naskah (Bab V) diputuskan bersama pemilik.

## 1. Pembukaan

| hal | isi |
|---|---|
| izin | izin tertulis pemilik, 19 November 2026 |
| catatan | `buka_uji.log`, satu baris: `2026-11-19T22:34:09`, commit `1ecdcbf81b35559249b554d655fe888a97c31e21` |
| perintah | perintah di `docs/KUNCI_UJI.md` bagian 10, dari folder `revisi/hasil/gerbang_batch`, cabang `uji/buka-e15-20261119` |
| mulai dan selesai | 19 November 2026 22.34.08 WIB sampai 21 November 2026 03.15.10 WIB (sekitar 28 jam 41 menit), kode keluar 0 |
| proses | 8 proses paralel, `OMP_NUM_THREADS=1`, pemeriksa jadwal (`PERIKSA=1`) hidup di semua episode |
| hash model | ke-14 hash di catatan pembukaan sama dengan `runs/e15_validasi/SHA256SUMS` (bagian 2 kunci uji); `sha256sum -c SHA256SUMS` sesudah pengujian juga lulus semua, jadi model tidak berubah |
| uji ulang | tidak ada; `--lanjut` tidak dipakai |

## 2. Kelengkapan berkas episode

Diperiksa dengan skrip yang membandingkan setiap berkas dengan `buka_uji.rencana_uji`. Skrip ini
tidak menghitung keterlambatan atau J.

| berkas | baris / rencana | kunci dobel | seed di luar 10000-10009 dan 11000-11004 | J bukan angka | `ubah` beda dengan pemeriksa | keputusan tanpa pemeriksaan | pelanggaran keras |
|---|---|---|---|---|---|---|---|
| `ep_aturan` (aor, seq, fix3_g, acak_g) | 44.280 / 44.280 | 0 | 0 | 0 | 0 | 0 | 0 |
| `ep_hgb` (il_g) | 11.070 / 11.070 | 0 | 0 | 0 | 0 | 0 | 0 |
| `ep_ppo_s0` (ppo_bc_ppo_g, ppo_bc_ppo, ppo_bc_g) | 33.210 / 33.210 | 0 | 0 | 0 | 0 | 0 | 0 |
| `ep_ppo_s1` | 33.210 / 33.210 | 0 | 0 | 0 | 0 | 0 | 0 |
| `ep_ppo_s2` | 33.210 / 33.210 | 0 | 0 | 0 | 0 | 0 | 0 |
| jumlah | 154.980 / 154.980 | | | | | | 0 |

Kunci setiap berkas sama persis dengan himpunan pekerjaan di rencana. Banyak `ms_keputusan` sama
dengan banyak keputusan di setiap baris. Berkas `.json` dan `.jsonl` berisi baris yang sama
banyaknya. Temuan lunak pemeriksa jadwal (K1 kapasitas dan A5 karena rework) ada di semua berkas;
keduanya memang lunak menurut `dfjsp.verify` dan tidak dihitung sebagai pelanggaran.

## 3. Berkas mentah

Berkas episode terlalu besar untuk repositori (sekitar 130 MB per berkas PPO, 1,1 GB semuanya),
jadi berkas itu disimpan di mesin penelitian, di `revisi/hasil/gerbang_batch/uji/`. Hash SHA-256-nya
dicatat di sini supaya angka di bawah dapat ditelusuri ke berkas yang sama.

```
60af931c9bf8de6404fd14f3079f4f526cc774d8d6e50d860a74d80e7391db15  ep_aturan.jsonl
2c9862f6812895457dc1982f803d4e0c15c5275609a03c44e5d49a457b51c46a  ep_hgb.jsonl
eb5fbcd568ead0163e41d180b40820199d9e84bb10a0cf29071251e243e48422  ep_ppo_s0.jsonl
f4b5cb3595c142b781b381cc1d383d34220d767054fe421c67ba0f51ccf05064  ep_ppo_s1.jsonl
6babc7e2b2acb35fd34eb54505d04786fc614ffa10030bbf7f8a872181a5e3b9  ep_ppo_s2.jsonl
2d48e6f8366b13e1071349dbae8084b59f15e2592f3359d11f06e9288cd131d8  ep_aturan.json
57dd41c79d0cc0c22d0b37252a8da2e3e1456cf2515fe06da4428c580be48a6d  ep_hgb.json
26b99ad90cfd2a738c94ff002eb816c70390938ad92adf4a0ebd9a4b8c411969  ep_ppo_s0.json
39d02f84bc87c537bf4f922d5fa08728a80287dc36ec946385866b8ed6469f1f  ep_ppo_s1.json
cc59114f0c08ca778a77b6212f08192f4766b001c8b0607ee6963da78b3e45e6  ep_ppo_s2.json
136dbfdca630f609d53df92247d12ff72d7905abb529083f0f2e6eab137590a7  sifat_uji.json
3262e87ab84011d1a1c564eb98ee1491b751f4b8a6948714a86cc677bc6566cb  buka_uji.log (folder induk)
```

`sifat_uji.json` (sifat instans uji, ditulis sebelum episode berjalan) dan `buka_uji.log` ikut
di-commit di PR ini.

## 4. Analisis

Dijalankan dengan `analisis_uji.py` pada commit `4e41520e` (main sesudah PR #186, cabang uji
di-fast-forward ke main sesudah pengujian selesai):

```
cd revisi/hasil/gerbang_batch
SIFAT=uji/sifat_uji.json PYTHONPATH=.:../../../exp1/dfjsp-ppo-repair/src \
    ../../../exp1/dfjsp-ppo-repair/.venv/bin/python analisis_uji.py ppo_bc_ppo_g \
    "Hasil uji E15 E9c (data uji, dibuka 19 November 2026)" \
    uji/analisis_uji.md uji/analisis_uji_sifat.csv \
    uji/ep_aturan.json uji/ep_hgb.json uji/ep_ppo_s0.json uji/ep_ppo_s1.json uji/ep_ppo_s2.json
```

J ketiga seed model PPO dirata-rata per kondisi lebih dulu (`pivot_table`, bagian 6 kunci uji).
Analisis selesai pada 21 November 2026 pukul 03.33 WIB dengan kode keluar 0.

## 5. Ringkasan menurut aturan kunci uji

Bagian ini hanya menerapkan aturan kalimat yang dikunci di `docs/KUNCI_UJI.md` bagian 6 dan 7
pada angka di laporan. Tafsiran dan susunan Bab V belum dibuat.

### 5.1 Keluarga uji utama (butir 2, selang 98,75%)

| pembanding | λ | uk. 1 | uk. 2 | kalimat menurut bagian 7 |
|---|---|---|---|---|
| acak + penjaga (`acak_g`) | 0,3 | −24,01 [−26,59; −21,38] | −23,62 [−25,58; −21,74] | lebih baik dari acak + penjaga di λ 0,3 |
| acak + penjaga (`acak_g`) | 3 | −14,88 [−16,74; −12,99] | −15,73 [−17,10; −14,11] | lebih baik dari acak + penjaga di λ 3 |
| tingkat 3 tetap + penjaga (`fix3_g`) | 0,3 | +18,14 [+7,86; +32,65] | −4,45 [−6,29; −0,91] | tidak ditemukan selisih yang tegas di λ 0,3 |
| tingkat 3 tetap + penjaga (`fix3_g`) | 3 | +0,49 [−2,17; +4,20] | −6,31 [−7,62; −4,30] | tidak ditemukan selisih yang tegas di λ 3 |

Lawan tingkat 3 tetap + penjaga, kedua ukuran tidak searah. Di λ 0,3 selang uk. 1 seluruhnya di
atas nol, sedangkan selang uk. 2 seluruhnya di bawah nol. Di λ 3 selang uk. 1 memuat nol. Karena
klaim hanya ditulis bila kedua ukuran searah (bagian 6), kalimatnya "tidak ditemukan selisih yang
tegas". Analisis kepekaan benchmark saja memberi arah yang sama untuk keempat baris ini.

### 5.2 Butir 1 (celah ke Sequential, selang 95%)

Kalimat menurut bagian 7: "celah di λ tanpa label paling banyak X poin lebih besar dari acuan",
dengan X = batas atas selang 95%.

| pasangan λ | X uk. 1 | X uk. 2 |
|---|---|---|
| 0,1 lawan 0,3 | 17,75 | 1,05 |
| 1 lawan interpolasi log 0,3 dan 3 | −1,64 | 0,38 |
| 10 lawan interpolasi log 3 dan 30 | 6,68 | 5,47 |
| pelengkap: rata-rata tanpa label lawan rata-rata berlabel | 10,51 | 2,92 |

X negatif (λ 1, uk. 1) berarti celah di λ tanpa label lebih kecil dari acuan, dengan seluruh
selang di bawah nol.

### 5.3 Indikator

| indikator | hasil | keterangan |
|---|---|---|
| I-1 cepat | tercapai | p95 waktu pemilih ARLS-PPO (`ppo_bc_ppo_g`) per instans 1,49 sampai 1,92 ms di ke-38 instans uji, di bawah 50 ms. p95 waktu total terbesar 466,42 ms, dilaporkan tanpa ambang. Diukur di 8 proses paralel. |
| I-2 layak | tercapai | 0 pelanggaran keras pemeriksa jadwal di 154.980 episode, semua metode. Pemeriksaan tambahan dari kolom `periksa`: di setiap episode, jumlah temuan lunak K1 (kapasitas) pada jadwal yang dipasang tidak lebih besar dari jumlahnya pada jadwal sebelum perbaikan. Jumlah ini menghitung pengembang yang kelebihan beban per keputusan, bukan banyaknya jam lebih. |
| I-3 stabil | deskriptif | tabel "Perubahan jadwal (I-3)" di laporan, tanpa ambang |
| I-4 | dihapus | `KERANGKA_NASKAH.md` bagian 8, catatan 19 November 2026 |

## 6. Laporan lengkap

Keluaran `analisis_uji.py` disimpan apa adanya di `uji/analisis_uji.md` (semua tabel: keluarga
utama, gabungan, kepekaan benchmark, butir 1, per keluarga, I-1, I-2, I-3, dan tingkat yang
dipilih). Tabel per sifat instans (bagian 9 kunci uji, deskriptif) ada di
`uji/analisis_uji_sifat.csv`.

## 7. Pemeriksaan deskriptif sesudah hasil dibaca: lawan Fixed-L3 + penjaga

Bagian ini dibuat sesudah laporan di atas dibaca, untuk memahami mengapa uk. 1 dan uk. 2
berbeda arah lawan tingkat 3 tetap + penjaga (`fix3_g`). Pemeriksaannya deskriptif, tidak
memakai selang, dan tidak menambah atau mengubah klaim di bagian 5. Tidak ada episode baru;
angkanya dihitung dari berkas episode yang sama.

Caranya: J ARLS-PPO dirata-rata atas 3 seed model per kondisi, lalu rasio per kondisi =
J ARLS-PPO / J `fix3_g`. Di dalam setiap instans, kondisi dibagi menjadi empat kelompok sama
banyak menurut besar J `fix3_g` (Q1 terkecil sampai Q4 terbesar), supaya instans dengan J besar
tidak menguasai pengelompokan. Rata-rata di tabel adalah rata-rata biasa atas kondisi.

| λ | kelompok | rata-rata rasio | median rasio | porsi kondisi ARLS-PPO lebih kecil | porsi J total `fix3_g` | rasio J total kelompok |
|---|---|---|---|---|---|---|
| 0,3 | Q1 (J terkecil) | 1,612 | 1,067 | 0,323 | 0,035 | 1,168 |
| 0,3 | Q2 | 1,147 | 1,031 | 0,339 | 0,105 | 1,088 |
| 0,3 | Q3 | 1,057 | 1,000 | 0,501 | 0,264 | 0,992 |
| 0,3 | Q4 (J terbesar) | 0,898 | 0,915 | 0,753 | 0,596 | 0,822 |
| 3 | Q1 (J terkecil) | 1,105 | 1,019 | 0,440 | 0,043 | 1,060 |
| 3 | Q2 | 1,053 | 1,031 | 0,364 | 0,118 | 1,052 |
| 3 | Q3 | 0,963 | 0,973 | 0,574 | 0,277 | 0,946 |
| 3 | Q4 (J terbesar) | 0,900 | 0,901 | 0,767 | 0,562 | 0,859 |

Yang terlihat:

- `fix3_g` lebih sering unggul di kondisi yang J-nya kecil, dan ARLS-PPO lebih sering unggul
  di kondisi yang J-nya besar. Kelompok Q4 memuat lebih dari separuh J total, sehingga rasio
  J total (uk. 2) memihak ARLS-PPO. uk. 1 memberi bobot yang sama pada setiap kondisi, sehingga
  banyak kondisi kecil dengan rasio besar menarik rata-ratanya ke atas.
- Menurut intensitas gangguan di λ 0,3, rata-rata rasio 1,460 di intensitas 0,05, 1,121 di
  0,15, dan 0,980 di 0,30. Rasio J total 1,093, 1,005, dan 0,859.
- Rasio terbesar di λ 0,3 sekitar 21,9 (hurink/vdata/la21, intensitas 0,05, seed 11004, ketiga
  kapasitas): `fix3_g` hampir menghapus keterlambatan dengan 15 perubahan, sedangkan ARLS-PPO
  menyisakan suku keterlambatan sekitar 150 dengan 28 sampai 33 perubahan.
- Di setiap kelompok, ARLS-PPO rata-rata membuat lebih sedikit perubahan daripada `fix3_g`.
  Di Q1 dan Q2, suku keterlambatannya lebih besar; di Q4 lebih kecil.
