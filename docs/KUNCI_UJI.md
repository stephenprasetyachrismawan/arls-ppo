# Dokumen kunci uji (disetujui pemilik 19 November 2026)

Status: **disetujui pemilik pada 19 November 2026, isinya dikunci.** Sembilan usul agen
disetujui pemilik pada 18 November 2026; bagian [kosong] diisi agen pada 19 November 2026
sesudah E17 dan E16 selesai, isinya diperiksa terhadap kode (bagian 13), dan pemilik
menyetujui seluruh dokumen pada 19 November 2026. Draf aslinya ada di
`revisi/kunci_uji_draf_20261118.md` (riwayat git). Isi dokumen ini tidak diubah lagi; hasil
pembukaan data uji ditulis di berkas terpisah. Data uji belum dibuka saat dokumen ini
disetujui.

Tanda di dokumen ini:

- **[diputuskan]**: sudah diputuskan pemilik, tanggal dan sumbernya disebut.
- **[kosong]**: diisi sesudah E17 dan E16 selesai, sebelum data uji dibuka.
- **[diisi 19 November 2026, menunggu persetujuan]**: bekas [kosong] yang diisi agen; semuanya
  disetujui pemilik pada 19 November 2026 bersama seluruh dokumen.

Sumber: `revisi/saran_fable_langkah_berikut_20261116.md` bagian 1.6 (12 butir isi dokumen
kunci), `revisi/rencana_benchmark_uji_20261118.md`, dan `docs/ARSITEKTUR_PIPELINE.md` 5.6.

## 1. Status data uji

**[diputuskan, 18 November 2026]** Kalimat untuk naskah:

> Data uji terdiri atas dua bagian. Bagian pertama, MK03, MK04, dan MK09 dengan seed episode
> 10000-10009, pernah dibuka satu kali pada 20 Oktober 2026 untuk run `resmi1` dengan
> resep lama (MPPO-SRS). Sesudah itu metode dirancang ulang hanya dengan seed validasi
> 5000-5001 dan seed guru 7000-an, dan bagian ini dibuka kedua kali untuk resep akhir.
> Bagian kedua, 35 instans benchmark FJSP publik dengan seed episode 11000-11004, dibuka
> pertama kali untuk resep akhir.

Kedua bagian dibuka sekali saja untuk resep akhir. Tidak ada uji ulang (butir 11).

## 2. Resep yang diuji

**[diisi 19 November 2026, menunggu persetujuan]** Satu resep saja, dipilih oleh E17
(`ringkas_e17.py`, kriteria K1-K4) lalu E16 (kriteria (a)-(d)) di
`docs/ARSITEKTUR_PIPELINE.md` 5.6. E17 tidak memilih P0, P1, atau P2
(`e17/HASIL_E17.md`), dan E16 gagal kriteria (a) dan (b) (`e16/HASIL_E16.md`, batas atas
λ 30 uk. 2 di 7 MK +3,06% dengan 10.000 sampel ulang). Jadi resep tetap E15 E9c. Pemilik
memilih jalan A pada 19 November 2026: tahap PPO ditulis "tidak melampaui imitasi".

| hal | isi |
|---|---|
| resep | E15 E9c: BC pada data guru Sequential (seed guru 7000-7001) ditambah satu putaran DAgger (keadaan dari seed 7002-7003 diberi label guru), lalu MaskablePPO |
| data latih | 7 MK saja (MK01, 02, 05, 06, 07, 08, 10); 86.832 baris BC + DAgger; 60 instans benchmark latih tidak dipakai karena E16 tidak lolos |
| langkah PPO | 100.000 (`latih_ppo_seq.py ... 100000`, `e15/jalan_e15.sh`); seed model 0, 1, 2 |
| folder model | `exp1/dfjsp-ppo-repair/runs/e15_validasi/e9c_s{0,1,2}/`: `bc_ppo.zip` (BC + DAgger + PPO), `bc.zip` (BC + DAgger saja, untuk metode sumbangan tahap PPO), `norm.pkl` (normalisasi keadaan) |
| model lain | `runs/e15_validasi/hgb7d.pkl` (boosting 7 MK + DAgger, data 86.832 baris yang sama) dan `runs/e15_validasi/b3_model.pkl` (dimuat `cek_episode._init` untuk semua metode) |
| hash | `runs/e15_validasi/SHA256SUMS`; `sha256sum -c` lulus pada 19 November 2026; isinya disalin di bawah |

Hash SHA-256 berkas yang dipakai pembukaan (dari `SHA256SUMS`):

```
58b40fed9b61faa75906dd855ae9f5e81d8076e7d30e42cfb178f70e5dc1563f  e9c_s0/bc_ppo.zip
b5bc3d0db83520f21dbda6ef7e7f01539006a18dd0e4fc4691ee2348df8a6744  e9c_s0/bc.zip
047dc19be6f268d9411f3f1fa8dc4abc0a6a9bb7196aa1aca6521d6efe51cdb1  e9c_s1/bc_ppo.zip
8dbe62a1a06b6096c6d5a5b24238cded1e5f648754929e055e613d15b28cb72f  e9c_s1/bc.zip
6a41d5a1fa8567b2581278a1fab4e7886ea10622294cf09f5e8aacc74e1d9bc3  e9c_s2/bc_ppo.zip
b2148ba981b5859b5dea2d2f1b3d0f4ea8210285288aaeec2570a9a8d81be6b0  e9c_s2/bc.zip
28353feeea419bec41bbbfb05cdabfad91af88e75dfe491f54a3993a0dcb5ee8  e9c_s{0,1,2}/norm.pkl
c43890937fac922332ccd3652e2561c617e37e4f74311fc8f69bb48628dd44f7  hgb7d.pkl
367de178ddd76c7d24554d784a0d83c4f4e6062e5872522a485d13e2b5effd9e  b3_model.pkl
```

## 3. Daftar metode yang dilaporkan

**[diputuskan, 18 November 2026]** Daftar tetap. Tidak ada metode yang ditambah atau dibuang sesudah angka uji
terlihat.

| metode | peran | nama di kode (`buka_uji.py`) |
|---|---|---|
| resep terpilih + penjaga (3 seed model) | metode usulan | `ppo_bc_ppo_g` |
| resep terpilih tanpa penjaga | sumbangan penjaga | `ppo_bc_ppo` |
| model awal resep (BC + DAgger) + penjaga | sumbangan tahap PPO | `ppo_bc_g` |
| tingkat 0 (AOR, right-shift) | pembanding tanpa perubahan urutan | `aor` |
| Sequential (guru) | acuan butir 1 | `seq` |
| tingkat 3 tetap + penjaga | pembanding aturan tetap (butir 2) | `fix3_g` |
| acak + penjaga | pembanding pemilih tanpa belajar (butir 2) | `acak_g` |
| boosting 7 MK + DAgger + penjaga | pemilih terlatih lain | `il_g` dengan `hgb7d.pkl` |

Boosting 7 MK + 50 acak + DAgger tidak dimasukkan: di E15 boosting ini jatuh di λ 10, dan
satu boosting sudah cukup sebagai pemilih terlatih pembanding. Bila E16 lolos, boosting yang
dilaporkan adalah versi E16 yang dilatih dengan data latih yang sama.

## 4. Instans, kondisi, dan seed

**[diputuskan, 18 November 2026]** (`rencana_benchmark_uji_20261118.md` bagian 2, 3, 6)

| bagian | instans | kondisi per instans | seed episode | kondisi per λ |
|---|---|---|---|---|
| MK | MK03, MK04, MK09 | 3 kapasitas × 3 intensitas | 10000-10009 | 270 |
| benchmark | 35 instans di bagian 5 | 3 kapasitas × 3 intensitas | 11000-11004 | 1.575 |

Benchmark validasi (23 instans) dan benchmark latih (60 instans) tidak pernah menjadi data
uji; pembagiannya menurut instans dasar sehingga tidak ada instans uji yang berbagi waktu
proses dengan instans latih atau validasi (bagian 11 rencana benchmark).

Kapasitas 1,0 / 1,5 / 2,0. Intensitas gangguan 0,05 / 0,15 / 0,30. λ 0,1 / 0,3 / 1 / 3 /
10 / 30; tiga berlabel guru (0,3 / 3 / 30), tiga tidak. Gugus bootstrap (instans,
intensitas, seed): 90 untuk MK, 525 untuk benchmark.

## 5. Benchmark

**[diputuskan, 18 November 2026]** Sumber `SchedulingLab/fjsp-instances` (MIT), commit
`ac4c3402312bfbeafcf4472d78be567d4e6b46ab`. 35 instans: mk11-mk15; Hurink la01, la06, la11,
la16, la21, la26, la36 masing-masing edata, rdata, vdata; Barnes mt10c1, setb4c9, seti5c12;
Dauzère-Pérès 01a, 02a, 03a, 07a, 08a, 09a. Checksum SHA-256 ke-35 berkas dikunci di
pemuat sebelum uji. Transformasi domain sama dengan MK (`k_tight` 1,5, dependensi
antar-job 0,10, `domain_seed` 0).

**[diputuskan, 18 November 2026]** Skala upaya: instans benchmark yang rata-rata upayanya di
luar rentang 7 MK latih (3,82-12,04) dikali `s = 7,53 / rata-rata upaya instans`
(`rencana_benchmark_uji_20261118.md` bagian 9; di kode `_skala` dengan `UPAYA_ACUAN` dan
`RENTANG_UPAYA_LATIH`). Menurut rencana itu ke-35 instans uji berada di luar rentang, supaya rata-rata upayanya sama dengan rata-rata 7 MK latih (gangguan berukuran mutlak dalam
jam, lihat `rencana_benchmark_uji_20261118.md` bagian 9). MK03, MK04, MK09 tidak diubah.
Faktor tiap instans dilaporkan.

## 6. Ukuran dan cara menggabung

**[diputuskan, 16 November 2026]** Dua ukuran agregat: uk. 1 rata-rata rasio J per kondisi,
uk. 2 rasio J total. Klaim "lebih baik" atau "tidak lebih buruk" hanya ditulis bila kedua
ukuran searah dan sama-sama memenuhi kriterianya. Bila hanya satu, hasilnya ditulis "tidak
tegas". J PPO dirata-rata atas 3 seed model per kondisi dulu.

**[diputuskan, 18 November 2026]** Bukti utama = gabungan MK dan benchmark, tiap instans
berbobot sama. Analisis kepekaan = benchmark saja, dilaporkan di samping gabungan. Bila arah
kesimpulan berbeda, keduanya ditulis apa adanya.

**[diputuskan, 18 November 2026]** Rumus bobot sama per instans:

- uk. 1 gabungan = rata-rata atas instans dari (rata-rata rasio J per kondisi di instans
  itu), dikurangi 1. MK punya 90 kondisi per instans, benchmark 45, jadi tanpa bobot ini
  MK berbobot dua kali.
- uk. 2 gabungan = rata-rata atas instans dari (total J metode / total J pembanding di
  instans itu), dikurangi 1. Dengan begitu instans dengan J besar tidak menguasai hasil.
- Bootstrap bergugus (instans, intensitas, seed) dengan strata = instans: gugus diambil
  ulang di dalam tiap instans, sehingga setiap instans selalu ada di setiap sampel ulang.
  `make_column_sampler(strata=...)` sudah mendukung ini. 10.000 sampel ulang.
- Kode baru untuk ini ditulis dengan /tdd dan dicoba dulu pada data validasi (gladi, butir
  12), bukan pada data uji.

## 7. Klaim dan kriterianya

**[diputuskan, 18 November 2026]** Keluarga uji utama kecil, dengan koreksi Bonferroni: butir 2 di λ 0,3 dan 3,
lawan tingkat 3 + penjaga dan lawan acak + penjaga (4 uji). Untuk keluarga ini dipakai
selang 98,75% (95% dibagi 4). Semua selang lain 95% dan berlabel pelengkap; selang pelengkap
tidak dipakai untuk kalimat klaim.

Kalimat klaim butir 2 per λ dan per pembanding:

| hasil (kedua ukuran) | kalimat |
|---|---|
| selang 98,75% seluruhnya di bawah 0 | "lebih baik dari [pembanding] di λ [λ]" |
| selang seluruhnya di atas 0 | "lebih buruk dari [pembanding] di λ [λ]" |
| selain itu | "tidak ditemukan selisih yang tegas di λ [λ]" |

Butir 1 (celah ke Sequential di λ tanpa label tidak lebih besar dari di λ berlabel):
pasangan λ di `ARSITEKTUR_PIPELINE.md` 5.6 dipakai apa adanya. **[diputuskan, 18 November 2026]** Di data uji
tidak dipakai kata "lolos". Yang ditulis: "celah di λ tanpa label paling banyak X poin
lebih besar dari acuan", dengan X = batas atas selang 95%. Alasannya, aturan "selang memuat
nol" makin mudah dipenuhi bila selang makin lebar.

Indikator I-1 (waktu) dan I-2 (layak) dihitung dengan cara di `KERANGKA_NASKAH.md`.
Untuk I-2, pemeriksa jadwal (`dfjsp.verify`) dipasang di semua episode uji; jumlah
pelanggaran keras dilaporkan apa adanya.

## 8. Kondisi dengan J pembanding = 0

**[diputuskan, 18 November 2026]** Bila J pembanding = 0 di suatu kondisi, kondisi itu dikeluarkan dari uk. 1
untuk kedua metode, dan banyaknya dilaporkan per instans. uk. 2 memakai semua kondisi. Bila
total J pembanding di satu instans = 0, instans itu dikeluarkan dari kedua ukuran dan
dilaporkan.

## 9. Analisis menurut sifat instans

**[diputuskan, 18 November 2026]** Deskriptif, tanpa hipotesis
(`rencana_benchmark_uji_20261118.md` bagian 7). Sifat instans dihitung sebelum episode:
ukuran, fleksibilitas, beban, keketatan, kepadatan dependensi, dan posisi terhadap rentang
data latih. Perilaku yang dicatat per episode: sebaran tingkat yang dipilih PPO dan
seberapa sering penjaga menolak pilihan PPO. Catatan ini butuh kode baru yang terbukti tidak
mengubah hasil episode.

## 10. Jalur uji

**[diputuskan, 18 November 2026]** Pilihan B dari saran Fable: tahap uji baru yang memakai kode evaluasi E15
(`cek_episode.py`, `cek_b3.py`) yang dipindah apa adanya ke `src/`, dengan bendera
`--buka-uji` dan catatan `buka_uji.log`. Pilihan A (menulis ulang di pustaka) lebih rapi
tetapi butuh beberapa hari dan berisiko mengubah perilaku.

Syarat jalur uji, apa pun pilihannya: di seed validasi, jalur baru menghasilkan J yang sama
persis (selisih 0,0) dengan `e15/ep_*.json` untuk semua metode dan semua λ.

**Yang dibangun (PR #180, digabung 19 November 2026): versi ringan pilihan B.** Kode
evaluasi tidak dipindah ke `src/`. `revisi/hasil/gerbang_batch/buka_uji.py` memanggil
`cek_episode.run` yang sama persis dengan yang menghasilkan berkas E15 dan E16, dengan
bendera `--buka-uji`, catatan `buka_uji.log`, dan `--lanjut` untuk meneruskan proses yang
mati di tengah (hanya bila hash model dan hash checksum data sama dengan catatan pertama).
Pemeriksa jadwal (`PERIKSA=1`) hidup di semua episode. Sifat instans uji disimpan ke
`<keluar>/sifat_uji.json` di dalam proses pembukaan. Status syarat selisih 0,0: lihat
bagian 13.

Perintah pembukaan (dari folder `revisi/hasil/gerbang_batch`; path model harus mengarah ke
`exp1/dfjsp-ppo-repair/runs`, bukan `runs/` di folder ini):

```
R=../../../exp1/dfjsp-ppo-repair/runs/e15_validasi
PYTHONPATH=.:../../../exp1/dfjsp-ppo-repair/src ../../../exp1/dfjsp-ppo-repair/.venv/bin/python \
    buka_uji.py --buka-uji --resep $R/e9c_s --b3 $R/b3_model.pkl --hgb $R/hgb7d.pkl --keluar uji
```

Penjaga data uji: `dfjsp.datasets.load_benchmark` menolak 35 instans uji dengan
`PermissionError` sebelum berkas dibuka, kecuali dipanggil dengan `buka_uji=True`; skrip
gerbang hanya bisa memilih `bench:latih` dan `bench:validasi`. Untuk MK03, MK04, MK09 tetap
berlaku penjaga PR #167 (`penjaga_uji.py`).

## 11. Bila klaim gagal di data uji

**[diputuskan, 18 November 2026]** Klaim dipersempit atau dibuang sesuai angka uji. Angka validasi tidak
dipakai sebagai gantinya. Tidak ada uji ulang, tidak ada latih ulang sesudah data uji
dibuka, dan tidak ada metode yang ditambah atau dibuang.

## 12. Urutan sebelum membuka data uji

Status 19 November 2026 dalam kurung.

1. E17 selesai, `ringkas_e17.py` memilih resep. (selesai: E15 E9c tetap, PR #181)
2. E16 dijalankan dengan resep itu, kriteria (a)-(d) memutuskan data latih. (selesai: 7 MK
   saja, PR #182)
3. Pemuat benchmark (checksum, skala upaya) dan perluasan penjaga data uji, dengan /tdd.
   (selesai: ke-35 instans uji punya checksum di `configs/checksums.yaml`)
4. Catatan tingkat dan penolakan penjaga di keluaran episode, dengan /tdd. (selesai)
5. Jalur uji (bagian 10) dengan syarat selisih 0,0 di validasi. (selesai: gladi kedua,
   selisih 0 di semua baris berpembanding, bagian 13)
6. Gladi: jalur uji dijalankan ujung ke ujung di seed validasi sampai semua tabel dan selang
   keluar, termasuk rumus gabungan bagian 6. (selesai: gladi keempat mengeluarkan semua tabel,
   termasuk butir 1, I-1, dan I-2, bagian 13)
7. Bagian [kosong] diisi; pemilik menyetujui dokumen ini; dokumen dipindah ke `docs/` dan
   di-commit. (selesai: temuan T1-T5 diperbaiki di PR #184, PR #183 dan
   #184 digabung, pemilik menyetujui pada 19 November 2026, dokumen dipindah ke
   `docs/KUNCI_UJI.md`)
8. Pemilik memberi izin tertulis; data uji dibuka sekali; catatan di `buka_uji.log`.

## Keputusan pemilik atas usul agen (disetujui semua, 18 November 2026)

| no | keputusan | yang disetujui |
|---|---|---|
| 1 | kalimat status data uji (bagian 1) | seperti tertulis |
| 2 | daftar metode (bagian 3) | 8 metode, boosting + 50 acak tidak dimasukkan |
| 3 | skala upaya benchmark (bagian 5) | ya, `s = 7,53 / rata-rata` |
| 4 | rumus bobot sama per instans dan bootstrap berstrata instans (bagian 6) | seperti tertulis |
| 5 | koreksi perbandingan berganda (bagian 7) | Bonferroni, keluarga 4 uji, selang 98,75% |
| 6 | butir 1 di data uji (bagian 7) | laporkan batas atas, tanpa kata "lolos" |
| 7 | J pembanding = 0 (bagian 8) | keluarkan dari uk. 1, laporkan banyaknya |
| 8 | jalur uji (bagian 10) | pilihan B |
| 9 | bila klaim gagal (bagian 11) | persempit, tanpa uji ulang |

## 13. Hasil pemeriksaan agen (19 November 2026)

Isi dokumen ini dicocokkan dengan kode yang akan dipakai: `buka_uji.py`, `cek_episode.py`,
`analisis_uji.py`, `banding_umum.py`, `dfjsp.datasets._benchmark`, dan berkas model di
`runs/e15_validasi`. Data uji tidak dibuka selama pemeriksaan.

**Sesuai.** Delapan metode bagian 3 sama dengan `METODE_PPO`, `METODE_ATURAN`, dan
`METODE_BOOSTING`. Instans, kapasitas, intensitas, seed, dan enam λ bagian 4 sama dengan
`rencana_uji` (270 kondisi MK dan 1.575 kondisi benchmark per λ per metode). Ke-35 instans
uji ada di `BENCHMARK_UJI` dan semuanya punya checksum. Rumus bobot sama per instans,
bootstrap berstrata instans dengan 10.000 sampel ulang, selang 98,75% untuk keluarga utama,
dan aturan J pembanding = 0 (bagian 6-8) sama dengan `bandingkan_gabungan` dan
`analisis_uji.analisis`. Hash berkas model sama dengan `SHA256SUMS`.

**Temuan yang harus diselesaikan sebelum data uji dibuka:**

| no | di mana | temuan | akibat bila dibiarkan |
|---|---|---|---|
| T1 | `buka_uji.py` | Bila path model salah, `_hash_berkas` diam-diam menghasilkan hash kosong, lalu pembukaan tetap dicatat di `buka_uji.log` sebelum model dimuat. Contoh perintah di docstring memakai `runs/...` dari folder gerbang_batch, padahal folder itu tidak ada di sana. | Satu-satunya kesempatan membuka data uji habis: proses gagal saat memuat model, pembukaan ulang ditolak, dan `--lanjut` juga ditolak karena hash berbeda. |
| T2 | `analisis_uji.py` | Butir 1 (celah ke Sequential di λ tanpa label dibanding λ berlabel, bagian 7) belum dihitung. `bandingkan_celah` yang ada belum memakai bobot sama per instans dan strata instans. | Kalimat butir 1 tidak punya angka uji. |
| T3 | `analisis_uji.py` | Jumlah pelanggaran keras dari pemeriksa jadwal (I-2) belum dilaporkan, walaupun datanya ada di kolom `periksa` tiap baris. | I-2 tidak tertulis di laporan. |
| T4 | `KERANGKA_NASKAH.md` I-1 | I-1 ditulis untuk resep lama: p95 waktu pemilih dari `measure_latency` dan dibandingkan dengan PPO-Pair, yang tidak ada di daftar metode bagian 3. `cek_episode` hanya mencatat rata-rata waktu per keputusan tiap episode (`ms`), bukan waktu tiap keputusan. | I-1 tidak bisa dihitung seperti tertulis. Perlu keputusan pemilik. |

**Gladi lengkap (syarat bagian 10).** `buka_uji.main` dijalankan pada data validasi:
MK01 dan Hurink edata la05, seed 5000-5001, ketiga kapasitas dan intensitas, keenam λ, dan
kedelapan metode (3.024 baris). Dari jumlah itu, 2.106 baris punya pasangan di berkas yang
sudah ada, dan selisih J-nya 0,0 di semua baris. Pasangannya diambil dari `e15/ep_*.json`
dan `e16/ep_v_*.json`, ditambah `episode_hasil.json` untuk tingkat 0 di λ 0,3, 3, dan 30.
Di MK01 kedelapan metode punya pasangan di keenam λ, kecuali tingkat 0 yang hanya punya
pasangan di tiga λ berlabel. Sisanya tidak punya pasangan karena memang belum pernah
dijalankan: tingkat 0 di λ tanpa label, serta boosting, PPO tanpa penjaga, dan BC + DAgger +
penjaga di la05. Pemeriksa jadwal hidup di semua 3.024 baris, dengan nol pelanggaran keras.
Gladi pertama sebelumnya hanya memakai seed 5000 dan λ 3.

**Status temuan (19 November 2026, cabang `fix/kunci-uji-temuan`).** Seam T1-T3 dan pilihan
I-1 disetujui pemilik pada 19 November 2026 ("sesuai saran anda").

| no | status |
|---|---|
| T1 | selesai: `buka_uji.main` memeriksa semua berkas model (ada dan sama dengan `SHA256SUMS` di folder induk model) sebelum mencatat; bila tidak, berhenti dengan kode 3. Docstring diperbaiki. |
| T2 | selesai: `bandingkan_celah_gabungan` (bobot sama per instans, strata instans) dan tabel butir 1 di laporan. Di 7 MK E15, uk. 1 sama dengan HASIL E15. |
| T3 | selesai: tabel pelanggaran keras per metode di laporan. |
| T4 | selesai bersama T5: `cek_episode.run` mencatat `ms_keputusan` (waktu total tiap keputusan) dan `ms_pemilih`; laporan memuat p95 per instans untuk semua metode. |
| T5 | selesai, diputuskan pemilik (lihat di bawah). Temuan awalnya: definisi lama I-1 memasang ambang 50 ms pada waktu pemilih saja, dengan waktu operator dan total dilaporkan tanpa ambang. `ms_keputusan` mengukur waktu total (pemilih, operator, dan penjaga). Untuk Sequential dan aturan tetap, pemilih dan operator tidak bisa dipisah, karena memilih berarti menjalankan operator. Di gladi validasi (8 proses paralel), p95 terbesar waktu total PPO + penjaga 55,37 ms, tingkat 3 tetap + penjaga 89,39 ms, Sequential 100,54 ms, dan boosting + penjaga 369,96 ms. |

**Gladi ketiga (kode baru).** Gladi lengkap yang sama diulang dengan kode sesudah T1-T4.
Ke-3.024 baris memberi J dan catatan tingkat yang sama persis dengan gladi kedua, dan
`ms_keputusan` lengkap di semua baris. Berarti pencatatan waktu tidak mengubah hasil
episode, dan pemeriksaan model T1 meloloskan model resmi. `analisis_uji.py` dengan
`SIFAT=<keluar>/sifat_uji.json` berjalan sampai semua tabel keluar: butir 1, I-1, I-2,
gabungan, kepekaan, per keluarga, dan tingkat.

**Kejadian 19 November 2026: berkas instans uji terbaca oleh tes.** Saat menulis tes merah
T1, tes memanggil `buka_uji.main` yang asli dengan path model palsu. Karena pemeriksaan T1
belum ada, proses itu mencatat pembukaan di log sementara (folder tmp pytest, bukan
`buka_uji.log` resmi), melepas penjaga, dan membaca berkas ke-38 instans uji untuk
menghitung `sifat_uji.json` (ukuran, fleksibilitas, beban, dan sifat lain sebelum episode).
Proses lalu macet karena model tidak ada, dan agen menghentikannya. Tidak ada episode yang
berjalan (`ep_aturan.jsonl` 0 baris), jadi tidak ada J, keterlambatan, atau hasil metode
apa pun dari data uji. Berkas sifat dihapus tanpa dibaca. Tes sekarang memasang
`_cegah_pembukaan`, sehingga kode yang belum benar gagal sebelum menyentuh berkas uji.
Pemilik memutuskan apakah kejadian ini perlu ditulis di naskah.

**T5 diputuskan (19 November 2026).** Pemilik memilih saran agen: I-1 tetap memakai
definisi lama. Ambang 50 ms berlaku untuk p95 waktu pemilih saja, dan waktu total
dilaporkan tanpa ambang (catatan 19 November di `KERANGKA_NASKAH.md` bagian 8).
`cek_b3.run_batch` sekarang mencatat waktu satu panggilan pemilih per sub-keputusan
(`ms_pemilih`), tanpa fitur dan operator, sama artinya dengan `selector_ms` di
`measure_latency`. `analisis_uji` memasang ambang pada waktu itu. Sequential dan tingkat 0
tidak punya pemilih terpisah, jadi hanya waktu totalnya yang dilaporkan.

**Gladi keempat (kode sesudah T5).** Gladi lengkap yang sama diulang. Ke-3.024 baris memberi
J dan catatan tingkat yang sama persis dengan gladi kedua, dan banyak `ms_pemilih` di setiap
baris sama dengan banyak tingkat yang diminta. Angka validasi I-1 (8 proses paralel):

| metode | p95 pemilih terbesar (ms) | di bawah 50 ms | p95 total terbesar (ms) |
|---|---|---|---|
| PPO + penjaga | 1,75 | ya | 55,93 |
| PPO tanpa penjaga | 1,73 | ya | 53,88 |
| BC + DAgger + penjaga | 1,71 | ya | 53,07 |
| boosting + penjaga | 61,02 | tidak | 385,57 |
| acak + penjaga | 0,11 | ya | 49,54 |
| tingkat 3 tetap + penjaga | 0,01 | ya | 84,97 |
| Sequential | - | - | 104,98 |
| tingkat 0 | - | - | 0,01 |

Semua temuan T1-T5 sudah selesai. Pemilik menyetujui dokumen ini pada 19 November 2026 dan
dokumen dipindah ke `docs/KUNCI_UJI.md`. Yang tersisa sebelum pembukaan: pemilik memberi izin
tertulis (bagian 12 langkah 8).
