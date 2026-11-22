# E15: latih ulang dengan hitungan perubahan baru (16 November 2026)

Kriteria ditulis sebelum menjalankan: `docs/ARSITEKTUR_PIPELINE.md` 5.6, butir "E15". Resep,
seed, dan pengaturan sama dengan E1, E7, E9c, E10; yang berubah hanya hitungan perubahan
(PR #164). Data guru seed 7000-7001 dan R000-R049 seed 7000, DAgger 7002-7003, evaluasi di
seed validasi 5000-5001, 7 MK latih. Data uji tidak dibuka.

Skrip `jalan_e15.sh` selesai dengan exit 0 (13.48 sampai sekitar 16.00 WIB). Tabel lengkap:
`banding_e15.md` (kode `banding_e15.py`). Selang 95% dari bootstrap bergugus (instans,
intensitas, seed), 10.000 sampel ulang, dengan dua ukuran: uk. 1 = rata-rata rasio J per
kondisi, uk. 2 = rasio J total. PPO dirata-rata atas 3 seed latih per kondisi dulu.
Selisih negatif berarti PPO lebih baik; tanda * berarti selang tidak memuat nol.

## Butir 2: PPO E9c + penjaga lawan pembanding tanpa latih

| λ | lawan tingkat 3 + penjaga, uk. 1 | uk. 2 | lawan acak + penjaga, uk. 1 | uk. 2 |
|---|---|---|---|---|
| 0,1 | +1,40 [−5,46; +8,85] | −9,45* | −18,12* | −16,08* |
| 0,3 | +1,36 [−4,60; +7,38] | −11,38* | −14,58* | −17,66* |
| 1 | −4,52 [−8,97; +0,18] | −12,48* | −13,10* | −18,46* |
| 3 | −6,69* | −10,22* | −7,20* | −10,66* |
| 10 | −1,86 [−6,17; +2,67] | −6,09* | +3,43 [−0,93; +8,01] | −1,82 [−7,24; +3,52] |
| 30 | +2,89* | +4,30 [−1,21; +9,91] | +1,77 [−1,13; +4,84] | −0,45 [−6,24; +5,42] |

Sesuai aturan (ditulis per λ menurut tanda selang, kedua ukuran dilaporkan):

- Lawan acak + penjaga: PPO lebih baik nyata di λ 0,1 sampai 3 dengan kedua ukuran. Di λ 10
  dan 30 selisihnya tidak nyata.
- Lawan tingkat 3 + penjaga: dengan uk. 2, PPO lebih baik nyata di λ 0,1 sampai 10. Dengan
  uk. 1, hanya nyata di λ 3. Di λ 0,1 dan 0,3 tanda uk. 1 malah positif (tidak nyata). Di
  λ 30 tingkat 3 + penjaga lebih baik nyata dengan uk. 1.
- Jadi butir 2 di λ berlabel ≤ 3: penuh di λ 3; di λ 0,3 hanya dengan uk. 2 untuk lawan
  tingkat 3 + penjaga. Gambaran ini sama dengan model lama (E14), jadi latih ulang tidak
  mengubah kesimpulan butir 2.

Uk. 1 dan uk. 2 berbeda di λ kecil. Di λ 0,3, rasio J PPO terhadap tingkat 3 + penjaga per
kondisi menyebar dari 0,54 sampai 1,80 dengan median 1,00. Uk. 2 sama dengan rata-rata rasio
yang diberi bobot J tingkat 3 + penjaga, jadi tanda yang berbeda berarti keunggulan PPO
terkumpul di kondisi dengan J besar, sedangkan di kondisi dengan J kecil PPO sering kalah.

## Butir 1: selisih PPO E9c + penjaga ke Sequential, λ berlabel dan tanpa label

| λ | label guru | uk. 1 | uk. 2 |
|---|---|---|---|
| 0,1 | tidak | +40,80 [+30,95; +51,30] | +19,68 [+10,22; +35,54] |
| 0,3 | ya | +42,05 [+30,94; +54,86] | +20,89 [+14,90; +33,08] |
| 1 | tidak | +30,49 [+23,21; +38,85] | +17,54 [+10,00; +26,23] |
| 3 | ya | +21,25 [+16,09; +26,61] | +17,28 [+12,03; +22,53] |
| 10 | tidak | +19,49 [+14,70; +24,52] | +16,11 [+12,60; +20,40] |
| 30 | ya | +8,78 [+4,99; +13,23] | +12,09 [+6,95; +16,78] |

Selisih ke Sequential mengecil saat λ naik, baik di λ berlabel maupun tanpa label. Di λ 0,1
dan 1 selisihnya tidak lebih besar dari tetangga berlabelnya. Di λ 10 selisihnya sedikit di
atas titik tengah λ 3 dan 30 (uk. 1: 19,5 lawan sekitar 15,0; uk. 2: 16,1 lawan sekitar 14,7).

**Uji formal** (pasangan λ diputuskan pemilik sesudah angka per λ terlihat, ditulis di
`docs/ARSITEKTUR_PIPELINE.md` 5.6 sebelum selang dihitung; fungsi `bandingkan_celah`). Nilai =
celah di λ tanpa label dikurangi celah acuan, poin persen; positif = celah di λ tanpa label
lebih besar.

| sasaran lawan acuan | uk. 1 | uk. 2 | butir 1 |
|---|---|---|---|
| 0,1 lawan 0,3 | −1,25 [−10,72; +6,78] | −1,21 [−8,41; +6,47] | ya |
| 1 lawan 0,477 × λ 0,3 + 0,523 × λ 3 | −0,68 [−6,43; +4,74] | −1,46 [−5,98; +2,66] | ya |
| 10 lawan 0,477 × λ 3 + 0,523 × λ 30 | +4,76 [−0,18; +9,95] | +1,54 [−2,55; +6,31] | ya |
| pelengkap: rata-rata 3 λ tanpa label lawan 3 λ berlabel | +6,24 [+2,38; +10,06]* | +1,02 [−3,12; +6,34] | tidak menentukan |

Butir 1 lolos di ketiga λ tanpa label dengan kedua ukuran, jadi butir 1 ditulis untuk model
E15. Di λ 10 dengan uk. 1 lolosnya tipis: batas bawah selang −0,18, dan titik dugaannya
+4,76 poin. Naskah perlu menyebut ini, tidak cukup hanya "lolos".

Pelengkap uk. 1 positif nyata, tetapi angka ini tidak menentukan dan juga tidak adil sebagai
uji. Ketiga λ tanpa label masing-masing lebih kecil dari pasangan berlabelnya (0,1 < 0,3,
1 < 3, 10 < 30), dan celah ke Sequential membesar saat λ mengecil. Jadi rata-rata λ tanpa
label pasti condong ke celah yang lebih besar, walaupun model sama baiknya di semua λ.
Karena itulah pasangan utama memakai interpolasi.

Pembanding yang menguatkan butir 1: di λ 10 boosting + DAgger (kedua varian) jatuh jauh.
PPO + penjaga lebih baik 17,6% sampai 22,5% dari keduanya (nyata dengan kedua ukuran),
padahal di λ 0,1 sampai 3 boosting setara atau lebih baik dari PPO. Boosting juga lebih
buruk dari acak + penjaga di λ 10. Boosting memakai log λ sebagai fitur, dan model pohon
membagi fitur itu dengan ambang di antara nilai latih (0,3 / 3 / 30), jadi λ 10 diperlakukan
seperti λ 3 atau λ 30; jaringan PPO tidak punya batasan itu. Ini dugaan penyebab, belum diuji.

## Hasil lain

- **Penjaga:** PPO + penjaga lebih baik nyata dari PPO tanpa penjaga di λ 3, 10, 30 dengan
  kedua ukuran (−8% sampai −15%), dan di λ ≤ 1 hanya dengan uk. 2.
- **Tahap PPO di atas BC + DAgger:** tidak ada selisih nyata kecuali λ 30 uk. 2 (−1,42%,
  selang hampir menyentuh nol). Tahap PPO tetap tidak menambah banyak, sama dengan temuan
  sebelumnya.
- **DAgger:** PPO E1 (tanpa DAgger) lebih buruk dari E9c di ketiga λ berlabel dengan uk. 2
  (+6,1% sampai +6,9%) dan di λ 3, 30 dengan uk. 1.
- **Lawan boosting di λ berlabel:** boosting 7 MK + DAgger + penjaga lebih baik nyata dari
  PPO di λ 0,3 dan 3 dengan uk. 1 (+14,2% dan +7,1%), tidak nyata dengan uk. 2. Ini sama
  dengan E10: di λ kecil dan sedang pembelajar tidak menentukan.
- **Sequential** tetap terbaik di semua λ.

## Langkah berikut

- Butir 1 dan 2 untuk model E15 sudah bisa ditulis ke naskah, dengan batasan di atas (λ 10
  uk. 1 tipis; butir 2 lawan tingkat 3 + penjaga di λ 0,3 hanya dengan uk. 2).
- Dugaan penyebab jatuhnya boosting di λ 10 belum diuji.
