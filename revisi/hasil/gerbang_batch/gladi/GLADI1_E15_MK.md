# Gladi 1: E15 di 7 MK validasi

Metode usulan: `ppo_bc_ppo_g`. Angka = selisih J usulan terhadap pembanding dalam persen (negatif = usulan lebih baik), tiap instans berbobot sama. uk. 1 rata-rata rasio per kondisi, uk. 2 rasio J total. * = selang tidak memuat nol.

## Keluarga uji utama (selang 98,75%)

| pembanding | λ | uk. 1 | uk. 2 | kondisi J=0 |
|---|---|---|---|---|
| acak_g | 0,3 | −14,58 [−21,11; −7,22]* | −18,66 [−22,38; −13,15]* | 0 |
| acak_g | 3 | −7,20 [−12,33; −2,36]* | −9,83 [−13,78; −4,83]* | 0 |
| fix3_g | 0,3 | +1,36 [−5,39; +8,56] | −3,65 [−8,58; +3,64] | 0 |
| fix3_g | 3 | −6,69 [−12,79; −0,61]* | −10,08 [−14,31; −3,89]* | 0 |

## Bukti utama: gabungan (selang 95%)

| pembanding | λ | uk. 1 | uk. 2 | kondisi J=0 |
|---|---|---|---|---|
| acak_g | 0,1 | −18,12 [−24,21; −11,88]* | −21,90 [−26,35; −16,16]* | 0 |
| acak_g | 0,3 | −14,58 [−19,88; −9,04]* | −18,66 [−21,52; −14,39]* | 0 |
| acak_g | 1 | −13,10 [−17,96; −8,27]* | −17,70 [−20,81; −12,44]* | 0 |
| acak_g | 3 | −7,20 [−11,16; −3,37]* | −9,83 [−12,77; −5,82]* | 0 |
| acak_g | 10 | +3,43 [−0,17; +7,22] | +1,43 [−1,95; +5,95] | 0 |
| acak_g | 30 | +1,77 [−0,94; +4,73] | +1,46 [−1,75; +4,72] | 0 |
| fix3_g | 0,1 | +1,40 [−4,69; +7,73] | −5,68 [−9,71; +2,43] | 0 |
| fix3_g | 0,3 | +1,36 [−4,10; +6,94] | −3,65 [−7,34; +2,08] | 0 |
| fix3_g | 1 | −4,52 [−8,57; −0,13]* | −7,29 [−10,06; −3,24]* | 0 |
| fix3_g | 3 | −6,69 [−11,40; −1,92]* | −10,08 [−13,14; −5,14]* | 0 |
| fix3_g | 10 | −1,86 [−5,92; +2,26] | −2,48 [−6,24; +2,05] | 0 |
| fix3_g | 30 | +2,89 [+0,33; +5,64]* | +4,58 [+0,85; +7,66]* | 0 |
| seq | 0,1 | +40,80 [+31,46; +50,31]* | +34,36 [+25,56; +43,15]* | 0 |
| seq | 0,3 | +42,05 [+31,66; +53,73]* | +33,29 [+25,12; +42,73]* | 0 |
| seq | 1 | +30,49 [+23,93; +37,62]* | +24,11 [+19,59; +30,28]* | 0 |
| seq | 3 | +21,25 [+16,65; +25,87]* | +16,96 [+13,85; +21,73]* | 0 |
| seq | 10 | +19,49 [+15,16; +23,83]* | +18,67 [+15,01; +22,41]* | 0 |
| seq | 30 | +8,78 [+5,34; +12,75]* | +10,36 [+6,29; +14,27]* | 0 |

## Kepekaan: benchmark saja (selang 95%)

Tidak ada baris.

## Per keluarga (titik, uk. 1 / uk. 2)

| keluarga | pembanding | λ | uk. 1 | uk. 2 |
|---|---|---|---|---|
| MK | acak_g | 0,1 | −18,12 | −21,90 |
| MK | acak_g | 0,3 | −14,58 | −18,66 |
| MK | acak_g | 1 | −13,10 | −17,70 |
| MK | acak_g | 3 | −7,20 | −9,83 |
| MK | acak_g | 10 | +3,43 | +1,43 |
| MK | acak_g | 30 | +1,77 | +1,46 |
| MK | fix3_g | 0,1 | +1,40 | −5,68 |
| MK | fix3_g | 0,3 | +1,36 | −3,65 |
| MK | fix3_g | 1 | −4,52 | −7,29 |
| MK | fix3_g | 3 | −6,69 | −10,08 |
| MK | fix3_g | 10 | −1,86 | −2,48 |
| MK | fix3_g | 30 | +2,89 | +4,58 |
| MK | seq | 0,1 | +40,80 | +34,36 |
| MK | seq | 0,3 | +42,05 | +33,29 |
| MK | seq | 1 | +30,49 | +24,11 |
| MK | seq | 3 | +21,25 | +16,96 |
| MK | seq | 10 | +19,49 | +18,67 |
| MK | seq | 30 | +8,78 | +10,36 |

## Tingkat yang dipilih usulan dan penolakan penjaga

Baris episode tidak punya kolom `tingkat`.


## Catatan gladi

Gladi 1 (18 November 2026) memakai berkas validasi E15 yang sudah ada (`e15/ep_e9c_s*.json`,
`e15/ep_aturan.json`), 7 MK, seed 5000-5001. Tujuannya memeriksa jalur analisis, bukan
menilai resep. Uk. 1 sama dengan `e15/HASIL_E15.md` karena di 7 MK setiap instans punya
jumlah kondisi yang sama; uk. 2 berbeda sedikit karena di sini tiap instans berbobot sama,
sedangkan HASIL_E15 memakai rasio J total gabungan. Kolom `tingkat` belum ada di berkas
E15. Tabel λ dirapikan sesudah dijalankan (3,0 menjadi 3).
