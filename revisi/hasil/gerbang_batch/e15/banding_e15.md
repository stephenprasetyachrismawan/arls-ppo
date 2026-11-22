# E15: hasil latih ulang dengan hitungan perubahan baru

Kode: `e15/banding_e15.py`. Episode: `e15/ep_*.json`. Seed validasi 5000-5001.

## Rasio J terhadap Sequential

Nilai = J metode / J Sequential. Uk. 1 = rata-rata rasio per kondisi; uk. 2 = rasio J total. 126 kondisi per λ (7 MK × 3 kapasitas × 3 intensitas × 2 seed).

### λ = 0,1

| metode | uk. 1 | uk. 2 |
|---|---|---|
| Sequential | 1,0000 | 1,0000 |
| boosting 7 MK + DAgger + penjaga | 1,3667 | 1,1415 |
| boosting 7 MK + DAgger | 1,3205 | 1,1736 |
| PPO E9c (BC + DAgger + PPO) + penjaga | 1,4080 | 1,1968 |
| BC + DAgger E9c + penjaga | 1,4236 | 1,2013 |
| boosting 7 MK + 50 acak + DAgger + penjaga | 1,4555 | 1,2049 |
| PPO E9c tanpa penjaga | 1,5113 | 1,2441 |
| boosting 7 MK + 50 acak + DAgger | 1,4991 | 1,2834 |
| tingkat 3 tetap + penjaga | 1,4951 | 1,3217 |
| acak + penjaga | 1,9463 | 1,4261 |

### λ = 0,3

| metode | uk. 1 | uk. 2 |
|---|---|---|
| Sequential | 1,0000 | 1,0000 |
| boosting 7 MK + 50 acak + penjaga | 1,2677 | 1,1853 |
| boosting 7 MK + DAgger | 1,2758 | 1,1955 |
| boosting 7 MK + DAgger + penjaga | 1,3022 | 1,1994 |
| PPO E9c (BC + DAgger + PPO) + penjaga | 1,4205 | 1,2089 |
| boosting 7 MK + 50 acak + DAgger + penjaga | 1,3746 | 1,2164 |
| BC + DAgger E9c + penjaga | 1,4208 | 1,2318 |
| PPO E9c tanpa penjaga | 1,4670 | 1,2552 |
| boosting 7 MK + penjaga | 1,4729 | 1,2562 |
| boosting 7 MK + 50 acak | 1,3104 | 1,2596 |
| BC E1 + penjaga | 1,4720 | 1,2848 |
| PPO E1 (BC + PPO) + penjaga | 1,4368 | 1,2918 |
| boosting 7 MK + 50 acak + DAgger | 1,4063 | 1,3027 |
| PPO E1 tanpa penjaga | 1,5668 | 1,3276 |
| tingkat 3 tetap + penjaga | 1,4540 | 1,3641 |
| boosting 7 MK | 1,5871 | 1,3796 |
| acak + penjaga | 1,8099 | 1,4682 |

### λ = 1

| metode | uk. 1 | uk. 2 |
|---|---|---|
| Sequential | 1,0000 | 1,0000 |
| BC + DAgger E9c + penjaga | 1,2857 | 1,1681 |
| PPO E9c (BC + DAgger + PPO) + penjaga | 1,3049 | 1,1754 |
| boosting 7 MK + DAgger + penjaga | 1,3083 | 1,1850 |
| boosting 7 MK + DAgger | 1,3474 | 1,2216 |
| PPO E9c tanpa penjaga | 1,4074 | 1,2261 |
| boosting 7 MK + 50 acak + DAgger + penjaga | 1,3608 | 1,2466 |
| boosting 7 MK + 50 acak + DAgger | 1,3874 | 1,2860 |
| tingkat 3 tetap + penjaga | 1,4051 | 1,3431 |
| acak + penjaga | 1,5938 | 1,4415 |

### λ = 3

| metode | uk. 1 | uk. 2 |
|---|---|---|
| Sequential | 1,0000 | 1,0000 |
| boosting 7 MK + DAgger + penjaga | 1,1459 | 1,1386 |
| PPO E9c (BC + DAgger + PPO) + penjaga | 1,2125 | 1,1728 |
| boosting 7 MK + 50 acak + penjaga | 1,1958 | 1,1799 |
| boosting 7 MK + 50 acak + DAgger + penjaga | 1,1952 | 1,1813 |
| BC + DAgger E9c + penjaga | 1,2053 | 1,1924 |
| boosting 7 MK + DAgger | 1,2292 | 1,2104 |
| boosting 7 MK + penjaga | 1,2152 | 1,2294 |
| PPO E1 (BC + PPO) + penjaga | 1,2668 | 1,2443 |
| BC E1 + penjaga | 1,2582 | 1,2476 |
| boosting 7 MK + 50 acak + DAgger | 1,2635 | 1,2650 |
| PPO E9c tanpa penjaga | 1,3621 | 1,2776 |
| boosting 7 MK + 50 acak | 1,3184 | 1,2859 |
| tingkat 3 tetap + penjaga | 1,3601 | 1,3064 |
| acak + penjaga | 1,3469 | 1,3127 |
| boosting 7 MK | 1,4276 | 1,3499 |
| PPO E1 tanpa penjaga | 1,4519 | 1,3595 |

### λ = 10

| metode | uk. 1 | uk. 2 |
|---|---|---|
| Sequential | 1,0000 | 1,0000 |
| PPO E9c (BC + DAgger + PPO) + penjaga | 1,1949 | 1,1611 |
| BC + DAgger E9c + penjaga | 1,2162 | 1,1754 |
| acak + penjaga | 1,1768 | 1,1826 |
| tingkat 3 tetap + penjaga | 1,2471 | 1,2364 |
| PPO E9c tanpa penjaga | 1,3093 | 1,3053 |
| boosting 7 MK + 50 acak + DAgger + penjaga | 1,5103 | 1,4845 |
| boosting 7 MK + DAgger + penjaga | 1,4997 | 1,4983 |
| boosting 7 MK + 50 acak + DAgger | 1,5467 | 1,5471 |
| boosting 7 MK + DAgger | 1,5256 | 1,5524 |

### λ = 30

| metode | uk. 1 | uk. 2 |
|---|---|---|
| Sequential | 1,0000 | 1,0000 |
| tingkat 3 tetap + penjaga | 1,0619 | 1,0747 |
| PPO E9c (BC + DAgger + PPO) + penjaga | 1,0878 | 1,1209 |
| acak + penjaga | 1,0727 | 1,1260 |
| BC + DAgger E9c + penjaga | 1,0960 | 1,1370 |
| boosting 7 MK + DAgger + penjaga | 1,0989 | 1,1627 |
| boosting 7 MK + 50 acak + DAgger + penjaga | 1,1046 | 1,1641 |
| boosting 7 MK + penjaga | 1,1146 | 1,1906 |
| PPO E1 (BC + PPO) + penjaga | 1,1151 | 1,1924 |
| BC E1 + penjaga | 1,1148 | 1,1956 |
| boosting 7 MK + 50 acak + penjaga | 1,1240 | 1,1992 |
| boosting 7 MK + DAgger | 1,1595 | 1,2738 |
| boosting 7 MK + 50 acak + DAgger | 1,1711 | 1,2885 |
| PPO E9c tanpa penjaga | 1,2100 | 1,3260 |
| boosting 7 MK | 1,1871 | 1,3325 |
| boosting 7 MK + 50 acak | 1,1956 | 1,3419 |
| PPO E1 tanpa penjaga | 1,2988 | 1,4662 |

## Butir 2: PPO E9c + penjaga lawan pembanding tanpa latih

Selisih J metode A terhadap B dalam persen; negatif = A lebih baik. Selang 95% bootstrap bergugus; * = selang tidak memuat nol.

| A | B | λ | uk. 1 (rata-rata rasio) | uk. 2 (rasio total) |
|---|---|---|---|---|
| PPO E9c (BC + DAgger + PPO) + penjaga | tingkat 3 tetap + penjaga | 0,1 | +1,40 [-5,46; +8,85] | -9,45 [-15,29; -2,26]* |
| PPO E9c (BC + DAgger + PPO) + penjaga | tingkat 3 tetap + penjaga | 0,3 | +1,36 [-4,60; +7,38] | -11,38 [-17,28; -1,51]* |
| PPO E9c (BC + DAgger + PPO) + penjaga | tingkat 3 tetap + penjaga | 1 | -4,52 [-8,97; +0,18] | -12,48 [-16,36; -5,53]* |
| PPO E9c (BC + DAgger + PPO) + penjaga | tingkat 3 tetap + penjaga | 3 | -6,69 [-11,96; -1,39]* | -10,22 [-13,41; -6,11]* |
| PPO E9c (BC + DAgger + PPO) + penjaga | tingkat 3 tetap + penjaga | 10 | -1,86 [-6,17; +2,67] | -6,09 [-11,12; -0,52]* |
| PPO E9c (BC + DAgger + PPO) + penjaga | tingkat 3 tetap + penjaga | 30 | +2,89 [+0,22; +5,77]* | +4,30 [-1,21; +9,91] |
| PPO E9c (BC + DAgger + PPO) + penjaga | acak + penjaga | 0,1 | -18,12 [-24,92; -11,13]* | -16,08 [-22,42; -11,55]* |
| PPO E9c (BC + DAgger + PPO) + penjaga | acak + penjaga | 0,3 | -14,58 [-20,29; -8,49]* | -17,66 [-22,63; -11,88]* |
| PPO E9c (BC + DAgger + PPO) + penjaga | acak + penjaga | 1 | -13,10 [-18,33; -7,77]* | -18,46 [-22,55; -13,78]* |
| PPO E9c (BC + DAgger + PPO) + penjaga | acak + penjaga | 3 | -7,20 [-11,52; -3,00]* | -10,66 [-14,64; -6,71]* |
| PPO E9c (BC + DAgger + PPO) + penjaga | acak + penjaga | 10 | +3,43 [-0,93; +8,01] | -1,82 [-7,24; +3,52] |
| PPO E9c (BC + DAgger + PPO) + penjaga | acak + penjaga | 30 | +1,77 [-1,13; +4,84] | -0,45 [-6,24; +5,42] |

## Selisih PPO E9c + penjaga ke Sequential per λ (bahan butir 1)

Selisih J metode A terhadap B dalam persen; negatif = A lebih baik. Selang 95% bootstrap bergugus; * = selang tidak memuat nol.

| A | B | λ | uk. 1 (rata-rata rasio) | uk. 2 (rasio total) |
|---|---|---|---|---|
| PPO E9c (BC + DAgger + PPO) + penjaga | Sequential | 0,1 | +40,80 [+30,95; +51,30]* | +19,68 [+10,22; +35,54]* |
| PPO E9c (BC + DAgger + PPO) + penjaga | Sequential | 0,3 | +42,05 [+30,94; +54,86]* | +20,89 [+14,90; +33,08]* |
| PPO E9c (BC + DAgger + PPO) + penjaga | Sequential | 1 | +30,49 [+23,21; +38,85]* | +17,54 [+10,00; +26,23]* |
| PPO E9c (BC + DAgger + PPO) + penjaga | Sequential | 3 | +21,25 [+16,09; +26,61]* | +17,28 [+12,03; +22,53]* |
| PPO E9c (BC + DAgger + PPO) + penjaga | Sequential | 10 | +19,49 [+14,70; +24,52]* | +16,11 [+12,60; +20,40]* |
| PPO E9c (BC + DAgger + PPO) + penjaga | Sequential | 30 | +8,78 [+4,99; +13,23]* | +12,09 [+6,95; +16,78]* |

## Butir 1: celah PPO E9c + penjaga ke Sequential, λ tanpa label dikurangi acuan

Celah = selisih J PPO + penjaga ke Sequential (persen). Nilai = celah sasaran dikurangi celah acuan, dalam poin persen; positif = celah di λ tanpa label lebih besar. Selang 95% bootstrap bergugus, gugus diambil ulang bersama untuk semua λ. Butir 1 ditulis untuk λ yang selangnya memuat nol atau negatif dengan kedua ukuran.

| sasaran lawan acuan | bobot acuan | uk. 1 | uk. 2 | butir 1 |
|---|---|---|---|---|
| 0,1 lawan 0,3 | 1,000 × λ 0,3 | -1,25 [-10,72; +6,78] | -1,21 [-8,41; +6,47] | ya |
| 1 lawan interpolasi log 0,3 dan 3 | 0,477 × λ 0,3 + 0,523 × λ 3 | -0,68 [-6,43; +4,74] | -1,46 [-5,98; +2,66] | ya |
| 10 lawan interpolasi log 3 dan 30 | 0,477 × λ 3 + 0,523 × λ 30 | +4,76 [-0,18; +9,95] | +1,54 [-2,55; +6,31] | ya |
| pelengkap: rata-rata tanpa label lawan rata-rata berlabel | 0,333 × λ 0,3 + 0,333 × λ 3 + 0,333 × λ 30 | +6,24 [+2,38; +10,06]* | +1,02 [-3,12; +6,34] | tidak (tidak menentukan) |

## PPO E9c + penjaga lawan pemilih terlatih lain (λ berlabel dan tanpa label)

Selisih J metode A terhadap B dalam persen; negatif = A lebih baik. Selang 95% bootstrap bergugus; * = selang tidak memuat nol.

| A | B | λ | uk. 1 (rata-rata rasio) | uk. 2 (rasio total) |
|---|---|---|---|---|
| PPO E9c (BC + DAgger + PPO) + penjaga | boosting 7 MK + DAgger + penjaga | 0,1 | +9,99 [+2,83; +17,87]* | +4,85 [+0,54; +8,30]* |
| PPO E9c (BC + DAgger + PPO) + penjaga | boosting 7 MK + DAgger + penjaga | 0,3 | +14,18 [+5,99; +24,04]* | +0,79 [-3,29; +6,22] |
| PPO E9c (BC + DAgger + PPO) + penjaga | boosting 7 MK + DAgger + penjaga | 1 | +3,42 [-3,59; +11,54] | -0,81 [-8,09; +6,77] |
| PPO E9c (BC + DAgger + PPO) + penjaga | boosting 7 MK + DAgger + penjaga | 3 | +7,11 [+2,45; +12,03]* | +3,00 [-0,41; +6,63] |
| PPO E9c (BC + DAgger + PPO) + penjaga | boosting 7 MK + DAgger + penjaga | 10 | -17,64 [-21,40; -14,18]* | -22,51 [-26,04; -18,70]* |
| PPO E9c (BC + DAgger + PPO) + penjaga | boosting 7 MK + DAgger + penjaga | 30 | -0,56 [-2,78; +1,94] | -3,59 [-8,17; +0,93] |
| PPO E9c (BC + DAgger + PPO) + penjaga | boosting 7 MK + 50 acak + DAgger + penjaga | 0,1 | +3,40 [-3,24; +10,21] | -0,67 [-4,82; +3,24] |
| PPO E9c (BC + DAgger + PPO) + penjaga | boosting 7 MK + 50 acak + DAgger + penjaga | 0,3 | +9,24 [+0,97; +19,33]* | -0,62 [-5,20; +4,32] |
| PPO E9c (BC + DAgger + PPO) + penjaga | boosting 7 MK + 50 acak + DAgger + penjaga | 1 | +0,41 [-6,52; +7,71] | -5,71 [-11,30; -0,98]* |
| PPO E9c (BC + DAgger + PPO) + penjaga | boosting 7 MK + 50 acak + DAgger + penjaga | 3 | +3,40 [-1,73; +8,82] | -0,72 [-4,10; +2,46] |
| PPO E9c (BC + DAgger + PPO) + penjaga | boosting 7 MK + 50 acak + DAgger + penjaga | 10 | -17,73 [-21,50; -14,16]* | -21,79 [-27,00; -17,07]* |
| PPO E9c (BC + DAgger + PPO) + penjaga | boosting 7 MK + 50 acak + DAgger + penjaga | 30 | -1,11 [-2,76; +0,41] | -3,71 [-7,70; +0,13] |

## Peran penjaga dan tahap PPO (E9c)

Selisih J metode A terhadap B dalam persen; negatif = A lebih baik. Selang 95% bootstrap bergugus; * = selang tidak memuat nol.

| A | B | λ | uk. 1 (rata-rata rasio) | uk. 2 (rasio total) |
|---|---|---|---|---|
| PPO E9c (BC + DAgger + PPO) + penjaga | PPO E9c tanpa penjaga | 0,1 | -1,27 [-5,20; +3,15] | -3,80 [-6,46; -1,14]* |
| PPO E9c (BC + DAgger + PPO) + penjaga | PPO E9c tanpa penjaga | 0,3 | -0,06 [-3,25; +3,25] | -3,69 [-6,18; -0,28]* |
| PPO E9c (BC + DAgger + PPO) + penjaga | PPO E9c tanpa penjaga | 1 | -2,61 [-6,54; +0,60] | -4,13 [-7,20; -0,72]* |
| PPO E9c (BC + DAgger + PPO) + penjaga | PPO E9c tanpa penjaga | 3 | -8,54 [-11,78; -5,81]* | -8,20 [-11,00; -5,28]* |
| PPO E9c (BC + DAgger + PPO) + penjaga | PPO E9c tanpa penjaga | 10 | -7,90 [-10,05; -5,83]* | -11,05 [-14,59; -7,31]* |
| PPO E9c (BC + DAgger + PPO) + penjaga | PPO E9c tanpa penjaga | 30 | -8,68 [-11,36; -6,06]* | -15,47 [-18,82; -12,13]* |
| PPO E9c (BC + DAgger + PPO) + penjaga | BC + DAgger E9c + penjaga | 0,1 | +1,14 [-2,03; +4,23] | -0,37 [-2,33; +2,39] |
| PPO E9c (BC + DAgger + PPO) + penjaga | BC + DAgger E9c + penjaga | 0,3 | +1,76 [-1,92; +5,88] | -1,86 [-6,33; +4,22] |
| PPO E9c (BC + DAgger + PPO) + penjaga | BC + DAgger E9c + penjaga | 1 | +1,77 [-0,78; +4,34] | +0,62 [-2,15; +3,55] |
| PPO E9c (BC + DAgger + PPO) + penjaga | BC + DAgger E9c + penjaga | 3 | +0,86 [-1,63; +3,31] | -1,64 [-3,37; +0,82] |
| PPO E9c (BC + DAgger + PPO) + penjaga | BC + DAgger E9c + penjaga | 10 | -1,49 [-3,33; +0,24] | -1,22 [-2,58; +0,22] |
| PPO E9c (BC + DAgger + PPO) + penjaga | BC + DAgger E9c + penjaga | 30 | -0,68 [-1,52; +0,08] | -1,42 [-2,71; -0,02]* |

## E1 lawan E9c (manfaat DAgger), λ berlabel

Selisih J metode A terhadap B dalam persen; negatif = A lebih baik. Selang 95% bootstrap bergugus; * = selang tidak memuat nol.

| A | B | λ | uk. 1 (rata-rata rasio) | uk. 2 (rasio total) |
|---|---|---|---|---|
| PPO E1 (BC + PPO) + penjaga | PPO E9c (BC + DAgger + PPO) + penjaga | 0,3 | +2,93 [-1,93; +7,69] | +6,86 [+1,27; +11,68]* |
| PPO E1 (BC + PPO) + penjaga | PPO E9c (BC + DAgger + PPO) + penjaga | 3 | +5,42 [+2,09; +8,87]* | +6,10 [+3,13; +9,01]* |
| PPO E1 (BC + PPO) + penjaga | PPO E9c (BC + DAgger + PPO) + penjaga | 30 | +2,68 [+0,21; +5,55]* | +6,37 [+1,03; +11,96]* |

