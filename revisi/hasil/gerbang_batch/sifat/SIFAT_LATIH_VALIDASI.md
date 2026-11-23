# Sifat instans latih dan validasi

Dihitung dengan `sifat_instans.py` sesudah transformasi domain, sebelum episode. Data uji tidak dimuat. Arti tiap kolom ada di kepala `sifat_instans.py`.

## Rentang per kelompok

| sifat | latih MK | latih benchmark | validasi |
|---|---|---|---|
| n_job | 10 sampai 20 | 10 sampai 20 | 10 sampai 20 |
| n_tugas | 55 sampai 240 | 50 sampai 293 | 50 sampai 293 |
| n_dev | 4 sampai 15 | 5 sampai 15 | 5 sampai 15 |
| tugas_per_dev | 9,17 sampai 26,50 | 10,00 sampai 39,20 | 10,00 sampai 39,20 |
| fleks_rata | 1,43 sampai 4,10 | 1,13 sampai 6,63 | 1,13 sampai 6,53 |
| fleks_sd | 0,45 sampai 1,85 | 0,34 sampai 1,76 | 0,34 sampai 1,90 |
| upaya_rata | 3,82 sampai 12,04 | 7,53 sampai 7,53 | 7,53 sampai 7,53 |
| skala_upaya | 1,00 sampai 1,00 | 0,13 sampai 0,17 | 0,13 sampai 0,17 |
| beban_per_dev | 35,19 sampai 279,35 | 70,51 sampai 294,70 | 68,59 sampai 288,72 |
| ketimpangan_keahlian | 0,09 sampai 0,75 | 0,04 sampai 0,15 | 0,04 sampai 0,16 |
| keketatan | 0,67 sampai 2,67 | 0,67 sampai 2,67 | 0,67 sampai 2,67 |
| dep_rata | 0,91 sampai 1,00 | 0,90 sampai 0,99 | 0,90 sampai 0,99 |

## Instans validasi di luar rentang latih (7 MK + 60 benchmark)

- hurink/vdata/la39: fleks_sd di atas (1,90)

## Per instans

| kelompok | instans | n_job | n_tugas | n_dev | tugas_per_dev | fleks_rata | fleks_sd | upaya_rata | skala_upaya | beban_per_dev | ketimpangan_keahlian | keketatan | dep_rata |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| latih MK | MK01 | 10 | 55 | 6 | 9,17 | 2,09 | 0,82 | 4,04 | 1,00 | 35,19 | 0,44 | 1,11 | 0,91 |
| latih MK | MK02 | 10 | 58 | 6 | 9,67 | 4,10 | 1,85 | 3,82 | 1,00 | 37,27 | 0,10 | 1,11 | 0,91 |
| latih MK | MK05 | 15 | 106 | 4 | 26,50 | 1,71 | 0,45 | 6,73 | 1,00 | 180,12 | 0,18 | 2,50 | 0,95 |
| latih MK | MK06 | 10 | 150 | 10 | 15,00 | 3,27 | 1,18 | 4,98 | 1,00 | 73,07 | 0,38 | 0,67 | 0,97 |
| latih MK | MK07 | 20 | 100 | 5 | 20,00 | 2,83 | 1,35 | 10,04 | 1,00 | 204,17 | 0,09 | 2,67 | 0,99 |
| latih MK | MK08 | 20 | 225 | 10 | 22,50 | 1,43 | 0,50 | 12,04 | 1,00 | 279,35 | 0,52 | 1,33 | 1,00 |
| latih MK | MK10 | 20 | 240 | 15 | 16,00 | 2,98 | 1,14 | 11,03 | 1,00 | 170,87 | 0,75 | 0,89 | 1,00 |
| latih benchmark | hurink/edata/la02 | 10 | 50 | 5 | 10,00 | 1,20 | 0,40 | 7,53 | 0,15 | 76,84 | 0,14 | 1,33 | 0,90 |
| latih benchmark | hurink/rdata/la02 | 10 | 50 | 5 | 10,00 | 1,88 | 0,71 | 7,53 | 0,14 | 76,03 | 0,10 | 1,33 | 0,90 |
| latih benchmark | hurink/vdata/la02 | 10 | 50 | 5 | 10,00 | 2,68 | 0,97 | 7,53 | 0,15 | 77,00 | 0,06 | 1,33 | 0,90 |
| latih benchmark | hurink/edata/la03 | 10 | 50 | 5 | 10,00 | 1,18 | 0,38 | 7,53 | 0,16 | 77,00 | 0,06 | 1,33 | 0,90 |
| latih benchmark | hurink/rdata/la03 | 10 | 50 | 5 | 10,00 | 1,98 | 0,73 | 7,53 | 0,15 | 73,13 | 0,10 | 1,33 | 0,90 |
| latih benchmark | hurink/vdata/la03 | 10 | 50 | 5 | 10,00 | 2,56 | 0,96 | 7,53 | 0,16 | 75,06 | 0,12 | 1,33 | 0,90 |
| latih benchmark | hurink/edata/la04 | 10 | 50 | 5 | 10,00 | 1,18 | 0,38 | 7,53 | 0,15 | 75,67 | 0,08 | 1,33 | 0,90 |
| latih benchmark | hurink/rdata/la04 | 10 | 50 | 5 | 10,00 | 2,02 | 0,76 | 7,53 | 0,14 | 70,51 | 0,06 | 1,33 | 0,90 |
| latih benchmark | hurink/vdata/la04 | 10 | 50 | 5 | 10,00 | 2,38 | 0,85 | 7,53 | 0,16 | 77,95 | 0,10 | 1,33 | 0,90 |
| latih benchmark | hurink/edata/la07 | 15 | 75 | 5 | 15,00 | 1,15 | 0,35 | 7,53 | 0,15 | 110,49 | 0,09 | 2,00 | 0,93 |
| latih benchmark | hurink/rdata/la07 | 15 | 75 | 5 | 15,00 | 1,96 | 0,77 | 7,53 | 0,15 | 114,44 | 0,04 | 2,00 | 0,93 |
| latih benchmark | hurink/vdata/la07 | 15 | 75 | 5 | 15,00 | 2,43 | 1,04 | 7,53 | 0,15 | 112,66 | 0,10 | 2,00 | 0,93 |
| latih benchmark | hurink/edata/la08 | 15 | 75 | 5 | 15,00 | 1,15 | 0,35 | 7,53 | 0,15 | 111,65 | 0,10 | 2,00 | 0,93 |
| latih benchmark | hurink/rdata/la08 | 15 | 75 | 5 | 15,00 | 1,93 | 0,79 | 7,53 | 0,15 | 112,04 | 0,08 | 2,00 | 0,93 |
| latih benchmark | hurink/vdata/la08 | 15 | 75 | 5 | 15,00 | 2,59 | 0,99 | 7,53 | 0,14 | 110,87 | 0,04 | 2,00 | 0,93 |
| latih benchmark | hurink/edata/la09 | 15 | 75 | 5 | 15,00 | 1,15 | 0,35 | 7,53 | 0,13 | 112,61 | 0,09 | 2,00 | 0,93 |
| latih benchmark | hurink/rdata/la09 | 15 | 75 | 5 | 15,00 | 1,92 | 0,78 | 7,53 | 0,13 | 113,85 | 0,05 | 2,00 | 0,93 |
| latih benchmark | hurink/vdata/la09 | 15 | 75 | 5 | 15,00 | 2,53 | 1,00 | 7,53 | 0,13 | 114,96 | 0,05 | 2,00 | 0,93 |
| latih benchmark | hurink/edata/la12 | 20 | 100 | 5 | 20,00 | 1,14 | 0,35 | 7,53 | 0,16 | 146,67 | 0,09 | 2,67 | 0,99 |
| latih benchmark | hurink/rdata/la12 | 20 | 100 | 5 | 20,00 | 1,99 | 0,81 | 7,53 | 0,17 | 161,73 | 0,07 | 2,67 | 0,99 |
| latih benchmark | hurink/vdata/la12 | 20 | 100 | 5 | 20,00 | 2,48 | 1,04 | 7,53 | 0,16 | 150,31 | 0,06 | 2,67 | 0,99 |
| latih benchmark | hurink/edata/la13 | 20 | 100 | 5 | 20,00 | 1,14 | 0,35 | 7,53 | 0,15 | 150,89 | 0,10 | 2,67 | 0,99 |
| latih benchmark | hurink/rdata/la13 | 20 | 100 | 5 | 20,00 | 1,98 | 0,81 | 7,53 | 0,15 | 154,97 | 0,07 | 2,67 | 0,99 |
| latih benchmark | hurink/vdata/la13 | 20 | 100 | 5 | 20,00 | 2,45 | 1,02 | 7,53 | 0,14 | 149,25 | 0,05 | 2,67 | 0,99 |
| latih benchmark | hurink/edata/la14 | 20 | 100 | 5 | 20,00 | 1,13 | 0,34 | 7,53 | 0,14 | 150,03 | 0,06 | 2,67 | 0,99 |
| latih benchmark | hurink/rdata/la14 | 20 | 100 | 5 | 20,00 | 1,97 | 0,82 | 7,53 | 0,14 | 152,00 | 0,05 | 2,67 | 0,99 |
| latih benchmark | hurink/vdata/la14 | 20 | 100 | 5 | 20,00 | 2,44 | 1,00 | 7,53 | 0,14 | 145,48 | 0,05 | 2,67 | 0,99 |
| latih benchmark | hurink/edata/la17 | 10 | 100 | 10 | 10,00 | 1,13 | 0,34 | 7,53 | 0,16 | 73,89 | 0,08 | 0,67 | 0,95 |
| latih benchmark | hurink/rdata/la17 | 10 | 100 | 10 | 10,00 | 1,93 | 0,78 | 7,53 | 0,16 | 74,60 | 0,14 | 0,67 | 0,95 |
| latih benchmark | hurink/vdata/la17 | 10 | 100 | 10 | 10,00 | 4,79 | 1,48 | 7,53 | 0,16 | 74,68 | 0,07 | 0,67 | 0,95 |
| latih benchmark | hurink/edata/la18 | 10 | 100 | 10 | 10,00 | 1,13 | 0,34 | 7,53 | 0,15 | 78,97 | 0,10 | 0,67 | 0,95 |
| latih benchmark | hurink/rdata/la18 | 10 | 100 | 10 | 10,00 | 1,99 | 0,78 | 7,53 | 0,15 | 75,94 | 0,15 | 0,67 | 0,95 |
| latih benchmark | hurink/vdata/la18 | 10 | 100 | 10 | 10,00 | 4,79 | 1,56 | 7,53 | 0,14 | 73,77 | 0,07 | 0,67 | 0,95 |
| latih benchmark | hurink/edata/la19 | 10 | 100 | 10 | 10,00 | 1,13 | 0,34 | 7,53 | 0,14 | 74,60 | 0,08 | 0,67 | 0,95 |
| latih benchmark | hurink/rdata/la19 | 10 | 100 | 10 | 10,00 | 1,96 | 0,80 | 7,53 | 0,14 | 76,54 | 0,09 | 0,67 | 0,95 |
| latih benchmark | hurink/vdata/la19 | 10 | 100 | 10 | 10,00 | 4,80 | 1,43 | 7,53 | 0,14 | 73,84 | 0,11 | 0,67 | 0,95 |
| latih benchmark | hurink/edata/la22 | 15 | 150 | 10 | 15,00 | 1,15 | 0,38 | 7,53 | 0,16 | 115,67 | 0,07 | 1,00 | 0,97 |
| latih benchmark | hurink/rdata/la22 | 15 | 150 | 10 | 15,00 | 2,04 | 0,81 | 7,53 | 0,16 | 114,76 | 0,12 | 1,00 | 0,97 |
| latih benchmark | hurink/vdata/la22 | 15 | 150 | 10 | 15,00 | 4,51 | 1,39 | 7,53 | 0,15 | 110,28 | 0,04 | 1,00 | 0,97 |
| latih benchmark | hurink/edata/la23 | 15 | 150 | 10 | 15,00 | 1,14 | 0,35 | 7,53 | 0,14 | 115,09 | 0,06 | 1,00 | 0,97 |
| latih benchmark | hurink/rdata/la23 | 15 | 150 | 10 | 15,00 | 2,04 | 0,80 | 7,53 | 0,14 | 110,76 | 0,11 | 1,00 | 0,97 |
| latih benchmark | hurink/vdata/la23 | 15 | 150 | 10 | 15,00 | 4,53 | 1,43 | 7,53 | 0,14 | 112,69 | 0,05 | 1,00 | 0,97 |
| latih benchmark | hurink/edata/la27 | 20 | 200 | 10 | 20,00 | 1,14 | 0,36 | 7,53 | 0,14 | 150,74 | 0,04 | 1,33 | 0,99 |
| latih benchmark | hurink/rdata/la27 | 20 | 200 | 10 | 20,00 | 1,96 | 0,80 | 7,53 | 0,14 | 150,78 | 0,09 | 1,33 | 0,99 |
| latih benchmark | hurink/vdata/la27 | 20 | 200 | 10 | 20,00 | 4,58 | 1,47 | 7,53 | 0,14 | 149,96 | 0,09 | 1,33 | 0,99 |
| latih benchmark | hurink/edata/la28 | 20 | 200 | 10 | 20,00 | 1,13 | 0,35 | 7,53 | 0,14 | 150,76 | 0,06 | 1,33 | 0,99 |
| latih benchmark | hurink/rdata/la28 | 20 | 200 | 10 | 20,00 | 2,01 | 0,81 | 7,53 | 0,14 | 148,40 | 0,09 | 1,33 | 0,99 |
| latih benchmark | hurink/vdata/la28 | 20 | 200 | 10 | 20,00 | 4,49 | 1,58 | 7,53 | 0,14 | 147,43 | 0,06 | 1,33 | 0,99 |
| latih benchmark | hurink/edata/la29 | 20 | 200 | 10 | 20,00 | 1,14 | 0,36 | 7,53 | 0,15 | 149,91 | 0,06 | 1,33 | 0,99 |
| latih benchmark | hurink/rdata/la29 | 20 | 200 | 10 | 20,00 | 2,00 | 0,80 | 7,53 | 0,15 | 151,76 | 0,10 | 1,33 | 0,99 |
| latih benchmark | hurink/vdata/la29 | 20 | 200 | 10 | 20,00 | 4,41 | 1,56 | 7,53 | 0,15 | 148,91 | 0,05 | 1,33 | 0,99 |
| latih benchmark | hurink/edata/la37 | 15 | 225 | 15 | 15,00 | 1,15 | 0,38 | 7,53 | 0,14 | 113,67 | 0,10 | 0,67 | 0,98 |
| latih benchmark | hurink/rdata/la37 | 15 | 225 | 15 | 15,00 | 1,94 | 0,80 | 7,53 | 0,14 | 114,17 | 0,14 | 0,67 | 0,98 |
| latih benchmark | hurink/vdata/la37 | 15 | 225 | 15 | 15,00 | 6,63 | 1,76 | 7,53 | 0,14 | 113,78 | 0,08 | 0,67 | 0,98 |
| latih benchmark | hurink/edata/la38 | 15 | 225 | 15 | 15,00 | 1,14 | 0,37 | 7,53 | 0,15 | 112,01 | 0,10 | 0,67 | 0,98 |
| latih benchmark | hurink/rdata/la38 | 15 | 225 | 15 | 15,00 | 1,97 | 0,80 | 7,53 | 0,15 | 112,96 | 0,14 | 0,67 | 0,98 |
| latih benchmark | hurink/vdata/la38 | 15 | 225 | 15 | 15,00 | 6,57 | 1,75 | 7,53 | 0,15 | 110,02 | 0,07 | 0,67 | 0,98 |
| latih benchmark | dauzere/05a | 10 | 196 | 5 | 39,20 | 1,69 | 0,77 | 7,53 | 0,13 | 294,70 | 0,10 | 1,33 | 0,97 |
| latih benchmark | dauzere/10a | 15 | 293 | 8 | 36,62 | 1,24 | 0,58 | 7,53 | 0,13 | 277,55 | 0,15 | 1,25 | 0,98 |
| latih benchmark | dauzere/11a | 15 | 293 | 8 | 36,62 | 2,42 | 1,21 | 7,53 | 0,13 | 273,82 | 0,07 | 1,25 | 0,98 |
| validasi | hurink/edata/la05 | 10 | 50 | 5 | 10,00 | 1,20 | 0,40 | 7,53 | 0,16 | 73,03 | 0,12 | 1,33 | 0,90 |
| validasi | hurink/rdata/la05 | 10 | 50 | 5 | 10,00 | 2,06 | 0,76 | 7,53 | 0,17 | 75,44 | 0,07 | 1,33 | 0,90 |
| validasi | hurink/vdata/la05 | 10 | 50 | 5 | 10,00 | 2,38 | 1,13 | 7,53 | 0,15 | 68,59 | 0,16 | 1,33 | 0,90 |
| validasi | hurink/edata/la10 | 15 | 75 | 5 | 15,00 | 1,16 | 0,37 | 7,53 | 0,14 | 114,53 | 0,12 | 2,00 | 0,93 |
| validasi | hurink/rdata/la10 | 15 | 75 | 5 | 15,00 | 1,96 | 0,76 | 7,53 | 0,14 | 113,69 | 0,10 | 2,00 | 0,93 |
| validasi | hurink/vdata/la10 | 15 | 75 | 5 | 15,00 | 2,67 | 1,02 | 7,53 | 0,14 | 116,15 | 0,07 | 2,00 | 0,93 |
| validasi | hurink/edata/la15 | 20 | 100 | 5 | 20,00 | 1,13 | 0,34 | 7,53 | 0,14 | 152,96 | 0,07 | 2,67 | 0,99 |
| validasi | hurink/rdata/la15 | 20 | 100 | 5 | 20,00 | 1,98 | 0,82 | 7,53 | 0,14 | 148,11 | 0,05 | 2,67 | 0,99 |
| validasi | hurink/vdata/la15 | 20 | 100 | 5 | 20,00 | 2,63 | 0,97 | 7,53 | 0,14 | 149,98 | 0,07 | 2,67 | 0,99 |
| validasi | hurink/edata/la20 | 10 | 100 | 10 | 10,00 | 1,13 | 0,34 | 7,53 | 0,14 | 75,02 | 0,08 | 0,67 | 0,95 |
| validasi | hurink/rdata/la20 | 10 | 100 | 10 | 10,00 | 1,99 | 0,82 | 7,53 | 0,14 | 76,56 | 0,12 | 0,67 | 0,95 |
| validasi | hurink/vdata/la20 | 10 | 100 | 10 | 10,00 | 4,87 | 1,44 | 7,53 | 0,14 | 75,91 | 0,12 | 0,67 | 0,95 |
| validasi | hurink/edata/la25 | 15 | 150 | 10 | 15,00 | 1,16 | 0,38 | 7,53 | 0,15 | 112,47 | 0,08 | 1,00 | 0,97 |
| validasi | hurink/rdata/la25 | 15 | 150 | 10 | 15,00 | 2,01 | 0,81 | 7,53 | 0,15 | 111,98 | 0,14 | 1,00 | 0,97 |
| validasi | hurink/vdata/la25 | 15 | 150 | 10 | 15,00 | 4,83 | 1,48 | 7,53 | 0,15 | 114,07 | 0,08 | 1,00 | 0,97 |
| validasi | hurink/edata/la30 | 20 | 200 | 10 | 20,00 | 1,14 | 0,36 | 7,53 | 0,14 | 150,35 | 0,08 | 1,33 | 0,99 |
| validasi | hurink/rdata/la30 | 20 | 200 | 10 | 20,00 | 1,96 | 0,81 | 7,53 | 0,14 | 150,85 | 0,09 | 1,33 | 0,99 |
| validasi | hurink/vdata/la30 | 20 | 200 | 10 | 20,00 | 4,65 | 1,54 | 7,53 | 0,14 | 150,82 | 0,07 | 1,33 | 0,99 |
| validasi | hurink/edata/la39 | 15 | 225 | 15 | 15,00 | 1,14 | 0,36 | 7,53 | 0,15 | 113,56 | 0,09 | 0,67 | 0,98 |
| validasi | hurink/rdata/la39 | 15 | 225 | 15 | 15,00 | 1,94 | 0,80 | 7,53 | 0,15 | 113,06 | 0,13 | 0,67 | 0,98 |
| validasi | hurink/vdata/la39 | 15 | 225 | 15 | 15,00 | 6,53 | 1,90 | 7,53 | 0,15 | 113,45 | 0,07 | 0,67 | 0,98 |
| validasi | dauzere/06a | 10 | 196 | 5 | 39,20 | 2,56 | 1,09 | 7,53 | 0,13 | 288,72 | 0,06 | 1,33 | 0,97 |
| validasi | dauzere/12a | 15 | 293 | 8 | 36,62 | 4,03 | 1,43 | 7,53 | 0,13 | 273,55 | 0,04 | 1,25 | 0,98 |
