#!/usr/bin/env bash
# Langkah E15 (16 November 2026): latih ulang semua model dengan hitungan perubahan baru
# (perbaikan T1, PR #164), karena gerbang E14 terlewati. Resep, seed, dan pengaturan sama
# dengan E1, E7, E9c, E10; yang berubah hanya kode hitungan perubahan. Dijalankan dari
# revisi/hasil/gerbang_batch. Data latih saja: seed guru 7000-7001, DAgger 7002-7003,
# instans acak R000-R049, evaluasi di seed validasi 5000-5001. Data uji tidak dibuka.
#   B3  = b3_model.pkl (dibutuhkan cek_episode.py saat mulai, tidak dipakai metode di sini)
#   OUT = folder keluaran model dan log
set -euo pipefail
PY=${PY:-/opt/My-Lovely_thesis/exp1/dfjsp-ppo-repair/.venv/bin/python}
SRC=../../../exp1/dfjsp-ppo-repair/src
: "${B3:?isi B3}" "${OUT:?isi OUT}"
mkdir -p "$OUT" e15
export OMP_NUM_THREADS=1
DATA="$OUT/bc_data_7000.pkl"
ACAK="$OUT/bc_data_acak50.pkl"
E1_ENV="NET=128 BC_EPOCHS=200 BC_TAHAN=0.1 BC_BOBOT=rugi REWARD_LAM=1"
SEMUA_LAM="0.1,0.3,1,3,10,30"     # λ berlabel 0,3 / 3 / 30 dan λ tanpa label 0,1 / 1 / 10 (E13)

# 1. Data guru Sequential dengan hitungan baru.
SEED_KUMPUL=7000,7001 $PY cek_b3.py kumpul $SRC "$DATA" > "$OUT/kumpul_mk.log" 2>&1
SEED_KUMPUL=7000 INSTANS_ACAK=0-49 $PY cek_b3.py kumpul $SRC "$ACAK" > "$OUT/kumpul_acak.log" 2>&1
echo DATA-SELESAI

# 2. Lima pelatihan satu proses berjalan bersamaan: PPO E1 tiga seed dan boosting dua varian.
for K in 0 1 2; do
  env $E1_ENV $PY latih_ppo_seq.py $SRC "$DATA" "$OUT/e1ppo_s$K" 100000 $K \
    > "$OUT/e1ppo_s$K.log" 2>&1 &
done
$PY latih_hgb.py "$OUT/hgb7.pkl" "$DATA" > "$OUT/hgb7.log" 2>&1 &
$PY latih_hgb.py "$OUT/hgb7r50.pkl" "$DATA" "$ACAK" > "$OUT/hgb7r50.log" 2>&1 &
wait
echo LATIH1-SELESAI

# 3. DAgger satu putaran dari lintasan murid + penjaga (seed 7002-7003).
PPO_DIR="$OUT/e1ppo_s0" $PY dagger_kumpul.py $SRC "$B3" "$OUT/dagger_e1s0.pkl" \
  > "$OUT/dagger_e1s0.log" 2>&1
DAGGER_HGB="$OUT/hgb7.pkl" $PY dagger_kumpul.py $SRC "$B3" "$OUT/dagger_hgb7.pkl" \
  > "$OUT/dagger_hgb7.log" 2>&1
DAGGER_HGB="$OUT/hgb7r50.pkl" $PY dagger_kumpul.py $SRC "$B3" "$OUT/dagger_hgb7r50.pkl" \
  > "$OUT/dagger_hgb7r50.log" 2>&1
echo DAGGER-SELESAI

# 4. PPO E9c tiga seed dan boosting + DAgger dua varian, bersamaan.
for K in 0 1 2; do
  env $E1_ENV BC_TAMBAH="$OUT/dagger_e1s0.pkl" $PY latih_ppo_seq.py $SRC "$DATA" \
    "$OUT/e9c_s$K" 100000 $K > "$OUT/e9c_s$K.log" 2>&1 &
done
$PY latih_hgb.py "$OUT/hgb7d.pkl" "$DATA" "$OUT/dagger_hgb7.pkl" > "$OUT/hgb7d.log" 2>&1 &
$PY latih_hgb.py "$OUT/hgb7r50d.pkl" "$DATA" "$ACAK" "$OUT/dagger_hgb7r50.pkl" \
  > "$OUT/hgb7r50d.log" 2>&1 &
wait
echo LATIH2-SELESAI

# 5. Evaluasi di seed validasi. Model utama (E9c, boosting + DAgger) dan pembanding tanpa
#    latih di enam λ; model antara (E1, boosting tanpa DAgger) di tiga λ berlabel.
LAMS=$SEMUA_LAM METODE=seq,fix3_g,acak_g $PY cek_episode.py $SRC "$B3" e15/ep_aturan.json \
  > "$OUT/ep_aturan.log" 2>&1
for K in 0 1 2; do
  LAMS=$SEMUA_LAM PPO_DIR="$OUT/e9c_s$K" METODE=ppo_bc_ppo_g,ppo_bc_ppo,ppo_bc_g \
    $PY cek_episode.py $SRC "$B3" "e15/ep_e9c_s$K.json" > "$OUT/ep_e9c_s$K.log" 2>&1
done
for H in hgb7d hgb7r50d; do
  LAMS=$SEMUA_LAM METODE=il,il_g $PY cek_episode.py $SRC "$OUT/$H.pkl" "e15/ep_$H.json" \
    > "$OUT/ep_$H.log" 2>&1
done
echo EVAL-UTAMA-SELESAI
for K in 0 1 2; do
  PPO_DIR="$OUT/e1ppo_s$K" METODE=ppo_bc_ppo_g,ppo_bc_ppo,ppo_bc_g \
    $PY cek_episode.py $SRC "$B3" "e15/ep_e1ppo_s$K.json" > "$OUT/ep_e1ppo_s$K.log" 2>&1
done
for H in hgb7 hgb7r50; do
  METODE=il,il_g $PY cek_episode.py $SRC "$OUT/$H.pkl" "e15/ep_$H.json" > "$OUT/ep_$H.log" 2>&1
done
echo E15-SELESAI
