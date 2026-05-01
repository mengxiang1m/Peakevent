#!/usr/bin/env bash
# ============================================================
# Run WLEL baseline experiments for pred168 and pred336 ONLY
# (pred96 already completed)
#
# Usage:
#   conda activate tslib_findpeaks
#   cd comparison_experiments
#   bash scripts/run_wlel_missing.sh
#
# Trains: 5 models × 2 pred_lens × 3 seeds = 30 runs
# Then evaluates all and collects results.
# ============================================================
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET="wlel"
DATA_PATH="${DATASET}.csv"
CONFIG_PATH="./data/dataset_configs.json"

# Only missing prediction lengths
PRED_LENS=(168 336)
SEEDS=(42 123 456)

EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"

# ---- Step 1: TSLib baselines (PatchTST, iTransformer, TimeMixer, DLinear) ----
MODELS=(PatchTST iTransformer TimeMixer DLinear)

echo "========================================="
echo " Step 1/3: Training TSLib baselines"
echo " ${#MODELS[@]} models × ${#PRED_LENS[@]} pred_lens × ${#SEEDS[@]} seeds = $(( ${#MODELS[@]} * ${#PRED_LENS[@]} * ${#SEEDS[@]} )) runs"
echo "========================================="

for model in "${MODELS[@]}"; do
  for pred_len in "${PRED_LENS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      model_id="${DATASET}_p${pred_len}_s${seed}_${model}"
      echo "[TSLib] model=${model} pred_len=${pred_len} seed=${seed}"
      python ./tslib/run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model_id "${model_id}" \
        --model "${model}" \
        --data stsep \
        --root_path ./data \
        --data_path "${DATA_PATH}" \
        --dataset_config "${CONFIG_PATH}" \
        --dataset_name "${DATASET}" \
        --window_stride 4 \
        --features S \
        --target OT \
        --seq_len 96 \
        --label_len 48 \
        --pred_len "${pred_len}" \
        --enc_in 1 \
        --dec_in 1 \
        --c_out 1 \
        --learning_rate "${LR}" \
        --batch_size "${BATCH_SIZE}" \
        --num_workers 0 \
        --train_epochs "${EPOCHS}" \
        --patience 5 \
        --itr 1 \
        --seed "${seed}" \
        --inverse
    done
  done
done

# ---- Step 2: Seq2Peak baseline (peak_Autoformer) ----
echo "========================================="
echo " Step 2/3: Training Seq2Peak baseline"
echo " 1 model × ${#PRED_LENS[@]} pred_lens × ${#SEEDS[@]} seeds = $(( ${#PRED_LENS[@]} * ${#SEEDS[@]} )) runs"
echo "========================================="

for pred_len in "${PRED_LENS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    model_id="${DATASET}_p${pred_len}_s${seed}_peak_Autoformer"
    echo "[Seq2Peak] model=peak_Autoformer pred_len=${pred_len} seed=${seed}"
    python ./seq2peak/run.py \
      --is_training 1 \
      --model_id "${model_id}" \
      --model peak_Autoformer \
      --data stsep \
      --root_path ./data \
      --data_path "${DATA_PATH}" \
      --dataset_config "${CONFIG_PATH}" \
      --dataset_name "${DATASET}" \
      --window_stride 4 \
      --features S \
      --target OT \
      --seq_len 96 \
      --label_len 48 \
      --pred_len "${pred_len}" \
      --enc_in 1 \
      --dec_in 1 \
      --c_out 1 \
      --learning_rate "${LR}" \
      --batch_size "${BATCH_SIZE}" \
      --num_workers 0 \
      --train_epochs "${EPOCHS}" \
      --patience 5 \
      --itr 1 \
      --seed "${seed}" \
      --inverse
  done
done

# ---- Step 3: Evaluate and collect results ----
echo "========================================="
echo " Step 3/3: Post-hoc evaluation"
echo "========================================="

RATE_FRAC=0.4
for pred_len in "${PRED_LENS[@]}"; do
  echo "[eval] dataset=${DATASET} pred_len=${pred_len} rate_frac=${RATE_FRAC}"
  python ./posthoc_evaluate.py --dataset "${DATASET}" --pred_len "${pred_len}" \
    --event_mode gradient_width --rate_frac "${RATE_FRAC}"
done

echo "========================================="
echo " Collecting all results"
echo "========================================="
python ./collect_results.py

echo ""
echo "Done! Check results/posthoc/wlel/pred168/ and results/posthoc/wlel/pred336/"
echo "Aggregated results: results/posthoc/summary_agg.csv"
