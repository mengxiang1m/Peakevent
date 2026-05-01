#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET="wlel"
DATA_PATH="${DATASET}.csv"
CONFIG_PATH="./data/dataset_configs.json"

PRED_LENS=(96 168 336)
SEEDS=(42 123 456)

EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"

for pred_len in "${PRED_LENS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    model_id="${DATASET}_p${pred_len}_s${seed}_Autoformer"
    echo "[TSLib] model=Autoformer pred_len=${pred_len} seed=${seed}"
    python ./tslib/run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --model_id "${model_id}" \
      --model Autoformer \
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
      --train_epochs "${EPOCHS}" \
      --patience 5 \
      --itr 1 \
      --seed "${seed}" \
      --inverse
  done
done

