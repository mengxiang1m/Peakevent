#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ "$#" -eq 0 ]; then
  DATASETS=(wlel)
else
  DATASETS=("$@")
fi
PRED_LENS=(96 168 336)

for dataset in "${DATASETS[@]}"; do
  # Match rate_frac to GT construction: WLEL=0.4, ETT/ELC=0.020
  if [ "${dataset}" = "wlel" ]; then
    RATE_FRAC=0.4
  else
    RATE_FRAC=0.020
  fi
  for pred_len in "${PRED_LENS[@]}"; do
    echo "[eval] dataset=${dataset} pred_len=${pred_len} rate_frac=${RATE_FRAC}"
    python ./posthoc_evaluate.py --dataset "${dataset}" --pred_len "${pred_len}" \
      --event_mode gradient_width --rate_frac "${RATE_FRAC}"
  done
done

python ./collect_results.py
