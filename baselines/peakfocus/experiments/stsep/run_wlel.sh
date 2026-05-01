#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."  # -> baselines/peakfocus

PYTHON=${PYTHON:-python}
GPU=${GPU:-0}
SEED=${SEED:-42}

DATASET_NAME=wlel
DATA=stsep_mixed
MODEL=proposed_model
MODEL_ID=wlel_pf_stsep

SEQ_LEN=96
LABEL_LEN=48
PRED_LEN=336
WINDOW_STRIDE=4
PEAK_THRESHOLD=0.4
TOLERANCE=3

DATASET_CONFIG=../data/dataset_configs.json
INPUT_EVENT_SERIES=../../dataset/wlel/event_v1/data/wlel_event_series_v1.csv
MIXED_CSV=./dataset/patchevent/wlel_mixed_for_peakfocus.csv
EVENTS_PATH=../../dataset/wlel/event_v1/data/wlel_events_v1.jsonl

SETTING="peak_detect_ltf_stsep_${DATA}_${MODEL}_${MODEL_ID}_${SEQ_LEN}_${PRED_LEN}_0"

echo "[1/3] Convert PatchEvent -> PeakFocus mixed CSV"
"$PYTHON" tools/convert_patchevent_to_peakfocus.py \
  --input "$INPUT_EVENT_SERIES" \
  --output "$MIXED_CSV" \
  --dataset "$DATASET_NAME"

echo "[2/3] Train + Test PeakFocus (STSEP-aligned split)"
"$PYTHON" -u run_stsep.py \
  --task_name peak_detect_ltf_stsep \
  --is_training 1 \
  --model_id "$MODEL_ID" \
  --model "$MODEL" \
  --data "$DATA" \
  --root_path ./ \
  --data_path "$MIXED_CSV" \
  --dataset_config "$DATASET_CONFIG" \
  --dataset_name "$DATASET_NAME" \
  --window_stride "$WINDOW_STRIDE" \
  --features S \
  --target OT \
  --freq h \
  --seq_len "$SEQ_LEN" \
  --label_len "$LABEL_LEN" \
  --pred_len "$PRED_LEN" \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --d_model 256 \
  --e_layers 1 \
  --d_layers 1 \
  --d_ff 256 \
  --n_heads 4 \
  --factor 3 \
  --batch_size 128 \
  --train_epochs 20 \
  --patience 5 \
  --learning_rate 0.001 \
  --lradj type3 \
  --peak_threshold "$PEAK_THRESHOLD" \
  --peak_tolerance "$TOLERANCE" \
  --enable_peak_eval 1 \
  --use_gpu True \
  --gpu "$GPU" \
  --itr 1 \
  --seed "$SEED"

echo "[3/3] Evaluate STSEP metrics"
"$PYTHON" tools/evaluate_stsep.py \
  --results_dir ./results \
  --setting "$SETTING" \
  --dataset "$DATASET_NAME" \
  --events_path "$EVENTS_PATH" \
  --pred_len "$PRED_LEN" \
  --seq_len "$SEQ_LEN" \
  --peak_threshold "$PEAK_THRESHOLD" \
  --tolerance "$TOLERANCE"

echo "Done. Metrics: ./results/${SETTING}/stsep_metrics.json"
