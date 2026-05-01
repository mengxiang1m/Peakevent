#!/bin/bash
cd "$(dirname "$0")/../../.."  # -> Code/
# Naive Baselines Evaluation (Responds to R1-W3/Q6)
# Tests persistence, seasonal, movavg on WLEL and ELC datasets

gpu=0
peak_tolerance_wlel=1
peak_tolerance_elc=1

# ==================== WLEL ====================
for naive_type in persistence seasonal movavg; do
  for pred_len in 336 720; do
    echo "============================================"
    echo "Running WLEL: naive_type=$naive_type, pred_len=$pred_len"
    echo "============================================"
    
    python -u run.py \
      --task_name peak_detect_ltf_basic \
      --is_training 0 \
      --root_path ./dataset/load_data/hf_load_data/ \
      --data_path hf_load_data_20210101-20250925_mixed_with_peaks_lookahead_3.csv \
      --model_id naive_${naive_type}_${pred_len} \
      --model NaiveBaseline \
      --data load_data_mixed \
      --features S \
      --seq_len 168 \
      --label_len 48 \
      --pred_len $pred_len \
      --input_col value_max \
      --target_col value_max \
      --des "Naive_${naive_type}" \
      --batch_size 128 \
      --enc_in 1 \
      --dec_in 1 \
      --c_out 1 \
      --d_model 64 \
      --enable_peak_eval 1 \
      --peak_tolerance $peak_tolerance_wlel \
      --freq t \
      --peak_lookahead 3 \
      --gpu $gpu \
      --naive_type $naive_type \
      --movavg_window 24 \
      --seasonal_period 168 \
      --itr 1
  done
done

# ==================== ELC ====================
for naive_type in persistence seasonal movavg; do
  for pred_len in 336 720; do
    echo "============================================"
    echo "Running ELC: naive_type=$naive_type, pred_len=$pred_len"
    echo "============================================"
    
    python -u run.py \
      --task_name peak_detect_ltf_basic \
      --is_training 0 \
      --root_path ./dataset/electricity \
      --data_path electricity_mixed_with_peaks_lookahead_5.csv \
      --model_id naive_${naive_type}_${pred_len} \
      --model NaiveBaseline \
      --data electricity_mixed \
      --features S \
      --seq_len 168 \
      --label_len 48 \
      --pred_len $pred_len \
      --input_col value_max \
      --target_col value_max \
      --des "Naive_${naive_type}" \
      --batch_size 128 \
      --enc_in 1 \
      --dec_in 1 \
      --c_out 1 \
      --d_model 64 \
      --enable_peak_eval 1 \
      --peak_tolerance $peak_tolerance_elc \
      --freq t \
      --peak_lookahead 5 \
      --gpu $gpu \
      --naive_type $naive_type \
      --movavg_window 24 \
      --seasonal_period 168 \
      --itr 1
  done
done

echo "Naive baselines evaluation complete!"
