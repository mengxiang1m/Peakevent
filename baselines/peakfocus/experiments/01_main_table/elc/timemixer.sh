#!/usr/bin/env bash
cd "$(dirname "$0")/../../.."  # -> Code/
model_name=TimeMixer
peak_tolerance=1
gpu=0

python run.py \
  --task_name peak_detect_ltf \
  --is_training 1 \
  --root_path ./dataset/electricity \
  --data_path electricity_mixed_with_peaks_lookahead_5.csv \
  --model_id maxIn_maxOut_MSE_244_23 \
  --model $model_name \
  --data electricity_mixed \
  --features S \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 336 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --down_sampling_layers 2 \
  --down_sampling_method max \
  --down_sampling_window 2 \
  --input_col value_max \
  --target_col value_max \
  --des 'MaxIn_MaxOut' \
  --batch_size 128 \
  --patience 5 \
  --train_epochs 20 \
  --enable_peak_eval 1 \
  --peak_tolerance $peak_tolerance \
  --freq t \
  --loss MSE \
  --learning_rate 0.001 \
  --lradj type3 \
  --itr 5 \
  --peak_lookahead 5 \
  --gpu $gpu \
  --d_model 64 \
  --e_layers 3

python run.py \
  --task_name peak_detect_ltf \
  --is_training 1 \
  --root_path ./dataset/electricity \
  --data_path electricity_mixed_with_peaks_lookahead_5.csv \
  --model_id maxIn_maxOut_MSE_244_23 \
  --model $model_name \
  --data electricity_mixed \
  --features S \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 720 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --down_sampling_layers 2 \
  --down_sampling_method avg \
  --down_sampling_window 2 \
  --input_col value_max \
  --target_col value_max \
  --des 'MaxIn_MaxOut' \
  --batch_size 128 \
  --patience 5 \
  --train_epochs 20 \
  --enable_peak_eval 1 \
  --peak_tolerance $peak_tolerance \
  --freq t \
  --loss MSE \
  --learning_rate 0.001 \
  --lradj type3 \
  --itr 5 \
  --peak_lookahead 5 \
  --gpu $gpu \
  --d_model 256 \
  --e_layers 3


# python run.py \
#   --task_name peak_detect_ltf_basic \
#   --is_training 1 \
#   --root_path ./dataset/electricity \
#   --data_path electricity_mixed_with_peaks_lookahead_5.csv \
#   --model_id maxIn_maxOut_MSE_244_23 \
#   --model $model_name \
#   --data electricity_mixed \
#   --features S \
#   --seq_len 168 \
#   --label_len 48 \
#   --pred_len 336 \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --down_sampling_layers 2 \
#   --down_sampling_method avg \
#   --down_sampling_window 2 \
#   --input_col value_max \
#   --target_col value_max \
#   --des 'MaxIn_MaxOut' \
#   --batch_size 128 \
#   --patience 5 \
#   --train_epochs 20 \
#   --enable_peak_eval 1 \
#   --peak_tolerance $peak_tolerance \
#   --freq t \
#   --loss MSE \
#   --learning_rate 0.001 \
#   --lradj type3 \
#   --itr 5 \
#   --peak_lookahead 5 \
#   --gpu $gpu \
#   --d_model 64 \
#   --e_layers 1



# python run.py \
#   --task_name peak_detect_ltf_basic \
#   --is_training 1 \
#   --root_path ./dataset/electricity \
#   --data_path electricity_mixed_with_peaks_lookahead_5.csv \
#   --model_id maxIn_maxOut_MSE_244_23 \
#   --model $model_name \
#   --data electricity_mixed \
#   --features S \
#   --seq_len 168 \
#   --label_len 48 \
#   --pred_len 720 \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --down_sampling_layers 2 \
#   --down_sampling_method avg \
#   --down_sampling_window 2 \
#   --input_col value_max \
#   --target_col value_max \
#   --des 'MaxIn_MaxOut' \
#   --batch_size 128 \
#   --patience 5 \
#   --train_epochs 20 \
#   --enable_peak_eval 1 \
#   --peak_tolerance $peak_tolerance \
#   --freq t \
#   --loss MSE \
#   --learning_rate 0.001 \
#   --lradj type3 \
#   --itr 5 \
#   --peak_lookahead 5 \
#   --gpu $gpu \
#   --d_model 64 \
#   --e_layers 1

