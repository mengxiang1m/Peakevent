#!/usr/bin/env bash
cd "$(dirname "$0")/../../.."  # -> Code/
model_name=SegRNN
peak_tolerance=1
gpu=2


python run.py \
  --task_name peak_detect_ltf_basic \
  --is_training 1 \
  --root_path ./dataset/load_data/hf_load_data/ \
  --data_path hf_load_data_20210101-20250925_mixed_with_peaks_lookahead_3.csv \
  --model_id maxIn_maxOut_MSE_244_23 \
  --model $model_name \
  --data load_data_mixed \
  --features S \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 336 \
  --seg_len 24 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
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
  --peak_lookahead 3 \
  --gpu $gpu



python run.py \
  --task_name peak_detect_ltf \
  --is_training 1 \
  --root_path ./dataset/load_data/hf_load_data/ \
  --data_path hf_load_data_20210101-20250925_mixed_with_peaks_lookahead_3.csv \
  --model_id maxIn_maxOut_MSE_244_23 \
  --model $model_name \
  --data load_data_mixed \
  --features S \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 336 \
  --seg_len 24 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
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
  --peak_lookahead 3 \
  --gpu $gpu

python run.py \
  --task_name peak_detect_ltf_basic \
  --is_training 1 \
  --root_path ./dataset/load_data/hf_load_data/ \
  --data_path hf_load_data_20210101-20250925_mixed_with_peaks_lookahead_3.csv \
  --model_id maxIn_maxOut_MSE_244_23 \
  --model $model_name \
  --data load_data_mixed \
  --features S \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 720 \
  --seg_len 24 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
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
  --peak_lookahead 3 \
  --gpu $gpu



python run.py \
  --task_name peak_detect_ltf \
  --is_training 1 \
  --root_path ./dataset/load_data/hf_load_data/ \
  --data_path hf_load_data_20210101-20250925_mixed_with_peaks_lookahead_3.csv \
  --model_id maxIn_maxOut_MSE_244_23 \
  --model $model_name \
  --data load_data_mixed \
  --features S \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 720 \
  --seg_len 24 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
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
  --peak_lookahead 3 \
  --gpu $gpu

