#!/usr/bin/env bash
cd "$(dirname "$0")/../../.."  # -> Code/
model_name=proposed_model
peak_tolerance=1
gpu=0

# ==========================Value Head wo LAD=========================
# python -u run.py \
#   --task_name peak_detect_ltf \
#   --is_training 1 \
#   --root_path ./dataset/electricity \
#   --data_path electricity_mixed_with_peaks_lookahead_5.csv \
#   --model_id maxIn_maxOut_MSE_244_23_wo_lad \
#   --model $model_name \
#   --data electricity_mixed \
#   --features S \
#   --seq_len 168 \
#   --label_len 48 \
#   --pred_len 336 \
#   --input_col value_max \
#   --target_col value_max \
#   --des 'MaxIn_MaxOut' \
#   --batch_size 128 \
#   --mlp_layers 2 \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --d_model 256 \
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
#   --e_layers 1 \
#   --d_ff 256 \
#   --n_heads 4 \
#   --factor 3 \
#   --if_lad 0 \
#   --if_msm_pl 1

# python -u run.py \
#   --task_name peak_detect_ltf \
#   --is_training 1 \
#   --root_path ./dataset/electricity \
#   --data_path electricity_mixed_with_peaks_lookahead_5.csv \
#   --model_id maxIn_maxOut_MSE_244_23_wo_lad \
#   --model $model_name \
#   --data electricity_mixed \
#   --features S \
#   --seq_len 168 \
#   --label_len 48 \
#   --pred_len 720 \
#   --input_col value_max \
#   --target_col value_max \
#   --des 'MaxIn_MaxOut' \
#   --batch_size 128 \
#   --mlp_layers 2 \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --d_model 256 \
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
#   --e_layers 1 \
#   --d_ff 256 \
#   --n_heads 4 \
#   --factor 3 \
#   --if_lad 0 \
#   --if_msm_pl 1
