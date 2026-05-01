#!/usr/bin/env bash
cd "$(dirname "$0")/../../.."  # -> Code/
model_name=peak_Transformer
peak_tolerance=1
peak_lookahead=5
gpu=0

# Peak Transformer model with busy decoder (maxpooling)
# This model outputs two values:
# 1. dec_out: Full prediction sequence [B, pred_len, C]
# 2. busy_out: Peak predictions (maxpooled) [B, pred_len//24, C]
# Task: seq2peak (专用于seq2peak任务)
# Loss: 0.5 * value_loss + 0.5 * peak_loss (peak_loss使用argmax找峰值位置后计算)

python run.py \
  --task_name seq2peak \
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
  --input_col value_max \
  --target_col value_max \
  --des 'Seq2Peak' \
  --batch_size 128 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 2 \
  --patience 5 \
  --train_epochs 20 \
  --freq t \
  --loss MSE \
  --learning_rate 0.001 \
  --lradj type3 \
  --itr 5 \
  --busy_decoder 1 \
  --busy_decoder_modal m \
  --hour_day h \
  --with_shift ws \
  --p_hidden_dims 128 128 \
  --p_hidden_layers 2 \
  --seq2peak_value_loss_weight 0.5 \
  --seq2peak_peak_loss_weight 0.5 \
  --enable_peak_eval 1 \
  --vis_test_peaks 1 \
  --peak_tolerance $peak_tolerance \
  --peak_lookahead $peak_lookahead \
  --gpu $gpu

python run.py \
  --task_name seq2peak \
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
  --input_col value_max \
  --target_col value_max \
  --des 'Seq2Peak' \
  --batch_size 128 \
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 2 \
  --patience 5 \
  --train_epochs 20 \
  --freq t \
  --loss MSE \
  --learning_rate 0.001 \
  --lradj type3 \
  --itr 5 \
  --busy_decoder 1 \
  --busy_decoder_modal m \
  --hour_day h \
  --with_shift ws \
  --p_hidden_dims 128 128 \
  --p_hidden_layers 2 \
  --seq2peak_value_loss_weight 0.5 \
  --seq2peak_peak_loss_weight 0.5 \
  --enable_peak_eval 1 \
  --vis_test_peaks 1 \
  --peak_tolerance $peak_tolerance \
  --peak_lookahead $peak_lookahead \
  --gpu $gpu


