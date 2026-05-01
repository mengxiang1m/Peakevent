#!/bin/bash
cd "$(dirname "$0")/../.."  # -> Code/
# Multi-Scale Layer Ablation (Responds to R4-Q3)
# Tests n_scales = 0, 1, 2, 3 on WLEL dataset with H=336

model_name=proposed_model
peak_tolerance=1
gpu=0
pred_len=336

for n_scales in 0 1 2 3; do
  echo "============================================"
  echo "Running: n_scales=$n_scales"
  echo "============================================"
  
  python -u run.py \
    --task_name peak_detect_ltf \
    --is_training 1 \
    --root_path ./dataset/load_data/hf_load_data/ \
    --data_path hf_load_data_20210101-20250925_mixed_with_peaks_lookahead_3.csv \
    --model_id ablation_nscales_${n_scales} \
    --model $model_name \
    --data load_data_mixed \
    --features S \
    --seq_len 168 \
    --label_len 48 \
    --pred_len $pred_len \
    --input_col value_max \
    --target_col value_max \
    --des "NScales_${n_scales}" \
    --batch_size 128 \
    --mlp_layers 2 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --d_model 256 \
    --patience 5 \
    --train_epochs 20 \
    --enable_peak_eval 1 \
    --peak_tolerance $peak_tolerance \
    --freq t \
    --loss MSE \
    --learning_rate 0.001 \
    --lradj type3 \
    --itr 3 \
    --peak_lookahead 3 \
    --gpu $gpu \
    --e_layers 1 \
    --d_ff 256 \
    --n_heads 4 \
    --factor 3 \
    --if_lad 1 \
    --if_msm_pl 1 \
    --n_scales $n_scales \
    --value_loss_weight 0.4 \
    --peak_loss_weight 0.4 \
    --tp_mse_loss_weight 0.2
done

echo "Multi-scale layer ablation complete!"
