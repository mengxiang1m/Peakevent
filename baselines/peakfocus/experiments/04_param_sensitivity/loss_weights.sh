#!/bin/bash
cd "$(dirname "$0")/../.."  # -> Code/
# Loss Weight Sensitivity Ablation (Responds to R1-Q5)
# Tests different combinations of (value_loss_weight, peak_loss_weight, tp_mse_loss_weight)
# on WLEL dataset with H=336

model_name=proposed_model
peak_tolerance=1
gpu=0
pred_len=336

# Configurations: value_w, peak_w, tp_mse_w
configs=(
  "0.4 0.4 0.2"   # Default (paper)
  "0.6 0.2 0.2"   # Heavy global
  "0.2 0.2 0.6"   # Heavy TP-MSE
  "0.33 0.33 0.33" # Equal weights
  "0.4 0.2 0.4"   # Swap peak/tp_mse
  "0.2 0.6 0.2"   # Heavy peak cls
  "0.2 0.4 0.4"   # Low global
  "0.8 0.1 0.1"   # Dominant global
)

for config in "${configs[@]}"; do
  read -r vw pw tw <<< "$config"
  echo "============================================"
  echo "Running: value_w=$vw, peak_w=$pw, tp_mse_w=$tw"
  echo "============================================"
  
  python -u run.py \
    --task_name peak_detect_ltf \
    --is_training 1 \
    --root_path ./dataset/load_data/hf_load_data/ \
    --data_path hf_load_data_20210101-20250925_mixed_with_peaks_lookahead_3.csv \
    --model_id ablation_lossw_${vw}_${pw}_${tw} \
    --model $model_name \
    --data load_data_mixed \
    --features S \
    --seq_len 168 \
    --label_len 48 \
    --pred_len $pred_len \
    --input_col value_max \
    --target_col value_max \
    --des "LossWeight_${vw}_${pw}_${tw}" \
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
    --value_loss_weight $vw \
    --peak_loss_weight $pw \
    --tp_mse_loss_weight $tw
done

echo "Loss weight sensitivity ablation complete!"
