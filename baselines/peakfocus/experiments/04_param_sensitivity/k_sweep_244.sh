#!/usr/bin/env bash
# A4: 重跑 WLEL K sweep (K=0/1/2/3), loss weights = 244
# 旧 424 数据已移至 results/pre/
source "$(dirname "$0")/../_common.sh"

MODEL=proposed_model
TASK=peak_detect_ltf

for K in 0 1 2 3; do
  LOG="${LOG_DIR}/wlel_${MODEL}_nscales_${K}_244_H336.log"
  echo "[$(date +%H:%M:%S)] WLEL K=$K 244 H=336 → $LOG"
  $PYTHON -u run.py \
    --task_name $TASK --is_training 1 \
    --root_path $WLEL_ROOT --data_path $WLEL_FILE \
    --model_id "ablation_nscales_${K}" --model $MODEL --data $WLEL_DATA \
    --features S --seq_len 168 --label_len 48 --pred_len 336 \
    --input_col value_max --target_col value_max --des 'MaxIn_MaxOut' \
    --batch_size $BATCH --mlp_layers 2 \
    --enc_in 1 --dec_in 1 --c_out 1 \
    --d_model 256 --d_ff 256 --e_layers 1 --n_heads 4 --factor 3 \
    --n_scales $K \
    --patience $PATIENCE --train_epochs $EPOCHS \
    --enable_peak_eval 1 --peak_tolerance 1 --freq t \
    --loss MSE --learning_rate $LR --lradj $LRADJ \
    --itr $SEEDS --peak_lookahead $LOOKAHEAD_WLEL --gpu $GPU \
    --if_lad 1 --if_msm_pl 1 \
    > "$LOG" 2>&1
done
echo "[$(date +%H:%M:%S)] realign_k_sweep_244 DONE"
