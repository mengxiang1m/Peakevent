#!/usr/bin/env bash
# A1: 重跑 ELC PeakFocus 主表，loss weights = 244 (V=0.2 P=0.4 T=0.4)
# 旧 424 数据已移至 results/pre/，新 run 直接用原 model_id
source "$(dirname "$0")/../_common.sh"

MODEL=proposed_model
TASK=peak_detect_ltf

for H in 336 720; do
  LOG="${LOG_DIR}/elc_${MODEL}_realign_244_H${H}.log"
  echo "[$(date +%H:%M:%S)] ELC PeakFocus 244 H=$H → $LOG"
  $PYTHON -u run.py \
    --task_name $TASK --is_training 1 \
    --root_path $ELC_ROOT --data_path $ELC_FILE \
    --model_id maxIn_maxOut_MSE_244_23 --model $MODEL --data $ELC_DATA \
    --features S --seq_len 168 --label_len 48 --pred_len $H \
    --input_col value_max --target_col value_max --des 'MaxIn_MaxOut' \
    --batch_size $BATCH --mlp_layers 2 \
    --enc_in 1 --dec_in 1 --c_out 1 \
    --d_model 256 --d_ff 256 --e_layers 1 --n_heads 4 --factor 3 \
    --patience $PATIENCE --train_epochs $EPOCHS \
    --enable_peak_eval 1 --peak_tolerance 1 --freq t \
    --loss MSE --learning_rate $LR --lradj $LRADJ \
    --itr $SEEDS --peak_lookahead $LOOKAHEAD_ELC --gpu $GPU \
    --if_lad 1 --if_msm_pl 1 \
    > "$LOG" 2>&1
done
echo "[$(date +%H:%M:%S)] realign_elc_peakfocus_244 DONE"
