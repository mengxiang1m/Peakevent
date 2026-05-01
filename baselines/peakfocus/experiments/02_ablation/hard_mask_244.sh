#!/usr/bin/env bash
# A2+A3: 重跑 Hard mask ablation (WLEL + ELC), loss weights = 244
# 旧 424 数据已移至 results/pre/
source "$(dirname "$0")/../_common.sh"

MODEL=proposed_model
TASK=peak_detect_ltf
MID=ablation_mask_hard_336

for DS_LABEL in WLEL ELC; do
  if [ "$DS_LABEL" = "WLEL" ]; then
    ROOT=$WLEL_ROOT; FILE=$WLEL_FILE; DATA=$WLEL_DATA; LA=$LOOKAHEAD_WLEL
  else
    ROOT=$ELC_ROOT; FILE=$ELC_FILE; DATA=$ELC_DATA; LA=$LOOKAHEAD_ELC
  fi
  LOG="${LOG_DIR}/${DS_LABEL,,}_${MODEL}_hard_mask_244_H336.log"
  echo "[$(date +%H:%M:%S)] $DS_LABEL Hard mask 244 H=336 → $LOG"
  $PYTHON -u run.py \
    --task_name $TASK --is_training 1 \
    --root_path $ROOT --data_path $FILE \
    --model_id "$MID" --model $MODEL --data $DATA \
    --features S --seq_len 168 --label_len 48 --pred_len 336 \
    --input_col value_max --target_col value_max --des 'MaskType_hard_244' \
    --batch_size $BATCH --mlp_layers 2 \
    --enc_in 1 --dec_in 1 --c_out 1 \
    --d_model 256 --d_ff 256 --e_layers 1 --n_heads 4 --factor 3 \
    --patience $PATIENCE --train_epochs $EPOCHS \
    --enable_peak_eval 1 --peak_tolerance 1 --freq t \
    --loss MSE --learning_rate $LR --lradj $LRADJ \
    --itr $SEEDS --peak_lookahead $LA --gpu $GPU \
    --if_lad 1 --if_msm_pl 1 --mask_type hard \
    > "$LOG" 2>&1
done
echo "[$(date +%H:%M:%S)] realign_hard_mask_244 DONE"
