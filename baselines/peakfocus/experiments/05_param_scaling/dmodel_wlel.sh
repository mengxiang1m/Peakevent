#!/usr/bin/env bash
# Exp-A on WLEL: d_model ∈ {64, 128, 512}, d_ff matches d_model.
# 3 values × 2 horizons = 6 script invocations, each with --itr 5.
source "$(dirname "$0")/../_common.sh"

MODEL=proposed_model
TASK=peak_detect_ltf

for D in 64 128 512; do
  for H in 336 720; do
    TAG="dmodel_${D}"
    LOG="${LOG_DIR}/wlel_${MODEL}_${TAG}_H${H}.log"
    echo "[$(date +%H:%M:%S)] WLEL $MODEL $TAG H=$H → $LOG"
    $PYTHON -u run.py \
      --task_name $TASK \
      --is_training 1 \
      --root_path $WLEL_ROOT \
      --data_path $WLEL_FILE \
      --model_id "maxIn_maxOut_MSE_244_23_${TAG}" \
      --model $MODEL \
      --data $WLEL_DATA \
      --features S \
      --seq_len 168 \
      --label_len 48 \
      --pred_len $H \
      --input_col value_max \
      --target_col value_max \
      --des 'MaxIn_MaxOut' \
      --batch_size $BATCH \
      --mlp_layers 2 \
      --enc_in 1 \
      --dec_in 1 \
      --c_out 1 \
      --d_model $D \
      --d_ff $D \
      --e_layers 1 \
      --n_heads 4 \
      --factor 3 \
      --patience $PATIENCE \
      --train_epochs $EPOCHS \
      --enable_peak_eval 1 \
      --peak_tolerance 1 \
      --freq t \
      --loss MSE \
      --learning_rate $LR \
      --lradj $LRADJ \
      --itr $SEEDS \
      --peak_lookahead $LOOKAHEAD_WLEL \
      --gpu $GPU \
      --if_lad 1 \
      --if_msm_pl 1 \
      > "$LOG" 2>&1
  done
done
echo "[$(date +%H:%M:%S)] scaling_dmodel_wlel DONE"
