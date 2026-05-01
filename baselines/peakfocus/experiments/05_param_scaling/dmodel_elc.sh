#!/usr/bin/env bash
# Exp-A on ELC: d_model ∈ {64, 128, 512}, d_ff matches d_model.
source "$(dirname "$0")/../_common.sh"

MODEL=proposed_model
TASK=peak_detect_ltf

for D in 64 128 512; do
  for H in 336 720; do
    TAG="dmodel_${D}"
    LOG="${LOG_DIR}/elc_${MODEL}_${TAG}_H${H}.log"
    echo "[$(date +%H:%M:%S)] ELC $MODEL $TAG H=$H → $LOG"
    $PYTHON -u run.py \
      --task_name $TASK \
      --is_training 1 \
      --root_path $ELC_ROOT \
      --data_path $ELC_FILE \
      --model_id "maxIn_maxOut_MSE_244_23_${TAG}" \
      --model $MODEL \
      --data $ELC_DATA \
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
      --peak_lookahead $LOOKAHEAD_ELC \
      --gpu $GPU \
      --if_lad 1 \
      --if_msm_pl 1 \
      --value_loss_weight $ELC_V \
      --peak_loss_weight $ELC_P \
      --tp_mse_loss_weight $ELC_T \
      > "$LOG" 2>&1
  done
done
echo "[$(date +%H:%M:%S)] scaling_dmodel_elc DONE"
