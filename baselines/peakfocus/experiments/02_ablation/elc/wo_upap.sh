#!/usr/bin/env bash
# Canonical w/o UPAP ablation on ELC (AUDIT_REPORT gap #1).
#
# "w/o UPAP" for this sweep means:
#   keep PeakFocus architecture on (MSM-PL + LAD enabled), but zero all
#   peak-aware losses via:
#     --if_msm_pl 1 --if_lad 1 --peak_loss_weight 0 --tp_mse_loss_weight 0
# ELC keeps its dataset-specific value-loss scaling from _common.sh
# (ELC_V=0.4), while peak/tp losses are disabled.
#
# Coverage: H in {336, 720}, each with --itr 5 (10 trainings total).

source "$(dirname "$0")/../../_common.sh"

MODEL=proposed_model
TASK=peak_detect_ltf
MODEL_ID=maxIn_maxOut_MSE_244_23_wo_upap

for H in 336 720; do
  LOG="${LOG_DIR}/elc_${MODEL}_ablation_wo_upap_H${H}.log"
  echo "[$(date +%H:%M:%S)] ELC $MODEL wo_upap H=$H -> $LOG"
  $PYTHON -u run.py \
    --task_name $TASK \
    --is_training 1 \
    --root_path $ELC_ROOT \
    --data_path $ELC_FILE \
    --model_id "$MODEL_ID" \
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
    --d_model 256 \
    --d_ff 256 \
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
    --peak_loss_weight 0 \
    --tp_mse_loss_weight 0 \
    > "$LOG" 2>&1
done

echo "[$(date +%H:%M:%S)] ablation_wo_upap_elc DONE"
