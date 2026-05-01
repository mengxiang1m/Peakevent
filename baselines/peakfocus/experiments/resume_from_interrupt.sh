#!/usr/bin/env bash
# 从中断处精准恢复，只跑缺失的实验
# 跳过已完成的 dmodel WLEL (30/30) 和 ELC d=64 (10/10) 和 ELC d=128 H=336 (5/5)
#
# 剩余清单：
#   1. ELC d=128 H=720 seed 3-4  (2 runs, seed 3 被杀需重跑)
#   2. ELC d=512 H=336+720       (10 runs)
#   3. elayers WLEL               (20 runs)
#   4. elayers ELC                (20 runs)
#   5. dff WLEL                   (20 runs)
#   6. dff ELC                    (20 runs)
#   总计 ~92 runs
#
# Usage: cd Code && nohup bash experiments/resume_from_interrupt.sh > experiments/logs/resume.log 2>&1 &

set -uo pipefail
source "$(dirname "$0")/_common.sh"

MODEL=proposed_model
TASK=peak_detect_ltf

echo "========================================================================"
echo "恢复实验 (~92 runs)"
echo "Start: $(date)"
echo "========================================================================"

# ── 1. ELC d=128 H=720 seed 3-4 ──
echo ""
echo "[$(date +%H:%M:%S)] ELC dmodel_128 H=720 seed 3-4 (补完)"
LOG="${LOG_DIR}/elc_${MODEL}_dmodel_128_H720_resume.log"
$PYTHON -u run.py \
  --task_name $TASK --is_training 1 \
  --root_path $ELC_ROOT --data_path $ELC_FILE --data $ELC_DATA \
  --model_id "maxIn_maxOut_MSE_244_23_dmodel_128" --model $MODEL \
  --features S --seq_len 168 --label_len 48 --pred_len 720 \
  --input_col value_max --target_col value_max --des 'MaxIn_MaxOut' \
  --batch_size $BATCH --mlp_layers 2 --enc_in 1 --dec_in 1 --c_out 1 \
  --d_model 128 --d_ff 128 --e_layers 1 --n_heads 4 --factor 3 \
  --patience $PATIENCE --train_epochs $EPOCHS \
  --enable_peak_eval 1 --peak_tolerance 1 --freq t --loss MSE \
  --learning_rate $LR --lradj $LRADJ \
  --start_seed 3 --itr 2 \
  --peak_lookahead $LOOKAHEAD_ELC --gpu $GPU \
  --if_lad 1 --if_msm_pl 1 \
  --value_loss_weight $ELC_V --peak_loss_weight $ELC_P --tp_mse_loss_weight $ELC_T \
  > "$LOG" 2>&1
echo "[$(date +%H:%M:%S)] ELC dmodel_128 H=720 补完 DONE"

# ── 2. ELC d=512 H=336+720 ──
for H in 336 720; do
  LOG="${LOG_DIR}/elc_${MODEL}_dmodel_512_H${H}.log"
  echo "[$(date +%H:%M:%S)] ELC dmodel_512 H=$H → $LOG"
  $PYTHON -u run.py \
    --task_name $TASK --is_training 1 \
    --root_path $ELC_ROOT --data_path $ELC_FILE --data $ELC_DATA \
    --model_id "maxIn_maxOut_MSE_244_23_dmodel_512" --model $MODEL \
    --features S --seq_len 168 --label_len 48 --pred_len $H \
    --input_col value_max --target_col value_max --des 'MaxIn_MaxOut' \
    --batch_size $BATCH --mlp_layers 2 --enc_in 1 --dec_in 1 --c_out 1 \
    --d_model 512 --d_ff 512 --e_layers 1 --n_heads 4 --factor 3 \
    --patience $PATIENCE --train_epochs $EPOCHS \
    --enable_peak_eval 1 --peak_tolerance 1 --freq t --loss MSE \
    --learning_rate $LR --lradj $LRADJ --itr $SEEDS \
    --peak_lookahead $LOOKAHEAD_ELC --gpu $GPU \
    --if_lad 1 --if_msm_pl 1 \
    --value_loss_weight $ELC_V --peak_loss_weight $ELC_P --tp_mse_loss_weight $ELC_T \
    > "$LOG" 2>&1
done
echo "[$(date +%H:%M:%S)] ELC dmodel_512 DONE"

# ── 3-6. elayers + dff (直接调用已有脚本) ──
DIR="$(dirname "$0")"
for s in 05_param_scaling/elayers_wlel.sh 05_param_scaling/elayers_elc.sh \
         05_param_scaling/dff_wlel.sh 05_param_scaling/dff_elc.sh; do
  echo ""
  echo "------------------------------------------------------------------------"
  echo "[$(date +%H:%M:%S)] RUNNING: $s"
  echo "------------------------------------------------------------------------"
  if bash "$DIR/$s"; then
    echo "[$(date +%H:%M:%S)] OK: $s"
  else
    echo "[$(date +%H:%M:%S)] FAILED: $s (continuing)"
  fi
done

echo ""
echo "========================================================================"
echo "ALL DONE. End: $(date)"
echo "========================================================================"
