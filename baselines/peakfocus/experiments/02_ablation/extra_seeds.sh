#!/usr/bin/env bash
# Fill low-seed Hard-mask ablation rows flagged by experiments/AUDIT_REPORT.md.
#
# Gaps being filled:
#   1) Hard mask WLEL H=336 (currently 3 seeds -> add seeds 3 & 4 to reach 5)
#   2) Hard mask ELC  H=336 (currently 3 seeds -> add seeds 3 & 4 to reach 5)
#
# NOT included (already at 5 seeds after classify_main fix):
#   - main ELC 336 Informer: already 5 seeds under the peak_detect_ltf task,
#     the 4-seed count was under peak_detect_ltf_basic which is no longer used.
#
# Mechanism:
#   run.py now supports `--start_seed N`: the itr loop runs ii in
#   [start_seed, start_seed + itr). Setting string suffix matches ii, so
#   new runs land at `..._3` and `..._4` without clobbering existing
#   `..._0`, `..._1`, `..._2` dirs.
#
#   Note: fix_seed=2021 is set globally in run.py (one-shot), so reruns rely
#   on GPU non-determinism for seed variance — same paradigm as the original
#   3 existing seeds. build_figure_data.py's classify_ablation will pick up
#   the 2 new rows automatically since they share the canonical model_id.

source "$(dirname "$0")/../_common.sh"

MODEL=proposed_model
TASK=peak_detect_ltf
HM_MODEL_ID=ablation_mask_hard_336   # MUST match existing dir basename

# WLEL hard-mask loss weights (verified from
#   results/peak_detect_ltf_load_data_mixed_proposed_model_ablation_mask_hard_336_168_336_0/experiment_log.txt
# header line "Val×0.4 Peak×0.4 TP_MSE×0.2").
HM_VALUE_W=0.4
HM_PEAK_W=0.4
HM_TP_W=0.2

START=3    # next seed slot after existing 0/1/2
N=2        # two more seeds → 3 & 4

# -----------------------------------------------------------------------------
# 1) Hard mask WLEL H=336 → add seeds 3 & 4
# -----------------------------------------------------------------------------
LOG="${LOG_DIR}/wlel_${MODEL}_ablation_mask_hard_336_H336_seed3-4.log"
echo "[$(date +%H:%M:%S)] WLEL hard-mask H=336 seeds ${START}..$((START+N-1)) → $LOG"
$PYTHON -u run.py \
  --task_name $TASK \
  --is_training 1 \
  --root_path $WLEL_ROOT \
  --data_path $WLEL_FILE \
  --model_id "$HM_MODEL_ID" \
  --model $MODEL \
  --data $WLEL_DATA \
  --features S \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 336 \
  --input_col value_max \
  --target_col value_max \
  --des 'MaskType_hard_seed3-4' \
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
  --start_seed $START \
  --itr $N \
  --peak_lookahead $LOOKAHEAD_WLEL \
  --gpu $GPU \
  --if_lad 1 \
  --if_msm_pl 1 \
  --mask_type hard \
  --value_loss_weight $HM_VALUE_W \
  --peak_loss_weight $HM_PEAK_W \
  --tp_mse_loss_weight $HM_TP_W \
  > "$LOG" 2>&1

# -----------------------------------------------------------------------------
# 2) Hard mask ELC H=336 → add seeds 3 & 4
# -----------------------------------------------------------------------------
LOG="${LOG_DIR}/elc_${MODEL}_ablation_mask_hard_336_H336_seed3-4.log"
echo "[$(date +%H:%M:%S)] ELC hard-mask H=336 seeds ${START}..$((START+N-1)) → $LOG"
$PYTHON -u run.py \
  --task_name $TASK \
  --is_training 1 \
  --root_path $ELC_ROOT \
  --data_path $ELC_FILE \
  --model_id "$HM_MODEL_ID" \
  --model $MODEL \
  --data $ELC_DATA \
  --features S \
  --seq_len 168 \
  --label_len 48 \
  --pred_len 336 \
  --input_col value_max \
  --target_col value_max \
  --des 'MaskType_hard_seed3-4' \
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
  --start_seed $START \
  --itr $N \
  --peak_lookahead $LOOKAHEAD_ELC \
  --gpu $GPU \
  --if_lad 1 \
  --if_msm_pl 1 \
  --mask_type hard \
  --value_loss_weight $ELC_V \
  --peak_loss_weight $ELC_P \
  --tp_mse_loss_weight $ELC_T \
  > "$LOG" 2>&1

echo "[$(date +%H:%M:%S)] ablation_extra_seeds DONE"
