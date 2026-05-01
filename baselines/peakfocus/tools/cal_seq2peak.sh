#!/bin/bash

# 切换到脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 基础配置
RESULTS_DIR="../results"
OUTPUT_DIR="../results"

# 创建输出目录（如不存在）
mkdir -p "$OUTPUT_DIR"

# ========== Seq2Peak 任务配置 ==========
echo "========== 处理 Seq2Peak 任务 =========="
SEQ2PEAK_DATAS=("electricity_mixed" "load_data_mixed")
SEQ2PEAK_MODELS=("peak_Transformer")
SEQ2PEAK_MODEL_IDS=("maxIn_maxOut_244_23" "maxIn_maxOut_MSE_244_23")
SEQ2PEAK_SEQ_LENS=(168)
SEQ2PEAK_PRED_LENS=(336 720)

for DATA in "${SEQ2PEAK_DATAS[@]}"; do
for MODEL in "${SEQ2PEAK_MODELS[@]}"; do
for MODEL_ID in "${SEQ2PEAK_MODEL_IDS[@]}"; do
for SEQ in "${SEQ2PEAK_SEQ_LENS[@]}"; do
for PRED in "${SEQ2PEAK_PRED_LENS[@]}"; do

  OUT_FILE="${OUTPUT_DIR}/mean_metrics_seq2peak_${DATA}_${MODEL}_${MODEL_ID}_${SEQ}_${PRED}.txt"
  
  echo "正在处理：seq2peak | ${DATA} | ${MODEL} | ${MODEL_ID} | ${SEQ} | ${PRED}"
  
  python calculate_mean_metrics.py \
    --task "seq2peak" \
    --data "$DATA" \
    --model "$MODEL" \
    --model_id "$MODEL_ID" \
    --seq_len "$SEQ" \
    --pred_len "$PRED" \
    --results_dir "$RESULTS_DIR" \
    --output_file "$OUT_FILE"
    
done; done; done; done; done

# ========== 旧格式任务配置 ==========
# echo ""
# echo "========== 处理旧格式任务 =========="
# OLD_TASKS=("peak_detect_ltf" "peak_detect_ltf_basic")
# OLD_DATAS=("electricity_mixed" "load_data_mixed")
# OLD_MODELS=("TimeMixer")
# OLD_INPUT_TYPES=("maxIn")
# OLD_OUTPUT_TYPES=("maxOut")
# OLD_LOSSES=("MSE_244_23" "MSE_244_23_wo_lad" "MSE_244_23_wo_msm_pl")
# OLD_SEQ_LENS=(168)
# OLD_PRED_LENS=(336 720)

# for TASK in "${OLD_TASKS[@]}"; do
# for DATA in "${OLD_DATAS[@]}"; do
# for MODEL in "${OLD_MODELS[@]}"; do
# for INPUT in "${OLD_INPUT_TYPES[@]}"; do
# for OUTPUT in "${OLD_OUTPUT_TYPES[@]}"; do
# for LOSS in "${OLD_LOSSES[@]}"; do
# for SEQ in "${OLD_SEQ_LENS[@]}"; do
# for PRED in "${OLD_PRED_LENS[@]}"; do

#   OUT_FILE="${OUTPUT_DIR}/mean_metrics_${TASK}_${DATA}_${MODEL}_${INPUT}_${OUTPUT}_${LOSS}_${SEQ}_${PRED}.txt"
  
#   echo "正在处理：${TASK} | ${DATA} | ${MODEL} | ${INPUT} | ${OUTPUT} | ${LOSS} | ${SEQ} | ${PRED}"
  
#   python calculate_mean_metrics.py \
#     --task "$TASK" \
#     --data "$DATA" \
#     --model "$MODEL" \
#     --input_type "$INPUT" \
#     --output_type "$OUTPUT" \
#     --loss "$LOSS" \
#     --seq_len "$SEQ" \
#     --pred_len "$PRED" \
#     --results_dir "$RESULTS_DIR" \
#     --output_file "$OUT_FILE"
    
# done; done; done; done; done; done; done; done

# echo ""
# echo "所有实验统计完成！"
