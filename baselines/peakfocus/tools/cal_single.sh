#!/bin/bash

# 切换到脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 基础配置
RESULTS_DIR="../results"
OUTPUT_DIR="../results"

# 可选参数列表（按实际需求修改）
TASKS=("peak_detect_ltf" "peak_detect_ltf_basic")
DATAS=("electricity_mixed" "load_data_mixed")
# MODELS=("proposed_model" "PatchTST" "STID" "CycleNet" "Informer" "TimeMixer" "DLinear" "SegRNN" "Transformer")
MODELS=("TimeMixer")
INPUT_TYPES=("maxIn")
OUTPUT_TYPES=("maxOut")
LOSSES=("MSE_244_23" "MSE_244_23_wo_lad" "MSE_244_23_wo_msm_pl")
SEQ_LENS=(168)
PRED_LENS=(336 720)

# 创建输出目录（如不存在）
mkdir -p "$OUTPUT_DIR"

# 遍历所有组合
for TASK in "${TASKS[@]}"; do
for DATA in "${DATAS[@]}"; do
for MODEL in "${MODELS[@]}"; do
for INPUT in "${INPUT_TYPES[@]}"; do
for OUTPUT in "${OUTPUT_TYPES[@]}"; do
for LOSS in "${LOSSES[@]}"; do
for SEQ in "${SEQ_LENS[@]}"; do
for PRED in "${PRED_LENS[@]}"; do

  # 自动生成输出文件名
  OUT_FILE="${OUTPUT_DIR}/mean_metrics_${TASK}_${DATA}_${MODEL}_${INPUT}_${OUTPUT}_${LOSS}_${SEQ}_${PRED}.txt"
  
  # 打印当前任务配置
  echo "正在处理：${TASK} | ${DATA} | ${MODEL} | ${INPUT} | ${OUTPUT} | ${LOSS} | ${SEQ} | ${PRED}"
  
  # 执行 Python 脚本
  python calculate_mean_metrics.py \
    --task "$TASK" \
    --data "$DATA" \
    --model "$MODEL" \
    --input_type "$INPUT" \
    --output_type "$OUTPUT" \
    --loss "$LOSS" \
    --seq_len "$SEQ" \
    --pred_len "$PRED" \
    --results_dir "$RESULTS_DIR" \
    --output_file "$OUT_FILE"
    
done; done; done; done; done; done; done; done

echo "所有实验统计完成！"
