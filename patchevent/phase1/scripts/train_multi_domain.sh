#!/bin/bash
# ============================================================================
# Phase 1: 多域联合 Encoder 预训练 + NPP (推荐配置)
# ============================================================================
# 用法 (从 repo 根目录运行):
#   bash phase1_encoder_pretrain_final/train_multi_domain.sh
#
# 配置: 3域联合 (wlel + ett + electricity) + NPP(λ=0.5) × 3seeds
#
# 输出:
#   phase1_encoder_pretrain_final/checkpoints/multi_npp05/s{42,123,456}/best_model.pth
# ============================================================================

PYTHON="D:/Anaconda/envs/tslib_findpeaks/python.exe"
SCRIPT="phase1_encoder_pretrain_final/train.py"
CKPT_BASE="phase1_encoder_pretrain_final/checkpoints/multi_npp05"

SEEDS=(42 123 456)

# 公共超参数
EPOCHS=50
BATCH=64
LR=1e-3
WD=1e-4
PATIENCE=5
LAMBDA_NPP=0.5

for SEED in "${SEEDS[@]}"; do
    OUT_DIR="${CKPT_BASE}/s${SEED}"

    # 跳过已有 checkpoint
    if [ -f "${OUT_DIR}/best_model.pth" ]; then
        echo "[SKIP] ${OUT_DIR}/best_model.pth already exists"
        continue
    fi

    echo "============================================================"
    echo "[START] multi-domain + NPP  seed=${SEED}  output=${OUT_DIR}"
    echo "============================================================"

    $PYTHON $SCRIPT \
        --train_mode multi \
        --multi_domains wlel ett electricity \
        --lambda_npp $LAMBDA_NPP \
        --output_dir "$OUT_DIR" \
        --seed "$SEED" \
        --train_epochs $EPOCHS \
        --batch_size $BATCH \
        --learning_rate $LR \
        --weight_decay $WD \
        --patience $PATIENCE \
        --seq_len 96 \
        --patch_len 8 \
        --stride 4 \
        --d_model 128 \
        --n_heads 4 \
        --e_layers 2 \
        --d_ff 256 \
        --dropout 0.1 \
        --lambda_bce 1.0 \
        --lambda_d 0.5 \
        --lambda_kl 1.0 \
        --lambda_off 0.5 \
        --log1p_d 1

    echo "[DONE] multi-domain + NPP  seed=${SEED}"
    echo ""
done

echo "============================================"
echo "All Phase1 multi-domain training complete"
echo "============================================"
