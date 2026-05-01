#!/bin/bash
# ============================================================================
# Phase 1: 单域 Encoder 预训练 (3域 × 3seeds = 9个实验)
# ============================================================================
# 用法 (从 repo 根目录运行):
#   bash phase1_encoder_pretrain_final/train_single_domain.sh
#
# 输出:
#   phase1_encoder_pretrain_final/checkpoints/wlel/s{42,123,456}/best_model.pth
#   phase1_encoder_pretrain_final/checkpoints/ett/s{42,123,456}/best_model.pth
#   phase1_encoder_pretrain_final/checkpoints/elc/s{42,123,456}/best_model.pth
# ============================================================================

PYTHON="D:/Anaconda/envs/tslib_findpeaks/python.exe"
SCRIPT="phase1_encoder_pretrain_final/train.py"
CKPT_BASE="phase1_encoder_pretrain_final/checkpoints"

SEEDS=(42 123 456)

# 域名映射: train.py 中 electricity 对应 elc 输出目录
DOMAINS=("wlel" "ett" "electricity")
OUT_NAMES=("wlel" "ett" "elc")

# 公共超参数
EPOCHS=50
BATCH=64
LR=1e-3
WD=1e-4
PATIENCE=5

for i in "${!DOMAINS[@]}"; do
    DOMAIN="${DOMAINS[$i]}"
    OUT_NAME="${OUT_NAMES[$i]}"

    for SEED in "${SEEDS[@]}"; do
        OUT_DIR="${CKPT_BASE}/${OUT_NAME}/s${SEED}"

        # 跳过已有 checkpoint
        if [ -f "${OUT_DIR}/best_model.pth" ]; then
            echo "[SKIP] ${OUT_DIR}/best_model.pth already exists"
            continue
        fi

        echo "============================================================"
        echo "[START] domain=${DOMAIN}  seed=${SEED}  output=${OUT_DIR}"
        echo "============================================================"

        $PYTHON $SCRIPT \
            --train_mode single \
            --single_domain "$DOMAIN" \
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

        echo "[DONE] domain=${DOMAIN}  seed=${SEED}"
        echo ""
    done
done

echo "=========================================="
echo "All Phase1 single-domain training complete"
echo "=========================================="
