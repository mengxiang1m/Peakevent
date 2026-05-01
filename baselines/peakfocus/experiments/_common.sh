#!/usr/bin/env bash
# Shared settings sourced by every experiment script.
# Run every script from the Code/ directory:
#   cd /mnt/d/.../Code && bash experiments/05_param_scaling/dmodel_wlel.sh

set -euo pipefail

export PYTHONUNBUFFERED=1

PYTHON=/home/yuwangzhi/miniconda3/envs/tslib_findpeaks/bin/python
GPU=0
SEEDS=5           # --itr
EPOCHS=20
PATIENCE=5
BATCH=128
LR=0.001
LRADJ=type3
LOOKAHEAD_WLEL=3
LOOKAHEAD_ELC=5

WLEL_ROOT=./dataset/load_data/hf_load_data/
WLEL_FILE=hf_load_data_20210101-20250925_mixed_with_peaks_lookahead_3.csv
WLEL_DATA=load_data_mixed

ELC_ROOT=./dataset/electricity
ELC_FILE=electricity_mixed_with_peaks_lookahead_5.csv
ELC_DATA=electricity_mixed

# Both datasets now use the same loss weights: 244 = (V=0.2, P=0.4, T=0.4).
# This matches the run.py defaults, so no explicit override is needed.
# The old ELC-specific (V=0.4, P=0.4, T=0.2) = 424 was deprecated in favour
# of uniform weights across datasets (see TODO.md for justification).
# Kept as comments for reference:
#   OLD: ELC_V=0.4  ELC_P=0.4  ELC_T=0.2   (424, deprecated)
ELC_V=0.2
ELC_P=0.4
ELC_T=0.4

# Per-run timestamped stdout log dir
LOG_DIR=experiments/logs
mkdir -p "$LOG_DIR"
