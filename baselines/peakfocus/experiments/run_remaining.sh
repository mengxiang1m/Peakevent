#!/usr/bin/env bash
# 继续跑重构前未完成的 scaling 实验（99 runs）
# Usage: cd Code && nohup bash experiments/run_remaining.sh > experiments/logs/run_remaining.log 2>&1 &
set -uo pipefail

DIR="$(dirname "$0")"
echo "========================================================================"
echo "PeakFocus 剩余实验（99 runs）"
echo "Start: $(date)"
echo "========================================================================"

REMAINING=(
  05_param_scaling/dmodel_elc.sh
  05_param_scaling/elayers_wlel.sh
  05_param_scaling/elayers_elc.sh
  05_param_scaling/dff_wlel.sh
  05_param_scaling/dff_elc.sh
)

for s in "${REMAINING[@]}"; do
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
