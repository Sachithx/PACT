#!/usr/bin/env bash
# Matched n450 baselines for UCIHAR/WISDM (same label budget as PACT n450 runs).
# FD/HHAR/Sleep stay at 5% in the main paper table.
set -u
cd "$(dirname "$0")/.."
GPU=${1:?usage: campaign_baselines_n450.sh <gpu_id>}
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONUNBUFFERED=1
ENTROPE=/home/AD/sachith/.conda/envs/entrope/bin/python
PACT2=/home/AD/sachith/.conda/envs/pact2/bin/python
export LD_PRELOAD=/home/AD/sachith/.conda/envs/entrope/lib/libstdc++.so.6

DATASETS=(ucihar wisdm)
TSFMS=(moment chronos-bolt moirai sundial time-moe)
HEAD_METHODS=(head-ft lwf srcdistill)
CALIB_N=450

py_for_tsfm() {
  case "$1" in
    sundial|time-moe) echo "$PACT2" ;;
    *) echo "$ENTROPE" ;;
  esac
}

controls_done() {
  [[ -f "results/controls/${1}_${2}_adatime_K16_n${CALIB_N}/summary.csv" ]]
}

head_done() {
  [[ -f "results/cl_baselines_tsfm/${1}_${2}_adatime_${3}_n${CALIB_N}/summary.csv" ]]
}

for DS in "${DATASETS[@]}"; do
  for TSFM in "${TSFMS[@]}"; do
    PY=$(py_for_tsfm "$TSFM")
    if ! controls_done "$DS" "$TSFM"; then
      echo ">>> [gpu$GPU] controls $DS / $TSFM n$CALIB_N"
      "$PY" scripts/run_controls.py --dataset "$DS" --backbone "$TSFM" \
        --domain_order adatime --k 16 --calib_n "$CALIB_N" \
        || echo "WARN: controls failed $DS $TSFM"
    else
      echo "=== skip controls $DS / $TSFM n$CALIB_N"
    fi
    for M in "${HEAD_METHODS[@]}"; do
      if ! head_done "$DS" "$TSFM" "$M"; then
        echo ">>> [gpu$GPU] $M $DS / $TSFM n$CALIB_N"
        "$PY" scripts/run_cl_baselines_tsfm.py --dataset "$DS" --tsfm "$TSFM" \
          --method "$M" --domain_order adatime --calib_n "$CALIB_N" \
          || echo "WARN: $M failed $DS $TSFM"
      else
        echo "=== skip $M $DS / $TSFM n$CALIB_N"
      fi
    done
  done
done
echo "BASELINES_N450_GPU${GPU}_DONE"
