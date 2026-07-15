#!/usr/bin/env bash
# PACT with fixed ~450-label calib budget (FD-equivalent; capped at domain train size).
set -u
cd "$(dirname "$0")/.."
GPU=${1:?usage: campaign_calib_n450.sh <gpu_id> [datasets...]}
shift
DATASETS=("$@")
if ((${#DATASETS[@]} == 0)); then
  DATASETS=(ucihar wisdm)
fi

export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONUNBUFFERED=1
ENTROPE=/home/AD/sachith/.conda/envs/entrope/bin/python
PACT2=/home/AD/sachith/.conda/envs/pact2/bin/python
export LD_PRELOAD=/home/AD/sachith/.conda/envs/entrope/lib/libstdc++.so.6
TSFMS=(moment chronos-bolt moirai sundial time-moe)
CALIB_N=450

py_for_tsfm() {
  case "$1" in
    sundial|time-moe) echo "$PACT2" ;;
    *) echo "$ENTROPE" ;;
  esac
}

for DS in "${DATASETS[@]}"; do
  for TSFM in "${TSFMS[@]}"; do
    OUT="results/sequential_tsfm/${DS}_${TSFM}_adatime_K16_n${CALIB_N}_learned/metrics.csv"
    if [[ -f "$OUT" ]]; then
      echo "=== skip $DS / $TSFM n${CALIB_N} (done)"
      continue
    fi
    PY=$(py_for_tsfm "$TSFM")
    echo ">>> [gpu$GPU] PACT $DS / $TSFM calib_n=$CALIB_N"
    "$PY" scripts/run_sequential_tsfm.py --dataset "$DS" --tsfm "$TSFM" \
      --domain_order adatime --k 16 --calib_n "$CALIB_N" --gate_kind learned \
      || echo "WARN: failed $DS $TSFM"
  done
done
echo "CALIB_N450_GPU${GPU}_DONE"
