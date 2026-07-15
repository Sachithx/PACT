#!/usr/bin/env bash
# TSFM plugin table (Option B): for each TSFM × dataset run
#   frozen+source head (no-adapt), head-FT, LwF-head, src-distill-head,
#   adapters-no-gate, +PACT on the AdaTime domain-incremental chain.
#
# Usage:
#   bash scripts/campaign_tsfm_plugin_table.sh <gpu_id> [dataset ...]
#   bash scripts/campaign_tsfm_plugin_table.sh 0          # all datasets on GPU 0
#   bash scripts/campaign_tsfm_plugin_table.sh 1 ucihar wisdm
#
# Skips runs whose summary/metrics already exist under results/.
set -u
cd "$(dirname "$0")/.."

GPU=${1:?usage: campaign_tsfm_plugin_table.sh <gpu_id> [datasets...]}
shift
DATASETS=("$@")
if ((${#DATASETS[@]} == 0)); then
  DATASETS=(fd hhar sleep ucihar wisdm)
fi

export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONUNBUFFERED=1

ENTROPE=/home/AD/sachith/.conda/envs/entrope/bin/python
PACT2=/home/AD/sachith/.conda/envs/pact2/bin/python
export LD_PRELOAD=/home/AD/sachith/.conda/envs/entrope/lib/libstdc++.so.6

TSFMS=(moment chronos-bolt moirai sundial time-moe)
HEAD_METHODS=(head-ft lwf srcdistill)

py_for_tsfm() {
  case "$1" in
    sundial|time-moe) echo "$PACT2" ;;
    *) echo "$ENTROPE" ;;
  esac
}

controls_done() {
  local ds=$1 tsfm=$2
  [[ -f "results/controls/${ds}_${tsfm}_adatime_K16_c5/summary.csv" ]]
}

head_done() {
  local ds=$1 tsfm=$2 method=$3
  [[ -f "results/cl_baselines_tsfm/${ds}_${tsfm}_adatime_${method}_c5/summary.csv" ]]
}

pact_done() {
  local ds=$1 tsfm=$2
  local d
  for d in results/sequential_tsfm/${ds}_${tsfm}_adatime_K16_c5*; do
    [[ -d "$d" && -f "$d/metrics.csv" && "$d" != *prototype* ]] && return 0
  done
  return 1
}

run_controls() {
  local ds=$1 tsfm=$2
  local PY
  PY=$(py_for_tsfm "$tsfm")
  echo ">>> [gpu$GPU] controls $ds / $tsfm"
  "$PY" scripts/run_controls.py --dataset "$ds" --backbone "$tsfm" \
    --domain_order adatime --k 16 --calib_frac 0.05
}

run_head() {
  local ds=$1 tsfm=$2 method=$3
  local PY
  PY=$(py_for_tsfm "$tsfm")
  echo ">>> [gpu$GPU] $method $ds / $tsfm"
  "$PY" scripts/run_cl_baselines_tsfm.py --dataset "$ds" --tsfm "$tsfm" \
    --method "$method" --domain_order adatime --calib_frac 0.05
}

run_pact() {
  local ds=$1 tsfm=$2
  local PY
  PY=$(py_for_tsfm "$tsfm")
  echo ">>> [gpu$GPU] PACT $ds / $tsfm"
  "$PY" scripts/run_sequential_tsfm.py --dataset "$ds" --tsfm "$tsfm" \
    --domain_order adatime --k 16 --calib_frac 0.05 --gate_kind learned
}

for DS in "${DATASETS[@]}"; do
  for TSFM in "${TSFMS[@]}"; do
    if ! controls_done "$DS" "$TSFM"; then
      run_controls "$DS" "$TSFM" || echo "WARN: controls failed $DS $TSFM"
    else
      echo "=== skip controls $DS / $TSFM (done)"
    fi
    for M in "${HEAD_METHODS[@]}"; do
      if ! head_done "$DS" "$TSFM" "$M"; then
        run_head "$DS" "$TSFM" "$M" || echo "WARN: $M failed $DS $TSFM"
      else
        echo "=== skip $M $DS / $TSFM (done)"
      fi
    done
    if ! pact_done "$DS" "$TSFM"; then
      run_pact "$DS" "$TSFM" || echo "WARN: PACT failed $DS $TSFM"
    else
      echo "=== skip PACT $DS / $TSFM (done)"
    fi
  done
done

echo "TSFM_PLUGIN_TABLE_GPU${GPU}_DONE datasets=${DATASETS[*]}"
