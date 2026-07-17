#!/usr/bin/env bash
# Dual-GPU runner for remaining external baseline jobs (skips completed outputs).
set -uo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-/home/AD/sachith/.conda/envs/entrope/bin/python}"
export LD_PRELOAD="${LD_PRELOAD:-/home/AD/sachith/.conda/envs/entrope/lib/libstdc++.so.6}"
export PYTHONUNBUFFERED=1
NUM_GPUS="${NUM_GPUS:-2}"
SEEDS="${SEEDS:-42,0,1}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="results/experiment_logs/external_remaining_${TS}.log"
mkdir -p results/experiment_logs

declare -a GPU_PID=()
for ((g=0; g<NUM_GPUS; g++)); do GPU_PID[$g]=""; done

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG" >&2; }

_reap_finished() {
  local g pid ec
  for ((g=0; g<NUM_GPUS; g++)); do
    pid="${GPU_PID[$g]:-}"
    [[ -z "$pid" ]] && continue
    if ! kill -0 "$pid" 2>/dev/null; then
      ec=0; wait "$pid" || ec=$?
      [[ "$ec" -ne 0 ]] && log "FAILED (exit $ec) on cuda:$g (pid $pid)"
      GPU_PID[$g]=""
    fi
  done
}

_acquire_gpu() {
  local g
  while true; do
    _reap_finished
    for ((g=0; g<NUM_GPUS; g++)); do
      [[ -z "${GPU_PID[$g]:-}" ]] && { echo "$g"; return 0; }
    done
    sleep 2
  done
}

run() {
  local gpu args=() skip=0 a
  gpu="$(_acquire_gpu)"
  for a in "$@"; do
    if [[ "$skip" -eq 1 ]]; then skip=0; continue; fi
    if [[ "$a" == "--device" ]]; then skip=1; continue; fi
    args+=("$a")
  done
  log ">>> [cuda:$gpu] ${args[*]}"
  "$PY" "${args[@]}" --device "cuda:$gpu" >>"$LOG" 2>&1 &
  GPU_PID[$gpu]=$!
}

wait_all() {
  local g pid ec
  for ((g=0; g<NUM_GPUS; g++)); do
    pid="${GPU_PID[$g]:-}"
    [[ -z "$pid" ]] && continue
    ec=0; wait "$pid" || ec=$?
    [[ "$ec" -ne 0 ]] && log "FAILED (exit $ec) on cuda:$g (pid $pid)"
    GPU_PID[$g]=""
  done
}

done_chain() {
  local tag=$1
  [[ -f "results/chain_baselines/${tag}/summary.csv" ]]
}

done_mapu() {
  local tag=$1
  [[ -f "results/mapu_baselines/${tag}/summary.csv" ]]
}

IFS=',' read -ra SEED_ARR <<< "$SEEDS"
queued=0
skipped=0

log "Starting external_remaining  num_gpus=$NUM_GPUS  seeds=$SEEDS  log=$LOG"

for seed in "${SEED_ARR[@]}"; do
  for ds in fd hhar sleep; do
    for method in sprompts dualprompt eata cotta; do
      tag="${ds}_moment_adatime_${method}_c5_s${seed}"
      if done_chain "$tag"; then
        ((skipped++)) || true
        continue
      fi
      run scripts/run_chain_baselines.py --dataset "$ds" --tsfm moment \
        --method "$method" --domain_order adatime --calib_frac 0.05 --seed "$seed"
      ((queued++)) || true
    done
    tag="${ds}_moment_adatime_mapu_c5_s${seed}"
    if done_mapu "$tag"; then
      ((skipped++)) || true
      continue
    fi
    run scripts/run_mapu_baseline.py --dataset "$ds" --tsfm moment \
      --domain_order adatime --calib_frac 0.05 --seed "$seed"
    ((queued++)) || true
  done
done

wait_all
log "Queued $queued jobs, skipped $skipped already done"
"$PY" scripts/aggregate_experiment_matrix.py --phase external
log "External remaining complete. Log: $LOG"
