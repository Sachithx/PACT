#!/usr/bin/env bash
# Re-run missing B7 WISDM (Sundial + TSLANet) and S-Prompts only (dual GPU).
set -uo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-/home/AD/sachith/.conda/envs/entrope/bin/python}"
export LD_PRELOAD="${LD_PRELOAD:-/home/AD/sachith/.conda/envs/entrope/lib/libstdc++.so.6}"
export PYTHONUNBUFFERED=1
NUM_GPUS="${NUM_GPUS:-2}"
SEEDS="${SEEDS:-42,0,1}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="results/experiment_logs/b7_sprompts_${TS}.log"
mkdir -p results/experiment_logs

declare -a GPU_PID=()
for ((g=0; g<NUM_GPUS; g++)); do GPU_PID[$g]=""; done
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG" >&2; }

_reap_finished() {
  local g pid ec
  for ((g=0; g<NUM_GPUS; g++)); do
    pid="${GPU_PID[$g]:-}"; [[ -z "$pid" ]] && continue
    if ! kill -0 "$pid" 2>/dev/null; then
      ec=0; wait "$pid" || ec=$?
      [[ "$ec" -ne 0 ]] && log "FAILED (exit $ec) on cuda:$g"
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
    [[ "$skip" -eq 1 ]] && { skip=0; continue; }
    [[ "$a" == "--device" ]] && { skip=1; continue; }
    args+=("$a")
  done
  log ">>> [cuda:$gpu] ${args[*]}"
  "$PY" "${args[@]}" --device "cuda:$gpu" >>"$LOG" 2>&1 &
  GPU_PID[$gpu]=$!
}

wait_all() {
  local g pid ec
  for ((g=0; g<NUM_GPUS; g++)); do
    pid="${GPU_PID[$g]:-}"; [[ -z "$pid" ]] && continue
    ec=0; wait "$pid" || ec=$?
    [[ "$ec" -ne 0 ]] && log "FAILED (exit $ec) on cuda:$g"
    GPU_PID[$g]=""
  done
}

IFS=',' read -ra SEED_ARR <<< "$SEEDS"
log "B7+sprompts rerun  num_gpus=$NUM_GPUS  seeds=$SEEDS"

for seed in "${SEED_ARR[@]}"; do
  pact_tag="wisdm_sundial_adatime_K16_c5_lora_learned_s${seed}"
  if [[ -f "results/sequential_tsfm/${pact_tag}/metrics.csv" ]]; then
    log "skip sundial PACT $pact_tag (done)"
  else
    run scripts/run_sequential_tsfm.py --dataset wisdm --tsfm sundial \
      --domain_order adatime --seed "$seed" --adapter_kind lora
  fi
  for method in head-ft lwf srcdistill; do
    bl_tag="wisdm_sundial_adatime_${method}_c5_s${seed}"
    if [[ -f "results/cl_baselines_tsfm/${bl_tag}/summary.csv" ]]; then
      log "skip sundial $method s${seed} (done)"
      continue
    fi
    run scripts/run_cl_baselines_tsfm.py --dataset wisdm --tsfm sundial \
      --method "$method" --domain_order adatime --calib_frac 0.05 --seed "$seed"
  done
  tsl_tag="wisdm_tslanet_adatime_K16_c5_t0.5_lora_learned_s${seed}"
  if [[ -f "results/sequential/${tsl_tag}/metrics.csv" ]]; then
    log "skip tslanet PACT $tsl_tag (done)"
  else
    run scripts/run_sequential.py --dataset wisdm --backbone tslanet \
      --domain_order adatime --seed "$seed"
  fi
done

for seed in "${SEED_ARR[@]}"; do
  for ds in fd hhar sleep; do
    tag="${ds}_moment_adatime_sprompts_c5_s${seed}"
    if [[ -f "results/chain_baselines/${tag}/summary.csv" ]]; then
      log "skip sprompts $tag (done)"
      continue
    fi
    run scripts/run_chain_baselines.py --dataset "$ds" --tsfm moment \
      --method sprompts --domain_order adatime --calib_frac 0.05 --seed "$seed"
  done
done

wait_all
"$PY" scripts/aggregate_experiment_matrix.py --phase b7
"$PY" scripts/aggregate_experiment_matrix.py --phase external
log "Done. Log: $LOG"
