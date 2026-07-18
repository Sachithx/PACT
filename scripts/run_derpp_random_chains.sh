#!/usr/bin/env bash
# Byte-matched Mammoth DER++ on the 4 random chains per dataset.
# Default: TSLANet full-FT DER++ (B12 primary) + Sundial head-only DER++.
set -uo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-/home/AD/sachith/.conda/envs/entrope/bin/python}"
export LD_PRELOAD="${LD_PRELOAD:-/home/AD/sachith/.conda/envs/entrope/lib/libstdc++.so.6}"
export PYTHONUNBUFFERED=1
NUM_GPUS="${NUM_GPUS:-2}"
SEED="${SEED:-42}"
CHAINS_JSON="${CHAINS_JSON:-experiments/random_chains.json}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="results/experiment_logs/derpp_chains_${TS}.log"
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
  local gpu
  gpu="$(_acquire_gpu)"
  log ">>> [cuda:$gpu] $*"
  "$PY" "$@" --device "cuda:$gpu" >>"$LOG" 2>&1 &
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

log "Computing byte-matched buffer sizes..."
"$PY" scripts/compute_derpp_buffer_size.py --chains_json "$CHAINS_JSON" | tee -a "$LOG"

# Parse chains with python (portable)
mapfile -t JOBS < <("$PY" - <<'PY'
import json
from pathlib import Path
chains = json.loads(Path("experiments/random_chains.json").read_text())
# (dataset, chain_str, calib_args...)
for ds, lst in chains.items():
    for ch in lst:
        cstr = ",".join(map(str, ch))
        if ds == "wisdm":
            print(f"{ds}|{cstr}|--calib_n|450")
        else:
            print(f"{ds}|{cstr}|--calib_frac|0.05")
PY
)

log "DER++ random-chains  num_gpus=$NUM_GPUS  n_jobs=$(( ${#JOBS[@]} * 2 ))"

for job in "${JOBS[@]}"; do
  IFS='|' read -r ds cstr carg cval <<<"$job"
  tag_tsl="results/derpp_baselines/${ds}_tslanet_${cstr}_derpp_*_s${SEED}/summary.csv"
  if compgen -G "$tag_tsl" > /dev/null; then
    log "skip tslanet $ds $cstr (done)"
  else
    run scripts/run_derpp_baseline.py --dataset "$ds" --backbone tslanet \
      --domain_order "$cstr" "$carg" "$cval" --seed "$SEED"
  fi
done

for job in "${JOBS[@]}"; do
  IFS='|' read -r ds cstr carg cval <<<"$job"
  tag_sun="results/derpp_baselines/${ds}_sundial_${cstr}_derpp_*_s${SEED}/summary.csv"
  if compgen -G "$tag_sun" > /dev/null; then
    log "skip sundial $ds $cstr (done)"
  else
    run scripts/run_derpp_baseline.py --dataset "$ds" --tsfm sundial \
      --domain_order "$cstr" "$carg" "$cval" --seed "$SEED" --freeze_backbone
  fi
done

wait_all
log "Done. Log: $LOG"
