#!/usr/bin/env bash
# Master experiment queue (audit section B). Paper edits (A) deferred.
#
# Usage:
#   bash scripts/run_experiment_matrix.sh [phase|all|remainder]
#   SEEDS=42,0,1 NUM_GPUS=2 bash scripts/run_experiment_matrix.sh b6
#
# NUM_GPUS (default 2) launches up to one job per GPU in parallel via --device cuda:N.
# Logs: results/experiment_logs/<phase>_<timestamp>.log
set -uo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-/home/AD/sachith/.conda/envs/entrope/bin/python}"
export LD_PRELOAD="${LD_PRELOAD:-/home/AD/sachith/.conda/envs/entrope/lib/libstdc++.so.6}"
export PYTHONUNBUFFERED=1
SEEDS="${SEEDS:-42,0,1}"
NUM_GPUS="${NUM_GPUS:-2}"
PHASE="${1:-all}"
TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="results/experiment_logs"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/${PHASE}_${TS}.log"

declare -a GPU_PID=()
for ((g=0; g<NUM_GPUS; g++)); do GPU_PID[$g]=""; done

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG" >&2; }

_reap_finished() {
  local g pid ec
  for ((g=0; g<NUM_GPUS; g++)); do
    pid="${GPU_PID[$g]:-}"
    [[ -z "$pid" ]] && continue
    if ! kill -0 "$pid" 2>/dev/null; then
      ec=0
      wait "$pid" || ec=$?
      if [[ "$ec" -ne 0 ]]; then
        log "FAILED (exit $ec) on cuda:$g (pid $pid)"
      fi
      GPU_PID[$g]=""
    fi
  done
}

_acquire_gpu() {
  local g
  while true; do
    _reap_finished
    for ((g=0; g<NUM_GPUS; g++)); do
      if [[ -z "${GPU_PID[$g]:-}" ]]; then
        echo "$g"
        return 0
      fi
    done
    sleep 2
  done
}

# Strip any caller --device and assign a free GPU.
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
    ec=0
    wait "$pid" || ec=$?
    if [[ "$ec" -ne 0 ]]; then
      log "FAILED (exit $ec) on cuda:$g (pid $pid)"
    fi
    GPU_PID[$g]=""
  done
}

run_sync() {
  log ">>> $*"
  if ! "$PY" "$@" >>"$LOG" 2>&1; then
    log "FAILED (exit $?): $*"
    return 1
  fi
  return 0
}

IFS=',' read -ra SEED_ARR <<< "$SEEDS"

# ── B1: multi-seed PACT + top-3 baselines ─────────────────────────────────────
phase_b1() {
  local ds method seed
  for seed in "${SEED_ARR[@]}"; do
    for ds in fd hhar sleep; do
      run scripts/run_sequential.py --dataset "$ds" --backbone tslanet \
        --domain_order adatime --seed "$seed"
      run scripts/run_sequential_tsfm.py --dataset "$ds" --tsfm moment \
        --domain_order adatime --seed "$seed" --adapter_kind lora
      for method in head-ft lwf srcdistill; do
        run scripts/run_cl_baselines.py --dataset "$ds" --backbone tslanet \
          --method "$method" --domain_order adatime --calib_frac 0.05 \
          --seed "$seed"
        run scripts/run_cl_baselines_tsfm.py --dataset "$ds" --tsfm moment \
          --method "$method" --domain_order adatime --calib_frac 0.05 \
          --seed "$seed"
      done
    done
  done
  wait_all
  run_sync scripts/aggregate_experiment_matrix.py --phase b1
}

# ── B5: K=1 chain + NCM + embedding replay ────────────────────────────────────
phase_b5() {
  local ds seed method
  for seed in "${SEED_ARR[@]}"; do
    for ds in fd hhar sleep; do
      run scripts/run_sequential.py --dataset "$ds" --backbone tslanet \
        --domain_order adatime --k 1 --seed "$seed"
      run scripts/run_sequential_tsfm.py --dataset "$ds" --tsfm moment \
        --domain_order adatime --k 1 --seed "$seed" --adapter_kind lora
      run scripts/run_ncm_baseline.py --dataset "$ds" --backbone tslanet \
        --seed "$seed"
      run scripts/run_ncm_baseline.py --dataset "$ds" --tsfm moment \
        --seed "$seed"
      for method in emb-replay-head global-adapter; do
        run scripts/run_embedding_replay.py --dataset "$ds" --method "$method" \
          --backbone tslanet --seed "$seed"
        run scripts/run_embedding_replay.py --dataset "$ds" --method "$method" \
          --tsfm moment --seed "$seed"
      done
    done
  done
  wait_all
  run_sync scripts/aggregate_experiment_matrix.py --phase b5
}

# ── B5 retry: MOMENT NCM + embedding replay only (after head_ep fix) ──────────
enqueue_b5_moment() {
  local ds seed method
  for seed in "${SEED_ARR[@]}"; do
    for ds in fd hhar sleep; do
      run scripts/run_ncm_baseline.py --dataset "$ds" --tsfm moment --seed "$seed"
      for method in emb-replay-head global-adapter; do
        run scripts/run_embedding_replay.py --dataset "$ds" --method "$method" \
          --tsfm moment --seed "$seed"
      done
    done
  done
}

phase_b5_moment() {
  enqueue_b5_moment
  wait_all
  run_sync scripts/aggregate_experiment_matrix.py --phase b5
}

# ── B6: EWC/MAS head on HHAR + Sleep MOMENT ────────────────────────────────────
enqueue_b6() {
  local ds seed method
  for seed in "${SEED_ARR[@]}"; do
    for ds in hhar sleep; do
      for method in ewc mas; do
        run scripts/run_cl_baselines_tsfm.py --dataset "$ds" --tsfm moment \
          --method "$method" --domain_order adatime --calib_frac 0.05 \
          --seed "$seed"
      done
    done
  done
}

phase_b6() {
  enqueue_b6
  wait_all
}

# ── B7: WISDM chain ───────────────────────────────────────────────────────────
enqueue_b7() {
  local seed tsfm method
  for seed in "${SEED_ARR[@]}"; do
    for tsfm in moment sundial; do
      run scripts/run_sequential_tsfm.py --dataset wisdm --tsfm "$tsfm" \
        --domain_order adatime --seed "$seed" --adapter_kind lora
      for method in head-ft lwf srcdistill; do
        run scripts/run_cl_baselines_tsfm.py --dataset wisdm --tsfm "$tsfm" \
          --method "$method" --domain_order adatime --calib_frac 0.05 \
          --seed "$seed"
      done
    done
    run scripts/run_sequential.py --dataset wisdm --backbone tslanet \
      --domain_order adatime --seed "$seed"
  done
}

phase_b7() {
  enqueue_b7
  wait_all
}

# ── B8: rank-8 MOMENT full sweep (explicit lora tag) ──────────────────────────
enqueue_b8() {
  local ds seed
  for seed in "${SEED_ARR[@]}"; do
    for ds in fd hhar sleep; do
      run scripts/run_sequential_tsfm.py --dataset "$ds" --tsfm moment \
        --domain_order adatime --adapter_kind lora --lora_rank 8 \
        --seed "$seed"
      run scripts/run_sequential_tsfm.py --dataset "$ds" --tsfm moment \
        --domain_order adatime --adapter_kind identity \
        --seed "$seed"
    done
  done
}

phase_b8() {
  enqueue_b8
  wait_all
}

# ── B9: rehearsal-free gates on MOMENT ────────────────────────────────────────
enqueue_b9() {
  local ds seed gk
  for seed in "${SEED_ARR[@]}"; do
    for ds in fd hhar sleep; do
      for gk in analytic prototype prototype_diag; do
        run scripts/run_sequential_tsfm.py --dataset "$ds" --tsfm moment \
          --domain_order adatime --gate_kind "$gk" --adapter_kind lora \
          --seed "$seed"
      done
    done
  done
}

phase_b9() {
  enqueue_b9
  wait_all
}

# ── B13: latency / buffer profiling ───────────────────────────────────────────
phase_b13() {
  run_sync scripts/profile_latency.py --device cuda:0 --seed 42
  run_sync scripts/compute_dataset_stats.py
  run_sync scripts/emit_sample_counts_tex.py
}

# ── External baselines (PACT-compatible wrappers) ─────────────────────────────
enqueue_external() {
  local ds seed method
  for seed in "${SEED_ARR[@]}"; do
    for ds in fd hhar sleep; do
      for method in sprompts dualprompt eata cotta; do
        run scripts/run_chain_baselines.py --dataset "$ds" --tsfm moment \
          --method "$method" --domain_order adatime --calib_frac 0.05 \
          --seed "$seed"
      done
      run scripts/run_mapu_baseline.py --dataset "$ds" --tsfm moment \
        --domain_order adatime --calib_frac 0.05 --seed "$seed"
    done
  done
}

phase_external() {
  enqueue_external
  wait_all
  run_sync scripts/aggregate_experiment_matrix.py --phase external
  log "B2 DualCD: integrated in run_cl_baselines_tsfm.py (--method dualcd); separate campaign may be running"
  log "B11 Mantis/UniTS: deferred — requires separate FM encoder campaign"
}

phase_fixes() {
  enqueue_b7
  wait_all
  run_sync scripts/aggregate_experiment_matrix.py --phase b7
  phase_b13
  enqueue_external
  wait_all
  run_sync scripts/aggregate_experiment_matrix.py --phase external
  log "Fixes complete (B7 WISDM re-run, B13 latency, external baselines)"
}

phase_remainder() {
  enqueue_b5_moment
  enqueue_b6
  enqueue_b7
  enqueue_b8
  enqueue_b9
  wait_all
  run_sync scripts/aggregate_experiment_matrix.py --phase b5
  phase_b13
  phase_external
}

log "Starting phase=$PHASE  seeds=$SEEDS  num_gpus=$NUM_GPUS  log=$LOG"

case "$PHASE" in
  b1) phase_b1 ;;
  b5) phase_b5 ;;
  b5_moment) phase_b5_moment ;;
  b6) phase_b6 ;;
  b7) phase_b7 ;;
  b8) phase_b8 ;;
  b9) phase_b9 ;;
  b13) phase_b13 ;;
  external) phase_external ;;
  fixes) phase_fixes ;;
  remainder) phase_remainder ;;
  all)
    phase_b1
    phase_b5
    phase_b6
    phase_b7
    phase_b8
    phase_b9
    phase_b13
    phase_external
    ;;
  *)
    echo "Unknown phase: $PHASE (b1|b5|b5_moment|b6|b7|b8|b9|b13|external|fixes|remainder|all)"
    exit 1
    ;;
esac

log "Phase $PHASE complete. Log: $LOG"
