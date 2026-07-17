#!/usr/bin/env bash
# Priority pre-submission experiments (review feedback order).
# Requires: datasets under dataset/, PYTHON pointing to env with torch.
#
# Usage:
#   PYTHON=/path/to/python bash scripts/campaign_priority.sh [phase]
# Phases: multiseed | k_ablation | replay | latency | all
set -euo pipefail

PY="${PYTHON:-python}"
PHASE="${1:-all}"
SEEDS="${SEEDS:-42,0,1,2,3}"
DEVICE="${DEVICE:-cuda}"

run_phase() {
  case "$1" in
    multiseed)
      $PY scripts/run_multiseed_campaign.py --campaign central --seeds "$SEEDS" --device "$DEVICE"
      ;;
    k_ablation)
      $PY scripts/run_multiseed_campaign.py --campaign k_ablation --seeds "$SEEDS" --device "$DEVICE"
      ;;
    replay)
      $PY scripts/run_multiseed_campaign.py --campaign replay --seeds "$SEEDS" --device "$DEVICE"
      ;;
    latency)
      $PY scripts/profile_latency.py --device "$DEVICE"
      $PY scripts/compute_dataset_stats.py
      $PY scripts/emit_sample_counts_tex.py
      ;;
    all)
      run_phase multiseed
      run_phase k_ablation
      run_phase replay
      run_phase latency
      ;;
    *) echo "Unknown phase: $1"; exit 1 ;;
  esac
}

run_phase "$PHASE"
echo "Campaign phase '$PHASE' complete."
