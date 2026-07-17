#!/usr/bin/env bash
# Config-based ablations that run entirely through the clean sequential driver.
# (DP and rehearsal-free variants have their own drivers under experiments/.)
#
# Usage:  bash scripts/run_ablations.sh [dataset]      # default: fd
set -euo pipefail

DATASET="${1:-fd}"
PY="${PYTHON:-python}"                       # override with PYTHON=/path/to/python
PRELOAD="${LD_PRELOAD:-}"                     # set LD_PRELOAD if your env needs libstdc++
export LD_PRELOAD="$PRELOAD"

run() { echo "=== $* ==="; $PY scripts/run_sequential.py "$@"; }

# 1. Backbone ablation (TSLANet vs PatchTST vs TimesNet)
for BB in tslanet patchtst timesnet; do
  run --dataset "$DATASET" --backbone "$BB"
done

# 2. Adapter ablation: full-rank identity vs LoRA
run --dataset "$DATASET" --adapter_kind identity
run --dataset "$DATASET" --adapter_kind lora --lora_rank 8

# 3. Number of charts K (include K=1 global-transport baseline)
for K in 1 4 8 16 32; do
  run --dataset "$DATASET" --k "$K"
done

# 4. Label budget (calibration fraction)
for C in 0.01 0.05 0.10 0.20 1.0; do
  run --dataset "$DATASET" --calib_frac "$C"
done

# 5. Softmax gate temperature (hard -> soft)
for T in 0.1 0.25 0.5 1.0 2.0; do
  run --dataset "$DATASET" --temperature "$T"
done

echo "All config-based ablations complete. See results/sequential/."
