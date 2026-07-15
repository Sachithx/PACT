#!/usr/bin/env bash
# TSLANet FD ablation grid: vary one hyper-parameter at a time around the base
# config (K16, temp0.5, calib5%, identity, adatime chain). Fast (~75s/run).
set -u
cd "$(dirname "$0")/.."

PY=/home/AD/sachith/.conda/envs/entrope/bin/python
export LD_PRELOAD=/home/AD/sachith/.conda/envs/entrope/lib/libstdc++.so.6
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

run() { echo ">>> $*"; $PY scripts/run_sequential.py --dataset fd --backbone tslanet "$@"; }

# K sweep (base temp0.5 c5 identity)
for K in 4 8 32 64; do run --domain_order adatime --k "$K"; done

# Temperature sweep
for T in 0.1 0.3 1.0 2.0; do run --domain_order adatime --temperature "$T"; done

# Label-budget (calibration fraction) sweep
for C in 0.01 0.02 0.10 0.20; do run --domain_order adatime --calib_frac "$C"; done

# Adapter kind
run --domain_order adatime --adapter_kind lora --lora_rank 8

# Domain-order robustness (explicit orders on FD's 4 domains)
for O in "0,1,2,3" "3,2,1,0" "1,3,0,2" "2,0,3,1"; do run --domain_order "$O"; done

echo "ABLATION_GRID_DONE"
