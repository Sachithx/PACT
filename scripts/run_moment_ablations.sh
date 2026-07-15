#!/usr/bin/env bash
# MOMENT hyperparameter sweeps for a given dataset (K + label-budget).
# Usage: run_moment_ablations.sh <dataset> <gpu>
set -u
cd "$(dirname "$0")/.."
PY=/home/AD/sachith/.conda/envs/entrope/bin/python
export LD_PRELOAD=/home/AD/sachith/.conda/envs/entrope/lib/libstdc++.so.6
export PYTHONUNBUFFERED=1
DS=$1; export CUDA_VISIBLE_DEVICES=$2
run() { echo ">>> $DS $*"; $PY scripts/run_sequential_tsfm.py --dataset "$DS" --tsfm moment "$@"; }
for K in 8 32; do run --domain_order adatime --k "$K"; done
for C in 0.01 0.02 0.10 0.20; do run --domain_order adatime --calib_frac "$C"; done
echo "MOMENT_ABLATIONS_${DS}_DONE"
