#!/usr/bin/env bash
# MOMENT (frozen TSFM) label-budget curve on FD: vary calibration fraction.
# Base c5 already exists; this fills in c1/c2/c10/c20. Also K sweep on MOMENT FD.
set -u
cd "$(dirname "$0")/.."

PY=/home/AD/sachith/.conda/envs/entrope/bin/python
export LD_PRELOAD=/home/AD/sachith/.conda/envs/entrope/lib/libstdc++.so.6
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

run() { echo ">>> $*"; $PY scripts/run_sequential_tsfm.py --dataset fd --tsfm moment "$@"; }

# Label-budget (calibration fraction) curve
for C in 0.01 0.02 0.10 0.20; do run --domain_order adatime --calib_frac "$C"; done

# K sweep on MOMENT FD
for K in 8 32; do run --domain_order adatime --k "$K"; done

echo "LABELBUDGET_MOMENT_DONE"
