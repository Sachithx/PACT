#!/usr/bin/env bash
# GPU1 campaign: MOMENT label-budget/K sweep + TSFM controls (no-adapt/no-gate).
set -u
cd "$(dirname "$0")/.."
PY=/home/AD/sachith/.conda/envs/entrope/bin/python
export LD_PRELOAD=/home/AD/sachith/.conda/envs/entrope/lib/libstdc++.so.6
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1

echo "===== [1/2] MOMENT FD label-budget + K sweep ====="
bash scripts/run_labelbudget_moment.sh

echo "===== [2/2] no-adapt + no-gate controls (moment, all datasets) ====="
for D in fd hhar sleep; do
  echo ">>> controls moment $D"
  $PY scripts/run_controls.py --dataset "$D" --backbone moment --domain_order adatime
done

echo "CAMPAIGN_GPU1_DONE"
