#!/usr/bin/env bash
# GPU0 campaign: TSLANet ablation grid + small-backbone controls + full-FT.
set -u
cd "$(dirname "$0")/.."
PY=/home/AD/sachith/.conda/envs/entrope/bin/python
export LD_PRELOAD=/home/AD/sachith/.conda/envs/entrope/lib/libstdc++.so.6
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

echo "===== [1/3] TSLANet FD ablation grid ====="
bash scripts/run_ablation_grid.sh

echo "===== [2/3] no-adapt + no-gate controls (tslanet, all datasets) ====="
for D in fd hhar sleep; do
  echo ">>> controls tslanet $D"
  $PY scripts/run_controls.py --dataset "$D" --backbone tslanet --domain_order adatime
done

echo "===== [3/3] full fine-tune per domain (tslanet, all datasets) ====="
for D in fd hhar sleep; do
  echo ">>> full-ft tslanet $D"
  $PY scripts/run_full_ft.py --dataset "$D" --backbone tslanet --domain_order adatime --epochs 20
done

echo "CAMPAIGN_GPU0_DONE"
