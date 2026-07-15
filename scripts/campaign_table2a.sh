#!/usr/bin/env bash
# Table 2A: continual baselines on TSLANet chain (5% calib, matched to PACT).
set -u
cd "$(dirname "$0")/.."
PY=/home/AD/sachith/.conda/envs/entrope/bin/python
export LD_PRELOAD=/home/AD/sachith/.conda/envs/entrope/lib/libstdc++.so.6
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

run() {
  echo ">>> Table2A $*"
  $PY scripts/run_cl_baselines.py --backbone tslanet --domain_order adatime \
    --calib_frac 0.05 "$@"
}

for DS in fd hhar sleep; do
  for M in full-ft head-ft ewc mas lwf srcdistill; do
    run --dataset "$DS" --method "$M"
  done
  for M in ortcl-lf ortcl-headft; do
    echo ">>> Table2A ORTCL $DS $M"
    $PY scripts/run_cl_baselines_ortcl.py --backbone tslanet --domain_order adatime \
      --calib_frac 0.05 --dataset "$DS" --method "$M"
  done
done
echo "TABLE2A_DONE"
