#!/usr/bin/env bash
# Table 2B: head-level continual baselines on frozen MOMENT chain.
set -u
cd "$(dirname "$0")/.."
PY=/home/AD/sachith/.conda/envs/entrope/bin/python
export LD_PRELOAD=/home/AD/sachith/.conda/envs/entrope/lib/libstdc++.so.6
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

run() {
  echo ">>> Table2B $*"
  $PY scripts/run_cl_baselines_tsfm.py --tsfm moment --domain_order adatime \
    --calib_frac 0.05 "$@"
}

for DS in fd hhar sleep; do
  for M in head-ft lwf srcdistill; do
    run --dataset "$DS" --method "$M"
  done
done
echo "TABLE2B_DONE"
