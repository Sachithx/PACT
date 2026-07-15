#!/usr/bin/env bash
# Table 3 extension: PACT chain runs for the new large TSFM backbones
# (sundial, time-moe, moirai) on FD/HHAR/Sleep-EDF. Runs in the `pact2`
# conda env (entrope's momentfm pins transformers==4.33.3, incompatible
# with these backbones' transformers==4.40.1 requirement).
#
# Usage: campaign_table3_newtsfm.sh <gpu_id> <job1> <job2> ...
#   each job is "backbone:dataset", e.g. sundial:hhar
set -u
cd "$(dirname "$0")/.."
PY=/home/AD/sachith/.conda/envs/pact2/bin/python
export PYTHONUNBUFFERED=1
GPU=$1; shift
export CUDA_VISIBLE_DEVICES=$GPU

for JOB in "$@"; do
  BACKBONE="${JOB%%:*}"
  DATASET="${JOB##*:}"
  echo ">>> [gpu$GPU] $BACKBONE / $DATASET"
  $PY scripts/run_sequential_tsfm.py --dataset "$DATASET" --tsfm "$BACKBONE" \
    --domain_order adatime --k 16 --calib_frac 0.05
done
echo "CAMPAIGN_TABLE3_NEWTSFM_GPU${GPU}_DONE"
