"""Aggregate the random-4-domain-chain K-sweep (Sundial backbone, FD dataset)
across 4 chains, for a given LoRA rank tag.

FD only has 4 domains total, so each chain is a full permutation (length 4,
no repeats). K=16 and K=32 exceed the domain count (more chart capacity than
domains).

Example:
  python scripts/aggregate_fdchains.py
  python scripts/aggregate_fdchains.py --tag_suffix _r768
"""

import argparse
import json

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import RESULTS_ROOT

TSFM = "sundial"
SEED = 42
KS = [1, 4, 8, 16, 32]
DATASET, TAG = "fd", "c5"

with open("/tmp/random_chains_fd.json") as f:
    CHAINS = json.load(f)["fd"]


def load(chain, k, tag_suffix):
    path = (RESULTS_ROOT / "sequential_tsfm" /
            f"{DATASET}_{TSFM}_{chain}_K{k}_{TAG}_lora_learned{tag_suffix}_s{SEED}" / "metrics.csv")
    if not path.exists():
        return None
    r = pd.read_csv(path).iloc[-1]
    return dict(AA=float(r["AA"]), BWT=float(r["BWT"]), F=float(r["F_all"]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag_suffix", default="", help="e.g. '' for rank=8 (default), or '_r768'")
    args = p.parse_args()

    rows = []
    print(f"{'K':<5}{'mean_AA':<10}{'std_AA':<10}{'mean_F':<10}{'n_chains'}")
    for k in KS:
        aas, fs = [], []
        for chain in CHAINS:
            chain_str = ",".join(str(x) for x in chain)
            r = load(chain_str, k, args.tag_suffix)
            if r is not None:
                aas.append(r["AA"])
                fs.append(r["F"])
        if aas:
            print(f"{k:<5}{np.mean(aas):<10.4f}{np.std(aas):<10.4f}{np.mean(fs):<10.4f}{len(aas)}")
            rows.append(dict(dataset=DATASET, K=k, mean_AA=np.mean(aas), std_AA=np.std(aas),
                             mean_F=np.mean(fs), n=len(aas)))
        else:
            print(f"{k:<5}{'--':<10}{'--':<10}{'--':<10}0")

    out = RESULTS_ROOT / "tables" / f"fdchains_sweep{args.tag_suffix}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
