"""Aggregate the random-9-domain-chain K-sweep (Sundial backbone) across
4 chains per dataset, for a given LoRA rank tag.

Example:
  python scripts/aggregate_kchains.py --rank 8
  python scripts/aggregate_kchains.py --rank 768 --suffix _r768
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
DATASETS = [("hhar", "c5"), ("sleep", "c5"), ("wisdm", "n450")]

with open("/tmp/random_chains.json") as f:
    CHAINS = json.load(f)


def load(ds, tag, chain, k, tag_suffix):
    path = (RESULTS_ROOT / "sequential_tsfm" /
            f"{ds}_{TSFM}_{chain}_K{k}_{tag}_lora_learned{tag_suffix}_s{SEED}" / "metrics.csv")
    if not path.exists():
        return None
    r = pd.read_csv(path).iloc[-1]
    return dict(AA=float(r["AA"]), BWT=float(r["BWT"]), F=float(r["F_all"]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag_suffix", default="", help="e.g. '' for rank=8 (default), or a custom tag")
    args = p.parse_args()

    rows = []
    print(f"{'dataset':<8}{'K':<5}{'mean_AA':<10}{'std_AA':<10}{'mean_F':<10}{'n_chains'}")
    for ds, tag in DATASETS:
        chains = CHAINS[ds]
        for k in KS:
            aas, fs = [], []
            for chain in chains:
                chain_str = ",".join(str(x) for x in chain)
                r = load(ds, tag, chain_str, k, args.tag_suffix)
                if r is not None:
                    aas.append(r["AA"])
                    fs.append(r["F"])
            if aas:
                print(f"{ds:<8}{k:<5}{np.mean(aas):<10.4f}{np.std(aas):<10.4f}{np.mean(fs):<10.4f}{len(aas)}")
                rows.append(dict(dataset=ds, K=k, mean_AA=np.mean(aas), std_AA=np.std(aas),
                                 mean_F=np.mean(fs), n=len(aas)))
            else:
                print(f"{ds:<8}{k:<5}{'--':<10}{'--':<10}{'--':<10}0")

    out = RESULTS_ROOT / "tables" / f"kchains_sweep{args.tag_suffix}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
