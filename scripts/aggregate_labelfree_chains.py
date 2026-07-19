"""Aggregate the genuinely label-free baselines (tent, mapu-free) across the
same 4 chains per dataset used everywhere else in the sundial random-chain
sweep (fd/hhar/sleep/wisdm/ucihar).

Example:
  python scripts/aggregate_labelfree_chains.py
"""

import json

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import RESULTS_ROOT

TSFM = "sundial"
SEED = 42
METHODS = ["tent", "mapu-free"]
DATASETS = [("fd", "c5"), ("hhar", "c5"), ("sleep", "c5"), ("wisdm", "n450"), ("ucihar", "n100")]

with open("/tmp/random_chains_fd.json") as f:
    FD_CHAINS = json.load(f)["fd"]
with open("/tmp/random_chains.json") as f:
    OTHER_CHAINS = json.load(f)
with open("/tmp/random_chains_ucihar.json") as f:
    UCIHAR_CHAINS = json.load(f)["ucihar"]


def chains_for(ds):
    if ds == "fd":
        return FD_CHAINS
    if ds == "ucihar":
        return UCIHAR_CHAINS
    return OTHER_CHAINS[ds]


def load(ds, chain, tag, method):
    path = (RESULTS_ROOT / "labelfree_baselines" /
            f"{ds}_{TSFM}_{chain}_{method}_{tag}_s{SEED}" / "summary.csv")
    if not path.exists():
        return None
    r = pd.read_csv(path).iloc[-1]
    return dict(AA=float(r["AA"]), RF=float(r["RF"]), PRF=float(r["PRF"]))


def main():
    rows = []
    print(f"{'method':<12}{'dataset':<8}{'mean_AA':<10}{'mean_RF':<10}{'mean_PRF':<10}{'n'}")
    for method in METHODS:
        for ds, tag in DATASETS:
            aas, rfs, prfs = [], [], []
            for chain in chains_for(ds):
                chain_str = ",".join(str(x) for x in chain)
                r = load(ds, chain_str, tag, method)
                if r is not None:
                    aas.append(r["AA"]); rfs.append(r["RF"]); prfs.append(r["PRF"])
            if aas:
                print(f"{method:<12}{ds:<8}{np.mean(aas):<10.4f}{np.mean(rfs):<10.4f}{np.mean(prfs):<10.4f}{len(aas)}")
                rows.append(dict(method=method, dataset=ds, mean_AA=np.mean(aas),
                                 mean_RF=np.mean(rfs), mean_PRF=np.mean(prfs), n=len(aas)))
            else:
                print(f"{method:<12}{ds:<8}{'--':<10}{'--':<10}{'--':<10}0")

    out = RESULTS_ROOT / "tables" / "labelfree_chains_sweep.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
