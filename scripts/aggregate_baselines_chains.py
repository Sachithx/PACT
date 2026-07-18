"""Aggregate the 7 CL baselines (dualcd, emb-replay-head, srcdistill, ewc, lwf,
head-ft, source-only) across the same 4 random chains per dataset used for the
PACT K-sweep (fd/hhar/sleep/wisdm), Sundial backbone, 5% calib (450 for wisdm).

Example:
  python scripts/aggregate_baselines_chains.py
"""

import json

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import RESULTS_ROOT

TSFM = "sundial"
SEED = 42
METHODS = ["dualcd", "emb-replay-head", "srcdistill", "ewc", "lwf", "head-ft", "source-only"]
DATASETS = [("fd", "c5"), ("hhar", "c5"), ("sleep", "c5"), ("wisdm", "n450")]

with open("/tmp/random_chains_fd.json") as f:
    FD_CHAINS = json.load(f)["fd"]
with open("/tmp/random_chains.json") as f:
    OTHER_CHAINS = json.load(f)


def chains_for(ds):
    return FD_CHAINS if ds == "fd" else OTHER_CHAINS[ds]


def load_head(ds, chain, tag, method):
    path = (RESULTS_ROOT / "cl_baselines_tsfm" /
            f"{ds}_{TSFM}_{chain}_{method}_{tag}_s{SEED}" / "summary.csv")
    if not path.exists():
        return None
    r = pd.read_csv(path).iloc[-1]
    return dict(AA=float(r["AA"]), F=float(r["F"]))


def load_embrep(ds, chain, tag):
    path = (RESULTS_ROOT / "embedding_replay" /
            f"{ds}_{TSFM}_{chain}_emb-replay-head_{tag}_s{SEED}" / "metrics.csv")
    if not path.exists():
        return None
    r = pd.read_csv(path).iloc[-1]
    return dict(AA=float(r["AA"]), F=float(r["F"]))


def load_srconly(ds, chain, tag):
    path = (RESULTS_ROOT / "controls" / f"{ds}_{TSFM}_{chain}_K16_{tag}" / "summary.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path)
    row = df[df["control"] == "no-adapt"]
    if row.empty:
        return None
    r = row.iloc[0]
    return dict(AA=float(r["AA"]), F=float(r["F"]))


def load(ds, chain, tag, method):
    if method == "emb-replay-head":
        return load_embrep(ds, chain, tag)
    if method == "source-only":
        return load_srconly(ds, chain, tag)
    return load_head(ds, chain, tag, method)


def main():
    rows = []
    print(f"{'method':<18}{'dataset':<8}{'mean_AA':<10}{'std_AA':<10}{'mean_F':<10}{'n'}")
    for method in METHODS:
        for ds, tag in DATASETS:
            aas, fs = [], []
            for chain in chains_for(ds):
                chain_str = ",".join(str(x) for x in chain)
                r = load(ds, chain_str, tag, method)
                if r is not None:
                    aas.append(r["AA"])
                    fs.append(r["F"])
            if aas:
                print(f"{method:<18}{ds:<8}{np.mean(aas):<10.4f}{np.std(aas):<10.4f}{np.mean(fs):<10.4f}{len(aas)}")
                rows.append(dict(method=method, dataset=ds, mean_AA=np.mean(aas),
                                 std_AA=np.std(aas), mean_F=np.mean(fs), n=len(aas)))
            else:
                print(f"{method:<18}{ds:<8}{'--':<10}{'--':<10}{'--':<10}0")

    out = RESULTS_ROOT / "tables" / "baselines_chains_sweep.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
