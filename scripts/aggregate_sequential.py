"""Aggregate all sequential-CL runs into one paper-ready table.

Scans results/sequential (small backbones) and results/sequential_tsfm (TSFMs)
for metrics.csv files and prints final-step AA/BWT/F_all/RF/PRF per
(dataset, backbone). Also writes a CSV.
"""

import re
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import RESULTS_ROOT

DATASETS = ["fd", "hhar", "sleep"]


def collect():
    rows = []
    for sub, kind in [("sequential", "small"), ("sequential_tsfm", "tsfm")]:
        root = RESULTS_ROOT / sub
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            mfile = d / "metrics.csv"
            if not mfile.exists():
                continue
            name = d.name
            ds = next((x for x in DATASETS if name.startswith(x + "_")), None)
            if ds is None:
                continue
            rest = name[len(ds) + 1:]
            backbone = rest.split("_")[0]
            m = pd.read_csv(mfile).iloc[-1]
            rows.append(dict(dataset=ds, backbone=backbone, tag=name,
                             T=int(m["step"]) + 1, AA=m["AA"], BWT=m["BWT"],
                             F_all=m["F_all"], RF=m.get("RF", float("nan")),
                             PRF=m.get("PRF", float("nan"))))
    return pd.DataFrame(rows)


def main():
    df = collect()
    if df.empty:
        print("No runs found yet.")
        return
    df = df.sort_values(["dataset", "backbone"]).reset_index(drop=True)
    print(f"{'dataset':<7}{'backbone':<12}{'T':>3}{'AA':>9}{'BWT':>9}"
          f"{'F_all':>9}{'RF':>9}{'PRF':>9}")
    print("-" * 66)
    for _, r in df.iterrows():
        print(f"{r['dataset']:<7}{r['backbone']:<12}{r['T']:>3}{r['AA']:>9.4f}"
              f"{r['BWT']:>+9.4f}{r['F_all']:>9.4f}{r['RF']:>9.4f}{r['PRF']:>9.4f}")
    out = RESULTS_ROOT / "sequential_summary.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
