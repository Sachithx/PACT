"""Aggregate multi-seed experiment outputs into mean±std CSVs.

Example:
  python scripts/aggregate_experiment_matrix.py --phase b1
  python scripts/aggregate_experiment_matrix.py --phase all
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import RESULTS_ROOT

PHASE_DIRS = {
    "b1": [
        RESULTS_ROOT / "sequential",
        RESULTS_ROOT / "sequential_tsfm",
        RESULTS_ROOT / "cl_baselines",
        RESULTS_ROOT / "cl_baselines_tsfm",
    ],
    "b5": [
        RESULTS_ROOT / "sequential",
        RESULTS_ROOT / "sequential_tsfm",
        RESULTS_ROOT / "ncm_baselines",
        RESULTS_ROOT / "embedding_replay",
    ],
    "b6": [RESULTS_ROOT / "cl_baselines_tsfm"],
    "b7": [
        RESULTS_ROOT / "sequential",
        RESULTS_ROOT / "sequential_tsfm",
        RESULTS_ROOT / "cl_baselines_tsfm",
    ],
    "b8": [RESULTS_ROOT / "sequential_tsfm"],
    "b9": [RESULTS_ROOT / "sequential_tsfm"],
    "external": [
        RESULTS_ROOT / "chain_baselines",
        RESULTS_ROOT / "mapu_baselines",
    ],
}


def load_metric(path: Path):
    if (path / "metrics.csv").exists():
        m = pd.read_csv(path / "metrics.csv").iloc[-1]
        return dict(AA=float(m["AA"]), F=float(m.get("F_all", m.get("F", np.nan))),
                    BWT=float(m.get("BWT", np.nan)), seed=_seed_from_name(path.name))
    if (path / "summary.csv").exists():
        m = pd.read_csv(path / "summary.csv").iloc[0]
        return dict(AA=float(m["AA"]), F=float(m["F"]),
                    BWT=float(m.get("BWT", np.nan)), seed=int(m.get("seed", _seed_from_name(path.name))))
    return None


def _seed_from_name(name: str) -> int:
    m = re.search(r"_s(\d+)$", name)
    return int(m.group(1)) if m else -1


def aggregate_dirs(dirs, min_seeds=1):
    rows = []
    for root in dirs:
        if not root.exists():
            continue
        for sub in sorted(root.iterdir()):
            if not sub.is_dir():
                continue
            rec = load_metric(sub)
            if rec:
                rows.append(dict(config=sub.name, root=root.name, **rec))
    if not rows:
        return None
    df = pd.DataFrame(rows)
    base = df["config"].str.replace(r"_s\d+$", "", regex=True)
    df["config_base"] = base
    summary = (df.groupby(["root", "config_base"])
               .agg(AA_mean=("AA", "mean"), AA_std=("AA", "std"),
                    F_mean=("F", "mean"), F_std=("F", "std"),
                    n_seeds=("seed", "nunique"))
               .reset_index())
    summary = summary[summary["n_seeds"] >= min_seeds]
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", default="all")
    p.add_argument("--min_seeds", type=int, default=1)
    args = p.parse_args()
    phases = list(PHASE_DIRS) if args.phase == "all" else [args.phase]
    out_root = RESULTS_ROOT / "experiment_matrix"
    out_root.mkdir(parents=True, exist_ok=True)
    for ph in phases:
        summary = aggregate_dirs(PHASE_DIRS.get(ph, []), args.min_seeds)
        if summary is None:
            print(f"{ph}: no results")
            continue
        out = out_root / f"{ph}_summary.csv"
        summary.to_csv(out, index=False)
        print(f"Wrote {out}\n{summary.to_string(index=False)}")


if __name__ == "__main__":
    main()
