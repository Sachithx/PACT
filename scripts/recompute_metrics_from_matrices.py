"""Recompute AA/BWT/F/RF from saved retention matrices using the unified F definition."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import RESULTS_ROOT

from src.method.sequential import compute_cl_metrics


def recompute_dir(d: Path):
    mat_path = d / "soft_f1_matrix.csv"
    if not mat_path.exists():
        return None
    A = pd.read_csv(mat_path, index_col=0).values.astype(float)
    t = A.shape[0] - 1
    aa, bwt, f, rf, prf = compute_cl_metrics(A, t)
    return dict(path=str(d.relative_to(RESULTS_ROOT)), AA=aa, BWT=bwt, F=f, RF=rf, PRF=prf)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(RESULTS_ROOT / "sequential"))
    args = p.parse_args()
    rows = []
    for d in sorted(Path(args.root).rglob("soft_f1_matrix.csv")):
        r = recompute_dir(d.parent)
        if r:
            rows.append(r)
    if not rows:
        print("No matrices found.")
        return
    df = pd.DataFrame(rows)
    out = RESULTS_ROOT / "tables" / "recomputed_metrics.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
