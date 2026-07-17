"""Compute per-domain sample counts and 5% calibration sizes for the supplement.

Example:
  python scripts/compute_dataset_stats.py
"""

import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import DATA_ROOT, RESULTS_ROOT

from src.data import get_spec
from src.data.calib import calib_indices
from src.data.datasets import adatime_chain


def main():
    rows = []
    for dataset in ("fd", "hhar", "sleep"):
        spec = get_spec(dataset)
        order, _ = adatime_chain(dataset)
        for did in order:
            lbl = spec.domain_label(did)
            for split in spec.splits:
                try:
                    n = len(spec.get_dataset(did, split, DATA_ROOT))
                except FileNotFoundError:
                    continue
                rows.append(dict(dataset=dataset, domain=lbl, split=split, n=n))
            ds = spec.get_dataset(did, "train", DATA_ROOT)
            y = ds.y.numpy()
            idx = calib_indices(y, 0.05, seed=42)
            rows.append(dict(dataset=dataset, domain=lbl, split="calib_5pct",
                             n=len(idx)))

    df = pd.DataFrame(rows)
    out = RESULTS_ROOT / "tables" / "sample_counts.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
