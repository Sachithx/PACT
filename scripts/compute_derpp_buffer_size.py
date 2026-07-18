"""Compute byte-matched DER++ buffer_size vs PACT embedding buffer.

PACT buffer (default learned gate) = float32 embedding keys for
  source-train + all target calibration slices (no class labels).
DER++ slot = float32 raw window + int64 label + float32 logits.

Example:
  python scripts/compute_derpp_buffer_size.py
  python scripts/compute_derpp_buffer_size.py --chains_json experiments/random_chains.json
"""

import argparse
import json
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import DATA_ROOT, RESULTS_ROOT

from src.data import get_spec
from src.data.calib import calib_indices
from src.data.datasets import adatime_chain
from src.method.derpp import (bytes_per_derpp_slot, matched_buffer_size,
                              pact_buffer_bytes)

# Embedding dims used in this repo
EMB_DIM = {
    "tslanet": 128,
    "moment": 1024,
    "sundial": 768,
}


def calib_count(spec, did, calib_frac, seed, calib_n=None):
    ds = spec.get_dataset(did, "train", DATA_ROOT)
    y = ds.y.numpy()
    return len(calib_indices(y, calib_frac, seed, calib_n=calib_n))


def budget_for_chain(dataset, order, backbone, calib_frac, calib_n, seed):
    spec = get_spec(dataset)
    emb_dim = EMB_DIM[backbone]
    n_src = len(spec.get_dataset(order[0], "train", DATA_ROOT))
    n_calib = sum(
        calib_count(spec, did, calib_frac, seed, calib_n)
        for did in order[1:]
    )
    pact_b = pact_buffer_bytes(emb_dim, n_src, n_calib)
    slot = bytes_per_derpp_slot(spec.num_channels, spec.seq_len, spec.num_classes)
    buf = matched_buffer_size(emb_dim, n_src, n_calib,
                              spec.num_channels, spec.seq_len, spec.num_classes)
    return dict(
        dataset=dataset, backbone=backbone, chain=",".join(map(str, order)),
        T=len(order), emb_dim=emb_dim, n_src=n_src, n_calib=n_calib,
        pact_buffer_mb=pact_b / (1024 ** 2),
        derpp_slot_bytes=slot,
        buffer_size=buf,
        derpp_buffer_mb=(buf * slot) / (1024 ** 2),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chains_json", default="experiments/random_chains.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--backbones", default="tslanet,moment,sundial")
    args = p.parse_args()

    chains = json.loads(Path(args.chains_json).read_text())
    rows = []
    for ds, ds_chains in chains.items():
        calib_frac, calib_n = (0.05, None) if ds != "wisdm" else (1.0, 450)
        for bb in args.backbones.split(","):
            for order in ds_chains:
                rows.append(budget_for_chain(ds, order, bb, calib_frac, calib_n, args.seed))
            # also adatime chain for reference
            try:
                order, _ = adatime_chain(ds)
                r = budget_for_chain(ds, order, bb, calib_frac, calib_n, args.seed)
                r["chain"] = "adatime:" + r["chain"]
                rows.append(r)
            except Exception:
                pass

    df = pd.DataFrame(rows)
    out = RESULTS_ROOT / "tables" / "derpp_byte_matched_buffer.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    # Per-dataset mean buffer_size (over the 4 random chains)
    summary = (df[~df["chain"].astype(str).str.startswith("adatime")]
               .groupby(["dataset", "backbone"], as_index=False)
               .agg(buffer_size_mean=("buffer_size", "mean"),
                    buffer_size_min=("buffer_size", "min"),
                    buffer_size_max=("buffer_size", "max"),
                    pact_mb_mean=("pact_buffer_mb", "mean")))
    print(summary.to_string(index=False))
    print(f"\nFull table -> {out}")
    sum_path = RESULTS_ROOT / "tables" / "derpp_byte_matched_buffer_summary.csv"
    summary.to_csv(sum_path, index=False)
    print(f"Summary   -> {sum_path}")


if __name__ == "__main__":
    main()
