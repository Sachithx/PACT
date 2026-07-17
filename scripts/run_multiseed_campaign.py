"""Run central experiments across multiple random seeds and aggregate mean±std.

Covers PACT sequential runs, embedding-replay baselines, and K ablations.
Results land under results/multiseed/<campaign>/seed_<s>/ and a summary CSV.

Example:
  python scripts/run_multiseed_campaign.py --campaign central --seeds 42,0,1
  python scripts/run_multiseed_campaign.py --campaign k_ablation --seeds 42,0,1,2,3
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import RESULTS_ROOT

ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--campaign", choices=["central", "k_ablation", "replay"], default="central")
    p.add_argument("--seeds", default="42,0,1,2,3",
                   help="Comma-separated seed list (3--5 recommended)")
    p.add_argument("--device", default=None)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def run_cmd(cmd, dry_run):
    print(" ".join(cmd))
    if dry_run:
        return 0
    return subprocess.call(cmd)


def jobs_for_campaign(campaign, device):
    dev = ["--device", device] if device else []
    if campaign == "central":
        specs = []
        for dataset in ("fd", "hhar", "sleep"):
            specs.append(["python", str(ROOT / "run_sequential.py"),
                          "--dataset", dataset, "--backbone", "tslanet",
                          "--domain_order", "adatime"] + dev)
            specs.append(["python", str(ROOT / "run_sequential_tsfm.py"),
                          "--dataset", dataset, "--tsfm", "moment",
                          "--domain_order", "adatime"] + dev)
        return specs
    if campaign == "k_ablation":
        specs = []
        for k in (1, 16):
            for dataset in ("fd", "hhar", "sleep"):
                specs.append(["python", str(ROOT / "run_sequential.py"),
                              "--dataset", dataset, "--backbone", "tslanet",
                              "--domain_order", "adatime", "--k", str(k)] + dev)
                specs.append(["python", str(ROOT / "run_sequential_tsfm.py"),
                              "--dataset", dataset, "--tsfm", "moment",
                              "--domain_order", "adatime", "--k", str(k)] + dev)
        return specs
    if campaign == "replay":
        specs = []
        for dataset in ("fd", "hhar", "sleep"):
            for method in ("emb-replay-head", "global-adapter"):
                specs.append(["python", str(ROOT / "run_embedding_replay.py"),
                              "--dataset", dataset, "--method", method,
                              "--backbone", "tslanet", "--domain_order", "adatime"] + dev)
                specs.append(["python", str(ROOT / "run_embedding_replay.py"),
                              "--dataset", dataset, "--method", method,
                              "--tsfm", "moment", "--domain_order", "adatime"] + dev)
        return specs
    raise ValueError(campaign)


def aggregate_campaign(campaign, seeds):
    rows = []
    for seed in seeds:
        seed_dir = RESULTS_ROOT / "multiseed" / campaign / f"seed_{seed}"
        if not seed_dir.exists():
            continue
        for metrics_path in seed_dir.rglob("metrics.csv"):
            rel = metrics_path.relative_to(seed_dir)
            m = pd.read_csv(metrics_path).iloc[-1 if "step" not in pd.read_csv(metrics_path).columns
                                               else -1]
            row = dict(campaign=campaign, seed=seed, path=str(rel.parent))
            if "AA" in m:
                row.update(AA=float(m["AA"]), F=float(m.get("F", m.get("F_all", np.nan))),
                           BWT=float(m.get("BWT", np.nan)))
            rows.append(row)
    if not rows:
        print("No metrics found to aggregate.")
        return
    df = pd.DataFrame(rows)
    summary = (df.groupby("path")
               .agg(AA_mean=("AA", "mean"), AA_std=("AA", "std"),
                    F_mean=("F", "mean"), F_std=("F", "std"),
                    n_seeds=("seed", "count"))
               .reset_index())
    out = RESULTS_ROOT / "multiseed" / f"{campaign}_summary.csv"
    summary.to_csv(out, index=False)
    print(f"Wrote {out}")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main():
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    jobs = jobs_for_campaign(args.campaign, args.device)

    for seed in seeds:
        seed_dir = RESULTS_ROOT / "multiseed" / args.campaign / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        for base_cmd in jobs:
            cmd = base_cmd + ["--seed", str(seed)]
            rc = run_cmd(cmd, args.dry_run)
            if rc != 0:
                print(f"Warning: seed={seed} command failed with code {rc}", file=sys.stderr)

    if not args.dry_run:
        aggregate_campaign(args.campaign, seeds)


if __name__ == "__main__":
    main()
