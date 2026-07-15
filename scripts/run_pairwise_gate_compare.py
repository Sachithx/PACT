"""Compare all gate variants (learned MLP + rehearsal-free prototype family +
confidence routing) on the standard AdaTime PAIRWISE scenarios (single
source->target shift each), as opposed to the sequential domain-incremental
chain in run_sequential.py. Each pair is its own 2-domain "chain": step 0 =
source only, step 1 = target adapted in. BWT/F at step 1 is exactly "did
adapting to the target damage the source" for that one pair.

Usage: python scripts/run_pairwise_gate_compare.py --dataset fd --backbone tslanet
"""

import argparse
import time

import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import CKPT_ROOT, DATA_ROOT, RESULTS_ROOT

from src.data import get_spec
from src.data.datasets import SCENARIOS
from src.method import PactConfig, run_sequential
from src.models.factory import BACKBONES, load_backbone
from src.training import train_or_load_source
from src.utils import get_device, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["fd", "hhar", "sleep"], default="fd")
    p.add_argument("--backbone", choices=BACKBONES, default="tslanet")
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--calib_frac", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--gate_kinds", nargs="+",
                   default=["learned", "prototype", "prototype_masked",
                            "prototype_diag", "prototype_lowrank",
                            "prototype_mixture", "analytic", "confidence"])
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    spec = get_spec(args.dataset)
    pairs = SCENARIOS[args.dataset]
    print(f"Device={device}  dataset={args.dataset}  backbone={args.backbone}  "
         f"{len(pairs)} AdaTime pairs x {len(args.gate_kinds)} gate kinds "
         f"({', '.join(args.gate_kinds)})")

    rows = []
    models = {}
    for gate_kind in args.gate_kinds:
        for s, t in pairs:
            if s not in models:
                train_or_load_source(args.backbone, args.dataset, s, CKPT_ROOT, DATA_ROOT, device)
                models[s] = load_backbone(args.backbone, spec, CKPT_ROOT, args.dataset, s,
                                          device, freeze=True)
            model = models[s]
            cfg = PactConfig(k=args.k, emb_dim=model.emb_dim, calib_frac=args.calib_frac,
                             seed=args.seed, gate_kind=gate_kind)
            t0 = time.time()
            res = run_sequential(model, spec, [s, t], DATA_ROOT, device, cfg,
                                 temperature=args.temperature, verbose=False)
            dt = time.time() - t0
            final = res["metrics"][-1]
            rows.append(dict(gate_kind=gate_kind, source=spec.domain_label(s),
                             target=spec.domain_label(t), AA=final["AA"], BWT=final["BWT"],
                             F_all=final["F_all"], F_tgt=final["F_tgt"], RF=final["RF"],
                             PRF=final["PRF"], seconds=dt))
            print(f"  [{gate_kind:>9}] {spec.domain_label(s)}->{spec.domain_label(t)}  "
                 f"AA={final['AA']:.4f}  BWT={final['BWT']:+.4f}  F_all={final['F_all']:.4f}  "
                 f"({dt:.1f}s)")

    df = pd.DataFrame(rows)
    out_dir = RESULTS_ROOT / "gate_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}_{args.backbone}_pairwise.csv"
    df.to_csv(out_path, index=False)

    print("\n=== Summary (mean over pairs) ===")
    summary = df.groupby("gate_kind")[["AA", "BWT", "F_all", "F_tgt", "RF", "PRF", "seconds"]].mean()
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
