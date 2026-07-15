"""Fast PACT-only reproduction check on the AdaTime scenarios.

Runs the *refactored* `src/` PACT (frozen backbone + concepts + per-domain
adapters + soft-mixture gate) over every AdaTime source->target shift, treating
each shift as a 2-domain sequential chain [source, target]. Reports per-scenario
AA / forgetting and the 10-scenario mean, then compares the mean against the
reference numbers from the original benchmark to confirm the clean tree
reproduces PACT's published results.

Usage:
  python scripts/validate_pact_scenarios.py                  # all datasets
  python scripts/validate_pact_scenarios.py --dataset fd     # single dataset
"""

import argparse
import time

import _bootstrap  # noqa: F401
from _bootstrap import CKPT_ROOT, DATA_ROOT

from src.data import get_spec
from src.data.datasets import SCENARIOS
from src.method import PactConfig, run_sequential
from src.models.factory import load_backbone
from src.utils import get_device, set_seed

# Reference PACT numbers (mean over 10 AdaTime scenarios) from the original run.
REFERENCE = {
    "fd":    {"AA": 0.981, "F": 0.000},
    "hhar":  {"AA": 0.963, "F": 0.001},
    "sleep": {"AA": 0.715, "F": 0.005},
}
TOL_AA = 0.02   # accept small run-to-run drift from k-means/gate init


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["fd", "hhar", "sleep", "all"], default="all")
    p.add_argument("--backbone", default="tslanet")
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--calib_frac", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    return p.parse_args()


def run_dataset(dataset, args, device):
    spec = get_spec(dataset)
    scen = SCENARIOS[dataset]
    print(f"\n{'='*72}\n{dataset.upper()}  —  {len(scen)} AdaTime scenarios  "
          f"(backbone={args.backbone}, K={args.k}, calib={args.calib_frac*100:.0f}%)"
          f"\n{'='*72}")
    print(f"  {'scenario':<18}{'AA':>10}{'F':>10}{'src->tgt F1':>16}")
    print(f"  {'-'*52}")

    aa_list, f_list = [], []
    for (src, tgt) in scen:
        order = [src, tgt]
        model = load_backbone(args.backbone, spec, CKPT_ROOT, dataset, src,
                              device, freeze=True)
        cfg = PactConfig(k=args.k, emb_dim=model.emb_dim,
                         calib_frac=args.calib_frac, seed=args.seed,
                         batch=spec.default_batch)
        res = run_sequential(model, spec, order, DATA_ROOT, device, cfg,
                             temperature=args.temperature, verbose=False)
        final = res["metrics"][-1]
        A = res["A_soft"]
        aa_list.append(final["AA"]); f_list.append(final["F_all"])
        name = f"{spec.domain_label(src)}->{spec.domain_label(tgt)}"
        print(f"  {name:<18}{final['AA']:>10.4f}{final['F_all']:>10.4f}"
              f"{A[1,0]:>8.3f}/{A[1,1]:<7.3f}")

    mean_aa = sum(aa_list) / len(aa_list)
    mean_f = sum(f_list) / len(f_list)
    ref = REFERENCE.get(dataset)
    print(f"  {'-'*52}")
    print(f"  {'MEAN (clean tree)':<18}{mean_aa:>10.4f}{mean_f:>10.4f}")
    if ref:
        d = mean_aa - ref["AA"]
        ok = abs(d) <= TOL_AA
        print(f"  {'REFERENCE (orig)':<18}{ref['AA']:>10.4f}{ref['F']:>10.4f}")
        print(f"  {'delta AA':<18}{d:>+10.4f}    "
              f"[{'PASS' if ok else 'CHECK'}  tol=±{TOL_AA}]")
    return mean_aa, mean_f


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    datasets = ["fd", "hhar", "sleep"] if args.dataset == "all" else [args.dataset]
    print(f"Device={device}  PACT reproduction check on AdaTime scenarios")
    t0 = time.time()
    summary = {}
    for ds in datasets:
        summary[ds] = run_dataset(ds, args, device)
    print(f"\n{'='*72}\nSUMMARY  (elapsed {time.time()-t0:.0f}s)\n{'='*72}")
    print(f"  {'dataset':<10}{'AA(clean)':>12}{'AA(ref)':>12}{'F(clean)':>12}")
    for ds, (aa, f) in summary.items():
        r = REFERENCE.get(ds, {})
        print(f"  {ds:<10}{aa:>12.4f}{r.get('AA', float('nan')):>12.4f}{f:>12.4f}")


if __name__ == "__main__":
    main()
