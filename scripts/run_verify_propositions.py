"""Empirical verification of Propositions 1 and 2 on one representative PACT run.

Proposition 1 (margin-bounded non-degradation) is already computed by
run_sequential() for source-domain test embeddings
(res["margin_diagnostics"]): checks (1-pi_0(z))B(z) < m(z)/(2L).

Proposition 2 (target recovery under chart relatedness) is computed here per
target domain via target_recovery_diagnostics(), reusing the fitted
centroids, adapters, gates, and frozen head from the same run_sequential()
call: checks eps_c(z) + 2(1-pi_own(z))B(z) < gamma(z)/(2L).

Both propositions are sufficient conditions, so the key empirical check is
that "condition satisfied" implies "actually correct" -- i.e. the violation
rate (satisfied but wrong) should be at or near zero.

Example:
  python scripts/run_verify_propositions.py --dataset fd --tsfm sundial \
      --domain_order 2,1,0,3 --k 8 --calib_frac 0.05 \
      --adapter_kind lora --lora_rank 8
"""

import argparse

import numpy as np
import pandas as pd
import torch

import _bootstrap  # noqa: F401
from _bootstrap import DATA_ROOT, RESULTS_ROOT

from src.data import get_spec
from src.method import PactConfig, run_sequential
from src.method.pact import extract_embs
from src.method.sequential import target_recovery_diagnostics
from src.models.tsfm import make_tsfm, TSFM_MODELS
from src.utils import get_device, set_seed
from run_sequential_tsfm import fit_standardizer_and_head


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["fd", "hhar", "sleep", "wisdm", "ucihar"], default="fd")
    p.add_argument("--tsfm", choices=list(TSFM_MODELS), default="sundial")
    p.add_argument("--domain_order", required=True)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--calib_frac", type=float, default=0.05)
    p.add_argument("--calib_n", type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--adapter_kind", choices=["identity", "lora"], default="lora")
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--gate_kind", default="learned")
    p.add_argument("--head_ep", type=int, default=150)
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    spec = get_spec(args.dataset)
    order = [int(x) for x in args.domain_order.split(",")]

    model = make_tsfm(args.tsfm, spec.num_classes, device=str(device)).to(device)
    fit_standardizer_and_head(model, spec, order[0], device,
                              args.head_ep, args.head_lr, args.batch, args.seed)

    cfg = PactConfig(k=args.k, emb_dim=model.emb_dim, calib_frac=args.calib_frac,
                     calib_n=args.calib_n, seed=args.seed, batch=args.batch,
                     adapter_kind=args.adapter_kind, lora_rank=args.lora_rank,
                     gate_kind=args.gate_kind)
    res = run_sequential(model, spec, order, DATA_ROOT, device, cfg,
                         temperature=args.temperature, verbose=True)

    calib_tag = f"n{args.calib_n}" if args.calib_n else f"c{int(args.calib_frac*100)}"
    tag = f"{args.dataset}_{args.tsfm}_{args.domain_order}_K{args.k}_{calib_tag}_s{args.seed}"
    out_dir = RESULTS_ROOT / "prop_verification"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Proposition 1: source non-degradation ------------------------------
    md = res["margin_diagnostics"]
    df1 = pd.DataFrame(md)
    df1.to_csv(out_dir / f"{tag}_prop1.csv", index=False)
    correct1 = df1["correct"].astype(bool)
    sat1 = df1.loc[correct1, "displacement_bound"] < df1.loc[correct1, "margin_threshold"]
    viol1 = correct1.sum() - sat1.sum()
    print(f"\n[Prop 1] source test samples={len(df1)}  correctly-classified={int(correct1.sum())}")
    print(f"[Prop 1] condition satisfied={sat1.mean()*100:.1f}%  violated={100-sat1.mean()*100:.1f}%"
          f"  (n_violations={int(viol1)})")

    # --- Proposition 2: target recovery, one domain at a time ---------------
    labels = res["labels"]
    src_ld = torch.utils.data.DataLoader(
        spec.get_dataset(order[0], "train", DATA_ROOT), batch_size=cfg.batch)
    src_embs, src_lbls = extract_embs(model, src_ld, device)

    frames = []
    for t in range(1, len(order)):
        test_ld = torch.utils.data.DataLoader(
            spec.get_dataset(order[t], "test", DATA_ROOT), batch_size=cfg.batch)
        tgt_embs, tgt_lbls = extract_embs(model, test_ld, device)
        diag = target_recovery_diagnostics(
            tgt_embs, tgt_lbls, own_domain_rank=t,
            src_embs_np=src_embs, src_lbls_np=src_lbls,
            centroids=res["centroids"], soft_gates=res["soft_gates"],
            adapters_list=res["adapters_list"], frozen_head=model.head,
            device=device, temperature=args.temperature, k=args.k)
        df_t = pd.DataFrame(diag)
        df_t["domain"] = labels[t]
        frames.append(df_t)
    df2 = pd.concat(frames, ignore_index=True)
    df2.to_csv(out_dir / f"{tag}_prop2.csv", index=False)

    has_anchor = df2["has_anchor"].astype(bool)
    n2 = int(has_anchor.sum())
    sat2 = df2.loc[has_anchor, "lhs"] < df2.loc[has_anchor, "rhs"]
    deployed_correct2 = df2.loc[has_anchor, "deployed_correct"].astype(bool)
    viol2 = int((sat2 & ~deployed_correct2).sum())
    print(f"\n[Prop 2] target test samples with an eligible same-chart same-label "
          f"source anchor={n2}/{len(df2)}")
    print(f"[Prop 2] condition satisfied={sat2.mean()*100:.1f}%  "
          f"deployed-correct among satisfied={deployed_correct2[sat2].mean()*100:.1f}%  "
          f"violations (satisfied but wrong)={viol2}")
    print(f"[Prop 2] deployed accuracy overall (all target test points)="
          f"{df2['deployed_correct'].mean()*100:.1f}%")

    print(f"\nSaved -> {out_dir}/{tag}_prop1.csv , {tag}_prop2.csv")


if __name__ == "__main__":
    main()
