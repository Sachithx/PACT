"""Sequential domain-incremental CL with PACT on a FROZEN TSFM backbone (MOMENT).

Flow (matches the paper's story):
  1. Load a pretrained TSFM, keep the encoder FROZEN.
  2. Fit a linear classification head on the SOURCE domain's frozen embeddings
     ("load pretrained TSFM, fit on source, freeze everything").
  3. Apply PACT (concepts + per-domain adapters + soft-mixture gate) over the
     domain stream exactly as for the small backbones -- the method is
     embedding-space and backbone-agnostic.

Examples:
  python scripts/run_sequential_tsfm.py --dataset fd --domain_order adatime
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

import _bootstrap  # noqa: F401
from _bootstrap import DATA_ROOT, RESULTS_ROOT

from src.data import DATASET_NAMES, get_spec
from src.data.datasets import adatime_chain
from src.method import PactConfig, run_sequential
from src.method.pact import extract_embs
from src.models.tsfm import make_tsfm, TSFM_MODELS
from src.utils import get_device, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["fd", "hhar", "sleep", "wisdm"], default="fd")
    p.add_argument("--tsfm", choices=list(TSFM_MODELS), default="moment")
    p.add_argument("--domain_order", default="adatime",
                   help="'adatime', comma-separated ids, or blank for 0..N")
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--calib_frac", type=float, default=0.05)
    p.add_argument("--calib_n", type=int, default=None,
                   help="Fixed calibration label budget (overrides calib_frac)")
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--adapter_kind", choices=["identity", "lora"], default="lora")
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--gate_kind", choices=["learned", "prototype", "prototype_diag",
                                            "prototype_lowrank", "prototype_masked",
                                            "prototype_mixture", "analytic", "confidence"],
                   default="learned")
    p.add_argument("--head_ep", type=int, default=150)
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--device", default=None)
    return p.parse_args()


def fit_standardizer_and_head(model, spec, src_id, device, epochs, lr, batch, seed):
    """Fit the z-score standardizer AND the linear head on frozen source
    embeddings (single embedding pass), then freeze the head."""
    ld = torch.utils.data.DataLoader(
        spec.get_dataset(src_id, "train", DATA_ROOT), batch_size=batch)
    X, y = extract_embs(model, ld, device)          # raw embeddings (no std yet)
    mean, std = X.mean(0), X.std(0)
    model.set_standardizer(mean, std)
    Xs = (X - mean) / np.clip(std, 1e-6, None)      # standardized source embs

    Xt = torch.from_numpy(Xs).to(device)
    yt = torch.from_numpy(y).long().to(device)
    opt = torch.optim.AdamW(model.head.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xt, yt), batch_size=256, shuffle=True)
    model.head.train()
    for _ in range(epochs):
        for eb, yb in dl:
            opt.zero_grad()
            loss = crit(model.head(eb), yb)
            loss.backward(); opt.step()
    model.freeze_head()
    with torch.no_grad():
        acc = (model.head(Xt).argmax(1) == yt).float().mean().item()
    return acc


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    spec = get_spec(args.dataset)

    if args.domain_order == "adatime":
        order, st = adatime_chain(args.dataset)
        print(f"AdaTime chain: {st['n_domains']} domains, "
              f"{st['forward']}/{st['n_pairs']} pairs forward")
    elif args.domain_order:
        order = [int(x) for x in args.domain_order.split(",")]
    else:
        order = list(range(spec.n_domains))

    calib_tag = f"n{args.calib_n}" if args.calib_n else f"c{int(args.calib_frac*100)}"
    print(f"Device={device}  tsfm={args.tsfm}(frozen)  dataset={args.dataset}"
          f"  T={len(order)}  K={args.k}  calib={calib_tag}")
    t0 = time.time()
    model = make_tsfm(args.tsfm, spec.num_classes, device=str(device)).to(device)
    print(f"{args.tsfm} loaded in {time.time()-t0:.1f}s  (emb_dim={model.emb_dim})")

    src_acc = fit_standardizer_and_head(model, spec, order[0], device,
                                        args.head_ep, args.head_lr, args.batch, args.seed)
    print(f"Standardizer + linear head fitted on source: train acc={src_acc:.4f}  "
          f"(source={spec.domain_label(order[0])}, multi_window={model.multi_window})")

    cfg = PactConfig(k=args.k, emb_dim=model.emb_dim, calib_frac=args.calib_frac,
                     calib_n=args.calib_n, seed=args.seed, batch=args.batch,
                     adapter_kind=args.adapter_kind, lora_rank=args.lora_rank,
                     gate_kind=args.gate_kind)
    res = run_sequential(model, spec, order, DATA_ROOT, device, cfg,
                         temperature=args.temperature, verbose=True)

    final = res["metrics"][-1]
    print(f"\nFinal (PACT on frozen {args.tsfm}): AA={final['AA']:.4f}  "
          f"BWT={final['BWT']:+.4f}  F_all={final['F_all']:.4f}  "
          f"RF={final['RF']:.4f}  PRF={final['PRF']:.4f}")

    tag = (f"{args.dataset}_{args.tsfm}_{args.domain_order}_K{args.k}"
           f"_{calib_tag}_{args.adapter_kind}_{args.gate_kind}_s{args.seed}")
    out_dir = RESULTS_ROOT / "sequential_tsfm" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = res["labels"]
    import pandas as pd
    pd.DataFrame(res["metrics"]).to_csv(out_dir / "metrics.csv", index=False)
    pd.DataFrame(res["A_soft"], index=labels, columns=labels).to_csv(
        out_dir / "soft_f1_matrix.csv")
    pd.DataFrame(res["A_orc"], index=labels, columns=labels).to_csv(
        out_dir / "oracle_f1_matrix.csv")
    print(f"Results -> {out_dir}")
    print(f"Elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
