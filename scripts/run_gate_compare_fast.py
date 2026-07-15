"""Fast gate-variant comparison: extract embeddings + train adapters ONCE,
then evaluate each gate variant using the same pre-computed artifacts.

This avoids re-running MOMENT forward passes for each gate kind."""

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

import _bootstrap  # noqa: F401
from _bootstrap import DATA_ROOT, RESULTS_ROOT

from src.data import DATASET_NAMES, get_spec
from src.data.calib import calib_indices
from src.data.datasets import adatime_chain
from src.method import PactConfig
from src.method.pact import extract_embs, train_adapters, eval_soft_mixture, eval_hard_gate, eval_oracle
from src.method.sequential import fit_gates, compute_cl_metrics, predict_soft_mixture
from src.models.concepts import assign_concepts, kmeans_numpy
from src.models.tsfm import make_tsfm, TSFM_MODELS
from src.utils import get_device, set_seed


GATE_KINDS = ["prototype", "prototype_diag", "prototype_lowrank", "confidence"]


def fit_standardizer_and_head(model, spec, src_id, device, epochs=150, lr=1e-3, batch=64):
    ld = torch.utils.data.DataLoader(
        spec.get_dataset(src_id, "train", DATA_ROOT), batch_size=batch)
    X, y = extract_embs(model, ld, device)
    mean, std = X.mean(0), X.std(0)
    model.set_standardizer(mean, std)
    Xs = (X - mean) / np.clip(std, 1e-6, None)

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


def evaluate_gate(gate_kind, src_embs, src_asgn, accumulated_calib, adapters_list,
                  frozen_head, test_embs, test_lbls, labels, centroids, device, cfg,
                  temperature=0.5):
    """Evaluate a single gate variant using pre-computed adapters and embeddings."""
    T = len(labels)
    A_soft = np.full((T, T), np.nan)
    metrics_rows = []

    # Step 0: source baseline (same for all gates)
    with torch.no_grad():
        et = torch.from_numpy(test_embs[labels[0]]).to(device)
        p0 = frozen_head(et).argmax(1).cpu().numpy()
    l0 = test_lbls[labels[0]]
    A_soft[0, 0] = f1_score(l0, p0, average="macro", zero_division=0)
    metrics_rows.append(dict(step=0, domain=labels[0], AA=A_soft[0, 0],
                             BWT=0.0, F_all=0.0, F_tgt=0.0, RF=0.0, PRF=0.0))

    for t in range(1, T):
        calib_up_to_t = accumulated_calib[:t]
        adapters_up_to_t = adapters_list[:t]

        cfg_gate = PactConfig(k=cfg.k, emb_dim=cfg.emb_dim, calib_frac=cfg.calib_frac,
                              seed=cfg.seed, batch=cfg.batch, gate_kind=gate_kind)
        soft_gates = fit_gates(src_embs, src_asgn, calib_up_to_t, device, cfg_gate,
                               adapters_list=adapters_up_to_t, frozen_head=frozen_head)

        for j in range(t + 1):
            lbl_j = labels[j]
            e_j, l_j = test_embs[lbl_j], test_lbls[lbl_j]
            A_soft[t, j] = eval_soft_mixture(e_j, l_j, centroids, soft_gates,
                                             adapters_up_to_t, frozen_head, device,
                                             cfg.k, temperature)

        AA, BWT, F_all, RF, PRF = compute_cl_metrics(A_soft, t, include_source=True)
        _, _, F_tgt, _, _ = compute_cl_metrics(A_soft, t, include_source=False)
        metrics_rows.append(dict(step=t, domain=labels[t], AA=AA, BWT=BWT,
                                 F_all=F_all, F_tgt=F_tgt, RF=RF, PRF=PRF))

    return A_soft, metrics_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=DATASET_NAMES, default="fd")
    p.add_argument("--tsfm", choices=list(TSFM_MODELS), default="moment")
    p.add_argument("--gate_kinds", nargs="+", default=GATE_KINDS)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--calib_frac", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    spec = get_spec(args.dataset)
    order, _ = adatime_chain(args.dataset)
    calib_tag = f"c{int(args.calib_frac*100)}"
    labels = [spec.domain_label(d) for d in order]
    T = len(order)

    print(f"=== Fast Gate Comparison: {args.dataset}/{args.tsfm}  K={args.k}  calib={calib_tag} ===")
    print(f"Gates to evaluate: {args.gate_kinds}")
    print(f"Chain: {' -> '.join(labels)}")
    print(f"Device: {device}\n")

    # ── Phase 1: Heavy lifting (done ONCE) ──
    t0 = time.time()
    model = make_tsfm(args.tsfm, spec.num_classes, device=str(device)).to(device)
    print(f"Model loaded in {time.time()-t0:.1f}s")

    src_acc = fit_standardizer_and_head(model, spec, order[0], device,
                                        batch=args.batch)
    print(f"Source head fitted: train acc={src_acc:.4f}")

    cfg = PactConfig(k=args.k, emb_dim=model.emb_dim, calib_frac=args.calib_frac,
                     seed=args.seed, batch=args.batch)
    frozen_head = model.head

    # Extract source embeddings + kmeans
    print("Extracting source embeddings...")
    src_ld = torch.utils.data.DataLoader(
        spec.get_dataset(order[0], "train", DATA_ROOT), batch_size=cfg.batch)
    src_embs, _ = extract_embs(model, src_ld, device)
    centroids, src_asgn = kmeans_numpy(src_embs, cfg.k, seed=cfg.seed)

    # Extract test embeddings for all domains
    print("Extracting test embeddings...")
    test_embs, test_lbls = {}, {}
    for did, lbl in zip(order, labels):
        ld = torch.utils.data.DataLoader(
            spec.get_dataset(did, "test", DATA_ROOT), batch_size=cfg.batch)
        test_embs[lbl], test_lbls[lbl] = extract_embs(model, ld, device)
        print(f"  {lbl}: {len(test_embs[lbl])} samples")

    # Extract calibration embeddings and train adapters for all target domains
    print("Training adapters (shared across all gate variants)...")
    accumulated_calib = []
    adapters_list = []
    for t in range(1, T):
        did_t = order[t]
        ds = spec.get_dataset(did_t, "train", DATA_ROOT)
        y_all = ds.y.numpy()
        idx = calib_indices(y_all, cfg.calib_frac, cfg.seed, calib_n=cfg.calib_n)
        calib_ld = torch.utils.data.DataLoader(
            torch.utils.data.Subset(ds, idx), batch_size=cfg.batch)
        calib_embs, _ = extract_embs(model, calib_ld, device)
        calib_lbls = y_all[idx]
        calib_asgn = assign_concepts(calib_embs, centroids)
        accumulated_calib.append((calib_embs, calib_asgn, calib_lbls))

        adapters = train_adapters(calib_embs, calib_lbls, centroids, frozen_head, device, cfg)
        adapters_list.append(adapters)
        print(f"  Step {t}: {labels[t]} adapters trained ({len(calib_embs)} calib samples)")

    phase1_time = time.time() - t0
    print(f"\nPhase 1 done in {phase1_time:.1f}s (embeddings + adapters)\n")

    # ── Phase 2: Gate-only evaluation (fast!) ──
    results_summary = []
    out_root = RESULTS_ROOT / "gate_compare"
    out_root.mkdir(parents=True, exist_ok=True)

    for gk in args.gate_kinds:
        print(f"{'─'*50}")
        print(f"  Evaluating gate_kind = {gk}")
        t1 = time.time()

        try:
            A_soft, metrics_rows = evaluate_gate(
                gk, src_embs, src_asgn, accumulated_calib, adapters_list,
                frozen_head, test_embs, test_lbls, labels, centroids, device, cfg,
                temperature=args.temperature)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ** FAILED: {e}")
            results_summary.append(dict(gate_kind=gk, AA=float("nan"), F=float("nan"),
                                        elapsed=0, error=str(e)))
            continue

        elapsed = time.time() - t1
        final = metrics_rows[-1]
        results_summary.append(dict(
            gate_kind=gk, AA=final["AA"], F=final["F_all"],
            BWT=final["BWT"], RF=final["RF"], PRF=final["PRF"],
            elapsed=elapsed, error=""))

        # Save per-variant results
        tag = f"{args.dataset}_{args.tsfm}_adatime_K{args.k}_{calib_tag}_{gk}"
        out_dir = RESULTS_ROOT / "sequential_tsfm" / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(metrics_rows).to_csv(out_dir / "metrics.csv", index=False)
        pd.DataFrame(A_soft, index=labels, columns=labels).to_csv(
            out_dir / "soft_f1_matrix.csv")

        for row in metrics_rows:
            step_str = f"    Step {row['step']}: {row['domain']:<10}"
            if row['step'] == 0:
                print(f"{step_str} F1={row['AA']:.4f}")
            else:
                print(f"{step_str} AA={row['AA']:.4f} F={row['F_all']:.4f}")
        print(f"  -> Final AA={final['AA']:.4f}  F={final['F_all']:.4f}  ({elapsed:.1f}s)\n")

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"  GATE COMPARISON SUMMARY: {args.dataset} / {args.tsfm}")
    print(f"{'='*70}")
    print(f"{'Gate Kind':<20} {'AA':>8} {'F':>8} {'BWT':>8} {'RF':>8} {'Time(s)':>8}")
    print("─" * 70)
    for r in results_summary:
        print(f"{r['gate_kind']:<20} {r.get('AA',0):>8.4f} {r.get('F',0):>8.4f} "
              f"{r.get('BWT',0):>+8.4f} {r.get('RF',0):>8.4f} {r.get('elapsed',0):>8.1f}")

    df = pd.DataFrame(results_summary)
    df.to_csv(out_root / f"{args.dataset}_{args.tsfm}_{calib_tag}_gate_compare.csv",
              index=False)
    print(f"\nTotal time: {time.time()-t0:.0f}s")
    print(f"Saved -> {out_root / f'{args.dataset}_{args.tsfm}_{calib_tag}_gate_compare.csv'}")


if __name__ == "__main__":
    main()
