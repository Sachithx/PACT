"""
run_exp8_rf_seqpact.py — Exp8: RF-SeqPACT (Rehearsal-Free Sequential PACT)
============================================================================

Rehearsal-free sequential domain-incremental adaptation.

Key constraints:
  · No raw time-series samples stored after training.
  · No calibration embeddings stored after stats are computed.
  · No accumulated_calib, no train_soft_gates, no rerun of k-means.
  · Source centroids fixed for all domains.
  · Old adapter banks frozen and never revisited.

Per-domain compact state:
  · adapter bank (frozen after training)
  · per-concept: mu, radius, conformal threshold q, count

Three routing modes (--router):
  distance   — argmin normalized distance to domain stats
  uncertainty — argmin frozen-head entropy after candidate correction
  conformal  — accept routes where composite score <= q_{j,k};
               fallback to lowest score if none accepted

Usage:
  python run_exp8_rf_seqpact.py --dataset fd --domain_order 0,1,2,3 --router conformal
  python run_exp8_rf_seqpact.py --dataset hhar --calib_frac 0.05 --router distance
"""

import sys
import importlib.util
import argparse

import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).parent
sys.path.insert(0, str(REPO / "TSLANet" / "Classification"))


def _load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e7  = _load_mod("e7",  REPO / "exp7_soft_gate.py")
tft = _load_mod("tft", REPO / "transfer_fd_tslanet.py")
tht = _load_mod("tht", REPO / "transfer_hhar_tslanet.py")

kmeans_numpy     = e7.kmeans_numpy
assign_concepts  = e7.assign_concepts
extract_embs     = e7.extract_embs
train_adapters   = e7.train_adapters
eval_oracle      = e7.eval_oracle
Args             = e7.Args
FD_MAP           = e7.FD_MAP
DATA_ROOT        = e7.DATA_ROOT
CKPT_DIR         = e7.CKPT_DIR


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",           choices=["fd", "hhar"], default="hhar")
    p.add_argument("--domain_order",      default=None,
                   help="Comma-separated domain IDs, first = source. "
                        "Default FD: 0,1,2,3  HHAR: 0,1,...,8")
    p.add_argument("--k",                 type=int,   default=16)
    p.add_argument("--calib_frac",        type=float, default=0.05)
    p.add_argument("--adapter_ep",        type=int,   default=200)
    p.add_argument("--batch",             type=int,   default=256)
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--radius_percentile", type=float, default=95.0)
    p.add_argument("--conformal_alpha",   type=float, default=0.10,
                   help="Conformal significance level (reject rate target)")
    p.add_argument("--router",
                   choices=["distance", "uncertainty", "score", "entropy_gain", "conformal"],
                   default="conformal",
                   help="distance: argmin dist | uncertainty: argmin entropy | "
                        "score: argmin composite (no thresholds) | "
                        "entropy_gain: route that most reduces entropy vs bypass | "
                        "conformal: composite + calibrated thresholds")
    p.add_argument("--score_alpha",           type=float, default=1.0,
                   help="Distance component weight")
    p.add_argument("--score_beta",            type=float, default=0.5,
                   help="Entropy component weight")
    p.add_argument("--score_gamma",           type=float, default=0.1,
                   help="Correction magnitude component weight")
    p.add_argument("--source_bias",           type=float, default=0.95,
                   help="Multiply source route score by this (<1 favours source)")
    p.add_argument("--min_count",             type=int,   default=3,
                   help="Min samples per concept to mark as available")
    p.add_argument("--entropy_gain_threshold", type=float, default=0.05,
                   help="Min entropy reduction ΔH = H_bypass - H_route to accept a non-source route "
                        "(entropy_gain router only)")
    p.add_argument("--reuse_threshold",       type=float, default=0.0,
                   help="Calibration F1 threshold for route reuse: if an existing route "
                        "achieves >= this F1 on the new domain's calibration set, "
                        "reuse it instead of training a new adapter (0 = disabled)")
    p.add_argument("--eval_oracle",           action="store_true",
                   help="Also evaluate oracle routing (uses eval_oracle from e7)")
    return p.parse_args()


def build_config(args):
    if args.dataset == "fd":
        order  = ([int(x) for x in args.domain_order.split(",")]
                  if args.domain_order else [0, 1, 2, 3])
        src_id = order[0]
        ckpt   = CKPT_DIR / f"tslanet_{FD_MAP[src_id]}_ps8_d2_dim128.pt"

        def load_model(device):
            m = tft._make_model(Args()).to(device)
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            m.load_state_dict(state["model_state_dict"])
            m.eval()
            for p in m.parameters(): p.requires_grad_(False)
            return m

        get_train = lambda did: tft.FDDataset(FD_MAP[did], "train", DATA_ROOT)
        get_test  = lambda did: tft.FDDataset(FD_MAP[did], "test",  DATA_ROOT)
        labels    = [FD_MAP[i] for i in order]

    elif args.dataset == "hhar":
        order  = ([int(x) for x in args.domain_order.split(",")]
                  if args.domain_order else list(range(9)))
        src_id = order[0]

        def load_model(device):
            m = tht.train_source(src_id, Args(), device)
            m.eval()
            for p in m.parameters(): p.requires_grad_(False)
            return m

        get_train = lambda did: tht.HHARDataset(did, "train", DATA_ROOT)
        get_test  = lambda did: tht.HHARDataset(did, "test",  DATA_ROOT)
        labels    = [f"User-{i}" for i in order]

    else:
        raise ValueError(args.dataset)

    tag = (f"{args.dataset}_{'_'.join(str(i) for i in order)}"
           f"_K{args.k}_c{int(args.calib_frac*100)}_{args.router}")
    return dict(load_model=load_model, get_train=get_train, get_test=get_test,
                domain_order=order, domain_labels=labels, out_tag=tag)


# ── CL metrics (per-domain F, includes source) ────────────────────────────────

def compute_cl_metrics(A, t, include_source=True):
    seen    = t + 1
    AA      = float(np.nanmean(A[t, :seen]))
    start_j = 0 if include_source else 1

    bwt_vals, f_vals = [], []
    for j in range(start_j, seen):
        if j == t or np.isnan(A[j, j]) or np.isnan(A[t, j]):
            continue
        delta = A[t, j] - A[j, j]
        bwt_vals.append(delta)
        f_vals.append(max(0.0, -delta))

    BWT = float(np.mean(bwt_vals)) if bwt_vals else 0.0
    F   = float(np.mean(f_vals))   if f_vals   else 0.0

    rf_list, prf_list = [], []
    for j in range(start_j, seen):
        if j == t or np.isnan(A[t, j]):
            continue
        peak_vals = [A[i, j] for i in range(j, t) if not np.isnan(A[i, j])]
        if not peak_vals:
            continue
        peak_j = max(peak_vals)
        drop   = max(0.0, peak_j - A[t, j])
        rf_j   = drop / (peak_j + 1e-9)
        prf_j  = (1.0 - A[t, j]) * rf_j
        rf_list.append(rf_j); prf_list.append(prf_j)

    RF  = float(np.mean(rf_list))  if rf_list  else 0.0
    PRF = float(np.mean(prf_list)) if prf_list else 0.0
    return AA, BWT, F, RF, PRF


# ── Compact domain statistics ─────────────────────────────────────────────────

def compute_domain_stats(embs_np, asgn_np, centroids, K,
                         radius_percentile=95.0, min_count=3):
    """Per-concept mu and radius. Does not retain embs_np after return."""
    stats = {}
    for k in range(K):
        mask = (asgn_np == k)
        cnt  = int(mask.sum())
        if cnt < min_count:
            stats[k] = {"available": False, "count": cnt}
            continue
        mu    = embs_np[mask].mean(0)
        dists = np.linalg.norm(embs_np[mask] - mu, axis=1)
        r     = float(np.percentile(dists, radius_percentile))
        stats[k] = {"available": True, "mu": mu.astype(np.float32),
                    "radius": r, "count": cnt}
    return stats


# ── Conformal score calibration ───────────────────────────────────────────────

@torch.no_grad()
def compute_calibration_scores(embs_np, asgn_np, stats, adapters,
                                frozen_head, device, args, is_source=False):
    """Compute per-concept composite scores for conformal threshold setting.

    is_source: if True, adapters=None and correction_norm=0.
    Returns dict[k -> np.array of scores].
    """
    K      = args.k
    embs_t = torch.from_numpy(embs_np).to(device)
    scores_by_k = {}

    for k in range(K):
        stat_k = stats.get(k, {"available": False})
        if not stat_k["available"]:
            continue
        mask = (asgn_np == k)
        if not mask.any():
            continue

        mu    = stat_k["mu"]
        r     = stat_k["radius"]
        batch = embs_t[mask]
        embs_k = embs_np[mask]

        dists = np.linalg.norm(embs_k - mu, axis=1) / (r + 1e-6)

        if is_source:
            probs    = torch.softmax(frozen_head(batch), dim=-1).cpu().numpy()
            entropy  = -(probs * np.log(probs + 1e-8)).sum(axis=1)
            corr_n   = np.zeros(len(dists), dtype=np.float32)
        else:
            adapters[k].eval()
            z_cand   = adapters[k](batch)
            probs    = torch.softmax(frozen_head(z_cand), dim=-1).cpu().numpy()
            entropy  = -(probs * np.log(probs + 1e-8)).sum(axis=1)
            corr_n   = torch.norm(z_cand - batch, dim=-1).cpu().numpy()

        scores_by_k[k] = (args.score_alpha * dists
                          + args.score_beta * entropy
                          + args.score_gamma * corr_n)
    return scores_by_k


def set_conformal_thresholds(stats, scores_by_k, args):
    """Attach q_{k} to stats in-place."""
    for k, stat_k in stats.items():
        if not stat_k["available"]:
            continue
        sc = scores_by_k.get(k)
        if sc is not None and len(sc) >= args.min_count:
            stat_k["q"] = float(np.quantile(sc, 1.0 - args.conformal_alpha))
        else:
            stat_k["q"] = float("inf")


# ── Rehearsal-free router + evaluation ────────────────────────────────────────

@torch.no_grad()
def eval_rf_router(embs_np, labels_np, centroids, domain_stats_list,
                   adapters_list, frozen_head, device, args):
    """Evaluate with the rehearsal-free hard router.

    Returns (macro_f1, accuracy, route_counts [n_routes], reject_rate).
    """
    n        = len(embs_np)
    n_routes = len(domain_stats_list)   # 0 = source, 1..t = targets
    K        = args.k
    eps      = 1e-6

    asgn   = assign_concepts(embs_np, centroids)
    embs_t = torch.from_numpy(embs_np).to(device)

    # Build per-route score components
    dist_arr    = np.full((n_routes, n), np.inf, dtype=np.float64)
    entropy_arr = np.full((n_routes, n), np.inf, dtype=np.float64)
    corr_arr    = np.zeros((n_routes, n),         dtype=np.float64)

    for j, stats_j in enumerate(domain_stats_list):
        for k in range(K):
            stat_jk = stats_j.get(k, {"available": False})
            if not stat_jk.get("available", False):
                continue
            mask = (asgn == k)
            if not mask.any():
                continue

            mu = stat_jk["mu"]
            r  = stat_jk["radius"]

            d = np.linalg.norm(embs_np[mask] - mu, axis=1) / (r + eps)
            dist_arr[j, mask] = d

            batch = embs_t[mask]
            if j == 0:
                z_cand = batch
            else:
                adapters_list[j - 1][k].eval()
                z_cand = adapters_list[j - 1][k](batch)

            probs = torch.softmax(frozen_head(z_cand), dim=-1).cpu().numpy()
            entropy_arr[j, mask] = -(probs * np.log(probs + 1e-8)).sum(axis=1)

            if j > 0:
                corr_arr[j, mask] = torch.norm(z_cand - batch, dim=-1).cpu().numpy()

    # Source bias on distance row 0
    finite0 = dist_arr[0] < np.inf
    dist_arr[0, finite0] *= args.source_bias

    # Mode-specific routing
    rejected = np.zeros(n, dtype=bool)

    if args.router == "distance":
        routes = (args.score_alpha * dist_arr).argmin(axis=0)

    elif args.router == "uncertainty":
        routes = (args.score_beta * entropy_arr).argmin(axis=0)

    elif args.router == "score":
        composite = (args.score_alpha * dist_arr
                     + args.score_beta * entropy_arr
                     + args.score_gamma * corr_arr)
        routes = composite.argmin(axis=0)

    elif args.router == "entropy_gain":
        # Select route with lowest entropy among those that reduce entropy vs bypass by > eta
        bypass_H  = entropy_arr[0:1, :]           # [1, n]
        delta_H   = bypass_H - entropy_arr         # [n_routes, n]; positive = improvement
        eta       = args.entropy_gain_threshold
        gain_mask = delta_H > eta
        gain_mask[0] = True                        # source always accepted as fallback
        candidate = np.where(gain_mask, entropy_arr, np.inf)
        routes = candidate.argmin(axis=0)

    else:  # conformal
        final_scores = (args.score_alpha * dist_arr
                        + args.score_beta * entropy_arr
                        + args.score_gamma * corr_arr)
        accept = (final_scores < np.inf)
        for j, stats_j in enumerate(domain_stats_list):
            for k in range(K):
                stat_jk = stats_j.get(k, {"available": False})
                if not stat_jk.get("available", False):
                    continue
                mask = (asgn == k)
                if not mask.any():
                    continue
                q_jk = stat_jk.get("q", np.inf)
                accept[j, mask] &= (final_scores[j, mask] <= q_jk)

        accepted_scores = np.where(accept, final_scores, np.inf)
        has_accepted    = accepted_scores.min(axis=0) < np.inf
        routes = np.where(has_accepted,
                          accepted_scores.argmin(axis=0),
                          final_scores.argmin(axis=0))
        rejected = ~has_accepted

    # Apply selected route
    corr = embs_t.clone()
    for j in range(1, n_routes):
        for k in range(K):
            mask = (routes == j) & (asgn == k)
            if not mask.any():
                continue
            adapters_list[j - 1][k].eval()
            corr[mask] = adapters_list[j - 1][k](embs_t[mask])

    preds = frozen_head(corr).argmax(1).cpu().numpy()
    f1  = f1_score(labels_np, preds, average="macro", zero_division=0)
    acc = accuracy_score(labels_np, preds)
    route_counts = np.bincount(routes, minlength=n_routes)
    return f1, acc, route_counts, float(rejected.mean())


# ── Memory accounting ─────────────────────────────────────────────────────────

def count_memory(adapters_list, domain_stats_list):
    seen_ids  = set()
    n_adapter = 0
    for adapters in adapters_list:
        for adp in adapters:   # ModuleList iteration
            if id(adp) not in seen_ids:
                seen_ids.add(id(adp))
                n_adapter += sum(p.numel() for p in adp.parameters())
    n_stats = 0
    for stats_j in domain_stats_list:
        for stat_k in stats_j.values():
            if stat_k.get("available", False):
                n_stats += len(stat_k["mu"]) + 1   # mu + radius
                if "q" in stat_k:
                    n_stats += 1
    return n_adapter, n_stats


@torch.no_grad()
def eval_route_on_calib(embs_np, labels_np, asgn_np, route_j,
                         adapters_list, frozen_head, device, K):
    """Evaluate single route on calibration set.

    route_j=0 → bypass (source head only); route_j>=1 → adapters_list[route_j-1].
    Returns macro-F1.
    """
    embs_t = torch.from_numpy(embs_np).to(device)
    corr   = embs_t.clone()
    if route_j > 0 and (route_j - 1) < len(adapters_list):
        adapters = adapters_list[route_j - 1]
        for k, adp in enumerate(adapters):   # ModuleList
            mask = (asgn_np == k)
            if not mask.any():
                continue
            adp.eval()
            corr[mask] = adp(embs_t[mask])
    preds = frozen_head(corr).argmax(1).cpu().numpy()
    return f1_score(labels_np, preds, average="macro", zero_division=0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = build_config(args)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Patch e7 globals so train_adapters / eval_oracle pick up CLI overrides
    e7.K          = args.k
    e7.ADAPTER_EP = args.adapter_ep
    e7.SEED       = args.seed
    e7.BATCH      = args.batch

    domain_order  = cfg["domain_order"]
    domain_labels = cfg["domain_labels"]
    T             = len(domain_order)
    out_tag       = cfg["out_tag"]

    out_dir = REPO / "results" / "exp8_rf_seqpact"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"EXP8 RF-SeqPACT  dataset={args.dataset.upper()}  T={T}  K={args.k}  "
          f"calib={args.calib_frac*100:.0f}%  router={args.router}")
    print(f"Domain order: {domain_labels}")

    # ── Frozen source model ───────────────────────────────────────────────────
    print(f"\nLoading frozen model for {domain_labels[0]} ...")
    model       = cfg["load_model"](device)
    frozen_head = model.head

    # ── Test embeddings (cached, never deleted) ───────────────────────────────
    print("Extracting test embeddings ...")
    test_embs, test_lbls = {}, {}
    for did, lbl in zip(domain_order, domain_labels):
        ld = torch.utils.data.DataLoader(
            cfg["get_test"](did), batch_size=args.batch)
        test_embs[lbl], test_lbls[lbl] = extract_embs(model, ld, device)

    # ── Step 0: source — extract, cluster, stat, discard ─────────────────────
    print(f"\n{'='*20} Step 0 — source only {'='*20}")
    print("Extracting source train embeddings ...")
    src_ld   = torch.utils.data.DataLoader(
        cfg["get_train"](domain_order[0]), batch_size=args.batch)
    src_embs, _ = extract_embs(model, src_ld, device)

    print(f"K-means (K={args.k}) ...")
    centroids, src_asgn = kmeans_numpy(src_embs, args.k, seed=args.seed)

    source_stats = compute_domain_stats(src_embs, src_asgn, centroids,
                                         args.k, args.radius_percentile, args.min_count)

    if args.router == "conformal":
        src_scores = compute_calibration_scores(
            src_embs, src_asgn, source_stats, None,
            frozen_head, device, args, is_source=True)
        set_conformal_thresholds(source_stats, src_scores, args)

    # Discard source embeddings — not needed again
    del src_embs, src_asgn

    # Source-only eval (no adapters)
    A_f1  = np.full((T, T), np.nan)
    A_acc = np.full((T, T), np.nan)
    A_orc = np.full((T, T), np.nan) if args.eval_oracle else None

    with torch.no_grad():
        et = torch.from_numpy(test_embs[domain_labels[0]]).to(device)
        p0 = frozen_head(et).argmax(1).cpu().numpy()
    l0 = test_lbls[domain_labels[0]]
    A_f1[0, 0]  = f1_score(l0, p0, average="macro", zero_division=0)
    A_acc[0, 0] = accuracy_score(l0, p0)
    if args.eval_oracle:
        A_orc[0, 0] = A_f1[0, 0]

    print(f"  {domain_labels[0]:<12} F1={A_f1[0,0]:.4f}  ACC={A_acc[0,0]:.4f}")

    domain_stats_list = [source_stats]
    adapters_list     = []
    cl_metrics_all    = [(A_f1[0, 0], 0.0, 0.0, 0.0, 0.0)]
    f_tgt_list        = [0.0]
    reject_log        = {}   # step -> {domain_idx -> reject_rate}
    routing_log       = {}   # step -> {domain_idx -> route_counts}

    n_avail_src = sum(1 for s in source_stats.values() if s["available"])
    print(f"  Source: {n_avail_src}/{args.k} concepts available")

    # ── Steps 1..T-1 ─────────────────────────────────────────────────────────
    for t in range(1, T):
        did_t = domain_order[t]
        lbl_t = domain_labels[t]
        print(f"\n{'='*20} Step {t}/{T-1}: {lbl_t} {'='*20}")

        # 1. Calibration split
        ds    = cfg["get_train"](did_t)
        y_all = ds.y.numpy()
        sss   = StratifiedShuffleSplit(n_splits=1,
                                       test_size=1 - args.calib_frac,
                                       random_state=args.seed)
        idx, _ = next(sss.split(np.zeros(len(y_all)), y_all))
        calib_ld = torch.utils.data.DataLoader(
            torch.utils.data.Subset(ds, idx), batch_size=args.batch)
        calib_embs, _ = extract_embs(model, calib_ld, device)
        calib_lbls    = y_all[idx]
        calib_asgn    = assign_concepts(calib_embs, centroids)
        print(f"  Calib: {len(calib_embs)} samples from {lbl_t}")

        # 2. Train adapter bank (or reuse an existing route)
        reused_route = -1
        if args.reuse_threshold > 0 and adapters_list:
            best_f1, best_j = 0.0, -1
            for j in range(len(adapters_list) + 1):   # 0=bypass, 1..t-1=adapters
                f1_j = eval_route_on_calib(
                    calib_embs, calib_lbls, calib_asgn,
                    j, adapters_list, frozen_head, device, args.k)
                if f1_j > best_f1:
                    best_f1, best_j = f1_j, j
            if best_f1 >= args.reuse_threshold and best_j > 0:
                reused_route = best_j
                adapters = adapters_list[best_j - 1]   # shared reference, no new params
                print(f"  Reusing route {best_j} ({domain_labels[best_j]}) "
                      f"F1={best_f1:.4f} >= {args.reuse_threshold:.2f}", flush=True)

        if reused_route < 0:
            print("  Training IdentityAdapter bank ...", flush=True)
            adapters = train_adapters(calib_embs, calib_lbls, centroids, frozen_head, device)

        # 3. Compact domain statistics
        stats_t = compute_domain_stats(calib_embs, calib_asgn, centroids,
                                        args.k, args.radius_percentile, args.min_count)
        n_avail = sum(1 for s in stats_t.values() if s["available"])
        print(f"  {n_avail}/{args.k} concepts available")

        # 4. Conformal thresholds (computed on current calib before deletion)
        if args.router == "conformal":
            cal_scores = compute_calibration_scores(
                calib_embs, calib_asgn, stats_t, adapters,
                frozen_head, device, args, is_source=False)
            set_conformal_thresholds(stats_t, cal_scores, args)

        # 5. Append to state
        domain_stats_list.append(stats_t)
        adapters_list.append(adapters)

        # 6. DELETE calibration embeddings — rehearsal-free
        del calib_embs, calib_lbls, calib_asgn

        # 7. Evaluate all seen domains [0..t]
        print("  Evaluating:")
        reject_log[t]  = {}
        routing_log[t] = {}
        for j in range(t + 1):
            lbl_j = domain_labels[j]
            e_j, l_j = test_embs[lbl_j], test_lbls[lbl_j]

            f1, acc, rc, rr = eval_rf_router(
                e_j, l_j, centroids, domain_stats_list, adapters_list,
                frozen_head, device, args)
            A_f1[t, j]  = f1
            A_acc[t, j] = acc
            reject_log[t][j]  = rr
            routing_log[t][j] = rc

            if args.eval_oracle:
                A_orc[t, j] = eval_oracle(
                    e_j, l_j, centroids, adapters_list, j, frozen_head, device)

            rp_str = " ".join(
                f"{domain_labels[r]}:{rc[r]/max(rc.sum(),1)*100:.0f}%"
                for r in range(len(rc)))
            print(f"    {lbl_j:<12}  F1={f1:.4f}  ACC={acc:.4f}  "
                  f"reject={rr*100:.1f}%")
            print(f"      routing → [{rp_str}]")
            if args.eval_oracle:
                print(f"      oracle  → F1={A_orc[t,j]:.4f}")

        # 8. CL metrics
        AA, BWT, F_all, RF, PRF = compute_cl_metrics(A_f1, t, include_source=True)
        _, _, F_tgt, _, _       = compute_cl_metrics(A_f1, t, include_source=False)
        cl_metrics_all.append((AA, BWT, F_all, RF, PRF))
        f_tgt_list.append(F_tgt)
        print(f"  AA={AA:.4f}  BWT={BWT:+.4f}  "
              f"F_all={F_all:.4f}  F_tgt={F_tgt:.4f}  "
              f"RF={RF:.4f}  PRF={PRF:.4f}")

    # ── Final summary ─────────────────────────────────────────────────────────
    AA_f, BWT_f, F_all_f, RF_f, PRF_f = cl_metrics_all[-1]
    F_tgt_f  = f_tgt_list[-1]
    n_adap, n_stat = count_memory(adapters_list, domain_stats_list)

    RED, RESET = "\033[91m", "\033[0m"
    print(f"\n{RED}{'='*58}{RESET}")
    print(f"  EXP8 RF-SeqPACT — {args.dataset.upper()}  "
          f"{args.calib_frac*100:.0f}% labels  router={args.router}")
    print(f"{RED}  AA     : {AA_f:.4f}{RESET}")
    print(f"{RED}  BWT    : {BWT_f:+.4f}{RESET}")
    print(f"{RED}  F_all  : {F_all_f:.4f}   (src + all targets){RESET}")
    print(f"{RED}  F_tgt  : {F_tgt_f:.4f}   (targets only){RESET}")
    print(f"{RED}  RF     : {RF_f:.4f}{RESET}")
    print(f"{RED}  PRF    : {PRF_f:.4f}{RESET}")
    print(f"{RED}  ACC-AA : {float(np.nanmean(A_acc[T-1, :T])):.4f}{RESET}")
    print(f"  Stored memory:")
    print(f"    adapter params    : {n_adap:,}")
    print(f"    statistics floats : {n_stat:,}")
    print(f"    cached embeddings : 0   (rehearsal-free)")
    print(f"{RED}{'='*58}{RESET}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    for name, mat in [("f1_matrix",  A_f1), ("acc_matrix", A_acc)]:
        pd.DataFrame(mat, index=domain_labels, columns=domain_labels).to_csv(
            out_dir / f"{out_tag}_{name}.csv")
    if args.eval_oracle and A_orc is not None:
        pd.DataFrame(A_orc, index=domain_labels, columns=domain_labels).to_csv(
            out_dir / f"{out_tag}_oracle_f1_matrix.csv")

    df_m = pd.DataFrame(
        [(m[0], m[1], m[2], ft, m[3], m[4])
         for m, ft in zip(cl_metrics_all, f_tgt_list)],
        columns=["AA", "BWT", "F_all", "F_tgt", "RF", "PRF"],
        index=domain_labels)
    df_m.to_csv(out_dir / f"{out_tag}_metrics.csv")

    # Final routing confusion matrix (step T-1)
    T_final = T - 1
    route_mat = np.full((T, T), np.nan)
    for j in range(T):
        rc = routing_log.get(T_final, {}).get(j)
        if rc is not None:
            total = max(rc.sum(), 1)
            route_mat[j, :len(rc)] = rc / total
    pd.DataFrame(route_mat, index=domain_labels, columns=domain_labels).to_csv(
        out_dir / f"{out_tag}_routing.csv")

    # Reject rates
    rr_mat = np.full((T, T), np.nan)
    for t_step, rlog in reject_log.items():
        for j, rr in rlog.items():
            rr_mat[t_step, j] = rr
    pd.DataFrame(rr_mat, index=domain_labels, columns=domain_labels).to_csv(
        out_dir / f"{out_tag}_reject_rates.csv")

    # ── Figures ───────────────────────────────────────────────────────────────
    fs = 7 if T > 6 else 9

    # 1. Accuracy matrix heatmap
    n_panels = 3 if args.eval_oracle else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(T * n_panels, max(4, T)))
    panels = [(axes[0], A_f1,  "RF router (Macro-F1)"),
              (axes[1], A_acc, "RF router (ACC)")]
    if args.eval_oracle:
        panels.append((axes[2], A_orc, "Oracle (Macro-F1)"))

    for ax, mat, title in panels:
        im = ax.imshow(np.nan_to_num(mat, nan=-0.05), vmin=0, vmax=1, cmap="YlGn")
        for i in range(T):
            for j in range(T):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center",
                            fontsize=fs,
                            color="black" if mat[i, j] > 0.3 else "white")
        ax.set_xticks(range(T)); ax.set_yticks(range(T))
        ax.set_xticklabels(domain_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(domain_labels, fontsize=7)
        ax.set_xlabel("Test domain"); ax.set_ylabel("After step")
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.03)
    fig.suptitle(
        f"Exp8 RF-SeqPACT — {args.dataset.upper()}  "
        f"K={args.k}  calib={args.calib_frac*100:.0f}%  router={args.router}\n"
        f"Final: AA={AA_f:.4f}  F_all={F_all_f:.4f}  F_tgt={F_tgt_f:.4f}",
        fontsize=9)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"{out_tag}_heatmap.{ext}", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. CL curve
    steps  = list(range(T))
    AAs    = [m[0] for m in cl_metrics_all]
    F_alls = [m[2] for m in cl_metrics_all]
    F_tgts = f_tgt_list
    fig2, ax2 = plt.subplots(figsize=(max(7, T + 1), 4))
    ax2.plot(steps, AAs,    "o-",  color="#2ecc71", lw=2, label="AA")
    ax2.plot(steps, F_alls, "s--", color="#e74c3c", lw=2, label="F_all (src+tgt)")
    ax2.plot(steps, F_tgts, "^:",  color="#f39c12", lw=2, label="F_tgt (targets)")
    ax2.set_xticks(steps)
    ax2.set_xticklabels(domain_labels, rotation=30, ha="right", fontsize=8)
    ax2.set_ylim(0, 1.05); ax2.set_ylabel("Macro-F1")
    ax2.legend(fontsize=8); ax2.grid(ls="--", alpha=0.4)
    ax2.set_title(
        f"Exp8 RF-SeqPACT CL Curve — {args.dataset.upper()}\n"
        f"Final: AA={AA_f:.4f}  F_all={F_all_f:.4f}  BWT={BWT_f:+.4f}",
        fontsize=10)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig2.savefig(out_dir / f"{out_tag}_cl_curve.{ext}", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Routing confusion heatmap (final step)
    fig3, ax3 = plt.subplots(figsize=(max(5, T), max(4, T - 1)))
    rm = np.nan_to_num(route_mat, nan=0.0)
    im3 = ax3.imshow(rm, vmin=0, vmax=1, cmap="Blues")
    for i in range(T):
        for j in range(T):
            if not np.isnan(route_mat[i, j]):
                ax3.text(j, i, f"{route_mat[i,j]*100:.0f}%",
                         ha="center", va="center", fontsize=fs,
                         color="white" if route_mat[i, j] > 0.5 else "black")
    ax3.set_xticks(range(T)); ax3.set_yticks(range(T))
    ax3.set_xticklabels(domain_labels, rotation=45, ha="right", fontsize=7)
    ax3.set_yticklabels(domain_labels, fontsize=7)
    ax3.set_xlabel("Route selected"); ax3.set_ylabel("Test domain")
    ax3.set_title(
        f"Routing confusion — {args.dataset.upper()} (final step)\n"
        f"router={args.router}", fontsize=9)
    plt.colorbar(im3, ax=ax3, fraction=0.03)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig3.savefig(out_dir / f"{out_tag}_routing.{ext}", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nOutputs → {out_dir}/{out_tag}_*")


if __name__ == "__main__":
    main()
