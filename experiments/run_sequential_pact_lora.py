"""
run_sequential_pact_lora.py — Sequential Domain-Incremental PACT with LoRA adapters

Source domain (D_0): train/load frozen TSLANet host.
For each new domain (D_t): train a small LoRA adapter bank using 5% labeled
  calibration data. Store raw-embedding statistics for distance-based routing.
At inference: route each sample to source (bypass) or a domain-specific adapter
  bank based on nearest-domain statistics. Exactly one route is selected.

Usage:
  CUDA_VISIBLE_DEVICES=0 python run_sequential_pact_lora.py \\
    --dataset hhar --domain_order 0,1,2,3 --rank 8 --k 16 --calib_frac 0.05

Do not modify run_unified_eval.py — this is a standalone extension.
"""

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

# ── Path setup ─────────────────────────────────────────────────────────────────
REPO = Path(__file__).parent
sys.path.insert(0, str(REPO / "TSLANet" / "Classification"))


def _load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Import dataset modules
tft = _load_mod("tft", REPO / "transfer_fd_tslanet.py")
tht = _load_mod("tht", REPO / "transfer_hhar_tslanet.py")
tst = _load_mod("tst", REPO / "transfer_sleep_tslanet.py")

# Import shared utilities from run_unified_eval without triggering its main()
rue = _load_mod("rue", REPO / "run_unified_eval.py")

load_frozen_model = rue.load_frozen_model
make_loader       = rue.make_loader
calib_split       = rue.calib_split
extract_embs      = rue.extract_embs
eval_f1_loader    = rue.eval_f1_loader
kmeans_numpy      = rue.kmeans_numpy
assign_concepts   = rue.assign_concepts

DATA_ROOT = rue.DATA_ROOT
EMB_DIM   = rue.EMB_DIM
FD_MAP    = rue.FD_MAP
SEED      = rue.SEED

OUT_DIR = REPO / "results" / "sequential_pact_lora"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── LoRA adapter ───────────────────────────────────────────────────────────────

class LoRAIdentityAdapter(nn.Module):
    """Residual LoRA adapter. Identity at init because up.weight = 0."""
    def __init__(self, dim: int = 128, rank: int = 8):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up   = nn.Linear(rank, dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(dim))

        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)   # residual starts at zero

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.down(x)) + self.bias


# ── Routing statistics ─────────────────────────────────────────────────────────

def build_domain_stats(
    embs: np.ndarray,
    asgn: np.ndarray,
    centroids: np.ndarray,
    K: int,
    radius_percentile: float = 95.0,
) -> dict:
    """Per-concept mean and radius for one domain's raw embeddings.

    Returns dict k → {"mu", "radius", "available"}.
    Concepts with no samples are marked unavailable (infinite distance at routing).
    """
    stats = {}
    for k in range(K):
        mask = (asgn == k)
        if mask.any():
            cluster = embs[mask]
            mu = cluster.mean(0)
            dists = np.linalg.norm(cluster - mu, axis=1)
            radius = float(np.percentile(dists, radius_percentile))
            stats[k] = {"mu": mu, "radius": radius, "available": True}
        else:
            stats[k] = {"available": False}
    return stats


def route_samples_batch(
    embs_np: np.ndarray,
    centroids: np.ndarray,
    domain_stats_list: list,
    source_bias: float = 0.95,
    eps: float = 1e-6,
):
    """Distance-based router. For each sample, select the domain whose
    per-concept statistics are closest (smallest normalised distance).

    source_bias: multiply source domain scores by this factor (<1 → favours bypass).

    Returns:
        routes      [N]  — best domain index (0 = bypass / source)
        concept_asgn [N] — source-concept assignment
    """
    N = len(embs_np)
    K = len(centroids)
    n_domains = len(domain_stats_list)

    concept_asgn = assign_concepts(embs_np, centroids)  # [N]

    # Build score matrix: scores[j, i] = normalised distance from sample i to domain j
    scores = np.full((n_domains, N), np.inf, dtype=np.float64)

    for j, stats_j in enumerate(domain_stats_list):
        for k in range(K):
            sk = stats_j.get(k, {"available": False})
            if not sk["available"]:
                continue
            mu     = sk["mu"]
            radius = sk["radius"]
            mask   = (concept_asgn == k)
            if not mask.any():
                continue
            dists = np.linalg.norm(embs_np[mask] - mu, axis=1)
            scores[j, mask] = dists / (radius + eps)

    # Apply source bias (slightly favours bypass when distances are close)
    scores[0] *= source_bias

    routes = scores.argmin(axis=0)  # [N]
    return routes, concept_asgn


# ── Adapter training ───────────────────────────────────────────────────────────

def train_lora_adapters(
    calib_embs: np.ndarray,
    calib_lbls: np.ndarray,
    centroids: np.ndarray,
    frozen_head: nn.Module,
    device: torch.device,
    K: int,
    rank: int,
    adapter_epochs: int,
    adapter_lr: float,
    penalty: float,
    weight_decay_lora: float = 1e-4,
) -> nn.ModuleList:
    """Train K LoRA adapters for a single new domain.

    Only this domain's adapter bank is trained; the frozen backbone, head,
    centroids, and existing domain stats/adapters are untouched.
    """
    adapters = nn.ModuleList(
        [LoRAIdentityAdapter(dim=EMB_DIM, rank=rank) for _ in range(K)]
    ).to(device)
    opt  = torch.optim.AdamW(adapters.parameters(), lr=adapter_lr, weight_decay=0.0)
    crit = nn.CrossEntropyLoss()

    asgn = assign_concepts(calib_embs, centroids)

    et = torch.from_numpy(calib_embs).to(device)
    lt = torch.from_numpy(calib_lbls).to(device)
    at = torch.from_numpy(asgn.astype(np.int64))

    batch_size = min(256, max(16, len(calib_embs)))
    ld = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(et, lt, at),
        batch_size=batch_size, shuffle=True,
    )

    for _ in range(adapter_epochs):
        for eb, yb, cb in ld:
            adapters.train()
            opt.zero_grad()

            # Apply concept-specific adapter to each sample
            corr = eb.clone()
            for k in range(K):
                mask = (cb == k)
                if mask.any():
                    corr[mask] = adapters[k](eb[mask])

            ce_loss       = crit(frozen_head(corr), yb)
            residual_loss = ((corr - eb) ** 2).mean()
            weight_loss   = sum(p.pow(2).sum() for p in adapters.parameters())

            loss = ce_loss + penalty * residual_loss + weight_decay_lora * weight_loss
            loss.backward()
            opt.step()

    return adapters


# ── Sequential evaluation ──────────────────────────────────────────────────────

@torch.no_grad()
def eval_sequential_f1(
    embs_np: np.ndarray,
    labels_np: np.ndarray,
    centroids: np.ndarray,
    domain_stats_list: list,
    adapters_per_domain: dict,
    domain_order_seen: list,
    frozen_head: nn.Module,
    device: torch.device,
    source_bias: float = 0.95,
    eps: float = 1e-6,
):
    """Evaluate macro-F1 using the distance router + LoRA adapter banks.

    adapters_per_domain: domain_id → nn.ModuleList of K LoRA adapters
    domain_order_seen:   list of domain IDs in order (index 0 = source)

    Returns:
        f1            — macro-F1 on this test split
        route_counts  — [n_domains_seen] samples routed to each bank
    """
    routes, concept_asgn = route_samples_batch(
        embs_np, centroids, domain_stats_list, source_bias, eps
    )

    embs_t   = torch.from_numpy(embs_np).to(device)
    corrected = embs_t.clone()

    n_seen       = len(domain_order_seen)
    route_counts = np.zeros(n_seen, dtype=int)

    for j, domain_id in enumerate(domain_order_seen):
        mask = (routes == j)
        route_counts[j] = int(mask.sum())

        if j == 0 or not mask.any():
            # Source → bypass (use raw embedding unchanged)
            continue

        domain_adapters = adapters_per_domain[domain_id]   # ModuleList of K adapters
        for k in range(len(centroids)):
            km = mask & (concept_asgn == k)
            if not km.any():
                continue
            idx   = np.where(km)[0]
            batch = embs_t[idx]
            domain_adapters[k].eval()
            corrected[idx] = domain_adapters[k](batch)

    preds = frozen_head(corrected).argmax(1).cpu().numpy()
    f1    = f1_score(labels_np, preds, average="macro", zero_division=0)
    return f1, route_counts


# ── Dataset helpers ────────────────────────────────────────────────────────────

def build_dataset_accessors(dataset: str):
    """Return (get_train_ds, get_test_ds, load_model, n_domains_max, batch_size)."""
    if dataset == "hhar":
        def get_train(uid): return tht.HHARDataset(uid, "train", DATA_ROOT)
        def get_test(uid):  return tht.HHARDataset(uid, "test",  DATA_ROOT)
        def load_model(uid, device): return load_frozen_model(uid, "hhar", device)
        return get_train, get_test, load_model, 9, 256

    elif dataset == "fd":
        def get_train(did): return tft.FDDataset(FD_MAP[did], "train", DATA_ROOT)
        def get_test(did):  return tft.FDDataset(FD_MAP[did], "test",  DATA_ROOT)
        def load_model(did, device): return load_frozen_model(FD_MAP[did], "fd", device)
        return get_train, get_test, load_model, 4, 256

    elif dataset == "sleep":
        def get_train(sid): return tst.SleepEDFDataset(sid, "train", DATA_ROOT)
        def get_test(sid):  return tst.SleepEDFDataset(sid, "test",  DATA_ROOT)
        def load_model(sid, device): return load_frozen_model(sid, "sleep", device)
        return get_train, get_test, load_model, 20, 64

    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# ── CL metrics ─────────────────────────────────────────────────────────────────

def compute_cl_metrics(A: np.ndarray, T: int):
    """Compute CL metrics at each adaptation step.

    A[t, j] = macro-F1 on domain j after adapting through step t (0-indexed).
    Only entries where j <= t are defined (lower-triangular).

    Metrics returned per step (AA, BWT, F from standard CL literature;
    RF, PRF from DualCD / Continual.py):

        AA   = mean F1 over all seen domains at step t
        BWT  = mean(A[t,j] - A[j,j])  for j < t   (signed backward transfer)
        F    = mean(max(0, A[j,j] - A[t,j])) for j < t  (absolute forgetting from diagonal)

        RF   = DualCD `forgetting`: mean relative drop from peak to current final
                 RF_j  = max(0, (peak_j - A[t,j]) / peak_j)
                 peak_j = max(A[i,j] for i in [j .. t-1])   ← excludes the current row
        PRF  = DualCD `performance_aware_forgetting`:
                 PRF_j = (1 - A[t,j]) * RF_j

    Reference: baselines/DualCD/Libs/utils/Continual.py

    Returns list of (AA_t, BWT_t, F_t, RF_t, PRF_t) for t = 0..T-1.
    """
    metrics = []
    for t in range(T):
        valid = [j for j in range(t + 1) if not np.isnan(A[t, j])]
        AA_t  = float(np.mean([A[t, j] for j in valid])) if valid else 0.0

        if t == 0:
            BWT_t = 0.0
            F_t   = 0.0
            RF_t  = 0.0
            PRF_t = 0.0
        else:
            bwt_terms = []
            f_terms   = []
            rf_terms  = []
            prf_terms = []

            for j in range(t):
                if np.isnan(A[t, j]) or np.isnan(A[j, j]):
                    continue

                final_j = A[t, j]

                # Standard BWT / absolute forgetting anchored to diagonal
                bwt_terms.append(final_j - A[j, j])
                f_terms.append(max(0.0, A[j, j] - final_j))

                # DualCD RF / PRF: peak over rows [j .. t-1], not including current row t
                peak_vals = [A[i, j] for i in range(j, t) if not np.isnan(A[i, j])]
                if peak_vals:
                    peak_j = max(peak_vals)
                    if peak_j > 0.0:
                        rf_j  = max(0.0, (peak_j - final_j) / peak_j)
                        prf_j = (1.0 - final_j) * rf_j
                    else:
                        rf_j = 0.0
                        prf_j = 0.0
                    rf_terms.append(rf_j)
                    prf_terms.append(prf_j)

            BWT_t = float(np.mean(bwt_terms)) if bwt_terms else 0.0
            F_t   = float(np.mean(f_terms))   if f_terms   else 0.0
            RF_t  = float(np.mean(rf_terms))   if rf_terms  else 0.0
            PRF_t = float(np.mean(prf_terms))  if prf_terms else 0.0

        metrics.append((AA_t, BWT_t, F_t, RF_t, PRF_t))
    return metrics


# ── Plotting ───────────────────────────────────────────────────────────────────

def save_heatmap(A: np.ndarray, domain_labels: list, out_path: Path, title: str):
    T = len(domain_labels)
    fig, ax = plt.subplots(figsize=(max(5, T), max(4, T * 0.8)))
    im = ax.imshow(A, vmin=0, vmax=1, cmap="YlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="Macro-F1")
    ax.set_xticks(range(T)); ax.set_xticklabels([f"D{d}" for d in domain_labels], fontsize=9)
    ax.set_yticks(range(T)); ax.set_yticklabels([f"Step {t}" for t in range(T)], fontsize=9)
    ax.set_xlabel("Eval domain"); ax.set_ylabel("Adaptation step")
    ax.set_title(title, fontsize=10, fontweight="bold")
    for i in range(T):
        for j in range(T):
            if not np.isnan(A[i, j]):
                ax.text(j, i, f"{A[i,j]:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if A[i, j] > 0.5 else "white")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_aa_curve(metrics: list, domain_labels: list, out_path: Path, title: str):
    T      = len(domain_labels)
    steps  = [f"D{d}" for d in domain_labels]
    AAs    = [m[0] for m in metrics]
    BWTs   = [m[1] for m in metrics]
    Fs     = [m[2] for m in metrics]
    RFs    = [m[3] for m in metrics]
    PRFs   = [m[4] for m in metrics]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    def _panel(ax, vals, color, ylabel, panel_title, hline=False):
        ax.plot(steps, vals, "o-", color=color)
        if hline:
            ax.axhline(0, ls="--", color="gray", lw=0.8)
        ax.set_title(panel_title, fontsize=9, fontweight="bold")
        ax.set_xlabel("Adaptation step", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(ls="--", alpha=0.4)
        ax.set_ylim(-0.05, 1.05)

    _panel(axes[0], AAs,  "#2471a3", "F1",  "Average Accuracy (AA)")
    _panel(axes[1], BWTs, "#1e8449", "F1",  "Backward Transfer (BWT)", hline=True)
    _panel(axes[2], Fs,   "#e74c3c", "F1",  "Forgetting (F)")
    _panel(axes[3], RFs,  "#f39c12", "RF",  "Relative Forgetting (RF)\n[DualCD]")
    _panel(axes[4], PRFs, "#8e44ad", "PRF", "Perf-Aware Forgetting (PRF)\n[DualCD]")

    fig.suptitle(title, fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Seq-PACT-LoRA: Sequential domain-incremental PACT")
    p.add_argument("--dataset",          choices=["fd", "hhar", "sleep"], default="hhar")
    p.add_argument("--domain_order",     default="0,1,2,3",
                   help="Comma-separated domain IDs, or 'auto' for all domains")
    p.add_argument("--k",                type=int,   default=16,   help="Number of source concepts")
    p.add_argument("--rank",             type=int,   default=8,    help="LoRA rank")
    p.add_argument("--calib_frac",       type=float, default=0.05, help="Calibration fraction")
    p.add_argument("--adapter_epochs",   type=int,   default=200)
    p.add_argument("--adapter_lr",       type=float, default=1e-3)
    p.add_argument("--penalty",          type=float, default=1e-3, help="Residual loss weight")
    p.add_argument("--radius_percentile",type=float, default=95.0)
    p.add_argument("--source_bias",      type=float, default=0.95,
                   help="Score multiplier for source domain (<1 → favours bypass)")
    p.add_argument("--seed",             type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}  rank={args.rank}  K={args.k}  "
          f"calib={args.calib_frac*100:.0f}%  epochs={args.adapter_epochs}")

    get_train, get_test, load_model, n_max, batch_size = build_dataset_accessors(args.dataset)

    # Parse domain order
    if args.domain_order.strip().lower() == "auto":
        domain_order = list(range(n_max))
    else:
        domain_order = [int(x.strip()) for x in args.domain_order.split(",")]
    T = len(domain_order)
    K = args.k

    print(f"Domain order: {domain_order}  (T={T})")

    # ── Step 0: Source domain ──────────────────────────────────────────────────
    src_id = domain_order[0]
    print(f"\n{'='*60}")
    print(f"Initialising from source domain D_{src_id}")
    print(f"{'='*60}")

    frozen_model = load_model(src_id, device)
    frozen_model.eval()
    for p in frozen_model.parameters():
        p.requires_grad_(False)
    frozen_head = frozen_model.head

    src_train_ds     = get_train(src_id)
    src_train_loader = make_loader(src_train_ds, batch_size=batch_size)
    src_embs, src_labels = extract_embs(frozen_model, src_train_loader, device)

    print(f"  Source embeddings: {src_embs.shape}")
    print(f"  Building concept space (K={K})...")
    centroids, src_asgn = kmeans_numpy(src_embs, K, seed=args.seed)

    # Source routing stats (j=0)
    src_stats = build_domain_stats(src_embs, src_asgn, centroids, K, args.radius_percentile)
    domain_stats_list  = [src_stats]     # domain_stats_list[j] → stats for j-th domain
    adapters_per_domain = {}             # domain_id → ModuleList of K LoRA adapters
    domain_order_seen  = [src_id]        # j-th entry = actual domain ID for step j

    # Pre-cache test embeddings to avoid re-extracting
    test_embs_cache: dict = {}

    def _get_test_embs(did):
        if did not in test_embs_cache:
            tds = get_test(did)
            tld = make_loader(tds, batch_size=batch_size)
            embs, labels = extract_embs(frozen_model, tld, device)
            test_embs_cache[did] = (embs, labels)
        return test_embs_cache[did]

    # Evaluate source only (step 0 diagonal)
    A = np.full((T, T), np.nan)
    src_test_embs, src_test_labels = _get_test_embs(src_id)
    f1_src, rc_src = eval_sequential_f1(
        src_test_embs, src_test_labels, centroids,
        domain_stats_list, adapters_per_domain, domain_order_seen,
        frozen_head, device, args.source_bias,
    )
    A[0, 0] = f1_src

    print(f"\nStep 0 — D_{src_id} (source): F1={f1_src:.4f}  routes={rc_src}")

    # Collect final-step routing counts per eval domain: routing_log[t][did] = route_counts
    routing_log: dict = {0: {src_id: rc_src}}

    # ── Sequential adaptation ──────────────────────────────────────────────────
    for t in range(1, T):
        domain_id = domain_order[t]
        print(f"\n{'='*60}")
        print(f"Step {t}: Adapting to domain D_{domain_id}")
        print(f"{'='*60}")

        # 5% calibration split
        train_ds  = get_train(domain_id)
        calib_ds  = calib_split(train_ds, frac=args.calib_frac)
        calib_ld  = make_loader(calib_ds, batch_size=batch_size)
        calib_embs, calib_lbls = extract_embs(frozen_model, calib_ld, device)
        print(f"  Calibration samples: {len(calib_embs)}")

        # Train LoRA adapter bank for this domain
        print(f"  Training LoRA adapters (rank={args.rank}, K={K}, "
              f"epochs={args.adapter_epochs})...")
        adapters = train_lora_adapters(
            calib_embs, calib_lbls, centroids, frozen_head, device,
            K=K, rank=args.rank,
            adapter_epochs=args.adapter_epochs,
            adapter_lr=args.adapter_lr,
            penalty=args.penalty,
        )
        adapters_per_domain[domain_id] = adapters

        # Build routing stats for this domain using raw calibration embeddings
        calib_asgn = assign_concepts(calib_embs, centroids)
        new_stats  = build_domain_stats(
            calib_embs, calib_asgn, centroids, K, args.radius_percentile
        )
        domain_stats_list.append(new_stats)
        domain_order_seen.append(domain_id)

        # Evaluate on all seen domains (j = 0 .. t)
        print(f"  Evaluating on {len(domain_order_seen)} domains...")
        routing_log[t] = {}
        for j, eval_did in enumerate(domain_order_seen):
            embs_j, labels_j = _get_test_embs(eval_did)
            f1_j, rc_j = eval_sequential_f1(
                embs_j, labels_j, centroids,
                domain_stats_list, adapters_per_domain, domain_order_seen,
                frozen_head, device, args.source_bias,
            )
            A[t, j] = f1_j
            routing_log[t][eval_did] = rc_j
            rc_str = " ".join(
                f"D{domain_order_seen[r]}:{rc_j[r]}" for r in range(len(rc_j))
            )
            print(f"    D_{eval_did}: F1={f1_j:.4f}  [{rc_str}]")

    # ── CL metrics ────────────────────────────────────────────────────────────
    cl_metrics = compute_cl_metrics(A, T)

    # ── Print report ──────────────────────────────────────────────────────────
    dataset_label = args.dataset.upper()
    rank_label    = f"rank={args.rank}"
    calib_label   = f"{args.calib_frac*100:.0f}% labels"
    print(f"\n{'#'*60}")
    print(f"SEQ-PACT-LoRA — {dataset_label}, {calib_label}, {rank_label}")
    print(f"{'#'*60}")
    header = (f"{'Step':<8} {'Seen domains':<22} "
              f"{'AA':>8} {'BWT':>8} {'F':>8} {'RF':>8} {'PRF':>8}")
    print(header)
    print("-" * len(header))
    for t in range(T):
        seen_str = ",".join(str(domain_order[j]) for j in range(t + 1))
        AA_t, BWT_t, F_t, RF_t, PRF_t = cl_metrics[t]
        step_label = f"D{domain_order[t]}"
        print(f"{step_label:<8} {seen_str:<22} "
              f"{AA_t:8.4f} {BWT_t:+8.4f} {F_t:8.4f} {RF_t:8.4f} {PRF_t:8.4f}")

    print(f"\nFinal F1 after last adaptation (row A[{T-1},:]):")
    header2 = "  ".join(f"D{domain_order[j]}" for j in range(T))
    vals2   = "  ".join(
        f"{A[T-1, j]:5.3f}" if not np.isnan(A[T-1, j]) else "  n/a" for j in range(T)
    )
    print(header2)
    print(vals2)

    AA_final, BWT_final, F_final, RF_final, PRF_final = cl_metrics[-1]
    print(f"\nFinal metrics:")
    print(f"  AA={AA_final:.4f}  BWT={BWT_final:+.4f}  F={F_final:.4f}")
    print(f"  RF={RF_final:.4f}  PRF={PRF_final:.4f}   (DualCD-style, relative / perf-aware)")

    # ── Save results ──────────────────────────────────────────────────────────
    tag = args.dataset

    # F1 matrix CSV
    matrix_path = OUT_DIR / f"{tag}_matrix.csv"
    domain_row_labels = [f"D{domain_order[j]}" for j in range(T)]
    with open(matrix_path, "w") as fh:
        fh.write("step," + ",".join(domain_row_labels) + "\n")
        for t in range(T):
            row = f"D{domain_order[t]}," + ",".join(
                f"{A[t,j]:.6f}" if not np.isnan(A[t,j]) else "" for j in range(T)
            )
            fh.write(row + "\n")
    print(f"\nSaved matrix → {matrix_path}")

    # CL metrics CSV (AA, BWT, F + DualCD RF, PRF)
    metrics_path = OUT_DIR / f"{tag}_metrics.csv"
    with open(metrics_path, "w") as fh:
        fh.write("step,seen_domains,AA,BWT,F,RF_dualcd,PRF_dualcd\n")
        for t in range(T):
            seen_str = "|".join(str(domain_order[j]) for j in range(t + 1))
            AA_t, BWT_t, F_t, RF_t, PRF_t = cl_metrics[t]
            fh.write(f"D{domain_order[t]},{seen_str},"
                     f"{AA_t:.6f},{BWT_t:.6f},{F_t:.6f},{RF_t:.6f},{PRF_t:.6f}\n")
    print(f"Saved metrics → {metrics_path}")

    # Routing counts CSV: final adaptation step, per eval domain
    routing_path = OUT_DIR / f"{tag}_routing_counts.csv"
    with open(routing_path, "w") as fh:
        # Header: eval_domain, routed_to_D0, routed_to_D1, ...
        bank_cols = ",".join(f"to_D{domain_order[j]}" for j in range(T))
        fh.write(f"eval_domain,step,{bank_cols}\n")
        for t in range(1, T):
            for j, eval_did in enumerate(domain_order[:t + 1]):
                if eval_did in routing_log.get(t, {}):
                    rc = routing_log[t][eval_did]
                    rc_padded = list(rc) + [0] * (T - len(rc))
                    fh.write(f"D{eval_did},D{domain_order[t]},"
                             + ",".join(str(rc_padded[r]) for r in range(T)) + "\n")
    print(f"Saved routing counts → {routing_path}")

    # Heatmap
    heatmap_path = OUT_DIR / f"{tag}_matrix_heatmap.png"
    save_heatmap(
        A, domain_order, heatmap_path,
        f"Seq-PACT-LoRA — {dataset_label}  (F1 matrix)\n"
        f"{calib_label}, {rank_label}, K={K}",
    )
    print(f"Saved heatmap → {heatmap_path}")

    # AA/Forgetting curve
    curve_path = OUT_DIR / f"{tag}_aa_forgetting_curve.png"
    save_aa_curve(
        cl_metrics, domain_order, curve_path,
        f"Seq-PACT-LoRA — {dataset_label}  ({calib_label}, {rank_label})",
    )
    print(f"Saved curve → {curve_path}")

    return A, cl_metrics


if __name__ == "__main__":
    main()
