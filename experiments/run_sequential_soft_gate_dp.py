"""
run_sequential_soft_gate_dp.py — DP-PACT: Differentially-Private Sequential Soft-Gate
========================================================================================

DP guarantee: the stored prototype bank (accumulated_prototypes) satisfies
(epsilon_total, delta)-DP under RDP composition of T_mech = T-1 independent
Gaussian mechanisms, one per stored target domain.

What is NOT covered:
  (a) The current domain's raw calibration embeddings, used transiently for
      adapter training and as its own gate class, then discarded.
  (b) The frozen TSLANet encoder weights, treated as public knowledge.

Threat model: honest-but-curious adversary with read access to the stored
prototype bank after all domains have arrived.

How to interpret ε: smaller = stronger privacy = more noise = lower utility.
The sweep_summary.csv and eps_utility figure show this tradeoff.

Algorithm at step t (t = 1 .. T-1):
  1. Draw calib_frac from train_t → calib_t  (same as base, ε-independent)
  2. Select k_proto_per_concept prototypes per concept from calib_t → stored clean
  3. Train IdentityAdapter bank A_{t,c} on raw calib_t  (ε-independent)
  4. Retrain SoftGateMLP on: src + DP-noised protos of domains 1..t-1 + raw calib_t
  5. Evaluate; compute AA / BWT / F / RF / PRF

Usage:
  python run_sequential_soft_gate_dp.py --dataset hhar --epsilon 1.0
  python run_sequential_soft_gate_dp.py --dataset hhar --no_dp
  python run_sequential_soft_gate_dp.py --dataset hhar \\
      --sweep_epsilons "0.1,0.5,1,2,5,10,inf"
  python run_sequential_soft_gate_dp.py --test    # fast correctness check
"""

import sys
import importlib.util
import argparse
import warnings

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

from dp_utils import (
    select_prototypes, clip_embeddings, add_gaussian_noise,
    compute_sigma_from_epsilon, verify_privacy_spent,
)


def _load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


e7  = _load_mod("e7",  REPO / "exp7_soft_gate.py")
tft = _load_mod("tft", REPO / "transfer_fd_tslanet.py")
tht = _load_mod("tht", REPO / "transfer_hhar_tslanet.py")

kmeans_numpy      = e7.kmeans_numpy
assign_concepts   = e7.assign_concepts
extract_embs      = e7.extract_embs
train_adapters    = e7.train_adapters
train_soft_gates  = e7.train_soft_gates
eval_soft_mixture = e7.eval_soft_mixture
eval_hard_gate    = e7.eval_hard_gate
eval_oracle       = e7.eval_oracle
Args              = e7.Args
FD_MAP            = e7.FD_MAP
DATA_ROOT         = e7.DATA_ROOT
CKPT_DIR          = e7.CKPT_DIR


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",      choices=["fd", "hhar"], default="hhar")
    p.add_argument("--domain_order", default=None,
                   help="Comma-separated domain IDs, first = source.")
    p.add_argument("--k",            type=int,   default=16)
    p.add_argument("--calib_frac",   type=float, default=0.05)
    p.add_argument("--temperature",  type=float, default=0.5)
    p.add_argument("--adapter_ep",   type=int,   default=200)
    p.add_argument("--gate_ep",      type=int,   default=200)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--batch",        type=int,   default=256)
    # DP flags
    p.add_argument("--epsilon",      type=float, default=1.0,
                   help="Target total ε across all T domains (default 1.0)")
    p.add_argument("--delta",        type=float, default=1e-5)
    p.add_argument("--clip_norm",    type=float, default=1.0,
                   help="L2 clip threshold for per-prototype sensitivity bounding")
    p.add_argument("--k_proto_per_concept", type=int, default=16,
                   help="Max prototypes per concept per domain (default 5)")
    p.add_argument("--prototype_method", choices=["kmeans", "random", "farthest"],
                   default="kmeans")
    p.add_argument("--no_dp",        action="store_true",
                   help="Skip noise (prototype compression only; utility upper bound)")
    p.add_argument("--sweep_epsilons", default=None,
                   help='Comma-separated ε values to sweep, e.g. "0.1,0.5,1,2,5,10,inf"')
    p.add_argument("--reference_metrics_csv", default=None,
                   help="Path to metrics CSV from run_sequential_soft_gate.py for overlay")
    p.add_argument("--test",         action="store_true",
                   help="Fast mode: T=3 domains, 20 epochs, single ε=1.0, no sweep")
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

    tag = (f"{args.dataset}_"
           f"{'_'.join(str(i) for i in order)}"
           f"_K{args.k}_c{int(args.calib_frac*100)}_t{args.temperature}"
           f"_kp{args.k_proto_per_concept}_pm{args.prototype_method}")
    return dict(
        load_model=load_model,
        get_train=get_train,
        get_test=get_test,
        domain_order=order,
        domain_labels=labels,
        out_tag=tag,
    )


# ── CL metrics (identical to base script) ────────────────────────────────────

def compute_cl_metrics(A, t, include_source=True):
    seen    = t + 1
    AA      = float(np.nanmean(A[t, :seen]))
    start_j = 0 if include_source else 1

    bwt_vals, f_vals = [], []
    for j in range(start_j, seen):
        if j == t:
            continue
        if np.isnan(A[j, j]) or np.isnan(A[t, j]):
            continue
        delta = A[t, j] - A[j, j]
        bwt_vals.append(delta)
        f_vals.append(max(0.0, -delta))

    BWT = float(np.mean(bwt_vals)) if bwt_vals else 0.0
    F   = float(np.mean(f_vals))   if f_vals   else 0.0

    rf_list, prf_list = [], []
    for j in range(start_j, seen):
        if j == t:
            continue
        final_j = A[t, j]
        if np.isnan(final_j):
            continue
        peak_vals = [A[i, j] for i in range(j, t) if not np.isnan(A[i, j])]
        if not peak_vals:
            continue
        peak_j = max(peak_vals)
        drop   = max(0.0, peak_j - final_j)
        rf_j   = drop / (peak_j + 1e-9)
        prf_j  = (1.0 - final_j) * rf_j
        rf_list.append(rf_j); prf_list.append(prf_j)

    RF  = float(np.mean(rf_list))  if rf_list  else 0.0
    PRF = float(np.mean(prf_list)) if prf_list else 0.0
    return AA, BWT, F, RF, PRF


# ── Routing helpers (identical to base script) ────────────────────────────────

@torch.no_grad()
def get_routing_probs(embs_np, centroids, soft_gates, temperature, K, device):
    asgn   = assign_concepts(embs_np, centroids)
    embs_t = torch.from_numpy(embs_np).to(device)

    n_domains = None
    for g in soft_gates:
        if g is not None:
            n_domains = g(embs_t[:1]).shape[1]
            break
    if n_domains is None:
        return np.array([1.0])

    weight_sum = np.zeros(n_domains)
    count = 0
    for c in range(K):
        mask = (asgn == c)
        if not mask.any():
            continue
        gate = soft_gates[c]
        if gate is None:
            w = np.zeros((int(mask.sum()), n_domains))
            w[:, 0] = 1.0
        else:
            gate.eval()
            logits = gate(embs_t[mask])
            w = torch.softmax(logits / temperature, dim=1).cpu().numpy()
        weight_sum += w.sum(axis=0)
        count += int(mask.sum())

    return weight_sum / max(count, 1)


@torch.no_grad()
def predict_soft_mixture(embs_np, centroids, soft_gates, adapters_list,
                         frozen_head, device, temperature, K):
    asgn   = assign_concepts(embs_np, centroids)
    embs_t = torch.from_numpy(embs_np).to(device)
    corr   = torch.zeros_like(embs_t)

    for c in range(K):
        mask_c = (asgn == c)
        if not mask_c.any():
            continue
        gate_c  = soft_gates[c]
        batch_c = embs_t[mask_c]
        if gate_c is None:
            corr[mask_c] = batch_c
            continue
        gate_c.eval()
        wts   = torch.softmax(gate_c(batch_c) / temperature, dim=1)
        mixed = wts[:, 0:1] * batch_c
        for r, adp_r in enumerate(adapters_list):
            adp_r[c].eval()
            mixed = mixed + wts[:, r + 1 : r + 2] * adp_r[c](batch_c)
        corr[mask_c] = mixed

    return frozen_head(corr).argmax(1).cpu().numpy()


# ── Per-domain prototype extraction ──────────────────────────────────────────

def build_proto_bank(calib_embs, calib_asgn, calib_lbls,
                     K, k_proto, method, domain_idx, base_seed):
    """Select clean prototypes from one domain's calibration data.

    Returns:
        (clean_protos, proto_asgn, proto_lbls): arrays with total
        sum(selection_counts_row) rows.
        counts: 1-D int array of length K with selection count per concept.
    """
    proto_e, proto_a, proto_l = [], [], []
    counts = np.zeros(K, dtype=np.int64)
    # Use a per-domain rng so concept order within a domain is reproducible
    rng = np.random.default_rng(base_seed + domain_idx)

    for c in range(K):
        mask_c = (calib_asgn == c)
        embs_c = calib_embs[mask_c]
        lbls_c = calib_lbls[mask_c]
        if len(embs_c) == 0:
            continue
        idx_c = select_prototypes(embs_c, k_proto, method, rng)
        counts[c] = len(idx_c)
        proto_e.append(embs_c[idx_c])
        proto_a.append(np.full(len(idx_c), c, dtype=np.int64))
        proto_l.append(lbls_c[idx_c])

    if proto_e:
        return (np.concatenate(proto_e),
                np.concatenate(proto_a),
                np.concatenate(proto_l)), counts
    else:
        dim = calib_embs.shape[1] if len(calib_embs) > 0 else 128
        return (np.empty((0, dim), np.float32),
                np.array([], np.int64),
                np.array([], calib_lbls.dtype)), counts


# ── Noise injection for one ε run ─────────────────────────────────────────────

def apply_dp_noise(clean_proto_banks, sigma, clip_norm, base_seed, eps_idx):
    """Return noised versions of all clean proto banks for a given ε.

    Noise is drawn once per domain per ε run (seed = base_seed + 1000*eps_idx),
    then reused at all subsequent gate retrainings.  This ensures each domain's
    stored data is protected by exactly one Gaussian mechanism application.
    """
    rng = np.random.default_rng(base_seed + 1000 * eps_idx)
    noised = []
    for clean_p, p_asgn, p_lbls in clean_proto_banks:
        if len(clean_p) == 0 or sigma <= 0.0:
            noised.append((clean_p.copy(), p_asgn.copy(), p_lbls.copy()))
        else:
            clipped = clip_embeddings(clean_p, clip_norm)
            noised_p = add_gaussian_noise(clipped, sigma, rng)
            noised.append((noised_p, p_asgn.copy(), p_lbls.copy()))
    return noised


# ── Core sequential evaluation for one ε ─────────────────────────────────────

def run_sequential_eval(
    src_embs, src_asgn, centroids,
    test_embs, test_lbls,
    raw_calib_banks,    # (calib_embs, calib_asgn, calib_lbls) per domain 1..T-1
    noised_proto_banks, # (noised_protos, proto_asgn, proto_lbls) per domain 1..T-1
    adapters_list,      # adapter bank per domain 1..T-1  (ε-independent)
    selection_counts,   # (T, K) int array
    domain_order, domain_labels, T,
    frozen_head, device, args,
    out_dir, out_tag, sigma, effective_eps,
):
    """Run one full sequential pass and save per-ε outputs.

    Gate training at step t uses:
      - noised protos of past domains 1..t-1  (DP-protected)
      - raw calib of current domain t          (not stored)

    Returns:
        metrics: dict with AA, BWT, F_all, F_tgt, RF, PRF
        final_domain_f1: {label: f1} for each domain at final step
    """
    K = args.k
    A_soft  = np.full((T, T), np.nan)
    A_hard  = np.full((T, T), np.nan)
    A_orc   = np.full((T, T), np.nan)

    cl_metrics_all = []
    f_tgt_list     = []
    routing_log    = {}

    # Step 0: source only
    with torch.no_grad():
        et = torch.from_numpy(test_embs[domain_labels[0]]).to(device)
        p0 = frozen_head(et).argmax(1).cpu().numpy()
    l0 = test_lbls[domain_labels[0]]
    f0 = f1_score(l0, p0, average="macro", zero_division=0)
    A_soft[0, 0] = A_hard[0, 0] = A_orc[0, 0] = f0
    cl_metrics_all.append((f0, 0.0, 0.0, 0.0, 0.0))
    f_tgt_list.append(0.0)

    for t in range(1, T):
        lbl_t = domain_labels[t]

        # Gate training data:
        #   past domains  1..t-1 → noised protos (DP-protected)
        #   current domain t     → raw calib (transient, not stored)
        gate_data = [noised_proto_banks[j] for j in range(t - 1)]
        gate_data.append(raw_calib_banks[t - 1])

        soft_gates = train_soft_gates(src_embs, src_asgn, gate_data, device)
        n_gated = sum(g is not None for g in soft_gates)

        routing_log[t] = {}
        for j in range(t + 1):
            lbl_j = domain_labels[j]
            e_j, l_j = test_embs[lbl_j], test_lbls[lbl_j]

            f1_soft = eval_soft_mixture(e_j, l_j, centroids, soft_gates,
                                        adapters_list[:t], frozen_head, device,
                                        temperature=args.temperature)
            f1_hard = eval_hard_gate(e_j, l_j, centroids, soft_gates,
                                     adapters_list[:t], frozen_head, device)
            f1_orc  = eval_oracle(e_j, l_j, centroids, adapters_list[:t],
                                  j, frozen_head, device)

            A_soft[t, j] = f1_soft
            A_hard[t, j] = f1_hard
            A_orc[t, j]  = f1_orc

            rp = get_routing_probs(e_j, centroids, soft_gates,
                                   args.temperature, K, device)
            routing_log[t][j] = rp

        AA, BWT, F_all, RF, PRF = compute_cl_metrics(A_soft, t, include_source=True)
        _, _, F_tgt, _, _       = compute_cl_metrics(A_soft, t, include_source=False)
        cl_metrics_all.append((AA, BWT, F_all, RF, PRF))
        f_tgt_list.append(F_tgt)

        print(f"  Step {t:2d}/{T-1}  {lbl_t:<12}  "
              f"AA={AA:.4f}  F_tgt={F_tgt:.4f}  "
              f"gates={n_gated}/{K}  σ={sigma:.4f}")

    # Final metrics
    AA_f, BWT_f, F_all_f, RF_f, PRF_f = cl_metrics_all[-1]
    F_tgt_f = f_tgt_list[-1]

    final_domain_f1 = {domain_labels[j]: float(A_soft[T - 1, j])
                       for j in range(T) if not np.isnan(A_soft[T - 1, j])}

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    for name, mat in [("soft_f1",   A_soft),
                      ("hard_f1",   A_hard),
                      ("oracle_f1", A_orc)]:
        pd.DataFrame(mat, index=domain_labels,
                     columns=domain_labels).to_csv(out_dir / f"{name}_matrix.csv")

    df_m = pd.DataFrame(
        [(m[0], m[1], m[2], ft, m[3], m[4])
         for m, ft in zip(cl_metrics_all, f_tgt_list)],
        columns=["AA", "BWT", "F_all", "F_tgt", "RF", "PRF"],
        index=domain_labels)
    df_m.to_csv(out_dir / "metrics.csv")

    # Routing confusion matrix (final step)
    T_fin = T - 1
    route_mat = np.full((T, T), np.nan)
    for j in range(T):
        rp = routing_log.get(T_fin, {}).get(j)
        if rp is not None:
            route_mat[j, :len(rp)] = rp
    pd.DataFrame(route_mat, index=domain_labels,
                 columns=domain_labels).to_csv(out_dir / "routing.csv")

    # sigma_per_step — constant by design; saved for auditability
    pd.DataFrame({"sigma": [sigma] * T},
                 index=domain_labels).to_csv(out_dir / "sigma_per_step.csv")

    # selection_counts (ε-independent copy)
    pd.DataFrame(selection_counts, index=domain_labels,
                 columns=[f"C{c}" for c in range(args.k)]).to_csv(
        out_dir / "selection_counts.csv")

    # ── Heatmap figure ────────────────────────────────────────────────────────
    fs_cell = 7 if T > 6 else 9
    fig, axes = plt.subplots(1, 3, figsize=(T * 3, max(4, T)))
    for ax, mat, title in [(axes[0], A_soft, "Soft mixture (main)"),
                           (axes[1], A_hard, "Hard gate"),
                           (axes[2], A_orc,  "Oracle (UB)")]:
        im = ax.imshow(np.nan_to_num(mat, nan=-0.05), vmin=0, vmax=1, cmap="YlGn")
        for i in range(T):
            for j in range(T):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center",
                            fontsize=fs_cell,
                            color="black" if mat[i, j] > 0.3 else "white")
        ax.set_xticks(range(T)); ax.set_yticks(range(T))
        ax.set_xticklabels(domain_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(domain_labels, fontsize=7)
        ax.set_xlabel("Test domain"); ax.set_ylabel("After step")
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.03)
    eps_str_display = "∞" if np.isinf(effective_eps) else f"{effective_eps:.2f}"
    fig.suptitle(
        f"DP-PACT — {args.dataset.upper()}  K={args.k}  "
        f"ε={eps_str_display}  σ={sigma:.3f}  "
        f"kp={args.k_proto_per_concept}  pm={args.prototype_method}\n"
        f"Final: AA={AA_f:.4f}  F_all={F_all_f:.4f}  F_tgt={F_tgt_f:.4f}",
        fontsize=9)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"{out_tag}_heatmap.{ext}", dpi=150, bbox_inches="tight")
    plt.close()

    # ── CL curve ─────────────────────────────────────────────────────────────
    steps  = list(range(T))
    AAs    = [m[0] for m in cl_metrics_all]
    F_alls = [m[2] for m in cl_metrics_all]
    F_tgts = f_tgt_list
    fig2, ax2 = plt.subplots(figsize=(max(7, T + 1), 4))
    ax2.plot(steps, AAs,    "o-",  color="#2ecc71", lw=2, label="AA")
    ax2.plot(steps, F_alls, "s--", color="#e74c3c", lw=2, label="F_all")
    ax2.plot(steps, F_tgts, "^:",  color="#f39c12", lw=2, label="F_tgt")
    ax2.set_xticks(steps)
    ax2.set_xticklabels(domain_labels, rotation=30, ha="right", fontsize=8)
    ax2.set_ylim(0, 1.05); ax2.set_ylabel("Macro-F1")
    ax2.legend(fontsize=8); ax2.grid(ls="--", alpha=0.4)
    ax2.set_title(
        f"DP-PACT CL Curve — {args.dataset.upper()}  ε={eps_str_display}  σ={sigma:.3f}\n"
        f"Final: AA={AA_f:.4f}  F_all={F_all_f:.4f}  BWT={BWT_f:+.4f}", fontsize=10)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig2.savefig(out_dir / f"{out_tag}_cl_curve.{ext}", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Routing heatmap ───────────────────────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(max(5, T), max(4, T - 1)))
    rm_plot = np.nan_to_num(route_mat, nan=0.0)
    im3 = ax3.imshow(rm_plot, vmin=0, vmax=1, cmap="Blues")
    for i in range(T):
        for j in range(T):
            if not np.isnan(route_mat[i, j]):
                ax3.text(j, i, f"{route_mat[i,j]*100:.0f}%",
                         ha="center", va="center", fontsize=fs_cell,
                         color="white" if route_mat[i, j] > 0.5 else "black")
    ax3.set_xticks(range(T)); ax3.set_yticks(range(T))
    ax3.set_xticklabels(domain_labels, rotation=45, ha="right", fontsize=7)
    ax3.set_yticklabels(domain_labels, fontsize=7)
    ax3.set_xlabel("Route (adapter bank)"); ax3.set_ylabel("Test domain")
    ax3.set_title(f"DP-PACT Routing — {args.dataset.upper()} (final step)\nε={eps_str_display}",
                  fontsize=9)
    plt.colorbar(im3, ax=ax3, fraction=0.03)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig3.savefig(out_dir / f"{out_tag}_routing.{ext}", dpi=150, bbox_inches="tight")
    plt.close()

    metrics = {"AA": AA_f, "BWT": BWT_f, "F_all": F_all_f, "F_tgt": F_tgt_f,
               "RF": RF_f, "PRF": PRF_f}
    return metrics, final_domain_f1


# ── Sweep figures ─────────────────────────────────────────────────────────────

def _eps_x_axis(eps_list):
    """Return (x_positions, xtick_labels) for a mixed finite/inf ε list."""
    finite = [e for e in eps_list if not np.isinf(e)]
    max_fin = max(finite) if finite else 1.0
    inf_x   = max_fin * 5.0
    xs      = [e if not np.isinf(e) else inf_x for e in eps_list]
    lbls    = [f"{e:.3g}" if not np.isinf(e) else "∞" for e in eps_list]
    return xs, lbls, inf_x, bool(finite)


def generate_sweep_figures(sweep_df, sweep_domain_f1, selection_counts,
                            domain_labels, args, out_dir, out_tag, ref_metrics):
    T = len(domain_labels)
    K = args.k
    eps_vals  = list(sweep_df["epsilon_target"])
    xs, xlbls, inf_x, has_finite = _eps_x_axis(eps_vals)

    # ── Figure 1: ε–utility curve ─────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    colors = {"AA": "#2ecc71", "F_all": "#e74c3c", "F_tgt": "#f39c12"}
    markers = {"AA": "o", "F_all": "s", "F_tgt": "^"}
    for metric in ["AA", "F_all", "F_tgt"]:
        ys = list(sweep_df[metric])
        ax1.plot(xs, ys, marker=markers[metric], color=colors[metric],
                 lw=2, label=metric)
        # Mark inf point with a dashed vertical line
        if any(np.isinf(e) for e in eps_vals):
            inf_y = [ys[i] for i, e in enumerate(eps_vals) if np.isinf(e)]
            if inf_y:
                ax1.axhline(inf_y[0], ls=":", color=colors[metric], alpha=0.4, lw=1)

    # Reference baseline (full bank, no DP)
    if ref_metrics is not None:
        for metric, ls, lbl in [("AA",    "--", "Full bank AA"),
                                  ("F_tgt", "--", "Full bank F_tgt")]:
            if metric in ref_metrics:
                ax1.axhline(float(ref_metrics[metric]), ls=ls,
                            color=colors[metric], alpha=0.7, lw=1.5,
                            label=f"Ref {lbl}")

    if has_finite:
        finite_xs = [x for x, e in zip(xs, eps_vals) if not np.isinf(e)]
        ax1.set_xscale("log")
        all_ticks = [x for x in xs]
        ax1.set_xticks(all_ticks)
        ax1.set_xticklabels(xlbls, fontsize=9)
    else:
        ax1.set_xticks(xs); ax1.set_xticklabels(xlbls, fontsize=9)

    ax1.set_xlabel("Privacy budget ε (log scale, ∞ = no noise)", fontsize=10)
    ax1.set_ylabel("Macro-F1", fontsize=10)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=9); ax1.grid(ls="--", alpha=0.4)
    ax1.set_title(
        f"DP-PACT ε–Utility — {args.dataset.upper()}  K={args.k}  "
        f"kp={args.k_proto_per_concept}  pm={args.prototype_method}",
        fontsize=11)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig1.savefig(out_dir / f"{out_tag}_eps_utility.{ext}", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Figure 2: selection counts heatmap (ε-independent) ───────────────────
    # Shape (T-1, K): rows = target domains, columns = concepts
    counts_tgt = selection_counts[1:, :]   # exclude source row (all zeros)
    tgt_labels = domain_labels[1:]

    fig2, ax2 = plt.subplots(figsize=(max(8, K * 0.7), max(3, (T - 1) * 0.6)))
    im2 = ax2.imshow(counts_tgt, aspect="auto", cmap="Blues",
                     vmin=0, vmax=args.k_proto_per_concept)
    for i in range(T - 1):
        for c in range(K):
            ax2.text(c, i, str(counts_tgt[i, c]),
                     ha="center", va="center", fontsize=7,
                     color="white" if counts_tgt[i, c] > args.k_proto_per_concept * 0.6
                     else "black")
    ax2.set_xticks(range(K))
    ax2.set_xticklabels([f"C{c}" for c in range(K)], fontsize=7, rotation=45, ha="right")
    ax2.set_yticks(range(T - 1))
    ax2.set_yticklabels(tgt_labels, fontsize=8)
    ax2.set_xlabel("Concept"); ax2.set_ylabel("Domain")
    ax2.set_title(
        f"Prototypes selected per (domain, concept) — ε-independent\n"
        f"method={args.prototype_method}  k_proto={args.k_proto_per_concept}",
        fontsize=9)
    plt.colorbar(im2, ax=ax2, fraction=0.02, label="# prototypes")
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig2.savefig(out_dir / f"{out_tag}_selection_per_concept.{ext}",
                     dpi=150, bbox_inches="tight")
    plt.close()

    # ── Figure 3: per-domain F1 vs ε ─────────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(9, 5))
    cmap3 = plt.get_cmap("tab10")
    for j, lbl in enumerate(domain_labels):
        ys = [sweep_domain_f1.get(eps_str(e), {}).get(lbl, np.nan)
              for e in eps_vals]
        ax3.plot(xs, ys, marker="o", color=cmap3(j % 10), lw=1.5,
                 label=lbl, alpha=0.85)
    if has_finite:
        ax3.set_xscale("log")
        ax3.set_xticks(xs); ax3.set_xticklabels(xlbls, fontsize=9)
    ax3.set_xlabel("Privacy budget ε", fontsize=10)
    ax3.set_ylabel("Final-step Macro-F1", fontsize=10)
    ax3.set_ylim(0, 1.05)
    ax3.legend(fontsize=7, ncol=3); ax3.grid(ls="--", alpha=0.4)
    ax3.set_title(
        f"DP-PACT Per-Domain F1 vs ε — {args.dataset.upper()}", fontsize=11)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig3.savefig(out_dir / f"{out_tag}_eps_per_domain_acc.{ext}",
                     dpi=150, bbox_inches="tight")
    plt.close()


def eps_str(e):
    return "inf" if np.isinf(e) else f"{e:.2f}"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Test-mode overrides
    if args.test:
        args.adapter_ep = 20
        args.gate_ep    = 20
        T_cap           = 3
        if args.sweep_epsilons is None and not args.no_dp:
            args.epsilon = 1.0
    else:
        T_cap = None

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Patch e7 globals (same pattern as run_sequential_soft_gate.py)
    e7.K          = args.k
    e7.ADAPTER_EP = args.adapter_ep
    e7.GATE_EP    = args.gate_ep
    e7.SEED       = args.seed
    e7.BATCH      = args.batch

    cfg = build_config(args)
    domain_order  = cfg["domain_order"][:T_cap] if T_cap else cfg["domain_order"]
    domain_labels = cfg["domain_labels"][:len(domain_order)]
    T             = len(domain_order)
    out_tag       = cfg["out_tag"]

    out_dir = REPO / "results" / "dp_pact" / out_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device : {device}")
    print(f"Dataset: {args.dataset.upper()}  T={T}  K={args.k}  "
          f"calib={args.calib_frac*100:.0f}%  temp={args.temperature}")
    print(f"DP     : clip={args.clip_norm}  delta={args.delta}  "
          f"k_proto={args.k_proto_per_concept}  method={args.prototype_method}")
    print(f"Domains: {domain_labels}")

    # ── Load frozen source model ──────────────────────────────────────────────
    print(f"\nLoading frozen model for {domain_labels[0]} ...")
    model       = cfg["load_model"](device)
    frozen_head = model.head

    # ── Source embeddings + fixed concept space ───────────────────────────────
    print("Extracting source train embeddings ...")
    src_ld = torch.utils.data.DataLoader(
        cfg["get_train"](domain_order[0]), batch_size=args.batch)
    src_embs, _ = extract_embs(model, src_ld, device)

    print(f"K-means (K={args.k}) ...")
    centroids, src_asgn = kmeans_numpy(src_embs, args.k, seed=args.seed)

    # ── Test embeddings (cached once, ε-independent) ──────────────────────────
    print("Extracting test embeddings ...")
    test_embs, test_lbls = {}, {}
    for did, lbl in zip(domain_order, domain_labels):
        ld = torch.utils.data.DataLoader(
            cfg["get_test"](did), batch_size=args.batch)
        test_embs[lbl], test_lbls[lbl] = extract_embs(model, ld, device)

    # ── Phase 1: per-domain calib / prototypes / adapters (ε-independent) ────
    print("\n── Phase 1: calib extraction, prototype selection, adapter training ──")
    raw_calib_banks  = []   # (calib_embs, calib_asgn, calib_lbls) per domain 1..T-1
    clean_proto_banks = []  # (clean_protos, proto_asgn, proto_lbls) per domain 1..T-1
    adapters_list    = []
    selection_counts = np.zeros((T, args.k), dtype=np.int64)
    # row 0 stays zero (source domain; no prototype selection)

    for t in range(1, T):
        did_t = domain_order[t]
        lbl_t = domain_labels[t]
        print(f"  Domain {t}/{T-1}: {lbl_t}", flush=True)

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
        raw_calib_banks.append((calib_embs, calib_asgn, calib_lbls))

        # Prototype selection
        bank, counts = build_proto_bank(calib_embs, calib_asgn, calib_lbls,
                                        args.k, args.k_proto_per_concept,
                                        args.prototype_method, t, args.seed)
        clean_proto_banks.append(bank)
        selection_counts[t] = counts
        n_proto = int(counts.sum())
        print(f"    calib={len(calib_embs)}  protos={n_proto}  "
              f"(max/concept={args.k_proto_per_concept})")

        # Adapter training (raw calib, ε-independent)
        print(f"    Training adapters ...", flush=True)
        adapters = train_adapters(calib_embs, calib_lbls, centroids, frozen_head, device)
        adapters_list.append(adapters)

    # ── ε run list ────────────────────────────────────────────────────────────
    T_mech = T - 1  # number of stored domains = number of Gaussian mechanisms

    if args.sweep_epsilons:
        eps_list = []
        for s in args.sweep_epsilons.split(","):
            s = s.strip()
            eps_list.append(float("inf") if s.lower() == "inf" else float(s))
    elif args.no_dp:
        eps_list = [float("inf")]
    else:
        eps_list = [args.epsilon]

    # ── Phase 2: sequential eval for each ε ──────────────────────────────────
    print("\n── Phase 2: ε sweep ──")
    sweep_rows        = []
    sweep_domain_f1   = {}   # eps_str -> {domain_label: f1}
    total_protos      = int(selection_counts[1:].sum())

    for eps_idx, eps_val in enumerate(eps_list):
        es = eps_str(eps_val)
        print(f"\n{'='*56}")
        print(f"  ε = {es}  (idx={eps_idx})")

        if np.isinf(eps_val) or args.no_dp:
            sigma        = 0.0
            effective_eps = float("inf")
        else:
            print(f"  Calibrating σ (T_mech={T_mech}) ...", flush=True)
            sigma        = compute_sigma_from_epsilon(eps_val, args.delta, T_mech, args.clip_norm)
            effective_eps = verify_privacy_spent(sigma, args.clip_norm, T_mech, args.delta)
        print(f"  σ = {sigma:.4f}   effective ε = {effective_eps:.4f}")

        # Build noised proto banks (noise drawn once per ε run)
        noised_proto_banks = apply_dp_noise(
            clean_proto_banks, sigma, args.clip_norm, args.seed, eps_idx)

        # Output directory
        if len(eps_list) > 1:
            eps_out_dir = out_dir / f"eps_{es}"
        else:
            eps_out_dir = out_dir
        eps_out_dir.mkdir(parents=True, exist_ok=True)

        metrics, final_f1 = run_sequential_eval(
            src_embs, src_asgn, centroids,
            test_embs, test_lbls,
            raw_calib_banks,
            noised_proto_banks,
            adapters_list,
            selection_counts,
            domain_order, domain_labels, T,
            frozen_head, device, args,
            eps_out_dir, out_tag, sigma, effective_eps,
        )

        sweep_domain_f1[es] = final_f1
        sweep_rows.append({
            "epsilon_target":        eps_val,
            "sigma":                 sigma,
            "effective_epsilon":     effective_eps,
            "AA":                    metrics["AA"],
            "BWT":                   metrics["BWT"],
            "F_all":                 metrics["F_all"],
            "F_tgt":                 metrics["F_tgt"],
            "RF":                    metrics["RF"],
            "PRF":                   metrics["PRF"],
            "total_prototypes_stored": total_protos,
        })

        RED, RESET = "\033[91m", "\033[0m"
        print(f"\n{RED}  ε={es}  AA={metrics['AA']:.4f}  "
              f"F_tgt={metrics['F_tgt']:.4f}  BWT={metrics['BWT']:+.4f}{RESET}")

    # ── Top-level selection counts (ε-independent) ───────────────────────────
    pd.DataFrame(selection_counts, index=domain_labels,
                 columns=[f"C{c}" for c in range(args.k)]).to_csv(
        out_dir / f"{out_tag}_selection_counts.csv")

    # ── Sweep-level outputs ───────────────────────────────────────────────────
    if len(eps_list) > 1:
        sweep_df = pd.DataFrame(sweep_rows)
        sweep_df.to_csv(out_dir / "sweep_summary.csv", index=False)
        print(f"\nSweep summary → {out_dir}/sweep_summary.csv")

        # Load reference metrics
        ref_metrics = None
        if args.reference_metrics_csv:
            try:
                ref_df      = pd.read_csv(args.reference_metrics_csv, index_col=0)
                ref_metrics = ref_df.iloc[-1]
                print(f"Reference metrics loaded from {args.reference_metrics_csv}")
            except Exception as exc:
                warnings.warn(f"Could not load reference metrics: {exc}")

        generate_sweep_figures(sweep_df, sweep_domain_f1, selection_counts,
                               domain_labels, args, out_dir, out_tag, ref_metrics)
        print(f"Sweep figures → {out_dir}/{out_tag}_eps_utility.png  (+ 2 more)")

    print(f"\nAll outputs → {out_dir}/")


if __name__ == "__main__":
    main()
