"""
run_sequential_soft_gate.py — Sequential Domain-Incremental with Soft-Mixture Gate
===================================================================================

True sequential CL: domains arrive one at a time.

Algorithm at step t (t = 1 .. T-1):
  1. Draw calib_frac from train_t → calib_t  (default 5%)
  2. Train IdentityAdapter bank  A_{t,c}  for each concept c  using calib_t
  3. Retrain SoftGateMLP per concept  g_c  on accumulated calib (src + 1..t)
       Gate output: 0 = source (bypass), j = domain-j's adapter bank
  4. Evaluate on all seen domains [0..t]:
       soft mixture (main) | hard gate (ablation) | oracle (upper bound)
  5. Compute AA / BWT / F / RF / PRF  — F reported both with and without source

Notes:
  · Host-level forgetting = 0 by construction (adapters frozen, host frozen).
  · System-level forgetting can still occur via routing changes as gate expands.
  · Near-zero BWT therefore has genuine meaning (gate does not destroy old routing).
  · calib_frac default = 0.05 (5% label budget, matching the paper).

Usage:
  python run_sequential_soft_gate.py --dataset hhar
  python run_sequential_soft_gate.py --dataset fd --domain_order 0,1,2,3
  python run_sequential_soft_gate.py --dataset hhar --calib_frac 0.20
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

# Functions look up K / ADAPTER_EP / GATE_EP / SEED / BATCH in e7.__globals__
# at call time — patching e7.K etc. below propagates automatically.
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
                   help="Comma-separated domain IDs, first = source. "
                        "Default FD: 0,1,2,3  HHAR: 0,1,...,8")
    p.add_argument("--k",            type=int,   default=16)
    p.add_argument("--calib_frac",   type=float, default=0.05,
                   help="Label budget as fraction of train_t (default 0.05 = 5%%)")
    p.add_argument("--temperature",  type=float, default=0.5)
    p.add_argument("--adapter_ep",   type=int,   default=200)
    p.add_argument("--gate_ep",      type=int,   default=200)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--batch",        type=int,   default=256)
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
           f"_K{args.k}_c{int(args.calib_frac*100)}_t{args.temperature}")
    return dict(
        load_model=load_model,
        get_train=get_train,
        get_test=get_test,
        domain_order=order,
        domain_labels=labels,
        out_tag=tag,
    )


# ── CL metrics ────────────────────────────────────────────────────────────────

def compute_cl_metrics(A, t, include_source=True):
    """AA, BWT, F, RF, PRF up to step t from NaN-padded matrix A (shape T×T).

    include_source: if True, domain 0 is included in BWT/F/RF/PRF.

    BWT = mean per-domain (final - first-seen).
    F   = mean per-domain max(0, -(final - first-seen)).
            NOT max(0, -mean_BWT) — that hides cancellations between domains.
    """
    seen    = t + 1
    AA      = float(np.nanmean(A[t, :seen]))
    start_j = 0 if include_source else 1

    bwt_vals, f_vals = [], []
    for j in range(start_j, seen):
        if j == t:
            continue  # current domain: first-seen = now, no forgetting yet
        if np.isnan(A[j, j]) or np.isnan(A[t, j]):
            continue
        delta = A[t, j] - A[j, j]
        bwt_vals.append(delta)
        f_vals.append(max(0.0, -delta))

    BWT = float(np.mean(bwt_vals)) if bwt_vals else 0.0
    F   = float(np.mean(f_vals))   if f_vals   else 0.0

    # RF / PRF: peak over intermediate rows [j..t-1] (DualCD definition)
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


# ── Routing probability extractor ─────────────────────────────────────────────

@torch.no_grad()
def get_routing_probs(embs_np, centroids, soft_gates, temperature, K, device):
    """Mean softmax gate probability per route, averaged over all samples.

    Returns array of shape [n_domains].
    None gates (no calib in concept) contribute source-bypass (all mass on route 0).
    """
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


# ── Soft mixture forward pass returning predictions (for accuracy) ─────────────

@torch.no_grad()
def predict_soft_mixture(embs_np, centroids, soft_gates, adapters_list,
                         frozen_head, device, temperature, K):
    """Returns predicted class indices from the soft-mixture forward pass."""
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = build_config(args)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Patch e7 globals so all shared functions pick up CLI overrides
    e7.K          = args.k
    e7.ADAPTER_EP = args.adapter_ep
    e7.GATE_EP    = args.gate_ep
    e7.SEED       = args.seed
    e7.BATCH      = args.batch

    domain_order  = cfg["domain_order"]
    domain_labels = cfg["domain_labels"]
    T             = len(domain_order)
    out_tag       = cfg["out_tag"]

    out_dir = REPO / "results" / "sequential_soft_gate"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Dataset: {args.dataset.upper()}  T={T}  K={args.k}  "
          f"calib={args.calib_frac*100:.0f}%  temp={args.temperature}")
    print(f"Domain order: {domain_labels}")

    # ── Frozen source model ───────────────────────────────────────────────────
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

    # ── Test embeddings extracted once and cached ─────────────────────────────
    print("Extracting test embeddings ...")
    test_embs, test_lbls = {}, {}
    for did, lbl in zip(domain_order, domain_labels):
        ld = torch.utils.data.DataLoader(
            cfg["get_test"](did), batch_size=args.batch)
        test_embs[lbl], test_lbls[lbl] = extract_embs(model, ld, device)

    # ── Sequential loop ───────────────────────────────────────────────────────
    # Three F1 matrices: soft mixture (main), hard gate (ablation), oracle (UB)
    A_soft  = np.full((T, T), np.nan)
    A_hard  = np.full((T, T), np.nan)
    A_orc   = np.full((T, T), np.nan)
    ACC_mat = np.full((T, T), np.nan)   # accuracy for DualCD comparability

    cl_metrics_all = []    # includes source in BWT/F
    f_tgt_list     = []    # F_tgt (targets only) per step

    # routing_log[t][j] = mean routing prob vector over test domain j at step t
    routing_log = {}

    accumulated_calib = []
    adapters_list     = []
    soft_gates        = None

    # ── Step 0: source only ───────────────────────────────────────────────────
    print(f"\n{'='*20} Step 0 — source only {'='*20}")
    with torch.no_grad():
        et = torch.from_numpy(test_embs[domain_labels[0]]).to(device)
        p0 = frozen_head(et).argmax(1).cpu().numpy()
    l0 = test_lbls[domain_labels[0]]
    A_soft[0, 0]  = f1_score(l0, p0, average="macro", zero_division=0)
    A_hard[0, 0]  = A_soft[0, 0]
    A_orc[0, 0]   = A_soft[0, 0]
    ACC_mat[0, 0] = accuracy_score(l0, p0)
    print(f"  {domain_labels[0]:<12} F1={A_soft[0,0]:.4f}  ACC={ACC_mat[0,0]:.4f}")
    cl_metrics_all.append((A_soft[0, 0], 0.0, 0.0, 0.0, 0.0))
    f_tgt_list.append(0.0)

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
        accumulated_calib.append((calib_embs, calib_asgn, calib_lbls))
        print(f"  Calib: {len(calib_embs)} samples from {lbl_t}")

        # 2. Train adapter bank (frozen after)
        print("  Training IdentityAdapter bank ...", flush=True)
        adapters = train_adapters(calib_embs, calib_lbls, centroids, frozen_head, device)
        adapters_list.append(adapters)

        # 3. Retrain (t+1)-way SoftGate on src + all accumulated calib
        print(f"  Retraining {t+1}-way SoftGate ...", flush=True)
        soft_gates = train_soft_gates(src_embs, src_asgn, accumulated_calib, device)
        n_gated = sum(g is not None for g in soft_gates)
        print(f"  {n_gated}/{args.k} concepts gated")

        # 4. Evaluate on all seen domains [0..t]
        print("  Evaluating:")
        routing_log[t] = {}
        for j in range(t + 1):
            lbl_j = domain_labels[j]
            e_j, l_j = test_embs[lbl_j], test_lbls[lbl_j]

            f1_soft = eval_soft_mixture(e_j, l_j, centroids, soft_gates,
                                        adapters_list, frozen_head, device,
                                        temperature=args.temperature)
            f1_hard = eval_hard_gate(e_j, l_j, centroids, soft_gates,
                                     adapters_list, frozen_head, device)
            f1_orc  = eval_oracle(e_j, l_j, centroids, adapters_list,
                                  j, frozen_head, device)

            preds_soft = predict_soft_mixture(e_j, centroids, soft_gates,
                                              adapters_list, frozen_head, device,
                                              args.temperature, args.k)
            acc_soft = accuracy_score(l_j, preds_soft)

            A_soft[t, j]  = f1_soft
            A_hard[t, j]  = f1_hard
            A_orc[t, j]   = f1_orc
            ACC_mat[t, j] = acc_soft

            rp = get_routing_probs(e_j, centroids, soft_gates,
                                   args.temperature, args.k, device)
            routing_log[t][j] = rp

            route_str = " ".join(f"{domain_labels[r]}:{rp[r]*100:.0f}%"
                                 for r in range(len(rp)))
            print(f"    {lbl_j:<12}  soft={f1_soft:.4f}  hard={f1_hard:.4f}  "
                  f"oracle={f1_orc:.4f}  acc={acc_soft:.4f}")
            print(f"      routing → [{route_str}]")

        # 5. CL metrics (per-domain F, not max(0, -BWT))
        AA, BWT, F_all, RF, PRF = compute_cl_metrics(A_soft, t, include_source=True)
        _, _, F_tgt, _, _       = compute_cl_metrics(A_soft, t, include_source=False)
        cl_metrics_all.append((AA, BWT, F_all, RF, PRF))
        f_tgt_list.append(F_tgt)

        print(f"  AA={AA:.4f}  BWT={BWT:+.4f}  "
              f"F_all={F_all:.4f}  F_tgt={F_tgt:.4f}  "
              f"RF={RF:.4f}  PRF={PRF:.4f}")

    # ── Final summary ─────────────────────────────────────────────────────────
    AA_f, BWT_f, F_all_f, RF_f, PRF_f = cl_metrics_all[-1]
    F_tgt_f = f_tgt_list[-1]
    RED, RESET = "\033[91m", "\033[0m"
    print(f"\n{RED}{'='*56}{RESET}")
    print(f"  Dataset : {args.dataset.upper()}  T={T}  K={args.k}  "
          f"calib={args.calib_frac*100:.0f}%  temp={args.temperature}")
    print(f"  Metrics: Macro-F1  (ACC also saved for DualCD comparability)")
    print(f"{RED}  AA     : {AA_f:.4f}{RESET}")
    print(f"{RED}  BWT    : {BWT_f:+.4f}{RESET}")
    print(f"{RED}  F_all  : {F_all_f:.4f}   (src + all targets){RESET}")
    print(f"{RED}  F_tgt  : {F_tgt_f:.4f}   (target domains only){RESET}")
    print(f"{RED}  RF     : {RF_f:.4f}{RESET}")
    print(f"{RED}  PRF    : {PRF_f:.4f}{RESET}")
    print(f"{RED}{'='*56}{RESET}")

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    for name, mat in [("soft_f1",   A_soft),
                      ("hard_f1",   A_hard),
                      ("oracle_f1", A_orc),
                      ("soft_acc",  ACC_mat)]:
        pd.DataFrame(mat, index=domain_labels, columns=domain_labels).to_csv(
            out_dir / f"{out_tag}_{name}_matrix.csv")

    df_m = pd.DataFrame(
        [(m[0], m[1], m[2], ft, m[3], m[4])
         for m, ft in zip(cl_metrics_all, f_tgt_list)],
        columns=["AA", "BWT", "F_all", "F_tgt", "RF", "PRF"],
        index=domain_labels)
    df_m.to_csv(out_dir / f"{out_tag}_metrics.csv")

    # Final routing confusion matrix
    T_final = T - 1
    route_mat = np.full((T, T), np.nan)
    for j in range(T):
        rp = routing_log.get(T_final, {}).get(j)
        if rp is not None:
            route_mat[j, :len(rp)] = rp
    pd.DataFrame(route_mat, index=domain_labels, columns=domain_labels).to_csv(
        out_dir / f"{out_tag}_routing.csv")

    # ── Figures ───────────────────────────────────────────────────────────────
    fs_cell = 7 if T > 6 else 9

    # 1. Three-panel accuracy matrix heatmap
    fig, axes = plt.subplots(1, 3, figsize=(T * 3, max(4, T)))
    for ax, mat, title in [(axes[0], A_soft, "Soft mixture (main)"),
                           (axes[1], A_hard, "Hard gate (ablation)"),
                           (axes[2], A_orc,  "Oracle (upper bound)")]:
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
    fig.suptitle(
        f"Seq Soft-Gate — {args.dataset.upper()}  "
        f"K={args.k}  calib={args.calib_frac*100:.0f}%  temp={args.temperature}\n"
        f"Final: AA={AA_f:.4f}  F_all={F_all_f:.4f}  F_tgt={F_tgt_f:.4f}",
        fontsize=9)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"{out_tag}_heatmap.{ext}", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. CL curve (AA + F_all + F_tgt over steps)
    steps = list(range(T))
    AAs    = [m[0] for m in cl_metrics_all]
    F_alls = [m[2] for m in cl_metrics_all]
    F_tgts = f_tgt_list
    fig2, ax2 = plt.subplots(figsize=(max(7, T + 1), 4))
    ax2.plot(steps, AAs,    "o-",  color="#2ecc71", lw=2, label="AA")
    ax2.plot(steps, F_alls, "s--", color="#e74c3c", lw=2, label="F_all (src + targets)")
    ax2.plot(steps, F_tgts, "^:",  color="#f39c12", lw=2, label="F_tgt (targets only)")
    ax2.set_xticks(steps)
    ax2.set_xticklabels(domain_labels, rotation=30, ha="right", fontsize=8)
    ax2.set_ylim(0, 1.05); ax2.set_ylabel("Macro-F1")
    ax2.legend(fontsize=8); ax2.grid(ls="--", alpha=0.4)
    ax2.set_title(
        f"Seq Soft-Gate CL Curve — {args.dataset.upper()}\n"
        f"Final: AA={AA_f:.4f}  F_all={F_all_f:.4f}  BWT={BWT_f:+.4f}", fontsize=10)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig2.savefig(out_dir / f"{out_tag}_cl_curve.{ext}", dpi=150, bbox_inches="tight")
    plt.close()

    # 3. Routing confusion heatmap (final step)
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
    ax3.set_title(
        f"Routing confusion — {args.dataset.upper()} (final step)\n"
        f"Diagonal = correct self-routing (mean gate probability)",
        fontsize=9)
    plt.colorbar(im3, ax=ax3, fraction=0.03)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig3.savefig(out_dir / f"{out_tag}_routing.{ext}", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nOutputs → {out_dir}/{out_tag}_*")


if __name__ == "__main__":
    main()
