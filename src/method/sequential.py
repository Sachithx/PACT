"""Sequential domain-incremental continual learning with PACT.

Domains arrive one at a time. At step t:
  1. draw a calib_frac labelled slice from domain t;
  2. train a frozen per-concept adapter bank A_t for it;
  3. retrain the (t+1)-way soft gate on accumulated calibration (src + 1..t);
  4. evaluate on all seen domains [0..t] (soft mixture / hard gate / oracle);
  5. accumulate CL metrics.

Host-level forgetting is 0 by construction (backbone + head + past adapters
all frozen); non-trivial BWT can only come from gate routing changes, so
near-zero BWT is a genuine result.

CL metrics follow Lopez-Paz & Ranzato (GEM, NeurIPS 2017) for AA/BWT/F and the
DualCD definitions for RF/PRF.
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from ..data.calib import calib_indices
from ..models.concepts import assign_concepts, kmeans_numpy
from ..models.gate import (build_prototype_gates, build_prototype_gates_diag,
                           build_prototype_gates_lowrank, build_confidence_gates,
                           build_prototype_gates_masked, build_mixture_prototype_gates,
                           build_analytic_gates)
from .pact import (PactConfig, eval_hard_gate, eval_oracle, eval_soft_mixture,
                   extract_embs, train_adapters, train_soft_gates)


def fit_gates(src_embs, src_asgn, accumulated_calib, device, cfg: PactConfig,
              adapters_list=None, frozen_head=None, src_lbls=None):
    """Dispatch on cfg.gate_kind: "learned" (MLP, re-fit on all accumulated
    calibration -- rehearsal) vs. the rehearsal-free family -- "prototype"
    (parameter-free, O(1) per new domain, see models.gate.build_prototype_gates),
    "prototype_diag"/"prototype_lowrank" (richer covariance), "prototype_masked"
    (isotropic with the absent-cell mask fix), "prototype_mixture"
    (class-conditional, uses calibration labels), "analytic" (ridge-regression
    domain classifier over fixed random features, see models.gate.
    build_analytic_gates), or "confidence" (no stored embeddings at all)."""
    if cfg.gate_kind == "learned":
        return train_soft_gates(src_embs, src_asgn, accumulated_calib, device, cfg)
    if cfg.gate_kind == "prototype":
        return build_prototype_gates(src_embs, src_asgn, accumulated_calib, cfg.k,
                                     cfg.emb_dim, device)
    if cfg.gate_kind == "prototype_diag":
        return build_prototype_gates_diag(src_embs, src_asgn, accumulated_calib,
                                         cfg.k, cfg.emb_dim, device)
    if cfg.gate_kind == "prototype_lowrank":
        return build_prototype_gates_lowrank(src_embs, src_asgn, accumulated_calib,
                                            cfg.k, cfg.emb_dim, device)
    if cfg.gate_kind == "prototype_masked":
        return build_prototype_gates_masked(src_embs, src_asgn, accumulated_calib,
                                           cfg.k, cfg.emb_dim, device)
    if cfg.gate_kind == "prototype_mixture":
        return build_mixture_prototype_gates(src_embs, src_asgn, accumulated_calib,
                                            cfg.k, cfg.emb_dim, device, src_lbls=src_lbls)
    if cfg.gate_kind == "analytic":
        gates, _, _ = build_analytic_gates(src_embs, src_asgn, accumulated_calib,
                                          cfg.k, cfg.emb_dim, device,
                                          n_features=cfg.analytic_n_features,
                                          lam_rel=cfg.analytic_lam_rel, seed=cfg.seed)
        return gates
    if cfg.gate_kind == "confidence":
        assert adapters_list is not None and frozen_head is not None
        return build_confidence_gates(adapters_list, frozen_head, cfg.k)
    raise ValueError(f"Unknown gate_kind: {cfg.gate_kind!r}")


# ── CL metrics ─────────────────────────────────────────────────────────────────

def compute_cl_metrics(A, t, include_source=True):
    """AA, BWT, F, RF, PRF up to step t from a NaN-padded T x T matrix A.

    BWT = mean per-domain (final - first-seen), i.e. A[t,j] - A[j,j].
    F   = standard GEM forgetting: mean_j [max_{l in j..t} A[l,j] - A[t,j]]_+.
    RF/PRF normalize the same peak-to-final drop used by F.
    """
    seen = t + 1
    AA = float(np.nanmean(A[t, :seen]))
    start_j = 0 if include_source else 1

    bwt_vals = []
    for j in range(start_j, seen):
        if j == t or np.isnan(A[j, j]) or np.isnan(A[t, j]):
            continue
        bwt_vals.append(A[t, j] - A[j, j])
    BWT = float(np.mean(bwt_vals)) if bwt_vals else 0.0

    f_vals, rf_list, prf_list = [], [], []
    for j in range(start_j, seen):
        if j == t:
            continue
        final_j = A[t, j]
        if np.isnan(final_j):
            continue
        peak_vals = [A[i, j] for i in range(j, t + 1) if not np.isnan(A[i, j])]
        if not peak_vals:
            continue
        peak_j = max(peak_vals)
        drop = max(0.0, peak_j - final_j)
        f_vals.append(drop)
        rf_j = drop / (peak_j + 1e-9)
        rf_list.append(rf_j)
        prf_list.append((1.0 - final_j) * rf_j)
    F = float(np.mean(f_vals)) if f_vals else 0.0
    RF = float(np.mean(rf_list)) if rf_list else 0.0
    PRF = float(np.mean(prf_list)) if prf_list else 0.0
    return AA, BWT, F, RF, PRF


@torch.no_grad()
def predict_soft_mixture(embs_np, centroids, soft_gates, adapters_list,
                         frozen_head, device, temperature, k):
    """Soft-mixture forward returning predicted class indices (for accuracy)."""
    asgn = assign_concepts(embs_np, centroids)
    embs_t = torch.from_numpy(embs_np).to(device)
    corr = torch.zeros_like(embs_t)
    for c in range(k):
        mask_c = (asgn == c)
        if not mask_c.any():
            continue
        gate_c = soft_gates[c]
        batch_c = embs_t[mask_c]
        if gate_c is None:
            corr[mask_c] = batch_c
            continue
        gate_c.eval()
        wts = torch.softmax(gate_c(batch_c) / temperature, dim=1)
        mixed = wts[:, 0:1] * batch_c
        for r, adp_r in enumerate(adapters_list):
            adp_r[c].eval()
            mixed = mixed + wts[:, r + 1: r + 2] * adp_r[c](batch_c)
        corr[mask_c] = mixed
    return frozen_head(corr).argmax(1).cpu().numpy()


@torch.no_grad()
def margin_condition_diagnostics(embs_np, labels_np, centroids, soft_gates,
                                 adapters_list, frozen_head, device,
                                 temperature, k):
    """Evaluate Proposition 1's pointwise condition on labeled embeddings."""
    assignments = assign_concepts(embs_np, centroids)
    embs = torch.from_numpy(embs_np).to(device)
    labels = torch.from_numpy(labels_np).long().to(device)
    identity_mass = torch.ones(len(embs), device=device)
    residual_bound = torch.zeros(len(embs), device=device)

    for c in range(k):
        mask_np = assignments == c
        if not mask_np.any():
            continue
        indices = torch.from_numpy(np.flatnonzero(mask_np)).long().to(device)
        batch = embs[indices]
        gate = soft_gates[c] if c < len(soft_gates) else None
        if gate is not None:
            gate.eval()
            identity_mass[indices] = torch.softmax(
                gate(batch) / temperature, dim=1
            )[:, 0]

        residual_norms = []
        for adapters in adapters_list:
            adapters[c].eval()
            residual_norms.append(torch.linalg.vector_norm(adapters[c](batch) - batch, dim=1))
        if residual_norms:
            residual_bound[indices] = torch.stack(residual_norms, dim=1).amax(dim=1)

    logits = frozen_head(embs)
    true_logits = logits.gather(1, labels[:, None]).squeeze(1)
    competing_logits = logits.clone()
    competing_logits[torch.arange(len(labels), device=device), labels] = -torch.inf
    margin = true_logits - competing_logits.amax(dim=1)
    lipschitz = torch.linalg.vector_norm(frozen_head.weight, dim=1).amax()
    displacement_bound = (1.0 - identity_mass) * residual_bound
    margin_threshold = margin / (2.0 * lipschitz)
    correct = margin > 0
    violation = correct & (displacement_bound >= margin_threshold)

    return {
        "chart": assignments,
        "correct": correct.cpu().numpy(),
        "identity_mass": identity_mass.cpu().numpy(),
        "residual_bound": residual_bound.cpu().numpy(),
        "displacement_bound": displacement_bound.cpu().numpy(),
        "margin": margin.cpu().numpy(),
        "margin_threshold": margin_threshold.cpu().numpy(),
        "violation": violation.cpu().numpy(),
    }


@torch.no_grad()
def target_recovery_diagnostics(tgt_embs_np, tgt_lbls_np, own_domain_rank,
                                src_embs_np, src_lbls_np, centroids, soft_gates,
                                adapters_list, frozen_head, device, temperature, k):
    """Evaluate Proposition 2's pointwise recovery condition for one target
    domain's test embeddings against the atlas fitted by run_sequential().

    own_domain_rank is the 1-indexed position of this domain among the
    adapted domains: adapters_list[own_domain_rank-1] is its own transport
    bank and gate column `own_domain_rank` is its own routing branch (column
    0 is always the source/identity branch).

    For each target embedding z assigned to chart c, we search source chart-c
    embeddings sigma(z) of the same true label that the frozen head already
    classifies correctly, and take the nearest one (by raw distance to z,
    not to the transported point, to avoid cherry-picking an anchor that
    flatters the transport). This realizes epsilon_c(z) =
    ||T_c^own(z) - sigma(z)||_2 and gamma(z) = m(sigma(z)) from the paper's
    epsilon-chart-relatedness condition. Samples with no eligible source
    anchor in their chart (has_anchor=False) cannot be evaluated and should
    be excluded before checking the condition.
    """
    tgt_asgn = assign_concepts(tgt_embs_np, centroids)
    src_asgn = assign_concepts(src_embs_np, centroids)
    tgt_embs = torch.from_numpy(tgt_embs_np).to(device)
    src_embs = torch.from_numpy(src_embs_np).to(device)
    src_labels = torch.from_numpy(src_lbls_np).long().to(device)

    src_logits = frozen_head(src_embs)
    src_true = src_logits.gather(1, src_labels[:, None]).squeeze(1)
    src_comp = src_logits.clone()
    src_comp[torch.arange(len(src_labels), device=device), src_labels] = -torch.inf
    src_margin = (src_true - src_comp.amax(dim=1)).cpu().numpy()
    src_correct = src_margin > 0

    own_bank = adapters_list[own_domain_rank - 1]
    lipschitz = float(torch.linalg.vector_norm(frozen_head.weight, dim=1).amax().cpu())

    n = len(tgt_embs_np)
    eps = np.full(n, np.nan)
    gamma = np.full(n, np.nan)
    has_anchor = np.zeros(n, dtype=bool)
    pi_own = np.zeros(n)
    residual_bound = np.zeros(n)
    isolated_correct = np.zeros(n, dtype=bool)
    deployed_correct = np.zeros(n, dtype=bool)

    for c in range(k):
        mask = tgt_asgn == c
        if not mask.any():
            continue
        idx = np.flatnonzero(mask)
        batch = tgt_embs[idx]
        y_np = tgt_lbls_np[idx]

        own_bank[c].eval()
        A_z = own_bank[c](batch)
        isolated_correct[idx] = frozen_head(A_z).argmax(1).cpu().numpy() == y_np

        anchor_mask = (src_asgn == c) & src_correct
        if anchor_mask.any():
            anchor_idx = np.flatnonzero(anchor_mask)
            anchor_embs = src_embs[anchor_idx]
            anchor_lbls = src_lbls_np[anchor_idx]
            anchor_margin = src_margin[anchor_idx]
            for row, gi in enumerate(idx):
                same_label = anchor_lbls == y_np[row]
                if not same_label.any():
                    continue
                cand = anchor_embs[same_label]
                # sigma(z) is existentially quantified in the relatedness
                # condition, so the tightest legitimate witness is the
                # correctly-classified same-label source point nearest to
                # the transported point A(z), not to the raw z.
                d = torch.linalg.vector_norm(cand - A_z[row:row + 1], dim=1)
                j = int(torch.argmin(d).item())
                eps[gi] = float(d[j].cpu())
                gamma[gi] = float(anchor_margin[same_label][j])
                has_anchor[gi] = True

        gate = soft_gates[c] if c < len(soft_gates) else None
        if gate is not None:
            gate.eval()
            wts = torch.softmax(gate(batch) / temperature, dim=1)
            pi_own[idx] = wts[:, own_domain_rank].cpu().numpy()
            mixed = wts[:, 0:1] * batch
            for r, bank in enumerate(adapters_list):
                bank[c].eval()
                mixed = mixed + wts[:, r + 1:r + 2] * bank[c](batch)
        else:
            mixed = batch
        deployed_correct[idx] = frozen_head(mixed).argmax(1).cpu().numpy() == y_np

        residual_norms = []
        for bank in adapters_list:
            bank[c].eval()
            residual_norms.append(torch.linalg.vector_norm(bank[c](batch) - batch, dim=1))
        if residual_norms:
            residual_bound[idx] = torch.stack(residual_norms, dim=1).amax(dim=1).cpu().numpy()

    lhs = eps + 2.0 * (1.0 - pi_own) * residual_bound
    rhs = gamma / (2.0 * lipschitz)

    return {
        "chart": tgt_asgn,
        "true_label": tgt_lbls_np,
        "eps": eps,
        "gamma": gamma,
        "pi_own": pi_own,
        "B": residual_bound,
        "lhs": lhs,
        "rhs": rhs,
        "has_anchor": has_anchor,
        "isolated_correct": isolated_correct,
        "deployed_correct": deployed_correct,
    }


# ── Sequential procedure ────────────────────────────────────────────────────────

def run_sequential(model, spec, domain_order, data_root, device, cfg: PactConfig,
                   temperature=0.5, verbose=True):
    """Run the sequential CL procedure. Returns a results dict of matrices/metrics.

    model: frozen source backbone (already trained on domain_order[0]).
    spec:  DatasetSpec.  domain_order: list of int domain ids (first = source).
    """
    frozen_head = model.head
    labels = [spec.domain_label(d) for d in domain_order]
    T = len(domain_order)

    src_ld = torch.utils.data.DataLoader(
        spec.get_dataset(domain_order[0], "train", data_root), batch_size=cfg.batch)
    src_embs, src_lbls_all = extract_embs(model, src_ld, device)
    # Class labels are not retained for the default embedding-key buffer.
    src_lbls = src_lbls_all if cfg.gate_kind == "prototype_mixture" else None
    del src_lbls_all
    centroids, src_asgn = kmeans_numpy(src_embs, cfg.k, seed=cfg.seed,
                                       n_init=cfg.kmeans_n_init)

    test_embs, test_lbls = {}, {}
    for did, lbl in zip(domain_order, labels):
        ld = torch.utils.data.DataLoader(
            spec.get_dataset(did, "test", data_root), batch_size=cfg.batch)
        test_embs[lbl], test_lbls[lbl] = extract_embs(model, ld, device)

    A_soft = np.full((T, T), np.nan)
    A_hard = np.full((T, T), np.nan)
    A_orc = np.full((T, T), np.nan)
    ACC = np.full((T, T), np.nan)
    metrics_rows = []

    # Step 0: source only
    with torch.no_grad():
        et = torch.from_numpy(test_embs[labels[0]]).to(device)
        p0 = frozen_head(et).argmax(1).cpu().numpy()
    l0 = test_lbls[labels[0]]
    A_soft[0, 0] = A_hard[0, 0] = A_orc[0, 0] = f1_score(l0, p0, average="macro", zero_division=0)
    ACC[0, 0] = accuracy_score(l0, p0)
    metrics_rows.append(dict(step=0, domain=labels[0], AA=A_soft[0, 0],
                             BWT=0.0, F_all=0.0, F_tgt=0.0, RF=0.0, PRF=0.0))
    if verbose:
        print(f"Step 0 — {labels[0]:<10} F1={A_soft[0,0]:.4f}")

    accumulated_calib, adapters_list = [], []
    for t in range(1, T):
        did_t, lbl_t = domain_order[t], labels[t]
        ds = spec.get_dataset(did_t, "train", data_root)
        y_all = ds.y.numpy()
        idx = calib_indices(y_all, cfg.calib_frac, cfg.seed, calib_n=cfg.calib_n)
        calib_ld = torch.utils.data.DataLoader(
            torch.utils.data.Subset(ds, idx), batch_size=cfg.batch)
        calib_embs, _ = extract_embs(model, calib_ld, device)
        calib_lbls = y_all[idx]
        calib_asgn = assign_concepts(calib_embs, centroids)
        # Default gates need embedding keys only (no class labels / values).
        # Keep labels in the buffer solely for prototype_mixture.
        if cfg.gate_kind == "prototype_mixture":
            accumulated_calib.append((calib_embs, calib_asgn, calib_lbls))
        else:
            accumulated_calib.append((calib_embs, calib_asgn))

        adapters = train_adapters(calib_embs, calib_lbls, centroids, frozen_head, device, cfg)
        adapters_list.append(adapters)
        soft_gates = fit_gates(src_embs, src_asgn, accumulated_calib, device, cfg,
                               adapters_list=adapters_list, frozen_head=frozen_head,
                               src_lbls=src_lbls if cfg.gate_kind == "prototype_mixture" else None)

        for j in range(t + 1):
            lbl_j = labels[j]
            e_j, l_j = test_embs[lbl_j], test_lbls[lbl_j]
            A_soft[t, j] = eval_soft_mixture(e_j, l_j, centroids, soft_gates,
                                             adapters_list, frozen_head, device,
                                             cfg.k, temperature)
            A_hard[t, j] = eval_hard_gate(e_j, l_j, centroids, soft_gates,
                                          adapters_list, frozen_head, device, cfg.k)
            A_orc[t, j] = eval_oracle(e_j, l_j, centroids, adapters_list, j,
                                      frozen_head, device, cfg.k)
            preds = predict_soft_mixture(e_j, centroids, soft_gates, adapters_list,
                                         frozen_head, device, temperature, cfg.k)
            ACC[t, j] = accuracy_score(l_j, preds)

        AA, BWT, F_all, RF, PRF = compute_cl_metrics(A_soft, t, include_source=True)
        _, _, F_tgt, _, _ = compute_cl_metrics(A_soft, t, include_source=False)
        metrics_rows.append(dict(step=t, domain=lbl_t, AA=AA, BWT=BWT,
                                 F_all=F_all, F_tgt=F_tgt, RF=RF, PRF=PRF))
        if verbose:
            print(f"Step {t} — {lbl_t:<10} AA={AA:.4f} BWT={BWT:+.4f} "
                  f"F_all={F_all:.4f} F_tgt={F_tgt:.4f} RF={RF:.4f} PRF={PRF:.4f}")

    margin_diagnostics = margin_condition_diagnostics(
        test_embs[labels[0]], test_lbls[labels[0]], centroids, soft_gates,
        adapters_list, frozen_head, device, temperature, cfg.k,
    )
    return dict(labels=labels, A_soft=A_soft, A_hard=A_hard, A_orc=A_orc,
                ACC=ACC, metrics=metrics_rows, centroids=centroids,
                adapters_list=adapters_list, soft_gates=soft_gates,
                margin_diagnostics=margin_diagnostics)
