"""Genuinely label-free baselines on frozen TSFM embeddings: TENT, source-free
MAPU, and ACCUP. Unlike scripts/run_mapu_baseline.py (which is a
label-supervised variant, kept in the paper with a dagger footnote), these
methods see NO calibration labels on target domains -- only the unlabeled
inputs of the same calibration-sized subset every other baseline uses (same
data budget, no label signal).

  tent      -- Wang et al. 2021: adapt the head by minimizing prediction
               entropy on unlabeled target batches. No pseudo-labels, no
               prototypes -- the purest test-time-adaptation baseline.
  mapu-free -- source-free MAPU: entropy minimization + self-training
               prototype alignment using the model's OWN pseudo-labels
               (argmax predictions), not ground truth. Faithful to the
               source-free premise of MAPU_SFDA_TS (labeled source, unlabeled
               target stream).
  accup     -- Gong et al., KDD 2025 (baselines/ACCUP-main): augmented
               contrastive clustering with uncertainty-aware prototyping.
               Maintains a low-entropy support memory (bootstrapped from the
               source head's own class weights, refreshed every batch to the
               filter_K lowest-entropy examples per pseudo-class), forms
               prototype logits from the memory, picks whichever of
               {prototype, classifier} logits has lower entropy as the
               pseudo-label, and adapts the head with a supervised
               contrastive loss on those pseudo-labels. Ported from the
               original CNN/raw-signal implementation to this repo's
               frozen-embedding-plus-linear-head setting: the same
               adaptation surface TENT and MAPU-free already use here, since
               there is no conv backbone to unfreeze under a frozen TSFM.
               The original also contrasts a raw/augmented view pair; with
               no time-series augmentation pipeline in embedding space this
               collapses to a single view, so the memory and support-set
               mechanics are preserved but the two-view contrastive term is
               not. filter_K/tau/temperature/lr are the paper's published
               per-dataset values for FD->fd, EEG->sleep, HAR->ucihar; HHAR
               and WISDM (not studied in the original paper) reuse the HAR
               setting as the closest available accelerometer-HAR analogue.

The source domain (step 0) is always labeled -- source-free DA assumes a
labeled source and only withholds labels on the target stream. Step 0 reuses
the standard source-fitted head (run_sequential_tsfm.fit_standardizer_and_head
on the FULL source train set), exactly like every other baseline in this repo.

Example:
  python scripts/run_labelfree_baselines.py --dataset fd --method tent --tsfm sundial
  python scripts/run_labelfree_baselines.py --dataset fd --method mapu-free --tsfm sundial
  python scripts/run_labelfree_baselines.py --dataset fd --method accup --tsfm sundial
"""

import argparse
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

import _bootstrap  # noqa: F401
from _bootstrap import DATA_ROOT, RESULTS_ROOT

from run_sequential_tsfm import fit_standardizer_and_head
from src.data import DATASET_NAMES, get_spec
from src.data.calib import calib_indices
from src.data.datasets import adatime_chain
from src.method import compute_cl_metrics
from src.method.pact import extract_embs
from src.models.tsfm import TSFM_MODELS, make_tsfm
from src.utils import get_device, set_seed

METHODS = ["tent", "mapu-free", "accup"]

# Published per-dataset ACCUP hyperparameters (configs/tta_hparams_new.py in
# baselines/ACCUP-main): FD, EEG (=Sleep-EDF/SSC), HAR (=UCI-HAR). HHAR/WISDM
# were not studied in the original paper and reuse the HAR setting.
ACCUP_HPARAMS = {
    "fd": dict(filter_K=100, tau=1.0, temperature=0.6, lr=3e-4),
    "sleep": dict(filter_K=50, tau=50.0, temperature=0.3, lr=1e-5),
    "ucihar": dict(filter_K=10, tau=20.0, temperature=0.7, lr=3e-4),
    "hhar": dict(filter_K=10, tau=20.0, temperature=0.7, lr=3e-4),
    "wisdm": dict(filter_K=10, tau=20.0, temperature=0.7, lr=3e-4),
}


def softmax_entropy(x, x_ema):
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)


def domain_contrastive_loss(features, pseudo_labels, temperature):
    """Supervised contrastive loss over same-pseudo-label pairs within the
    batch (ported from baselines/ACCUP-main/loss/sup_contrast_loss.py)."""
    device = features.device
    anchor = F.normalize(features, dim=1)
    labels = pseudo_labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float()
    logits = (anchor @ anchor.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True)[0].detach()
    self_mask = torch.eye(features.shape[0], device=device)
    logits_mask = 1.0 - self_mask
    mask = mask * logits_mask
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
    mask_sum = mask.sum(1).clamp_min(1.0)
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
    return (-mean_log_prob_pos).mean()


class AccupState:
    """Low-entropy support memory + prototype logits (ported from
    baselines/ACCUP-main/algorithms/accup.py: ACCUP.__init__/select_supports/
    compute_logits). Bootstrapped from the source head's own class weight
    rows, then refreshed every adaptation batch."""

    def __init__(self, head, num_classes, filter_K, tau, device):
        self.num_classes = num_classes
        self.filter_K = filter_K
        self.tau = tau
        with torch.no_grad():
            warmup_supports = head.weight.data.detach().clone()
            warmup_prob = head(warmup_supports)
            warmup_labels = F.one_hot(warmup_prob.argmax(1), num_classes=num_classes).float()
            warmup_ent = softmax_entropy(warmup_prob, warmup_prob)
        self.supports = warmup_supports.to(device)
        self.labels = warmup_labels.to(device)
        self.ents = warmup_ent.to(device)

    def update(self, z, yhat, ent):
        self.supports = torch.cat([self.supports, z.detach()])
        self.labels = torch.cat([self.labels, yhat])
        self.ents = torch.cat([self.ents, ent.detach()])

    def select_supports(self):
        if self.filter_K == -1:
            return
        y_hat = self.labels.argmax(dim=1).long()
        base = torch.arange(len(self.ents), device=self.ents.device)
        indices = []
        for c in range(self.num_classes):
            mask = y_hat == c
            if mask.any():
                order = torch.argsort(self.ents[mask])
                indices.append(base[mask][order][: self.filter_K])
        if not indices:
            return
        indices = torch.cat(indices)
        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ents = self.ents[indices]

    def compute_logits(self, z):
        temp_centroids = (self.labels / (self.labels.sum(0, keepdim=True) + 1e-12)).T @ self.supports
        temp_z = F.normalize(z, dim=1)
        temp_centroids = F.normalize(temp_centroids, dim=1)
        return self.tau * temp_z @ temp_centroids.T


def adapt_head_accup(head, z, state, epochs, lr, temperature):
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(z), batch_size=256, shuffle=True)
    head.train()
    for _ in range(epochs):
        for (zb,) in dl:
            p = head(zb)
            yhat = F.one_hot(p.argmax(1), num_classes=state.num_classes).float()
            ent = softmax_entropy(p, p)
            state.update(zb, yhat, ent)
            state.select_supports()
            prt_scores = state.compute_logits(zb)
            prt_ent = softmax_entropy(prt_scores, prt_scores)
            use_proto = (prt_ent < ent).unsqueeze(1).expand(-1, state.num_classes)
            select_pred = torch.where(use_proto, prt_scores, F.softmax(p, 1))
            pseudo_labels = select_pred.argmax(1).float()

            loss = domain_contrastive_loss(p, pseudo_labels, temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
    head.eval()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=DATASET_NAMES, default="fd")
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--tsfm", choices=list(TSFM_MODELS), default="moment")
    p.add_argument("--domain_order", default="adatime")
    p.add_argument("--calib_frac", type=float, default=0.05)
    p.add_argument("--calib_n", type=int, default=None,
                   help="Fixed unlabeled adaptation budget (overrides calib_frac); "
                        "same instance count as the labeled baselines, no labels used")
    p.add_argument("--head_ep", type=int, default=80)
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--ent_weight", type=float, default=0.1)
    p.add_argument("--proto_weight", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--device", default=None)
    return p.parse_args()


def unlabeled_calib_subset(spec, did, calib_frac, seed, calib_n=None):
    """Same index-selection as the labeled baselines (stratified by the
    hidden true labels, matching everyone else's instance budget) -- but the
    labels themselves are never passed to the adaptation loss below."""
    ds = spec.get_dataset(did, "train", DATA_ROOT)
    idx = calib_indices(ds.y.numpy(), calib_frac, seed, calib_n=calib_n)
    return torch.utils.data.Subset(ds, idx)


def entropy(logits):
    p = F.softmax(logits, 1)
    return -(p * F.log_softmax(logits, 1)).sum(1).mean()


def prototype_align_loss(head, z, pseudo_y, num_classes):
    """Pull each class's classifier weight row toward the mean embedding of
    samples currently pseudo-labeled that class (self-training / nearest-
    class-mean calibration). Embedding-space prototype loss can't move
    anything under a frozen encoder, so this is the gradient-effective
    analogue: it targets the one trainable component (the head) instead."""
    loss = z.new_zeros(())
    for c in range(num_classes):
        m = pseudo_y == c
        if m.any():
            proto = z[m].mean(0).detach()
            loss = loss + F.mse_loss(head.weight[c], proto)
    return loss


def adapt_head_labelfree(head, z, num_classes, method, epochs, lr, ent_weight, proto_weight):
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(z), batch_size=256, shuffle=True)
    head.train()
    for _ in range(epochs):
        for (zb,) in dl:
            opt.zero_grad()
            logits = head(zb)
            loss = ent_weight * entropy(logits)
            if method == "mapu-free":
                pseudo_y = logits.argmax(1).detach()
                loss = loss + proto_weight * prototype_align_loss(head, zb, pseudo_y, num_classes)
            loss.backward()
            opt.step()
    head.eval()


@torch.no_grad()
def eval_f1(model, head, spec, did, device, batch):
    model.eval()
    head.eval()
    ld = torch.utils.data.DataLoader(
        spec.get_dataset(did, "test", DATA_ROOT), batch_size=batch)
    ys, ps = [], []
    for xb, yb in ld:
        z = model.extract(xb.to(device))
        ps.append(head(z).argmax(1).cpu().numpy())
        ys.append(yb.numpy())
    return f1_score(np.concatenate(ys), np.concatenate(ps),
                    average="macro", zero_division=0)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    spec = get_spec(args.dataset)

    if args.domain_order == "adatime":
        order, _ = adatime_chain(args.dataset)
    else:
        order = [int(x) for x in args.domain_order.split(",")]
    labels = [spec.domain_label(d) for d in order]
    T = len(order)
    calib_tag = f"n{args.calib_n}" if args.calib_n else f"c{int(args.calib_frac * 100)}"

    print(f"Label-free baseline={args.method}  tsfm={args.tsfm}  dataset={args.dataset}  "
          f"T={T}  adapt_budget={calib_tag}  seed={args.seed}")

    model = make_tsfm(args.tsfm, spec.num_classes, device=str(device)).to(device)
    fit_standardizer_and_head(model, spec, order[0], device, 150, 1e-3, args.batch, args.seed)
    head = copy.deepcopy(model.head).to(device)  # labeled source head, step 0 reuses this as-is
    for p in head.parameters():
        p.requires_grad_(True)  # fit_standardizer_and_head froze the source model.head

    accup_state, accup_hp = None, None
    if args.method == "accup":
        accup_hp = ACCUP_HPARAMS[args.dataset]
        accup_state = AccupState(head, spec.num_classes, accup_hp["filter_K"],
                                 accup_hp["tau"], device)

    A = np.full((T, T), np.nan)
    A[0, 0] = eval_f1(model, head, spec, order[0], device, args.batch)
    print(f"[{args.method}] Step 0 — {labels[0]:<10} F1={A[0,0]:.4f}  (labeled source)")

    for t in range(1, T):
        calib_ds = unlabeled_calib_subset(spec, order[t], args.calib_frac, args.seed, args.calib_n)
        calib_ld = torch.utils.data.DataLoader(calib_ds, batch_size=args.batch)
        embs, _ = extract_embs(model, calib_ld, device)  # labels discarded, unlabeled target
        z = torch.from_numpy(embs).to(device)
        if args.method == "accup":
            adapt_head_accup(head, z, accup_state, args.head_ep, accup_hp["lr"],
                            accup_hp["temperature"])
        else:
            adapt_head_labelfree(head, z, spec.num_classes, args.method,
                                 args.head_ep, args.head_lr, args.ent_weight, args.proto_weight)
        for j in range(t + 1):
            A[t, j] = eval_f1(model, head, spec, order[j], device, args.batch)
        aa, bwt, f, rf, prf = compute_cl_metrics(A, t)
        print(f"[{args.method}] Step {t} — {labels[t]:<10} AA={aa:.4f} "
              f"BWT={bwt:+.4f} F={f:.4f} RF={rf:.4f}")

    tag = f"{args.dataset}_{args.tsfm}_{args.domain_order}_{args.method}_{calib_tag}_s{args.seed}"
    out_dir = RESULTS_ROOT / "labelfree_baselines" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(A, index=labels, columns=labels).to_csv(out_dir / "f1_matrix.csv")
    aa, bwt, f, rf, prf = compute_cl_metrics(A, T - 1)
    pd.DataFrame([dict(method=args.method, AA=aa, BWT=bwt, F=f, RF=rf, PRF=prf,
                       seed=args.seed)]).to_csv(out_dir / "summary.csv", index=False)
    print(f"\n[{args.method}] Final: AA={aa:.4f}  F={f:.4f}  RF={rf:.4f}\nResults -> {out_dir}")


if __name__ == "__main__":
    main()
