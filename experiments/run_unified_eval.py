"""
run_unified_eval.py — Unified Evaluation Harness

Datasets  : FD (fault diagnosis, 10 AdaTime scenarios)
            HHAR (cross-user HAR, 10 AdaTime scenarios)
Label budget : 5% stratified target-domain train
Backbone  : TSLANet (same pre-trained checkpoints for all methods)

Methods:
  ── Label-free ──────────────────────────────────────────────────────────────
  NO_ADAPT    source-only lower bound
  DANN        adversarial gradient reversal (port to TSLANet embeddings)
  Deep_Coral  CORAL covariance alignment   (port to TSLANet embeddings)
  MMDA        MMD + CORAL + cond. entropy  (port to TSLANet embeddings)
  TENT        LayerNorm entropy minimisation (uses full unlabeled target train)

  ── Calibration (5% labeled target train) ───────────────────────────────────
  Full FT     full model fine-tune
  Head FT     head-only fine-tune
  EWC         Elastic Weight Consolidation (standard Fisher)
  LwF         Learning without Forgetting (distill on source, clf on calib)
  MAS         Memory Aware Synapses
  PACT (ng)   frozen host + k=16 adapters, no gate
  PACT        frozen host + k=16 adapters + learned gate MLP  ← main result

Run:
  CUDA_VISIBLE_DEVICES=0 \\
  LD_PRELOAD=/home/AD/sachith/.conda/envs/entrope/lib/libstdc++.so.6 \\
  /home/AD/sachith/.conda/envs/entrope/bin/python -u run_unified_eval.py
"""

import sys, copy, itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit

# ── Path setup ─────────────────────────────────────────────────────────────────
REPO = Path(__file__).parent
sys.path.insert(0, str(REPO / "TSLANet" / "Classification"))
sys.path.append(str(REPO / "baselines" / "AdaTime"))

import importlib.util

def _load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

tft = _load_mod("tft", REPO / "transfer_fd_tslanet.py")
tht = _load_mod("tht", REPO / "transfer_hhar_tslanet.py")
tst = _load_mod("tst", REPO / "transfer_sleep_tslanet.py")

# AdaTime losses (no relative imports inside loss.py)
_loss       = _load_mod("adatime_loss", REPO / "baselines" / "AdaTime" / "models" / "loss.py")
CORAL       = _loss.CORAL
MMD_loss    = _loss.MMD_loss
CondEntLoss = _loss.ConditionalEntropyLoss
NTXentLoss  = _loss.NTXentLoss
SupConLoss  = _loss.SupConLoss

from torch.autograd import Function as _Fn
class ReverseLayerF(_Fn):
    @staticmethod
    def forward(ctx, x, alpha): ctx.alpha = alpha; return x.view_as(x)
    @staticmethod
    def backward(ctx, g): return g.neg() * ctx.alpha, None

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ─────────────────────────────────────────────────────────────────────

DATA_ROOT  = REPO / "dataset"
CKPT_DIR   = REPO / "TSLANet" / "Classification" / "output" / "tslanet"
OUT_DIR    = REPO / "results" / "unified_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FD_MAP    = {0: "FD-A", 1: "FD-B", 2: "FD-C", 3: "FD-D"}
HHAR_NAMES = {i: f"User-{chr(ord('a')+i)}" for i in range(9)}

FD_SCENARIOS   = [(0,1),(0,3),(1,0),(1,2),(1,3),(2,1),(2,3),(3,0),(3,1),(3,2)]
HHAR_SCENARIOS = [(0,6),(1,6),(2,7),(3,8),(4,5),(5,0),(6,1),(7,4),(8,3),(0,2)]
SLEEP_SCENARIOS = tst.SLEEP_SCENARIOS   # 10 AdaTime EEG cross-subject scenarios

EMB_DIM     = 128
CALIB_FRAC  = 0.05        # 5% label budget
SEED        = 42
BATCH       = 256

# UDA hyperparams
UDA_EPOCHS  = 40
UDA_LR      = 1e-4
DISC_HID    = 64

# Calibration-method hyperparams
FT_EPOCHS   = 80
FT_LR       = 1e-4
EWC_LAMBDA  = 1_000_000   # large: Fisher is degenerate for near-perfect src models (see run_ewc)
LWF_TEMP    = 2.0
LWF_ALPHA   = 0.5
MAS_LAMBDA  = 10          # diagnostic: lam=1000 → penalty 22×CE (degenerate); lam=10 → ~22% CE
ADAPTER_EP  = 200
ADAPTER_LR  = 1e-3
GATE_EP     = 200
GATE_LR     = 1e-3
K           = 16
PENALTY     = 1e-3


class Args:
    emb_dim=128; depth=2; patch_size=8; dropout_rate=0.15; masking_ratio=0.4
    ICB=True; ASB=True; adaptive_filter=True
    data_root=str(DATA_ROOT); ckpt_dir=str(CKPT_DIR)
    batch_size=256; pretrain=True; pretrain_epochs=40; pretrain_lr=1e-4
    cls_epochs=80; cls_lr=1e-4; retrain=False


# ── Data helpers ───────────────────────────────────────────────────────────────

def make_loader(ds, shuffle=False, drop_last=False, batch_size=None):
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size or BATCH, shuffle=shuffle, drop_last=drop_last)


def calib_split(full_train_ds, frac=CALIB_FRAC):
    """Stratified frac-split → calib subset."""
    y = torch.stack([full_train_ds[i][1] for i in range(len(full_train_ds))]).numpy()
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1-frac, random_state=SEED)
    idx, _ = next(sss.split(np.zeros(len(y)), y))
    return torch.utils.data.Subset(full_train_ds, idx)


# ── Dataset loaders (FD and HHAR share the same interface) ────────────────────

def load_fd_datasets(src_id, tgt_id):
    src_name = FD_MAP[src_id]; tgt_name = FD_MAP[tgt_id]
    return dict(
        src_name=src_name, tgt_name=tgt_name,
        src_train=tft.FDDataset(src_name, "train", DATA_ROOT),
        src_test =tft.FDDataset(src_name, "test",  DATA_ROOT),
        tgt_train=tft.FDDataset(tgt_name, "train", DATA_ROOT),
        tgt_test =tft.FDDataset(tgt_name, "test",  DATA_ROOT),
    )


def load_hhar_datasets(src_id, tgt_id):
    src_name = HHAR_NAMES[src_id]; tgt_name = HHAR_NAMES[tgt_id]
    return dict(
        src_name=src_name, tgt_name=tgt_name,
        src_train=tht.HHARDataset(src_id, "train", DATA_ROOT),
        src_test =tht.HHARDataset(src_id, "test",  DATA_ROOT),
        tgt_train=tht.HHARDataset(tgt_id, "train", DATA_ROOT),
        tgt_test =tht.HHARDataset(tgt_id, "test",  DATA_ROOT),
    )


def load_sleep_datasets(src_id, tgt_id):
    src_name = f"Sleep-{src_id}"; tgt_name = f"Sleep-{tgt_id}"
    return dict(
        src_name=src_name, tgt_name=tgt_name,
        src_train=tst.SleepEDFDataset(src_id, "train", DATA_ROOT),
        src_test =tst.SleepEDFDataset(src_id, "test",  DATA_ROOT),
        tgt_train=tst.SleepEDFDataset(tgt_id, "train", DATA_ROOT),
        tgt_test =tst.SleepEDFDataset(tgt_id, "test",  DATA_ROOT),
    )


class SleepArgs(Args):
    """Smaller batch for Sleep-EDF (seq_len=3000 uses more GPU memory per sample)."""
    batch_size = 64


def load_frozen_model(src_id_or_name, dataset, device):
    if dataset == "hhar":
        model = tht.train_source(src_id_or_name, Args(), device)
    elif dataset == "sleep":
        model = tst.train_source(src_id_or_name, SleepArgs(), device)
    else:
        model = tft.train_source(src_id_or_name, Args(), device)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model


# ── Eval helpers ───────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_f1_loader(model, loader, device):
    model.eval()
    preds, targets = [], []
    for xb, yb in loader:
        preds.extend(model(xb.to(device)).argmax(1).cpu().numpy())
        targets.extend(yb.numpy())
    return f1_score(targets, preds, average="macro", zero_division=0)


@torch.no_grad()
def extract_embs(model, loader, device):
    model.eval()
    embs, lbls = [], []
    for xb, yb in loader:
        embs.append(model.extract(xb.to(device)).cpu().numpy())
        lbls.append(yb.numpy())
    return np.concatenate(embs).astype(np.float32), np.concatenate(lbls)


# ── DANN discriminator ─────────────────────────────────────────────────────────

class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim=EMB_DIM, hid=DISC_HID):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid),   nn.ReLU(),
            nn.Linear(hid, 2))
    def forward(self, x): return self.net(x)


# ── PACT modules ───────────────────────────────────────────────────────────────

class IdentityAdapter(nn.Module):
    def __init__(self, dim=EMB_DIM):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        nn.init.eye_(self.linear.weight); nn.init.zeros_(self.linear.bias)
    def forward(self, x): return self.linear(x)


class GateMLP(nn.Module):
    """Binary classifier: 0=source-like, 1=drifted (apply adapter)."""
    def __init__(self, in_dim=EMB_DIM, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Dropout(0.2), nn.Linear(hidden, 2))
    def forward(self, x): return self.net(x)


def kmeans_numpy(embs, k, seed=SEED, n_iter=100):
    rng = np.random.default_rng(seed)
    centroids = embs[rng.choice(len(embs), k, replace=False)].copy()
    asgn = np.zeros(len(embs), dtype=int)
    for _ in range(n_iter):
        d = np.linalg.norm(embs[:,None] - centroids[None], axis=2)
        new = d.argmin(1)
        if np.all(new == asgn): break
        asgn = new
        for c in range(k):
            m = embs[asgn == c]
            if len(m): centroids[c] = m.mean(0)
    return centroids, asgn


def assign_concepts(embs, centroids):
    return np.linalg.norm(embs[:,None] - centroids[None], axis=2).argmin(1)


def train_learned_gates(src_embs, src_asgn, calib_embs, calib_asgn, device):
    gates = []
    for c in range(K):
        neg = src_embs[src_asgn == c]
        pos = calib_embs[calib_asgn == c]
        if len(pos) == 0:
            gates.append(None); continue
        rng = np.random.default_rng(SEED + c)
        if len(neg) > len(pos):
            pos = pos[rng.choice(len(pos), len(neg), replace=True)]
        elif len(pos) > len(neg):
            neg = neg[rng.choice(len(neg), len(pos), replace=True)]
        X = np.concatenate([neg, pos])
        y_g = np.concatenate([np.zeros(len(neg), dtype=np.int64),
                              np.ones(len(pos),  dtype=np.int64)])
        perm = np.random.default_rng(SEED + c + 100).permutation(len(X))
        X, y_g = X[perm], y_g[perm]
        Xt = torch.from_numpy(X).to(device)
        yt = torch.from_numpy(y_g).to(device)
        ld = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xt, yt),
            batch_size=min(BATCH, len(X)), shuffle=True)
        gate = GateMLP().to(device)
        opt  = torch.optim.AdamW(gate.parameters(), lr=GATE_LR, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()
        for _ in range(GATE_EP):
            gate.train()
            for xb, yb in ld:
                opt.zero_grad(); crit(gate(xb), yb).backward(); opt.step()
        gates.append(gate)
    return gates


@torch.no_grad()
def eval_adapter_f1(embs_np, labels_np, centroids, adapters, frozen_head, device,
                    gate_fn=None):
    asgn  = assign_concepts(embs_np, centroids)
    embs  = torch.from_numpy(embs_np).to(device)
    corr  = embs.clone()
    apply = (gate_fn(embs_np, asgn) if gate_fn is not None
             else np.ones(len(embs_np), dtype=bool))
    for c in range(K):
        mask = (asgn == c) & apply
        if mask.any():
            adapters[c].eval()
            idx = torch.from_numpy(mask).to(device)
            corr[idx] = adapters[c](embs[idx])
    return f1_score(labels_np,
                    frozen_head(corr).argmax(1).cpu().numpy(),
                    average="macro", zero_division=0)


# ══════════════════════════════════════════════════════════════════════════════
# METHOD IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════════════════════

def run_no_adapt(src_model, tgt_test_ld, src_test_ld, device):
    return (eval_f1_loader(src_model, src_test_ld, device),
            eval_f1_loader(src_model, tgt_test_ld, device))


def run_dann(src_model, src_train_ld, tgt_train_ld, src_test_ld, tgt_test_ld, device):
    """DANN ported to TSLANet 128-dim embeddings. Label-free."""
    model    = copy.deepcopy(src_model)
    for p in model.parameters(): p.requires_grad_(True)
    disc     = DomainDiscriminator().to(device)
    opt_fe   = torch.optim.Adam(model.parameters(), lr=UDA_LR, weight_decay=1e-4)
    opt_disc = torch.optim.Adam(disc.parameters(),  lr=UDA_LR, weight_decay=1e-4,
                                betas=(0.5, 0.99))
    crit     = nn.CrossEntropyLoss()
    n_batches = max(len(src_train_ld), len(tgt_train_ld))
    for ep in range(UDA_EPOCHS):
        model.train(); disc.train()
        for step, ((sx, sy), (tx, _)) in enumerate(
                zip(src_train_ld, itertools.cycle(tgt_train_ld))):
            sx, sy, tx = sx.to(device), sy.to(device), tx.to(device)
            p = float(step + ep * n_batches) / (UDA_EPOCHS * n_batches)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            opt_fe.zero_grad(); opt_disc.zero_grad()
            src_emb = model.extract(sx); tgt_emb = model.extract(tx)
            cls_loss = crit(model.head(src_emb), sy)
            dom_loss = (crit(disc(ReverseLayerF.apply(src_emb, alpha)),
                             torch.ones(len(sx),  dtype=torch.long, device=device)) +
                        crit(disc(ReverseLayerF.apply(tgt_emb, alpha)),
                             torch.zeros(len(tx), dtype=torch.long, device=device)))
            (cls_loss + dom_loss).backward()
            opt_fe.step(); opt_disc.step()
    return (eval_f1_loader(model, src_test_ld, device),
            eval_f1_loader(model, tgt_test_ld, device))


def run_coral(src_model, src_train_ld, tgt_train_ld, src_test_ld, tgt_test_ld, device):
    """Deep CORAL ported to TSLANet 128-dim embeddings. Label-free."""
    model = copy.deepcopy(src_model)
    for p in model.parameters(): p.requires_grad_(True)
    coral = CORAL()
    opt   = torch.optim.Adam(model.parameters(), lr=UDA_LR, weight_decay=1e-4)
    crit  = nn.CrossEntropyLoss()
    for ep in range(UDA_EPOCHS):
        model.train()
        for (sx, sy), (tx, _) in zip(src_train_ld, itertools.cycle(tgt_train_ld)):
            sx, sy, tx = sx.to(device), sy.to(device), tx.to(device)
            src_emb = model.extract(sx); tgt_emb = model.extract(tx)
            loss = crit(model.head(src_emb), sy) + coral(src_emb, tgt_emb)
            opt.zero_grad(); loss.backward(); opt.step()
    return (eval_f1_loader(model, src_test_ld, device),
            eval_f1_loader(model, tgt_test_ld, device))


def run_mmda(src_model, src_train_ld, tgt_train_ld, src_test_ld, tgt_test_ld, device):
    """MMDA ported to TSLANet 128-dim embeddings. Label-free."""
    model    = copy.deepcopy(src_model)
    for p in model.parameters(): p.requires_grad_(True)
    mmd_loss = MMD_loss(); coral = CORAL(); cond_ent = CondEntLoss()
    opt      = torch.optim.Adam(model.parameters(), lr=UDA_LR, weight_decay=1e-4)
    crit     = nn.CrossEntropyLoss()
    for ep in range(UDA_EPOCHS):
        model.train()
        for (sx, sy), (tx, _) in zip(src_train_ld, itertools.cycle(tgt_train_ld)):
            sx, sy, tx = sx.to(device), sy.to(device), tx.to(device)
            src_emb = model.extract(sx); tgt_emb = model.extract(tx)
            loss = (crit(model.head(src_emb), sy)
                    + mmd_loss(src_emb, tgt_emb)
                    + coral(src_emb, tgt_emb)
                    + cond_ent(model.head(tgt_emb)))
            opt.zero_grad(); loss.backward(); opt.step()
    return (eval_f1_loader(model, src_test_ld, device),
            eval_f1_loader(model, tgt_test_ld, device))


# ══════════════════════════════════════════════════════════════════════════════
# SEMI-SUPERVISED DA METHODS  (label-free alignment + 5% labeled target CE/contrastive)
# ══════════════════════════════════════════════════════════════════════════════

class CDANDiscriminator(nn.Module):
    """CDAN discriminator on the outer product of classifier prediction and embedding."""
    def __init__(self, emb_dim=EMB_DIM, n_classes=3, hid=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim * n_classes, hid), nn.ReLU(),
            nn.Linear(hid, hid // 2),            nn.ReLU(),
            nn.Linear(hid // 2, 2))
    def forward(self, x): return self.net(x)


def run_dann_ss(src_model, src_train_ld, tgt_train_ld, calib_ds,
                src_test_ld, tgt_test_ld, device):
    """DANN† : label-free adversarial alignment + supervised CE on 5% labeled target."""
    model    = copy.deepcopy(src_model)
    for p in model.parameters(): p.requires_grad_(True)
    disc     = DomainDiscriminator().to(device)
    opt_fe   = torch.optim.Adam(model.parameters(), lr=UDA_LR, weight_decay=1e-4)
    opt_disc = torch.optim.Adam(disc.parameters(), lr=UDA_LR, weight_decay=1e-4,
                                betas=(0.5, 0.99))
    crit     = nn.CrossEntropyLoss()
    calib_ld = make_loader(calib_ds, shuffle=True)
    n_batches = max(len(src_train_ld), len(tgt_train_ld))

    for ep in range(UDA_EPOCHS):
        model.train(); disc.train()
        # ── adversarial alignment (label-free) ──────────────────────────────
        for step, ((sx, sy), (tx, _)) in enumerate(
                zip(src_train_ld, itertools.cycle(tgt_train_ld))):
            sx, sy, tx = sx.to(device), sy.to(device), tx.to(device)
            p     = float(step + ep * n_batches) / (UDA_EPOCHS * n_batches)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            opt_fe.zero_grad(); opt_disc.zero_grad()
            src_emb = model.extract(sx); tgt_emb = model.extract(tx)
            cls_loss = crit(model.head(src_emb), sy)
            dom_loss = (crit(disc(ReverseLayerF.apply(src_emb, alpha)),
                             torch.ones(len(sx),  dtype=torch.long, device=device)) +
                        crit(disc(ReverseLayerF.apply(tgt_emb, alpha)),
                             torch.zeros(len(tx), dtype=torch.long, device=device)))
            (cls_loss + dom_loss).backward()
            opt_fe.step(); opt_disc.step()
        # ── supervised CE on labeled calibration ────────────────────────────
        model.train()
        for xb, yb in calib_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt_fe.zero_grad(); crit(model(xb), yb).backward(); opt_fe.step()

    return (eval_f1_loader(model, src_test_ld, device),
            eval_f1_loader(model, tgt_test_ld, device))


def run_cdan_ss(src_model, src_train_ld, tgt_train_ld, calib_ds,
                src_test_ld, tgt_test_ld, device):
    """CDAN† : outer-product conditional adversarial + supervised CE on 5% labeled target.
    Ported to TSLANet's 128-dim embedding + n_classes logits (disc input = emb_dim × n_classes).
    """
    n_cls    = src_model.head.out_features
    model    = copy.deepcopy(src_model)
    for p in model.parameters(): p.requires_grad_(True)
    disc     = CDANDiscriminator(EMB_DIM, n_cls).to(device)
    cond_ent = CondEntLoss()
    opt_fe   = torch.optim.Adam(model.parameters(), lr=UDA_LR, weight_decay=1e-4)
    opt_disc = torch.optim.Adam(disc.parameters(),  lr=UDA_LR, weight_decay=1e-4)
    crit     = nn.CrossEntropyLoss()
    calib_ld = make_loader(calib_ds, shuffle=True)

    for ep in range(UDA_EPOCHS):
        model.train(); disc.train()
        for (sx, sy), (tx, _) in zip(src_train_ld, itertools.cycle(tgt_train_ld)):
            sx, sy, tx = sx.to(device), sy.to(device), tx.to(device)
            src_feat = model.extract(sx); tgt_feat = model.extract(tx)
            src_pred = model.head(src_feat); tgt_pred = model.head(tgt_feat)

            feat_cat = torch.cat([src_feat, tgt_feat], 0)
            pred_cat = torch.cat([src_pred, tgt_pred], 0)
            B2       = feat_cat.shape[0]
            # outer product → [B2, n_cls, emb_dim] → flatten → [B2, n_cls*emb_dim]
            outer    = torch.bmm(pred_cat.unsqueeze(2), feat_cat.unsqueeze(1))
            outer_flat = outer.view(B2, -1).detach()

            dom_real = torch.cat([torch.ones(len(sx), dtype=torch.long, device=device),
                                  torch.zeros(len(tx), dtype=torch.long, device=device)])
            disc_loss = crit(disc(outer_flat), dom_real)
            opt_disc.zero_grad(); disc_loss.backward(); opt_disc.step()

            # feature extractor: fool discriminator with flipped labels
            outer_flat2 = torch.bmm(pred_cat.unsqueeze(2), feat_cat.unsqueeze(1)).view(B2, -1)
            dom_fake    = torch.cat([torch.zeros(len(sx), dtype=torch.long, device=device),
                                     torch.ones(len(tx),  dtype=torch.long, device=device)])
            loss = (crit(src_pred, sy)
                    + crit(disc(outer_flat2), dom_fake)
                    + 0.1 * cond_ent(tgt_pred))
            opt_fe.zero_grad(); loss.backward(); opt_fe.step()

        # supervised CE on labeled calibration
        model.train()
        for xb, yb in calib_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt_fe.zero_grad(); crit(model(xb), yb).backward(); opt_fe.step()

    return (eval_f1_loader(model, src_test_ld, device),
            eval_f1_loader(model, tgt_test_ld, device))


def run_cotmix_ss(src_model, src_train_ld, tgt_train_ld, calib_ds,
                  src_test_ld, tgt_test_ld, device):
    """CoTMix† : temporal mixup + supervised contrastive on 5% labeled target.
    Ported to TSLANet signals (roll on time dim).
    Losses: src CE + tgt conditional entropy + src SupCon + tgt NTXent + tgt SupCon (new).
    """
    model      = copy.deepcopy(src_model)
    for p in model.parameters(): p.requires_grad_(True)
    opt        = torch.optim.Adam(model.parameters(), lr=UDA_LR, weight_decay=1e-4)
    crit       = nn.CrossEntropyLoss()
    cond_ent   = CondEntLoss()
    sup_con    = SupConLoss(device, temperature=0.2)
    crit_ntx   = None   # instantiated per-batch with actual batch size
    calib_ld   = make_loader(calib_ds, shuffle=True, drop_last=False)
    MIX        = 0.8     # source/target dominance weight

    ntx_cache = {}  # cache NTXentLoss instances by batch size

    def temporal_mixup(sx, tx):
        L = sx.shape[-1]
        h = min(3, max(1, L // 200))   # cap at 3 to avoid O(h) overhead for long signals
        with torch.no_grad():           # shifts are not differentiable
            rolled_tx = torch.stack([torch.roll(tx, i, -1) for i in range(-h, h+1)], dim=0).mean(0)
            rolled_sx = torch.stack([torch.roll(sx, i, -1) for i in range(-h, h+1)], dim=0).mean(0)
        src_dom = MIX * sx + (1 - MIX) * rolled_tx
        tgt_dom = MIX * tx + (1 - MIX) * rolled_sx
        return src_dom, tgt_dom

    for ep in range(UDA_EPOCHS):
        model.train()
        # ── main CoTMix update on unlabeled source + target ──────────────────
        for (sx, sy), (tx, _) in zip(src_train_ld, itertools.cycle(tgt_train_ld)):
            sx, sy, tx = sx.to(device), sy.to(device), tx.to(device)
            src_dom, tgt_dom = temporal_mixup(sx, tx)

            # use 128-dim embeddings for contrastive losses (not 3-dim logits)
            so_emb = model.extract(sx);    sd_emb = model.extract(src_dom)
            to_emb = model.extract(tx);    td_emb = model.extract(tgt_dom)
            so_logit = model.head(so_emb); to_logit = model.head(to_emb)

            src_cls  = crit(so_logit, sy)
            tgt_ent  = cond_ent(to_logit)

            # source supervised contrastive on embeddings
            sc_views = torch.stack([so_emb, sd_emb], dim=1)
            src_sc   = sup_con(sc_views, sy)

            # unsupervised target contrastive via NTXent on embeddings
            B = sx.shape[0]
            if B not in ntx_cache:
                ntx_cache[B] = NTXentLoss(device, B, temperature=0.2, use_cosine_similarity=True)
            tgt_ntx = ntx_cache[B](to_emb, td_emb)

            loss = src_cls + 0.5 * tgt_ent + 0.5 * src_sc + 0.5 * tgt_ntx
            opt.zero_grad(); loss.backward(); opt.step()

        # ── supervised contrastive on labeled calibration ────────────────────
        model.train()
        for xb, yb in calib_ld:
            xb, yb = xb.to(device), yb.to(device)
            xb_dom, _ = temporal_mixup(xb, xb)   # self-mixup for second view
            orig_emb = model.extract(xb); dom_emb = model.extract(xb_dom)
            # supervised contrastive using labeled target embeddings
            views    = torch.stack([orig_emb, dom_emb], dim=1)
            tgt_sc   = sup_con(views, yb)
            # also CE for direct supervision
            tgt_cls  = crit(model.head(orig_emb), yb)
            opt.zero_grad(); (tgt_cls + 0.5 * tgt_sc).backward(); opt.step()

    return (eval_f1_loader(model, src_test_ld, device),
            eval_f1_loader(model, tgt_test_ld, device))


def run_tent(src_model, tgt_train_ds, src_test_ld, tgt_test_ld, device, lr=1e-3):
    """TENT: entropy min on LayerNorm params. Uses full unlabeled target train."""
    model = copy.deepcopy(src_model)
    model.eval(); model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            if m.weight is not None: m.weight.requires_grad_(True)
            if m.bias   is not None: m.bias.requires_grad_(True)
    params = [p for m in model.modules()
              if isinstance(m, nn.LayerNorm) for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    for xb, _ in make_loader(tgt_train_ds, shuffle=True):
        xb = xb.to(device)
        with torch.enable_grad():
            out  = model(xb)
            loss = -(out.softmax(1) * out.log_softmax(1)).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return (eval_f1_loader(model, src_test_ld, device),
            eval_f1_loader(model, tgt_test_ld, device))


def run_fullft(src_model, calib_ds, src_test_ld, tgt_test_ld, device):
    model = copy.deepcopy(src_model)
    for p in model.parameters(): p.requires_grad_(True)
    opt  = torch.optim.AdamW(model.parameters(), lr=FT_LR, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    for _ in range(FT_EPOCHS):
        model.train()
        for xb, yb in make_loader(calib_ds, shuffle=True):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
    model.eval()
    return (eval_f1_loader(model, src_test_ld, device),
            eval_f1_loader(model, tgt_test_ld, device))


def run_headft(src_model, calib_ds, src_test_ld, tgt_test_ld, device):
    model = copy.deepcopy(src_model)
    for p in model.parameters(): p.requires_grad_(False)
    for p in model.head.parameters(): p.requires_grad_(True)
    opt  = torch.optim.AdamW(model.head.parameters(), lr=FT_LR, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    for _ in range(FT_EPOCHS):
        model.train()
        for xb, yb in make_loader(calib_ds, shuffle=True):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
    model.eval()
    return (eval_f1_loader(model, src_test_ld, device),
            eval_f1_loader(model, tgt_test_ld, device))


def run_ewc(src_model, src_train_ds, calib_ds, src_test_ld, tgt_test_ld, device):
    """EWC with Fisher floor.
    Near-perfect source models (F1≈0.998) produce degenerate Fisher (mean≈1e-8)
    because log-likelihood gradients vanish when the model is already optimal.
    A floor at 1% of the max Fisher ensures all parameters receive some regularization,
    preventing EWC from collapsing to unconstrained fine-tuning.
    EWC_LAMBDA = 1e6 to compensate for the small Fisher magnitudes.
    """
    model = copy.deepcopy(src_model)
    for p in model.parameters(): p.requires_grad_(True)
    omega = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    model.eval(); n_batches = 0
    for xb, yb in make_loader(src_train_ds):
        xb, yb = xb.to(device), yb.to(device)
        model.zero_grad()
        F.nll_loss(F.log_softmax(model(xb), dim=1), yb).backward()
        for n, p in model.named_parameters():
            if p.grad is not None: omega[n] += p.grad.pow(2).detach()
        n_batches += 1
    for n in omega: omega[n] /= n_batches
    # Apply floor at 1% of global max to prevent full degeneracy
    omega_max = max(v.max().item() for v in omega.values() if v.numel() > 0)
    floor = omega_max * 0.01
    for n in omega: omega[n] = omega[n].clamp(min=floor)
    mean = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
    opt  = torch.optim.AdamW(model.parameters(), lr=FT_LR, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    for _ in range(FT_EPOCHS):
        model.train()
        for xb, yb in make_loader(calib_ds, shuffle=True):
            xb, yb = xb.to(device), yb.to(device)
            loss = crit(model(xb), yb) + EWC_LAMBDA * sum(
                torch.sum(omega[n] * (p - mean[n]).pow(2)) / 2
                for n, p in model.named_parameters() if n in omega)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return (eval_f1_loader(model, src_test_ld, device),
            eval_f1_loader(model, tgt_test_ld, device))


def run_src_distill(src_model, src_train_ds, calib_ds, src_test_ld, tgt_test_ld, device):
    """Source-distillation: target CE + teacher/student KL on SOURCE data.
    Requires source data at adaptation time. Preserves source behaviour directly.
    NOT canonical LwF — canonical LwF uses only target data (see run_lwf below).
    """
    teacher = src_model
    student = copy.deepcopy(src_model)
    for p in student.parameters(): p.requires_grad_(True)
    opt          = torch.optim.AdamW(student.parameters(), lr=FT_LR, weight_decay=1e-4)
    crit         = nn.CrossEntropyLoss()
    T            = LWF_TEMP
    src_loader   = make_loader(src_train_ds, shuffle=True)
    calib_loader = make_loader(calib_ds, shuffle=True)
    for _ in range(FT_EPOCHS):
        student.train()
        for xb, yb in calib_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); crit(student(xb), yb).backward(); opt.step()
        for xb, _ in src_loader:
            xb = xb.to(device)
            with torch.no_grad():
                soft_targets = F.softmax(teacher(xb) / T, dim=1)
            distill_loss = -(soft_targets * F.log_softmax(student(xb) / T, dim=1)).sum(1).mean() * T**2
            opt.zero_grad(); (LWF_ALPHA * distill_loss).backward(); opt.step()
    student.eval()
    return (eval_f1_loader(student, src_test_ld, device),
            eval_f1_loader(student, tgt_test_ld, device))


def run_lwf(src_model, src_train_ds, calib_ds, src_test_ld, tgt_test_ld, device):
    """Canonical LwF (Li & Hoiem 2016): target CE + teacher/student KL on TARGET calib.
    Source data NOT used at adaptation time. Preserves old behaviour on new-domain inputs.
    """
    teacher = src_model
    student = copy.deepcopy(src_model)
    for p in student.parameters(): p.requires_grad_(True)
    opt          = torch.optim.AdamW(student.parameters(), lr=FT_LR, weight_decay=1e-4)
    crit         = nn.CrossEntropyLoss()
    T            = LWF_TEMP
    calib_loader = make_loader(calib_ds, shuffle=True)
    for _ in range(FT_EPOCHS):
        student.train()
        for xb, yb in calib_loader:
            xb, yb = xb.to(device), yb.to(device)
            with torch.no_grad():
                soft_targets = F.softmax(teacher(xb) / T, dim=1)
            distill_loss = -(soft_targets * F.log_softmax(student(xb) / T, dim=1)).sum(1).mean() * T**2
            clf_loss     = crit(student(xb), yb)
            opt.zero_grad(); (clf_loss + LWF_ALPHA * distill_loss).backward(); opt.step()
    student.eval()
    return (eval_f1_loader(student, src_test_ld, device),
            eval_f1_loader(student, tgt_test_ld, device))


def run_mas(src_model, src_train_ds, calib_ds, src_test_ld, tgt_test_ld, device):
    """MAS: importance = gradient of output norm on source (no labels)."""
    model = copy.deepcopy(src_model)
    for p in model.parameters(): p.requires_grad_(True)
    omega = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    model.eval(); n_batches = 0
    for xb, _ in make_loader(src_train_ds):
        xb = xb.to(device); model.zero_grad()
        model(xb).pow(2).sum().backward()
        for n, p in model.named_parameters():
            if p.grad is not None: omega[n] += p.grad.abs().clone()
        n_batches += 1
    for n in omega: omega[n] /= n_batches
    mean = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
    opt  = torch.optim.AdamW(model.parameters(), lr=FT_LR, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    for _ in range(FT_EPOCHS):
        model.train()
        for xb, yb in make_loader(calib_ds, shuffle=True):
            xb, yb = xb.to(device), yb.to(device)
            loss = crit(model(xb), yb) + MAS_LAMBDA * sum(
                torch.sum(omega[n] * (p - mean[n]).pow(2)) / 2
                for n, p in model.named_parameters() if n in omega)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return (eval_f1_loader(model, src_test_ld, device),
            eval_f1_loader(model, tgt_test_ld, device))


def _train_pact_adapters(calib_embs, calib_lbls, centroids, frozen_head, device):
    adapters = nn.ModuleList([IdentityAdapter() for _ in range(K)]).to(device)
    opt  = torch.optim.AdamW(adapters.parameters(), lr=ADAPTER_LR)
    crit = nn.CrossEntropyLoss()
    asgn = assign_concepts(calib_embs, centroids)
    et   = torch.from_numpy(calib_embs).to(device)
    lt   = torch.from_numpy(calib_lbls).to(device)
    at   = torch.from_numpy(asgn.astype(np.int64))
    ld   = make_loader(torch.utils.data.TensorDataset(et, lt, at), shuffle=True)
    eye  = torch.eye(EMB_DIM, device=device)
    for _ in range(ADAPTER_EP):
        for eb, yb, cb in ld:
            adapters.train(); opt.zero_grad()
            corr = eb.clone()
            for c in range(K):
                m = (cb == c)
                if m.any(): corr[m] = adapters[c](eb[m])
            loss = crit(frozen_head(corr), yb)
            for a in adapters:
                loss = loss + PENALTY * (torch.sum((a.linear.weight - eye)**2)
                                          + torch.sum(a.linear.bias**2))
            loss.backward(); opt.step()
    return adapters


def run_pact(src_model, src_train_ld, calib_ds, src_test_ld, tgt_test_ld,
             device, gate=True):
    """PACT: frozen host + k=16 adapters [+ learned gate]."""
    frozen_head = src_model.head
    src_embs, _ = extract_embs(src_model, src_train_ld, device)
    centroids, src_asgn = kmeans_numpy(src_embs, K)

    calib_embs, calib_lbls = extract_embs(src_model, make_loader(calib_ds), device)
    adapters = _train_pact_adapters(calib_embs, calib_lbls, centroids, frozen_head, device)

    gate_fn = None
    if gate:
        calib_asgn = assign_concepts(calib_embs, centroids)
        gates      = train_learned_gates(src_embs, src_asgn, calib_embs, calib_asgn, device)
        @torch.no_grad()
        def gate_fn(embs_np, asgn):
            result = np.zeros(len(embs_np), dtype=bool)
            for c in range(K):
                mask = (asgn == c)
                if not mask.any(): continue
                g = gates[c]
                if g is None: result[mask] = False; continue
                g.eval()
                p = g(torch.from_numpy(embs_np[mask]).to(device)).argmax(1).cpu().numpy()
                result[mask] = (p == 1)
            return result

    def _f1(ld):
        embs, lbls = extract_embs(src_model, ld, device)
        return eval_adapter_f1(embs, lbls, centroids, adapters, frozen_head,
                               device, gate_fn)
    return _f1(src_test_ld), _f1(tgt_test_ld)


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO RUNNER
# ══════════════════════════════════════════════════════════════════════════════

METHODS = [
    ("no_adapt",   "label-free",    "NO_ADAPT"),
    ("dann",       "label-free",    "DANN"),
    ("coral",      "label-free",    "Deep_Coral"),
    ("mmda",       "label-free",    "MMDA"),
    ("tent",       "label-free",    "TENT"),
    # Semi-supervised DA: label-free alignment + 5% labeled target supervision
    ("dann_ss",    "semi-sup DA",   "DANN†"),
    ("cdan_ss",    "semi-sup DA",   "CDAN†"),
    ("cotmix_ss",  "semi-sup DA",   "CoTMix†"),
    ("full_ft",    "calibration",   "Full FT"),
    ("head_ft",   "calibration", "Head FT"),
    ("ewc",        "calibration", "EWC"),
    ("src_distill","calibration", "Src-Distill"),  # source-data distillation (NOT canonical LwF)
    ("lwf",        "calibration", "LwF"),           # canonical LwF (target-data distillation only)
    ("mas",       "calibration", "MAS"),
    ("pact_ng",   "calibration", "PACT (no gate)"),
    ("pact",      "calibration", "PACT"),
]


def run_all_methods(datasets, frozen_model, device, batch_override=None):
    """Run all 16 methods for one scenario. batch_override sets loader batch size
    (use 64 for Sleep-EDF due to longer signals)."""
    src_train_ds = datasets["src_train"]
    src_test_ds  = datasets["src_test"]
    tgt_train_ds = datasets["tgt_train"]
    tgt_test_ds  = datasets["tgt_test"]

    calib_ds     = calib_split(tgt_train_ds)
    bs = batch_override   # None → make_loader uses global BATCH
    src_train_ld = make_loader(src_train_ds, shuffle=True, drop_last=True, batch_size=bs)
    src_test_ld  = make_loader(src_test_ds,  batch_size=bs)
    tgt_train_ld = make_loader(tgt_train_ds, shuffle=True, drop_last=True, batch_size=bs)
    tgt_test_ld  = make_loader(tgt_test_ds,  batch_size=bs)

    f1_src_before = eval_f1_loader(frozen_model, src_test_ld, device)
    f1_tgt_nocorr = eval_f1_loader(frozen_model, tgt_test_ld, device)
    print(f"  src_before={f1_src_before:.4f}  tgt_nocorr={f1_tgt_nocorr:.4f}")

    results = {}
    for key, budget, name in METHODS:
        print(f"  [{budget:12s}] {name:<18}", end=" … ", flush=True)
        try:
            if key == "no_adapt":
                r = run_no_adapt(frozen_model, tgt_test_ld, src_test_ld, device)
            elif key == "dann":
                r = run_dann(frozen_model, src_train_ld, tgt_train_ld,
                             src_test_ld, tgt_test_ld, device)
            elif key == "coral":
                r = run_coral(frozen_model, src_train_ld, tgt_train_ld,
                              src_test_ld, tgt_test_ld, device)
            elif key == "mmda":
                r = run_mmda(frozen_model, src_train_ld, tgt_train_ld,
                             src_test_ld, tgt_test_ld, device)
            elif key == "tent":
                r = run_tent(frozen_model, tgt_train_ds, src_test_ld, tgt_test_ld, device)
            elif key == "dann_ss":
                r = run_dann_ss(frozen_model, src_train_ld, tgt_train_ld, calib_ds,
                                src_test_ld, tgt_test_ld, device)
            elif key == "cdan_ss":
                r = run_cdan_ss(frozen_model, src_train_ld, tgt_train_ld, calib_ds,
                                src_test_ld, tgt_test_ld, device)
            elif key == "cotmix_ss":
                r = run_cotmix_ss(frozen_model, src_train_ld, tgt_train_ld, calib_ds,
                                  src_test_ld, tgt_test_ld, device)
            elif key == "full_ft":
                r = run_fullft(frozen_model, calib_ds, src_test_ld, tgt_test_ld, device)
            elif key == "head_ft":
                r = run_headft(frozen_model, calib_ds, src_test_ld, tgt_test_ld, device)
            elif key == "ewc":
                r = run_ewc(frozen_model, src_train_ds, calib_ds,
                            src_test_ld, tgt_test_ld, device)
            elif key == "src_distill":
                r = run_src_distill(frozen_model, src_train_ds, calib_ds,
                                    src_test_ld, tgt_test_ld, device)
            elif key == "lwf":
                r = run_lwf(frozen_model, None, calib_ds,
                            src_test_ld, tgt_test_ld, device)
            elif key == "mas":
                r = run_mas(frozen_model, src_train_ds, calib_ds,
                            src_test_ld, tgt_test_ld, device)
            elif key == "pact_ng":
                r = run_pact(frozen_model, src_train_ld, calib_ds,
                             src_test_ld, tgt_test_ld, device, gate=False)
            elif key == "pact":
                r = run_pact(frozen_model, src_train_ld, calib_ds,
                             src_test_ld, tgt_test_ld, device, gate=True)
            else:
                r = (0.0, 0.0)
            src_f1, tgt_f1 = r
            forget = f1_src_before - src_f1
            results[key] = (src_f1, tgt_f1, forget)
            print(f"src={src_f1:.4f}  tgt={tgt_f1:.4f}  forget={forget:+.4f}")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback; traceback.print_exc()
            results[key] = (0.0, 0.0, 0.0)

    return results, f1_src_before


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_dataset_sweep(dataset_tag, scenarios, load_fn, dataset_key, device):
    """Run all 16 methods over all scenarios for one dataset.
    dataset_key: 'fd' | 'hhar' | 'sleep'  (controls model loading + batch size)
    """
    print(f"\n{'#'*72}")
    print(f"## {dataset_tag} SWEEP  ({len(scenarios)} scenarios, 5% labels, learned gate)")
    print(f"{'#'*72}")

    mkeys   = [m[0] for m in METHODS]
    mnames  = {m[0]: m[2] for m in METHODS}
    mbudgets = {m[0]: m[1] for m in METHODS}
    pop_src = {k: [] for k in mkeys}
    pop_tgt = {k: [] for k in mkeys}
    pop_fgt = {k: [] for k in mkeys}
    all_results = {}

    for src_id, tgt_id in scenarios:
        datasets  = load_fn(src_id, tgt_id)
        scen_key  = f"{datasets['src_name']}→{datasets['tgt_name']}"
        print(f"\n{'='*60}\n  Scenario: {scen_key}\n{'='*60}")
        src_lookup = src_id if dataset_key in ("hhar", "sleep") else datasets['src_name']
        frozen_model = load_frozen_model(src_lookup, dataset_key, device)
        # Use smaller batch for Sleep-EDF (seq_len=3000)
        batch_override = 64 if dataset_key == "sleep" else None
        res, src_bef = run_all_methods(datasets, frozen_model, device,
                                       batch_override=batch_override)
        all_results[scen_key] = res
        for k in mkeys:
            if k in res:
                s, t, f = res[k]
                pop_src[k].append(s); pop_tgt[k].append(t); pop_fgt[k].append(f)

    # Population summary
    print(f"\n{'='*72}")
    print(f"POPULATION SUMMARY — {dataset_tag}  (mean ± std, {len(scenarios)} scenarios)")
    print(f"{'='*72}")
    hdr = f"  {'Method':<20} {'Budget':>12}  {'Src F1':>12}  {'Tgt F1':>12}  {'Forgetting':>14}"
    sep = "  " + "-"*74
    print(hdr); print(sep)
    lines = [f"POPULATION SUMMARY — {dataset_tag}\n", hdr, sep]
    for k in mkeys:
        sm = np.mean(pop_src[k]); ss = np.std(pop_src[k])
        tm = np.mean(pop_tgt[k]); ts = np.std(pop_tgt[k])
        fm = np.mean(pop_fgt[k]); fs = np.std(pop_fgt[k])
        row = (f"  {mnames[k]:<20} {mbudgets[k]:>12}  "
               f"{sm:.4f}±{ss:.4f}  {tm:.4f}±{ts:.4f}  {fm:+.4f}±{fs:.4f}")
        print(row); lines.append(row)
    (OUT_DIR / f"{dataset_tag.lower()}_population_summary.txt").write_text("\n".join(lines))

    # Per-scenario table
    scen_lines = [f"Per-scenario results — {dataset_tag}\n"]
    for sk, res in all_results.items():
        scen_lines.append(f"\nScenario: {sk}")
        scen_lines.append(f"  {'Method':<20} {'Src F1':>8}  {'Tgt F1':>8}  {'Forget':>8}")
        for k in mkeys:
            if k in res:
                s, t, f = res[k]
                scen_lines.append(f"  {mnames[k]:<20} {s:>8.4f}  {t:>8.4f}  {f:>+8.4f}")
    (OUT_DIR / f"{dataset_tag.lower()}_per_scenario.txt").write_text("\n".join(scen_lines))

    # ── Figure with method-category annotations ───────────────────────────────
    # Method order in mkeys matches category order already:
    # [0]=NO_ADAPT | [1-3]=DANN/CORAL/MMDA | [4]=TENT | [5-6]=FullFT/HeadFT
    # [7-9]=EWC/LwF/MAS | [10-11]=PACT_ng/PACT

    # Category definitions: (display_label, [indices_into_mkeys], base_color)
    GROUPS = [
        ("Lower bound",                   [0],         "#777777"),
        ("UDA Benchmark\n(label-free)",   [1, 2, 3],   "#2471a3"),
        ("Test-time\n(label-free)",       [4],          "#76448a"),
        ("Semi-supervised DA\n(5% labels)", [5, 6, 7], "#17a589"),
        ("Fine-tuning\n(5% labels)",        [8, 9],          "#d35400"),
        ("Continual Learning\n(5% labels)", [10, 11, 12, 13], "#922b21"),
        ("Ours\n(5% labels)",               [14, 15],         "#1e8449"),
    ]

    # One color per bar — same hue family within each category
    BAR_COLORS = [
        "#999999",                               # NO_ADAPT
        "#3498db", "#5dade2", "#85c1e9",         # DANN, CORAL, MMDA
        "#9b59b6",                               # TENT
        "#1abc9c", "#17a589", "#148f77",         # DANN†, CDAN†, CoTMix†
        "#e67e22", "#f0a500",                    # Full FT, Head FT
        "#e74c3c", "#8e44ad", "#c0392b", "#f1948a",  # EWC, Src-Distill, LwF, MAS
        "#52be80", "#1e8449",                    # PACT (ng), PACT
    ]

    def _annotate_categories(ax, groups, mkeys_list, y_span_top=1.0,
                              y_line=-0.27, y_label=-0.35):
        """Draw category bracket lines + labels below x-axis tick labels."""
        trans = ax.get_xaxis_transform()   # x=data, y=axes fraction
        for cat_label, idxs, color in groups:
            x1 = idxs[0]  - 0.45
            x2 = idxs[-1] + 0.45
            xc = (x1 + x2) / 2
            # Light background span
            ax.axvspan(x1, x2, ymin=0, ymax=y_span_top,
                       alpha=0.07, color=color, zorder=0, lw=0)
            # Bracket line below ticks
            ax.annotate("",
                        xy=(x1, y_line), xycoords=trans,
                        xytext=(x2, y_line), textcoords=trans,
                        arrowprops=dict(arrowstyle="-[", color=color,
                                        lw=1.4, mutation_scale=5))
            # Category label
            ax.text(xc, y_label, cat_label, transform=trans,
                    ha="center", va="top", fontsize=6.5,
                    color=color, fontweight="bold", linespacing=1.3)

    x    = np.arange(len(mkeys))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(bottom=0.30, top=0.88, wspace=0.28)

    # Panel 1 — Target F1
    ax = axes[0]
    means_tgt = [np.mean(pop_tgt[k]) for k in mkeys]
    stds_tgt  = [np.std(pop_tgt[k])  for k in mkeys]
    bars = ax.bar(x, means_tgt, yerr=stds_tgt, capsize=4, color=BAR_COLORS,
                  edgecolor="white", linewidth=0.5,
                  error_kw=dict(elinewidth=1.0, ecolor="#555"))
    ax.axhline(np.mean(pop_tgt["no_adapt"]), color="#aaa", lw=1.1, ls="--",
               label="NO_ADAPT baseline")
    ax.set_ylim(0, 1.12); ax.set_xlim(-0.6, len(mkeys)-0.4)
    ax.set_ylabel("Target macro-F1", fontsize=10)
    ax.set_title("Target domain adaptation", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([mnames[k] for k in mkeys],
                       rotation=35, ha="right", fontsize=8)
    ax.tick_params(axis="x", pad=2)
    ax.legend(fontsize=8, loc="upper left"); ax.grid(axis="y", ls="--", alpha=0.4)
    for bar, v in zip(bars, means_tgt):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                f"{v:.3f}", ha="center", fontsize=6.5)
    _annotate_categories(ax, GROUPS, mkeys)

    # Panel 2 — Forgetting
    ax = axes[1]
    means_fgt = [np.mean(pop_fgt[k]) for k in mkeys]
    stds_fgt  = [np.std(pop_fgt[k])  for k in mkeys]
    ax.bar(x, means_fgt, yerr=stds_fgt, capsize=4, color=BAR_COLORS,
           edgecolor="white", linewidth=0.5,
           error_kw=dict(elinewidth=1.0, ecolor="#555"))
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlim(-0.6, len(mkeys)-0.4)
    ax.set_ylabel("Forgetting  (source F1 drop)", fontsize=10)
    ax.set_title("Catastrophic forgetting", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([mnames[k] for k in mkeys],
                       rotation=35, ha="right", fontsize=8)
    ax.tick_params(axis="x", pad=2)
    ax.grid(axis="y", ls="--", alpha=0.4)
    for bar, v in zip(bars if False else ax.patches, means_fgt):
        pass   # value labels skipped on forgetting panel for clarity
    _annotate_categories(ax, GROUPS, mkeys)

    fig.suptitle(
        f"Unified Evaluation — {dataset_tag}  (5% target labels, learned gate)\n"
        f"TSLANet backbone · 10 AdaTime scenarios · mean ± std",
        fontsize=11, fontweight="bold")

    for ext in ["png", "pdf"]:
        fig.savefig(OUT_DIR / f"{dataset_tag.lower()}_figure.{ext}",
                    dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    return all_results, pop_src, pop_tgt, pop_fgt


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Label budget: 5%  |  PACT gate: learned MLP  |  Datasets: FD + HHAR + Sleep-EDF")
    print(f"Methods: {len(METHODS)}  |  Scenarios per dataset: 10  |  Total: 30 scenarios\n")

    # FD
    fd_all, fd_ps, fd_pt, fd_pf = run_dataset_sweep(
        "FD", FD_SCENARIOS, load_fd_datasets, dataset_key="fd", device=device)

    # HHAR
    hhar_all, hh_ps, hh_pt, hh_pf = run_dataset_sweep(
        "HHAR", HHAR_SCENARIOS, load_hhar_datasets, dataset_key="hhar", device=device)

    # Sleep-EDF
    sl_all, sl_ps, sl_pt, sl_pf = run_dataset_sweep(
        "Sleep-EDF", SLEEP_SCENARIOS, load_sleep_datasets, dataset_key="sleep", device=device)

    # Combined cross-dataset summary
    mkeys  = [m[0] for m in METHODS]
    mnames = {m[0]: m[2] for m in METHODS}
    print(f"\n{'#'*72}")
    print("## CROSS-DATASET SUMMARY  (FD / HHAR / Sleep-EDF, 5% labels, learned gate)")
    print(f"{'#'*72}")
    hdr = (f"  {'Method':<20} {'FD tgt':>9} {'FD fgt':>9}  "
           f"{'HHAR tgt':>9} {'HHAR fgt':>9}  {'Sleep tgt':>9} {'Sleep fgt':>9}")
    sep = "  " + "-"*80
    print(hdr); print(sep)
    combined_lines = ["CROSS-DATASET SUMMARY\n", hdr, sep]
    for k in mkeys:
        fd_t = np.mean(fd_pt[k]); fd_f = np.mean(fd_pf[k])
        hh_t = np.mean(hh_pt[k]); hh_f = np.mean(hh_pf[k])
        sl_t = np.mean(sl_pt[k]); sl_f = np.mean(sl_pf[k])
        row = (f"  {mnames[k]:<20} {fd_t:>9.4f} {fd_f:>+9.4f}  "
               f"{hh_t:>9.4f} {hh_f:>+9.4f}  {sl_t:>9.4f} {sl_f:>+9.4f}")
        print(row); combined_lines.append(row)
    (OUT_DIR / "combined_summary.txt").write_text("\n".join(combined_lines))
    print(f"\nAll outputs → {OUT_DIR}")


if __name__ == "__main__":
    main()
