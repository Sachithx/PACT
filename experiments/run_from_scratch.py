"""
run_from_scratch.py — Train TSLANet from scratch on 5% target calib per scenario.

Computes b_target for FWT = f1_tgt_nocorr − b_target.
b_target is the target F1 of a randomly-initialised TSLANet trained ONLY on
the 5% calibration labels — no source pretraining, no domain knowledge.

Covers FD, HHAR, Sleep-EDF (30 scenarios total).
Output: results/from_scratch/b_target.txt  (one "scenario_key: float" line per scenario)
"""

import sys, copy
import torch, torch.nn as nn, numpy as np
from pathlib import Path
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader

REPO = Path(__file__).parent
sys.path.insert(0, str(REPO / "TSLANet" / "Classification"))

import importlib.util
def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

tft = _load("tft", REPO / "transfer_fd_tslanet.py")
tht = _load("tht", REPO / "transfer_hhar_tslanet.py")
tst = _load("tst", REPO / "transfer_sleep_tslanet.py")

OUT_DIR = REPO / "results" / "from_scratch"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CALIB_FRAC = 0.05
SEED       = 42
FT_EPOCHS  = 100          # more epochs since starting from random init
FT_LR      = 1e-3         # higher LR for scratch training
BATCH      = 256


class FDArgs:
    emb_dim=128; depth=2; patch_size=8; dropout_rate=0.15; masking_ratio=0.4
    ICB=True; ASB=True; adaptive_filter=True
    data_root=str(REPO/"dataset"); ckpt_dir=str(REPO/"TSLANet/Classification/output/tslanet")
    batch_size=256; pretrain=False; pretrain_epochs=0; pretrain_lr=1e-4
    cls_epochs=100; cls_lr=1e-3; retrain=False


class HHARArgs(FDArgs):
    batch_size=256


class SleepArgs(FDArgs):
    batch_size=64; cls_lr=1e-3


FD_MAP     = {0:"FD-A",1:"FD-B",2:"FD-C",3:"FD-D"}
HHAR_NAMES = {i:f"User-{chr(ord('a')+i)}" for i in range(9)}

FD_SCENARIOS   = [(0,1),(0,3),(1,0),(1,2),(1,3),(2,1),(2,3),(3,0),(3,1),(3,2)]
HHAR_SCENARIOS = [(0,6),(1,6),(2,7),(3,8),(4,5),(5,0),(6,1),(7,4),(8,3),(0,2)]
SLEEP_SCENARIOS= tst.SLEEP_SCENARIOS


@torch.no_grad()
def eval_f1(model, loader, device):
    model.eval()
    preds, lbls = [], []
    for x, y in loader:
        preds.extend(model(x.to(device)).argmax(1).cpu().numpy())
        lbls.extend(y.numpy())
    return f1_score(lbls, preds, average="macro", zero_division=0)


def train_scratch(model_class_fn, calib_ds, test_ds, device, batch=BATCH):
    """Train a fresh (randomly-initialised) model on calib_ds, eval on test_ds."""
    model = model_class_fn().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=FT_LR, weight_decay=1e-4)
    crit  = nn.CrossEntropyLoss()
    ld    = DataLoader(calib_ds, batch_size=min(batch, len(calib_ds)), shuffle=True)
    te_ld = DataLoader(test_ds,  batch_size=batch, shuffle=False)

    best_f1, best_state = 0.0, None
    for ep in range(FT_EPOCHS):
        model.train()
        for xb, yb in ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        if (ep+1) % 20 == 0:
            f1 = eval_f1(model, te_ld, device)
            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return eval_f1(model, te_ld, device)


def calib_subset(tgt_train_ds):
    y = torch.stack([tgt_train_ds[i][1] for i in range(len(tgt_train_ds))]).numpy()
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1-CALIB_FRAC, random_state=SEED)
    idx, _ = next(sss.split(np.zeros(len(y)), y))
    return torch.utils.data.Subset(tgt_train_ds, idx)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Training from scratch on 5% calib per scenario")

    results = {}

    # ── FD ─────────────────────────────────────────────────────────────────────
    print("\n## FD")
    for src_id, tgt_id in FD_SCENARIOS:
        src_name = FD_MAP[src_id]; tgt_name = FD_MAP[tgt_id]
        scen_key = f"{src_name}→{tgt_name}"
        tgt_tr = tft.FDDataset(tgt_name, "train", REPO/"dataset")
        tgt_te = tft.FDDataset(tgt_name, "test",  REPO/"dataset")
        calib  = calib_subset(tgt_tr)
        mk = lambda: tft.TSLANet(
            seq_len=tft.FD_SEQ_LEN, num_channels=tft.FD_CHANNELS,
            num_classes=tft.FD_N_CLASSES, emb_dim=128, depth=2, patch_size=8)
        b = train_scratch(mk, calib, tgt_te, device, BATCH)
        results[scen_key] = b
        print(f"  {scen_key:<22}  b_target={b:.4f}  (calib={len(calib)} samples)")

    # ── HHAR ───────────────────────────────────────────────────────────────────
    print("\n## HHAR")
    for src_id, tgt_id in HHAR_SCENARIOS:
        src_name = HHAR_NAMES[src_id]; tgt_name = HHAR_NAMES[tgt_id]
        scen_key = f"{src_name}→{tgt_name}"
        tgt_tr = tht.HHARDataset(tgt_id, "train", REPO/"dataset")
        tgt_te = tht.HHARDataset(tgt_id, "test",  REPO/"dataset")
        calib  = calib_subset(tgt_tr)
        mk = lambda: tht.TSLANet(
            seq_len=tht.HHAR_SEQ_LEN, num_channels=tht.HHAR_CHANNELS,
            num_classes=tht.HHAR_N_CLASSES, emb_dim=128, depth=2, patch_size=8)
        b = train_scratch(mk, calib, tgt_te, device, BATCH)
        results[scen_key] = b
        print(f"  {scen_key:<22}  b_target={b:.4f}  (calib={len(calib)} samples)")

    # ── Sleep-EDF ──────────────────────────────────────────────────────────────
    print("\n## Sleep-EDF")
    for src_id, tgt_id in SLEEP_SCENARIOS:
        scen_key = f"Sleep-{src_id}→Sleep-{tgt_id}"
        tgt_tr = tst.SleepEDFDataset(tgt_id, "train", REPO/"dataset")
        tgt_te = tst.SleepEDFDataset(tgt_id, "test",  REPO/"dataset")
        calib  = calib_subset(tgt_tr)
        mk = lambda: tst.TSLANet(
            seq_len=tst.SLEEP_SEQ_LEN, num_channels=tst.SLEEP_CHANNELS,
            num_classes=tst.SLEEP_N_CLASSES, emb_dim=128, depth=2, patch_size=8)
        b = train_scratch(mk, calib, tgt_te, device, 64)
        results[scen_key] = b
        print(f"  {scen_key:<22}  b_target={b:.4f}  (calib={len(calib)} samples)")

    # ── Save ───────────────────────────────────────────────────────────────────
    lines = ["# b_target per scenario (train from scratch on 5% calib, no source pretraining)"]
    for k, v in results.items():
        lines.append(f"{k}: {v:.6f}")
    (OUT_DIR / "b_target.txt").write_text("\n".join(lines))
    print(f"\n  Saved → {OUT_DIR / 'b_target.txt'}")
    print("  Re-run compute_cl_metrics.py to add FWT column.")


if __name__ == "__main__":
    main()
