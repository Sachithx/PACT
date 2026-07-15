"""
run_ortcl_tslanet.py — ORTCL adapted for TSLANet backbone, all 3 datasets.

Applies ORTCL's SVD-based orthogonal rotation to TSLANet's last ICB output
projection (blocks[-1].icb.conv3.weight), which is the closest structural
analogue to the transformer's out_projection targeted in the original ORTCL.

Three variants per scenario:
  ORTCL (LF)      — SVD rotation only (label-free, no target labels)
  ORTCL + HeadFT  — rotation → head-only fine-tune on 5% calib
  ORTCL + FullFT  — rotation → full fine-tune on 5% calib

Datasets: FD (10), HHAR (10), Sleep-EDF (10)  →  30 scenarios total
Output:   results/ortcl_tslanet/{fd,hhar,sleep}_per_scenario.txt + figures
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

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = REPO / "results" / "ortcl_tslanet"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CALIB_FRAC = 0.05
SEED       = 42
FT_EPOCHS  = 80
FT_LR      = 1e-4
BATCH      = 256
SVD_CAP    = 50_000   # max patch tokens for SVD (memory)

FD_MAP     = {0:"FD-A",1:"FD-B",2:"FD-C",3:"FD-D"}
HHAR_NAMES = {i:f"User-{chr(ord('a')+i)}" for i in range(9)}

FD_SCENARIOS   = [(0,1),(0,3),(1,0),(1,2),(1,3),(2,1),(2,3),(3,0),(3,1),(3,2)]
HHAR_SCENARIOS = [(0,6),(1,6),(2,7),(3,8),(4,5),(5,0),(6,1),(7,4),(8,3),(0,2)]
SLEEP_SCENARIOS= tst.SLEEP_SCENARIOS


# ── Helpers ────────────────────────────────────────────────────────────────────

class FDArgs:
    emb_dim=128; depth=2; patch_size=8; dropout_rate=0.15; masking_ratio=0.4
    ICB=True; ASB=True; adaptive_filter=True
    data_root=str(REPO/"dataset"); ckpt_dir=str(REPO/"TSLANet/Classification/output/tslanet")
    batch_size=256; pretrain=True; pretrain_epochs=40; pretrain_lr=1e-4
    cls_epochs=80; cls_lr=1e-4; retrain=False

class HHARArgs(FDArgs): batch_size=256
class SleepArgs(FDArgs): batch_size=64


@torch.no_grad()
def eval_f1(model, loader, device):
    model.eval()
    preds, lbls = [], []
    for x, y in loader:
        preds.extend(model(x.to(device)).argmax(1).cpu().numpy())
        lbls.extend(y.numpy())
    return f1_score(lbls, preds, average="macro", zero_division=0)


def calib_subset(tgt_train_ds):
    y = torch.stack([tgt_train_ds[i][1] for i in range(len(tgt_train_ds))]).numpy()
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1-CALIB_FRAC, random_state=SEED)
    idx, _ = next(sss.split(np.zeros(len(y)), y))
    return torch.utils.data.Subset(tgt_train_ds, idx)


def make_loader(ds, batch=BATCH, shuffle=False, drop_last=False):
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, drop_last=drop_last)


# ── ORTCL rotation on TSLANet ─────────────────────────────────────────────────

def collect_icb_features(model, loader, device, cap=SVD_CAP):
    """
    Collect (K, V) features from the last TSLANetLayer's ICB:
      K = input  to conv3  (the pre-output features) — shape [B, hidden, N_patches]
      V = output of conv3  (the post-output features) — shape [B, emb_dim, N_patches]

    Returns K_2d [emb_dim, min(N_total, cap)], V_2d [emb_dim, min(N_total, cap)].
    hidden = emb_dim * mlp_ratio = 128*3 = 384.  We'll reshape K to emb_dim via mean.
    """
    icb = model.blocks[-1].icb
    K_list, V_list = [], []

    def k_hook(module, inp, out):
        # inp[0] is the input to conv3: [B, hidden_dim, N_patches]
        x = inp[0].detach().cpu()   # [B, hidden, N]
        # Reshape hidden → emb_dim by taking mean over groups
        # hidden = 3 * emb_dim; average triplets
        B, H, N = x.shape
        emb = H // 3
        x = x.view(B, 3, emb, N).mean(1)   # [B, emb_dim, N]
        K_list.append(x)

    def v_hook(module, inp, out):
        V_list.append(out.detach().cpu())   # [B, emb_dim, N]

    h1 = icb.conv3.register_forward_hook(k_hook)
    h2 = icb.conv3.register_forward_hook(v_hook)

    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            model(xb.to(device))

    h1.remove(); h2.remove()
    K_raw = torch.cat(K_list, dim=0)   # [N_samples, emb_dim, N_patches]
    V_raw = torch.cat(V_list, dim=0)   # [N_samples, emb_dim, N_patches]

    N, D, P = K_raw.shape
    K_2d = K_raw.permute(1, 0, 2).reshape(D, N*P)   # [emb_dim, N*P]
    V_2d = V_raw.permute(1, 0, 2).reshape(D, N*P)

    total = K_2d.shape[1]
    if total > cap:
        rng = np.random.default_rng(SEED)
        idx = torch.from_numpy(rng.choice(total, cap, replace=False))
        K_2d = K_2d[:, idx]; V_2d = V_2d[:, idx]

    return K_2d.to(device), V_2d.to(device)


def apply_ortcl_rotation(model, src_loader, tgt_loader, device, batch):
    """
    Apply ORTCL SVD rotation to model.blocks[-1].icb.conv3.weight.
    conv3 is a Conv1d(hidden, emb_dim, 1) so weight shape is [emb_dim, hidden, 1].
    We treat it as a linear [emb_dim, hidden] and rotate.
    """
    model = copy.deepcopy(model)
    icb   = model.blocks[-1].icb
    W     = icb.conv3.weight.data.squeeze(-1).clone()  # [emb_dim, hidden]
    D     = W.shape[0]
    eps   = torch.eye(D, device=device) * 1e-6

    src_ld = make_loader(src_loader.dataset, batch=batch, shuffle=False)
    tgt_ld = make_loader(tgt_loader.dataset, batch=batch, shuffle=False)

    print("    Collecting ICB features …")
    K_a, V_a = collect_icb_features(model, src_ld, device)
    K_b, V_b = collect_icb_features(model, tgt_ld, device)

    print("    Computing SVD rotation …")
    U0, S0, V0_T = torch.linalg.svd(W, full_matrices=False)
    V0 = V0_T.T

    K_combined = torch.cat([K_a, K_b], dim=1)
    # Project into W's right singular subspace (need W to be [emb_dim, hidden])
    # K has shape [emb_dim, N] — project via V0 [hidden, emb_dim]
    # Actually K_2d is [emb_dim, N_total], and W is [emb_dim, hidden]
    # We'll do a simplified ORTCL in the emb_dim × emb_dim space
    K_proj = U0 @ (U0.T @ K_combined)   # [emb_dim, N]

    U_K, S_K, X_T = torch.linalg.svd(K_proj.T, full_matrices=False)
    S_mat = X_T.T   # [emb_dim, emb_dim]

    A     = K_a   # use directly (in emb_dim space already)
    B_mat = K_b
    M     = V_a @ A.T + V_b @ B_mat.T + eps
    U_M, S_M, V_M_T = torch.linalg.svd(M)
    R = U_M @ V_M_T   # [emb_dim, emb_dim] orthogonal

    # Apply rotation: W_new = R @ W  (rotate emb_dim space)
    W_new = R @ W   # [emb_dim, hidden]
    icb.conv3.weight.data = W_new.unsqueeze(-1)

    orth_err = torch.norm(R.T @ R - torch.eye(D, device=device), p='fro').item()
    print(f"    Orthogonality error: {orth_err:.2e}")
    return model


def finetune(model, calib_ds, src_test_ld, tgt_test_ld, device, batch, head_only):
    model = copy.deepcopy(model)
    if head_only:
        for p in model.parameters(): p.requires_grad_(False)
        for p in model.head.parameters(): p.requires_grad_(True)
        opt = torch.optim.AdamW(model.head.parameters(), lr=FT_LR, weight_decay=1e-4)
    else:
        for p in model.parameters(): p.requires_grad_(True)
        opt = torch.optim.AdamW(model.parameters(), lr=FT_LR, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    ld   = make_loader(calib_ds, batch=min(batch, len(calib_ds)), shuffle=True)
    for _ in range(FT_EPOCHS):
        model.train()
        for xb, yb in ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
    return eval_f1(model, src_test_ld, device), eval_f1(model, tgt_test_ld, device)


# ── One scenario ───────────────────────────────────────────────────────────────

VARIANTS = ["No correction", "ORTCL (LF)", "ORTCL+HeadFT", "ORTCL+FullFT"]

def run_scenario(src_model, src_tr_ds, tgt_tr_ds, src_te_ds, tgt_te_ds,
                 device, batch):
    src_tr_ld = make_loader(src_tr_ds, batch=batch, shuffle=False)
    tgt_tr_ld = make_loader(tgt_tr_ds, batch=batch, shuffle=False)
    src_te_ld = make_loader(src_te_ds, batch=batch)
    tgt_te_ld = make_loader(tgt_te_ds, batch=batch)
    calib_ds  = calib_subset(tgt_tr_ds)

    f1_src_before = eval_f1(src_model, src_te_ld, device)
    f1_tgt_before = eval_f1(src_model, tgt_te_ld, device)

    results = {}
    results["No correction"] = (f1_src_before, f1_tgt_before, 0.0)

    print("    Applying ORTCL rotation …")
    rotated = apply_ortcl_rotation(src_model, src_tr_ld, tgt_tr_ld, device, batch)
    src_rot = eval_f1(rotated, src_te_ld, device)
    tgt_rot = eval_f1(rotated, tgt_te_ld, device)
    results["ORTCL (LF)"] = (src_rot, tgt_rot, f1_src_before - src_rot)

    print("    ORTCL + HeadFT …")
    src_hft, tgt_hft = finetune(rotated, calib_ds, src_te_ld, tgt_te_ld, device, batch, head_only=True)
    results["ORTCL+HeadFT"] = (src_hft, tgt_hft, f1_src_before - src_hft)

    print("    ORTCL + FullFT …")
    src_fft, tgt_fft = finetune(rotated, calib_ds, src_te_ld, tgt_te_ld, device, batch, head_only=False)
    results["ORTCL+FullFT"] = (src_fft, tgt_fft, f1_src_before - src_fft)

    for name, (s, t, f) in results.items():
        print(f"    {name:<22} src={s:.4f}  tgt={t:.4f}  forget={f:+.4f}")
    return results


# ── Dataset sweep ──────────────────────────────────────────────────────────────

def run_sweep(tag, scenarios, load_fn, device, batch):
    print(f"\n{'#'*60}\n## {tag}  ({len(scenarios)} scenarios)\n{'#'*60}")
    pop = {v: {"src":[],"tgt":[],"fgt":[]} for v in VARIANTS}
    all_res = {}

    for src_id, tgt_id in scenarios:
        src_ds, tgt_ds, src_name, tgt_name, src_model = load_fn(src_id, tgt_id, device)
        scen_key = f"{src_name}→{tgt_name}"
        print(f"\n{'='*55}\n  {scen_key}\n{'='*55}")
        res = run_scenario(src_model,
                           src_ds["train"], tgt_ds["train"],
                           src_ds["test"],  tgt_ds["test"],
                           device, batch)
        all_res[scen_key] = res
        for v in VARIANTS:
            s, t, f = res[v]
            pop[v]["src"].append(s); pop[v]["tgt"].append(t); pop[v]["fgt"].append(f)

    # Summary
    print(f"\n{'='*60}")
    print(f"POPULATION SUMMARY — {tag} ORTCL (TSLANet backbone)\n")
    hdr = f"  {'Method':<22} {'Src F1':>14}  {'Tgt F1':>14}  {'Forgetting':>14}"
    sep = "  " + "-"*68
    print(hdr); print(sep)
    lines = [f"POPULATION SUMMARY — {tag} ORTCL on TSLANet\n", hdr, sep]
    for v in VARIANTS:
        sm = np.mean(pop[v]["src"]); ss = np.std(pop[v]["src"])
        tm = np.mean(pop[v]["tgt"]); ts = np.std(pop[v]["tgt"])
        fm = np.mean(pop[v]["fgt"]); fs = np.std(pop[v]["fgt"])
        row = (f"  {v:<22} {sm:.4f}±{ss:.4f}  {tm:.4f}±{ts:.4f}  {fm:+.4f}±{fs:.4f}")
        print(row); lines.append(row)
    (OUT_DIR / f"{tag.lower()}_population_summary.txt").write_text("\n".join(lines))

    # Per-scenario
    scen_lines = [f"Per-scenario results — {tag} ORTCL\n"]
    for sk, res in all_res.items():
        scen_lines.append(f"\nScenario: {sk}")
        scen_lines.append(f"  {'Method':<22} {'Src F1':>8}  {'Tgt F1':>8}  {'Forget':>8}")
        for v in VARIANTS:
            s, t, f = res[v]
            scen_lines.append(f"  {v:<22} {s:>8.4f}  {t:>8.4f}  {f:>+8.4f}")
    (OUT_DIR / f"{tag.lower()}_per_scenario.txt").write_text("\n".join(scen_lines))
    return all_res, pop


# ── Dataset loaders ────────────────────────────────────────────────────────────

def load_fd(src_id, tgt_id, device):
    src_name = FD_MAP[src_id]; tgt_name = FD_MAP[tgt_id]
    src_ds = {"train": tft.FDDataset(src_name,"train",REPO/"dataset"),
              "test" : tft.FDDataset(src_name,"test", REPO/"dataset")}
    tgt_ds = {"train": tft.FDDataset(tgt_name,"train",REPO/"dataset"),
              "test" : tft.FDDataset(tgt_name,"test", REPO/"dataset")}
    model  = tft.train_source(src_name, FDArgs(), device)
    model.eval(); [p.requires_grad_(False) for p in model.parameters()]
    return src_ds, tgt_ds, src_name, tgt_name, model

def load_hhar(src_id, tgt_id, device):
    src_name = HHAR_NAMES[src_id]; tgt_name = HHAR_NAMES[tgt_id]
    src_ds = {"train": tht.HHARDataset(src_id,"train",REPO/"dataset"),
              "test" : tht.HHARDataset(src_id,"test", REPO/"dataset")}
    tgt_ds = {"train": tht.HHARDataset(tgt_id,"train",REPO/"dataset"),
              "test" : tht.HHARDataset(tgt_id,"test", REPO/"dataset")}
    model  = tht.train_source(src_id, HHARArgs(), device)
    model.eval(); [p.requires_grad_(False) for p in model.parameters()]
    return src_ds, tgt_ds, src_name, tgt_name, model

def load_sleep(src_id, tgt_id, device):
    src_name = f"Sleep-{src_id}"; tgt_name = f"Sleep-{tgt_id}"
    src_ds = {"train": tst.SleepEDFDataset(src_id,"train",REPO/"dataset"),
              "test" : tst.SleepEDFDataset(src_id,"test", REPO/"dataset")}
    tgt_ds = {"train": tst.SleepEDFDataset(tgt_id,"train",REPO/"dataset"),
              "test" : tst.SleepEDFDataset(tgt_id,"test", REPO/"dataset")}
    model  = tst.train_source(src_id, SleepArgs(), device)
    model.eval(); [p.requires_grad_(False) for p in model.parameters()]
    return src_ds, tgt_ds, src_name, tgt_name, model


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("ORTCL on TSLANet — 3 datasets × 10 scenarios × 4 variants\n")

    run_sweep("FD",       FD_SCENARIOS,    load_fd,    device, BATCH)
    run_sweep("HHAR",     HHAR_SCENARIOS,  load_hhar,  device, BATCH)
    run_sweep("SleepEDF", SLEEP_SCENARIOS, load_sleep, device, 64)

    print(f"\nAll outputs → {OUT_DIR}")
