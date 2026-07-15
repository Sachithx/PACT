"""
run_ortcl_fd.py — ORTCL baseline on FD cross-domain scenarios (3 variants)

Backbone: TrmEncoderBackbone (patch embed + transformer + mean-pool head).
Training: source domain cross-entropy, trained from scratch per scenario.

Three variants evaluated per scenario:

  ORTCL (LF)      — SVD rotation only, 0 target labels (original)
  ORTCL + HeadFT  — rotation → head-only fine-tune on 5% target calib
  ORTCL + FullFT  — rotation → full model fine-tune on 5% target calib

All compared against PACT (learned gate, 5%) from exp_crossfd_lg.py.

NOTE: ORTCL uses TrmEncoderBackbone (standard MHA), not TSLANet (ASB+ICB).

Run:
  CUDA_VISIBLE_DEVICES=0 LD_PRELOAD=...libstdc++.so.6 \\
  python -u run_ortcl_fd.py
"""

import sys, copy
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit

REPO  = Path(__file__).parent
ORTCL = REPO / "baselines" / "ORTCL"

import importlib.util, importlib.machinery, types

def _load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# ORTCL's utils/ is a package — register it as a package module first,
# then register utils.masking so the submodule import in SelfAttention_Family works
_ortcl_utils_pkg = types.ModuleType("utils")
_ortcl_utils_pkg.__path__ = [str(ORTCL / "utils")]
_ortcl_utils_pkg.__package__ = "utils"
sys.modules["utils"] = _ortcl_utils_pkg
_load_file("utils.masking", ORTCL / "utils" / "masking.py")

# Now load ORTCL layers by file path
_embed     = _load_file("ortcl_embed",      ORTCL / "layers" / "Embed.py")
_attn      = _load_file("ortcl_attn",       ORTCL / "layers" / "SelfAttention_Family.py")
_encdec    = _load_file("ortcl_encdec",     ORTCL / "layers" / "Transformer_EncDec.py")

PatchEmbedding = _embed.PatchEmbedding
AttentionLayer = _attn.AttentionLayer
FullAttention  = _attn.FullAttention
Encoder        = _encdec.Encoder
EncoderLayer   = _encdec.EncoderLayer

# Clean cached utils so TSLANet's utils.py can be found correctly
del sys.modules["utils"]
del sys.modules["utils.masking"]

# Load TSLANet
sys.path.insert(0, str(REPO / "TSLANet" / "Classification"))
_spec = importlib.util.spec_from_file_location("tft", REPO / "transfer_fd_tslanet.py")
tft   = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(tft)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────

DATA_ROOT    = REPO / "dataset"
OUT_DIR      = REPO / "results" / "ortcl_fd"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FD_MAP       = {0: "FD-A", 1: "FD-B", 2: "FD-C", 3: "FD-D"}
ALL_DOMAINS  = ["FD-A", "FD-B", "FD-C", "FD-D"]
# AdaTime's 10 FD cross-domain scenarios
SCENARIOS    = [(0,1),(0,3),(1,0),(1,2),(1,3),(2,1),(2,3),(3,0),(3,1),(3,2)]

SEQ_LEN      = 5120
N_CHANNELS   = 1
N_CLASSES    = 3
PATCH_LEN    = 64       # 5120/64 = 80 patches
STRIDE       = 64
D_MODEL      = 128      # match TSLANet embedding dim
D_FF         = 256
E_LAYERS     = 2
N_HEADS      = 4
DROPOUT      = 0.1
SEED         = 42
BATCH        = 256
TRAIN_EPOCHS = 80
LR           = 1e-4
CALIB_FRAC   = 0.05   # 5% target labels for HeadFT / FullFT variants
FT_EPOCHS    = 80
FT_LR        = 1e-4

# Cap patches used in SVD (8184 src × 80 patches = 655K → cap for memory)
SVD_PATCH_CAP = 50_000

# PACT (learned gate, 5%) per-scenario results — loaded for comparison table
PACT_LG_FILE = Path(__file__).parent / "results" / "crossfd_lg" / "per_scenario_table.txt"


# ── ORTCL Backbone with Classification Head ───────────────────────────────────

class ORTCLClassifier(nn.Module):
    """
    TrmEncoderBackbone + mean-pool classification head.
    Input:  [B, 1, 5120]  (FD channel-first)
    Output: [B, 3]        logits
    Also exposes extract() → [B, d_model] (mean-pooled patch embeddings).
    """
    def __init__(self, seq_len=SEQ_LEN, n_ch=N_CHANNELS, n_cls=N_CLASSES,
                 d_model=D_MODEL, d_ff=D_FF, e_layers=E_LAYERS, n_heads=N_HEADS,
                 patch_len=PATCH_LEN, stride=STRIDE, dropout=DROPOUT):
        super().__init__()
        n_patches = seq_len // stride   # 80

        self.patch_embedding = PatchEmbedding(d_model, patch_len, stride, 0, dropout)

        self.encoder = Encoder(
            [EncoderLayer(
                AttentionLayer(
                    FullAttention(False, factor=1, attention_dropout=dropout,
                                  output_attention=True),
                    d_model, n_heads),
                d_model, d_ff, dropout=dropout, activation="gelu")
             for _ in range(e_layers)],
            norm_layer=nn.LayerNorm(d_model))

        self.head = nn.Linear(d_model, n_cls)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def _encode(self, x):
        """x: [B, 1, T] → enc: [B, N_patches, d_model]"""
        # PatchEmbedding expects [B, M, T] where M=n_channels
        enc, n_vars = self.patch_embedding(x)   # [B*M, N, d_model], M=1
        enc, _      = self.encoder(enc)          # [B*M, N, d_model]
        enc = enc.reshape(-1, n_vars, enc.shape[-2], enc.shape[-1])  # [B, M, N, d_model]
        enc = enc.squeeze(1)                     # [B, N, d_model]  (M=1)
        return enc

    def extract(self, x):
        """Returns mean-pooled embedding [B, d_model]."""
        return self._encode(x).mean(1)

    def forward(self, x):
        return self.head(self.extract(x))


# ── Data helpers ──────────────────────────────────────────────────────────────

def make_loader(ds, shuffle=False, batch=BATCH):
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=shuffle)


@torch.no_grad()
def eval_f1(model, loader, device):
    model.eval()
    preds, targets = [], []
    for xb, yb in loader:
        preds.extend(model(xb.to(device)).argmax(1).cpu().numpy())
        targets.extend(yb.numpy())
    return f1_score(targets, preds, average="macro", zero_division=0)


# ── Training from scratch ─────────────────────────────────────────────────────

def train_ortcl(src_ds, src_test_ld, device):
    model = ORTCLClassifier().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    crit  = nn.CrossEntropyLoss()
    ld    = make_loader(src_ds, shuffle=True)
    best_f1, best_sd = 0.0, None

    for ep in range(TRAIN_EPOCHS):
        model.train()
        for xb, yb in ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        if (ep + 1) % 20 == 0:
            f1 = eval_f1(model, src_test_ld, device)
            if f1 > best_f1:
                best_f1 = f1
                best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"    ep {ep+1}/{TRAIN_EPOCHS}  src_F1={f1:.4f}")

    if best_sd is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_sd.items()})
    return model


# ── Feature collection for ORTCL rotation ────────────────────────────────────

def collect_kv_features(model, loader, device, cap=SVD_PATCH_CAP):
    """
    Collect K (attention output, input to out_projection) and
    V (out_projection output) from encoder layer 0.
    Both have shape collected as list of [B, N_patches, d_model].
    Returns K_2d [d_model, min(N_total, cap)], V_2d [d_model, same].
    """
    attn_layer = model.encoder.attn_layers[0].attention  # AttentionLayer

    K_list, V_list = [], []

    def k_hook(module, inp, out):
        # FullAttention output: (V_contiguous [B, L, H, D], attn_weights)
        v_out = out[0]              # [B, L, H, D_v]
        B, L, H, D = v_out.shape
        K_list.append(v_out.detach().reshape(B, L, -1).cpu())  # [B, L, H*D]

    def v_hook(module, inp, out):
        V_list.append(out.detach().cpu())   # [B, L, d_model]

    h1 = attn_layer.inner_attention.register_forward_hook(k_hook)
    h2 = attn_layer.out_projection.register_forward_hook(v_hook)

    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            model(xb.to(device))

    h1.remove(); h2.remove()

    K = torch.cat(K_list, dim=0)   # [N_samples, N_patches, d_model]
    V = torch.cat(V_list, dim=0)   # [N_samples, N_patches, d_model]

    # Reshape: [d_model, N_samples * N_patches]
    N, P, D = K.shape
    K_2d = K.reshape(N * P, D).T   # [d_model, N*P]
    V_2d = V.reshape(N * P, D).T   # [d_model, N*P]

    # Subsample for tractable SVD
    total = K_2d.shape[1]
    if total > cap:
        rng  = np.random.default_rng(SEED)
        idx  = rng.choice(total, cap, replace=False)
        idx  = torch.from_numpy(idx)
        K_2d = K_2d[:, idx]
        V_2d = V_2d[:, idx]

    return K_2d.to(device), V_2d.to(device)


# ── ORTCL rotation ────────────────────────────────────────────────────────────

def apply_ortcl_rotation(model, src_loader, tgt_loader, device):
    """
    Apply ORTCL's SVD-based orthogonal rotation to out_projection weight of
    encoder layer 0.

      W* = R @ (W @ S^T)

    where R is orthogonal, S encodes the shared subspace of source and target
    attention features.

    Label-free: uses only raw input signals from both domains.
    """
    attn_layer = model.encoder.attn_layers[0].attention
    W = attn_layer.out_projection.weight.data.clone()   # [d_model, d_model]
    D = W.shape[0]
    eps = torch.eye(D, device=device) * 1e-6

    print("    Collecting source features …")
    K_a, V_a = collect_kv_features(model, src_loader, device)

    print("    Collecting target features …")
    K_b, V_b = collect_kv_features(model, tgt_loader, device)

    print("    Computing SVD rotation …")
    # Step 1: SVD of W
    U0, S0, V0_T = torch.linalg.svd(W, full_matrices=False)
    V0 = V0_T.T     # [d_model, d_model]

    # Step 2: project combined K into W's column subspace
    K_combined = torch.cat([K_a, K_b], dim=1)          # [d_model, N_a+N_b]
    K_proj     = V0 @ (V0_T @ K_combined)              # [d_model, N_a+N_b]

    # Step 3: SVD of K_proj^T to find shared subspace S
    U_K, S_K, X_T = torch.linalg.svd(K_proj.T, full_matrices=False)  # X_T [d_model, d_model]
    S = X_T.T       # [d_model, d_model]

    # Step 4: compute transformed source and target projections
    A     = W @ (S.T @ K_a)   # [d_model, N_a]
    B_mat = W @ (S.T @ K_b)   # [d_model, N_b]

    # Step 5: compute rotation-inducing matrix M
    M = V_a @ A.T + V_b @ B_mat.T + eps   # [d_model, d_model]
    U_M, S_M, V_M_T = torch.linalg.svd(M)
    R = U_M @ V_M_T   # orthogonal [d_model, d_model]

    # Step 6: update weight
    W_star = R @ (W @ S.T)
    attn_layer.out_projection.weight.data = W_star

    orth_err = torch.norm(R.T @ R - torch.eye(D, device=device), p='fro').item()
    print(f"    Rotation orthogonality error: {orth_err:.2e}")
    return model


# ── Post-rotation supervised fine-tuning ─────────────────────────────────────

def ft_after_rotation(model_rotated, calib_ds, src_test_ld, tgt_test_ld,
                      device, head_only=False):
    """Fine-tune either head-only or full model on 5% target calib after rotation."""
    model = copy.deepcopy(model_rotated)
    if head_only:
        for p in model.parameters(): p.requires_grad_(False)
        for p in model.head.parameters(): p.requires_grad_(True)
        opt = torch.optim.AdamW(model.head.parameters(), lr=FT_LR, weight_decay=1e-4)
    else:
        for p in model.parameters(): p.requires_grad_(True)
        opt = torch.optim.AdamW(model.parameters(), lr=FT_LR, weight_decay=1e-4)

    crit = nn.CrossEntropyLoss()
    ld   = torch.utils.data.DataLoader(calib_ds, batch_size=BATCH, shuffle=True)
    for _ in range(FT_EPOCHS):
        model.train()
        for xb, yb in ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
    model.eval()
    return eval_f1(model, src_test_ld, device), eval_f1(model, tgt_test_ld, device)


# ── One scenario ──────────────────────────────────────────────────────────────

def run_scenario(src_id, tgt_id, device):
    src_name = FD_MAP[src_id]
    tgt_name = FD_MAP[tgt_id]
    print(f"\n  Scenario: {src_name}→{tgt_name}")

    src_train_ds = tft.FDDataset(src_name, "train", DATA_ROOT)
    src_test_ds  = tft.FDDataset(src_name, "test",  DATA_ROOT)
    tgt_train_ds = tft.FDDataset(tgt_name, "train", DATA_ROOT)
    tgt_test_ds  = tft.FDDataset(tgt_name, "test",  DATA_ROOT)

    src_train_ld = make_loader(src_train_ds, shuffle=True)
    src_test_ld  = make_loader(src_test_ds)
    tgt_train_ld = make_loader(tgt_train_ds)
    tgt_test_ld  = make_loader(tgt_test_ds)

    # 5% stratified calibration from target train (shared by HeadFT + FullFT)
    y_all = tgt_train_ds.y.numpy()
    sss   = StratifiedShuffleSplit(n_splits=1, test_size=1-CALIB_FRAC, random_state=SEED)
    calib_idx, _ = next(sss.split(np.zeros(len(y_all)), y_all))
    calib_ds  = torch.utils.data.Subset(tgt_train_ds, calib_idx)
    print(f"  Calib: {len(calib_idx)} samples (5% of {len(y_all)})")

    # Train from scratch on source
    print("  Training …")
    model = train_ortcl(src_train_ds, src_test_ld, device)

    f1_src_before = eval_f1(model, src_test_ld, device)
    f1_tgt_before = eval_f1(model, tgt_test_ld, device)
    print(f"  No correction: src={f1_src_before:.4f}  tgt={f1_tgt_before:.4f}")

    # Apply ORTCL rotation (label-free)
    print("  Applying ORTCL rotation …")
    model_rotated = copy.deepcopy(model)
    model_rotated = apply_ortcl_rotation(model_rotated, src_train_ld, tgt_train_ld, device)

    # Variant A: label-free (rotation only)
    s_lf = eval_f1(model_rotated, src_test_ld, device)
    t_lf = eval_f1(model_rotated, tgt_test_ld, device)
    print(f"  [ORTCL LF    ] src={s_lf:.4f}  tgt={t_lf:.4f}  forget={f1_src_before-s_lf:+.4f}",
          flush=True)

    # Variant B: rotation → head-only fine-tune on 5% calib
    print("  [ORTCL+HeadFT] fine-tuning head …", flush=True)
    s_hft, t_hft = ft_after_rotation(model_rotated, calib_ds,
                                      src_test_ld, tgt_test_ld, device, head_only=True)
    print(f"  [ORTCL+HeadFT] src={s_hft:.4f}  tgt={t_hft:.4f}  "
          f"forget={f1_src_before-s_hft:+.4f}", flush=True)

    # Variant C: rotation → full model fine-tune on 5% calib
    print("  [ORTCL+FullFT] fine-tuning all …", flush=True)
    s_fft, t_fft = ft_after_rotation(model_rotated, calib_ds,
                                      src_test_ld, tgt_test_ld, device, head_only=False)
    print(f"  [ORTCL+FullFT] src={s_fft:.4f}  tgt={t_fft:.4f}  "
          f"forget={f1_src_before-s_fft:+.4f}", flush=True)

    return {
        "src_before": f1_src_before,
        "no_corr": (f1_src_before, f1_tgt_before, 0.0),
        "lf":  (s_lf,  t_lf,  f1_src_before - s_lf),
        "hft": (s_hft, t_hft, f1_src_before - s_hft),
        "fft": (s_fft, t_fft, f1_src_before - s_fft),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Backbone: TrmEncoderBackbone (d_model={D_MODEL}, e_layers={E_LAYERS}, "
          f"n_heads={N_HEADS}, patch_len={PATCH_LEN})")
    print(f"Variants: ORTCL (LF, 0 labels) | ORTCL+HeadFT (5%) | ORTCL+FullFT (5%)\n")

    VARIANTS = ["no_corr", "lf", "hft", "fft"]
    VAR_LABELS = {"no_corr": "No correction",
                  "lf": "ORTCL (LF)", "hft": "ORTCL+HeadFT", "fft": "ORTCL+FullFT"}
    pop = {v: {"src": [], "tgt": [], "forget": []} for v in VARIANTS}
    records = []

    for src_id, tgt_id in SCENARIOS:
        res = run_scenario(src_id, tgt_id, device)
        records.append((FD_MAP[src_id], FD_MAP[tgt_id], res))
        for v in VARIANTS:
            s, t, f = res[v]
            pop[v]["src"].append(s); pop[v]["tgt"].append(t); pop[v]["forget"].append(f)

    # ── Per-scenario table ─────────────────────────────────────────────────────
    scen_lines = ["ORTCL FD Results — No correction + 3 variants\n"]
    for src, tgt, res in records:
        scen_lines.append(f"\nScenario: {src}→{tgt}")
        scen_lines.append(f"  {'Method':<18} {'Src F1':>8}  {'Tgt F1':>8}  {'Forget':>8}")
        for v in VARIANTS:
            s, t, f = res[v]
            scen_lines.append(f"  {VAR_LABELS[v]:<18} {s:>8.4f}  {t:>8.4f}  {f:>+8.4f}")
    (OUT_DIR / "per_scenario_table.txt").write_text("\n".join(scen_lines))

    # ── Population summary ─────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("POPULATION SUMMARY (mean ± std, 10 scenarios)")
    print(f"{'='*72}")
    hdr = f"  {'Method':<20} {'Src F1':>14}  {'Tgt F1':>14}  {'Forgetting':>14}"
    sep = "  " + "-"*66
    print(hdr); print(sep)

    pop_lines = ["POPULATION SUMMARY — ORTCL FD (3 variants vs PACT)\n", hdr, sep]

    for v in VARIANTS:
        sm = np.mean(pop[v]["src"]);    ss = np.std(pop[v]["src"])
        tm = np.mean(pop[v]["tgt"]);    ts = np.std(pop[v]["tgt"])
        fm = np.mean(pop[v]["forget"]); fs = np.std(pop[v]["forget"])
        row = (f"  {VAR_LABELS[v]:<20} {sm:.4f}±{ss:.4f}  "
               f"{tm:.4f}±{ts:.4f}  {fm:+.4f}±{fs:.4f}")
        print(row); pop_lines.append(row)

    # PACT reference line
    pact_ref = ("  PACT (lg, 5%)        0.9970±0.0027  "
                "0.9718±0.0234  +0.0002±0.0005")
    print(sep)
    print(pact_ref + "  ← PACT (learned gate)")
    pop_lines.extend([sep, pact_ref + "  ← from exp_crossfd_lg.py"])
    pop_lines.append(f"\n  Backbone note: ORTCL uses TrmEncoderBackbone; "
                     f"PACT uses TSLANet (ASB+ICB). Not directly comparable.")
    (OUT_DIR / "population_summary.txt").write_text("\n".join(pop_lines))

    # ── Figure: grouped bar chart (3 metrics × 4 methods) ─────────────────────
    # Add PACT population numbers for visual comparison
    pact_pop = {"src": 0.9970, "src_s": 0.0027,
                "tgt": 0.9718, "tgt_s": 0.0234,
                "forget": 0.0002, "forget_s": 0.0005}

    # "lf" kept in table/summary but omitted from bar chart
    PLOT_VARIANTS = ["no_corr", "hft", "fft"]
    all_methods = PLOT_VARIANTS + ["pact"]
    all_labels  = [VAR_LABELS[v] for v in PLOT_VARIANTS] + ["PACT (lg, 5%)"]
    colors      = ["#888888", "#f4a460", "#e05c5c", "#1b7a2e"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    x = np.arange(len(all_methods))

    for ax, metric, title in zip(
            axes,
            ["src", "tgt", "forget"],
            ["Source F1 (retention)", "Target F1 (drift recovery)", "Forgetting"]):
        means = ([np.mean(pop[v][metric]) for v in PLOT_VARIANTS] +
                 [pact_pop[metric]])
        stds  = ([np.std(pop[v][metric])  for v in PLOT_VARIANTS] +
                 [pact_pop[metric + "_s"]])
        # no_corr forget is always 0 by definition — zero the std too
        if metric == "forget":
            stds[0] = 0.0
        bars  = ax.bar(x, means, yerr=stds, capsize=5, color=colors,
                       edgecolor="black", linewidth=0.6)
        if metric == "forget":
            ax.axhline(0, color="black", lw=0.8)
        else:
            ax.axhline(0.9, color="gray", ls="--", lw=0.8, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(all_labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylim((-0.05 if metric == "forget" else 0), 1.05)
        ax.set_ylabel("Macro F1" if metric != "forget" else "F1 drop")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(axis="y", ls="--", alpha=0.4)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height() + (0.01 if m >= 0 else -0.04),
                    f"{m:.3f}", ha="center", fontsize=8)

    fig.suptitle("ORTCL variants vs PACT (learned gate, 5%) on FD Cross-Domain\n"
                 "(10 AdaTime scenarios — note: different backbones)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "ortcl_vs_pact.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT_DIR / "ortcl_vs_pact.pdf", bbox_inches="tight")
    plt.close()

    print(f"\nOutputs → {OUT_DIR}")
    print("  population_summary.txt  per_scenario_table.txt  ortcl_vs_pact.{png,pdf}")


if __name__ == "__main__":
    main()
