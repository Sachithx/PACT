"""
run_sleep_unified_eval.py — Unified Evaluation for Sleep-EDF

Same 15 methods as run_unified_eval.py but applied to Sleep-EDF
cross-subject scenarios (10 AdaTime EEG scenarios, 20 subjects).

Methods (identical implementations, imported from run_unified_eval.py):
  Label-free:  NO_ADAPT, DANN, Deep_Coral, MMDA, TENT
  Semi-sup DA: DANN†, CDAN†, CoTMix†  (5% labeled target)
  Calibration: Full FT, Head FT, EWC, LwF, MAS, PACT (ng), PACT

Run:
  CUDA_VISIBLE_DEVICES=1 \\
  LD_PRELOAD=/home/AD/sachith/.conda/envs/entrope/lib/libstdc++.so.6 \\
  /home/AD/sachith/.conda/envs/entrope/bin/python -u run_sleep_unified_eval.py
"""

import sys, importlib.util
import numpy as np
from pathlib import Path

REPO = Path(__file__).parent
sys.path.insert(0, str(REPO / "TSLANet" / "Classification"))
sys.path.append(str(REPO / "baselines" / "AdaTime"))

def _load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Import all method implementations from run_unified_eval
ue  = _load_mod("ue",  REPO / "run_unified_eval.py")
tst = _load_mod("tst", REPO / "transfer_sleep_tslanet.py")

import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ─────────────────────────────────────────────────────────────────────

DATA_ROOT = ue.DATA_ROOT
CKPT_DIR  = ue.CKPT_DIR
OUT_DIR   = REPO / "results" / "sleep_unified_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SLEEP_SCENARIOS = tst.SLEEP_SCENARIOS   # 10 AdaTime EEG scenarios

METHODS   = ue.METHODS
mkeys     = [m[0] for m in METHODS]
mnames    = {m[0]: m[2] for m in METHODS}
mbudgets  = {m[0]: m[1] for m in METHODS}


class SleepArgs:
    """Args for Sleep-EDF TSLANet (smaller batch due to longer signals)."""
    emb_dim=128; depth=2; patch_size=8; dropout_rate=0.15; masking_ratio=0.4
    ICB=True; ASB=True; adaptive_filter=True
    data_root=str(DATA_ROOT)
    ckpt_dir=str(CKPT_DIR)
    batch_size=64; pretrain=True; pretrain_epochs=40; pretrain_lr=1e-4
    cls_epochs=80; cls_lr=1e-4; retrain=False


# ── Dataset loader for Sleep-EDF ───────────────────────────────────────────────

def load_sleep_datasets(src_id, tgt_id):
    src_name = f"Sleep-{src_id}"
    tgt_name = f"Sleep-{tgt_id}"
    return dict(
        src_name=src_name, tgt_name=tgt_name,
        src_train=tst.SleepEDFDataset(src_id, "train", DATA_ROOT),
        src_test =tst.SleepEDFDataset(src_id, "test",  DATA_ROOT),
        tgt_train=tst.SleepEDFDataset(tgt_id, "train", DATA_ROOT),
        tgt_test =tst.SleepEDFDataset(tgt_id, "test",  DATA_ROOT),
    )


def load_frozen_sleep_model(src_id, device):
    model = tst.train_source(src_id, SleepArgs(), device)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model


# ── Scenario runner (mirrors run_unified_eval.run_all_methods) ─────────────────

def run_all_methods_sleep(datasets, frozen_model, device):
    """Run all 15 methods for one Sleep-EDF scenario."""
    # Use smaller batch for Sleep-EDF (seq_len=3000 → more memory per sample)
    SLEEP_BATCH = 64

    def _loader(ds, shuffle=False, drop_last=False):
        return torch.utils.data.DataLoader(
            ds, batch_size=SLEEP_BATCH, shuffle=shuffle, drop_last=drop_last)

    src_train_ds = datasets["src_train"]
    src_test_ds  = datasets["src_test"]
    tgt_train_ds = datasets["tgt_train"]
    tgt_test_ds  = datasets["tgt_test"]

    calib_ds     = ue.calib_split(tgt_train_ds)
    src_train_ld = _loader(src_train_ds, shuffle=True, drop_last=True)
    src_test_ld  = _loader(src_test_ds)
    tgt_train_ld = _loader(tgt_train_ds, shuffle=True, drop_last=True)
    tgt_test_ld  = _loader(tgt_test_ds)

    f1_src_before = ue.eval_f1_loader(frozen_model, src_test_ld, device)
    f1_tgt_nocorr = ue.eval_f1_loader(frozen_model, tgt_test_ld, device)
    print(f"  src_before={f1_src_before:.4f}  tgt_nocorr={f1_tgt_nocorr:.4f}")

    results = {}
    for key, budget, name in METHODS:
        print(f"  [{budget:12s}] {name:<18}", end=" … ", flush=True)
        try:
            if key == "no_adapt":
                r = ue.run_no_adapt(frozen_model, tgt_test_ld, src_test_ld, device)
            elif key == "dann":
                r = ue.run_dann(frozen_model, src_train_ld, tgt_train_ld,
                                src_test_ld, tgt_test_ld, device)
            elif key == "coral":
                r = ue.run_coral(frozen_model, src_train_ld, tgt_train_ld,
                                 src_test_ld, tgt_test_ld, device)
            elif key == "mmda":
                r = ue.run_mmda(frozen_model, src_train_ld, tgt_train_ld,
                                src_test_ld, tgt_test_ld, device)
            elif key == "tent":
                r = ue.run_tent(frozen_model, tgt_train_ds, src_test_ld, tgt_test_ld, device)
            elif key == "dann_ss":
                r = ue.run_dann_ss(frozen_model, src_train_ld, tgt_train_ld, calib_ds,
                                   src_test_ld, tgt_test_ld, device)
            elif key == "cdan_ss":
                r = ue.run_cdan_ss(frozen_model, src_train_ld, tgt_train_ld, calib_ds,
                                   src_test_ld, tgt_test_ld, device)
            elif key == "cotmix_ss":
                r = ue.run_cotmix_ss(frozen_model, src_train_ld, tgt_train_ld, calib_ds,
                                     src_test_ld, tgt_test_ld, device)
            elif key == "full_ft":
                r = ue.run_fullft(frozen_model, calib_ds, src_test_ld, tgt_test_ld, device)
            elif key == "head_ft":
                r = ue.run_headft(frozen_model, calib_ds, src_test_ld, tgt_test_ld, device)
            elif key == "ewc":
                r = ue.run_ewc(frozen_model, src_train_ds, calib_ds,
                               src_test_ld, tgt_test_ld, device)
            elif key == "src_distill":
                r = ue.run_src_distill(frozen_model, src_train_ds, calib_ds,
                                       src_test_ld, tgt_test_ld, device)
            elif key == "lwf":
                r = ue.run_lwf(frozen_model, None, calib_ds,
                               src_test_ld, tgt_test_ld, device)
            elif key == "mas":
                r = ue.run_mas(frozen_model, src_train_ds, calib_ds,
                               src_test_ld, tgt_test_ld, device)
            elif key == "pact_ng":
                r = ue.run_pact(frozen_model, src_train_ld, calib_ds,
                                src_test_ld, tgt_test_ld, device, gate=False)
            elif key == "pact":
                r = ue.run_pact(frozen_model, src_train_ld, calib_ds,
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


# ── Dataset sweep ──────────────────────────────────────────────────────────────

def run_sleep_sweep(device):
    print(f"\n{'#'*72}")
    print(f"## SLEEP-EDF SWEEP  ({len(SLEEP_SCENARIOS)} scenarios, 5% labels, learned gate)")
    print(f"{'#'*72}")

    pop_src = {k: [] for k in mkeys}
    pop_tgt = {k: [] for k in mkeys}
    pop_fgt = {k: [] for k in mkeys}
    all_results = {}

    for src_id, tgt_id in SLEEP_SCENARIOS:
        datasets  = load_sleep_datasets(src_id, tgt_id)
        scen_key  = f"{datasets['src_name']}→{datasets['tgt_name']}"
        print(f"\n{'='*60}\n  Scenario: {scen_key}\n{'='*60}")
        frozen_model = load_frozen_sleep_model(src_id, device)
        res, src_bef = run_all_methods_sleep(datasets, frozen_model, device)
        all_results[scen_key] = res
        for k in mkeys:
            if k in res:
                s, t, f = res[k]
                pop_src[k].append(s); pop_tgt[k].append(t); pop_fgt[k].append(f)

    # ── Population summary ─────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"POPULATION SUMMARY — Sleep-EDF  (mean ± std, {len(SLEEP_SCENARIOS)} scenarios)")
    print(f"{'='*72}")
    hdr = f"  {'Method':<20} {'Budget':>12}  {'Src F1':>12}  {'Tgt F1':>12}  {'Forgetting':>14}"
    sep = "  " + "-"*74
    print(hdr); print(sep)
    lines = [f"POPULATION SUMMARY — Sleep-EDF\n", hdr, sep]
    for k in mkeys:
        sm = np.mean(pop_src[k]); ss = np.std(pop_src[k])
        tm = np.mean(pop_tgt[k]); ts = np.std(pop_tgt[k])
        fm = np.mean(pop_fgt[k]); fs = np.std(pop_fgt[k])
        row = (f"  {mnames[k]:<20} {mbudgets[k]:>12}  "
               f"{sm:.4f}±{ss:.4f}  {tm:.4f}±{ts:.4f}  {fm:+.4f}±{fs:.4f}")
        print(row); lines.append(row)
    (OUT_DIR / "sleep_population_summary.txt").write_text("\n".join(lines))

    # ── Per-scenario table ─────────────────────────────────────────────────────
    scen_lines = ["Per-scenario results — Sleep-EDF\n"]
    for sk, res in all_results.items():
        scen_lines.append(f"\nScenario: {sk}")
        scen_lines.append(f"  {'Method':<20} {'Src F1':>8}  {'Tgt F1':>8}  {'Forget':>8}")
        for k in mkeys:
            if k in res:
                s, t, f = res[k]
                scen_lines.append(f"  {mnames[k]:<20} {s:>8.4f}  {t:>8.4f}  {f:>+8.4f}")
    (OUT_DIR / "sleep_per_scenario.txt").write_text("\n".join(scen_lines))

    # ── Figure ─────────────────────────────────────────────────────────────────
    GROUPS = [
        ("Lower bound",                     [0],         "#777777"),
        ("UDA Benchmark\n(label-free)",     [1, 2, 3],   "#2471a3"),
        ("Test-time\n(label-free)",         [4],          "#76448a"),
        ("Semi-supervised DA\n(5% labels)", [5, 6, 7],   "#17a589"),
        ("Fine-tuning\n(5% labels)",        [8, 9],          "#d35400"),
        ("Continual Learning\n(5% labels)", [10, 11, 12, 13], "#922b21"),
        ("Ours\n(5% labels)",               [14, 15],         "#1e8449"),
    ]
    BAR_COLORS = [
        "#999999",
        "#3498db", "#5dade2", "#85c1e9",
        "#9b59b6",
        "#1abc9c", "#17a589", "#148f77",
        "#e67e22", "#f0a500",
        "#e74c3c", "#8e44ad", "#c0392b", "#f1948a",
        "#52be80", "#1e8449",
    ]

    def _annotate_categories(ax, groups, y_line=-0.27, y_label=-0.35):
        trans = ax.get_xaxis_transform()
        for cat_label, idxs, color in groups:
            x1 = idxs[0] - 0.45; x2 = idxs[-1] + 0.45; xc = (x1+x2)/2
            ax.axvspan(x1, x2, ymin=0, ymax=1.0, alpha=0.07, color=color, zorder=0, lw=0)
            ax.annotate("", xy=(x1, y_line), xycoords=trans,
                        xytext=(x2, y_line), textcoords=trans,
                        arrowprops=dict(arrowstyle="-[", color=color, lw=1.4, mutation_scale=5))
            ax.text(xc, y_label, cat_label, transform=trans, ha="center", va="top",
                    fontsize=6.5, color=color, fontweight="bold", linespacing=1.3)

    x = np.arange(len(mkeys))
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(bottom=0.30, top=0.88, wspace=0.28)

    ax = axes[0]
    means_tgt = [np.mean(pop_tgt[k]) if pop_tgt[k] else 0 for k in mkeys]
    stds_tgt  = [np.std(pop_tgt[k])  if pop_tgt[k] else 0 for k in mkeys]
    bars = ax.bar(x, means_tgt, yerr=stds_tgt, capsize=4, color=BAR_COLORS,
                  edgecolor="white", linewidth=0.5,
                  error_kw=dict(elinewidth=1.0, ecolor="#555"))
    ax.axhline(means_tgt[0], color="#aaa", lw=1.1, ls="--", label="NO_ADAPT baseline")
    ax.set_ylim(0, 1.12); ax.set_xlim(-0.6, len(mkeys)-0.4)
    ax.set_ylabel("Target macro-F1", fontsize=10)
    ax.set_title("Target domain adaptation", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([mnames[k] for k in mkeys], rotation=35, ha="right", fontsize=8)
    ax.tick_params(axis="x", pad=2)
    ax.legend(fontsize=8, loc="upper left"); ax.grid(axis="y", ls="--", alpha=0.4)
    for bar, v in zip(bars, means_tgt):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                f"{v:.3f}", ha="center", fontsize=6.5)
    _annotate_categories(ax, GROUPS)

    ax = axes[1]
    means_fgt = [np.mean(pop_fgt[k]) if pop_fgt[k] else 0 for k in mkeys]
    stds_fgt  = [np.std(pop_fgt[k])  if pop_fgt[k] else 0 for k in mkeys]
    ax.bar(x, means_fgt, yerr=stds_fgt, capsize=4, color=BAR_COLORS,
           edgecolor="white", linewidth=0.5,
           error_kw=dict(elinewidth=1.0, ecolor="#555"))
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlim(-0.6, len(mkeys)-0.4)
    ax.set_ylabel("Forgetting  (source F1 drop)", fontsize=10)
    ax.set_title("Catastrophic forgetting", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([mnames[k] for k in mkeys], rotation=35, ha="right", fontsize=8)
    ax.tick_params(axis="x", pad=2)
    ax.grid(axis="y", ls="--", alpha=0.4)
    _annotate_categories(ax, GROUPS)

    fig.suptitle(
        f"Unified Evaluation : Sleep-EDF  (5% target labels, learned gate)\n"
        f"TSLANet backbone · 10 AdaTime EEG scenarios · mean ± std",
        fontsize=11, fontweight="bold")

    for ext in ["png", "pdf"]:
        out = OUT_DIR / f"sleep_figure.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Saved → {out}")
    plt.close()

    print(f"\nAll outputs → {OUT_DIR}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Label budget: 5%  |  PACT gate: learned MLP  |  Dataset: Sleep-EDF")
    print(f"Methods: {len(METHODS)}  |  Scenarios: {len(SLEEP_SCENARIOS)}")
    run_sleep_sweep(device)
    print("Done.")
