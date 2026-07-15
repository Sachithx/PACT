"""
run_semisup_only.py — Incremental addition of semi-supervised DA methods.

Adds DANN†, CDAN†, CoTMix† to the existing 12-method results in
  results/unified_eval/{fd,hhar}_per_scenario.txt

Workflow:
  1. Parse existing per_scenario files (12 methods, all 20 scenarios)
  2. Load each scenario's frozen model + datasets
  3. Run only the 3 semi-supervised methods
  4. Insert results into per_scenario dicts (maintaining METHODS order)
  5. Write updated per_scenario.txt files
  6. Recalculate population summary (all 15 methods)
  7. Regenerate figures (all 15 methods)

Run:
  CUDA_VISIBLE_DEVICES=0 \\
  LD_PRELOAD=/home/AD/sachith/.conda/envs/entrope/lib/libstdc++.so.6 \\
  /home/AD/sachith/.conda/envs/entrope/bin/python -u run_semisup_only.py
"""

import sys, importlib.util
import numpy as np
from pathlib import Path

REPO = Path(__file__).parent
sys.path.insert(0, str(REPO / "TSLANet" / "Classification"))
sys.path.append(str(REPO / "baselines" / "AdaTime"))

# Import the full unified eval module (reuses all helpers)
def _load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

ue = _load_mod("ue", REPO / "run_unified_eval.py")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Constants ──────────────────────────────────────────────────────────────────

OUT_DIR  = ue.OUT_DIR
METHODS  = ue.METHODS
mkeys    = [m[0] for m in METHODS]
mnames   = {m[0]: m[2] for m in METHODS}
mbudgets = {m[0]: m[1] for m in METHODS}

SS_KEYS  = ["dann_ss", "cdan_ss", "cotmix_ss"]

# ── Parser ─────────────────────────────────────────────────────────────────────

def parse_per_scenario(path):
    """Read existing per_scenario.txt → {scenario_key: {method_name: (src,tgt,forget)}}."""
    data = {}
    text = Path(path).read_text()
    for block in text.split("\nScenario:")[1:]:
        lines = block.strip().splitlines()
        scen_key = lines[0].strip()
        data[scen_key] = {}
        for line in lines[2:]:   # skip header
            if len(line) < 25: continue
            name  = line[2:22].strip()
            nums  = line[22:].split()
            if len(nums) < 3: continue
            try:
                data[scen_key][name] = (float(nums[0]), float(nums[1]), float(nums[2]))
            except ValueError:
                pass
    return data


def write_per_scenario(path, data, dataset_tag):
    """Write back per_scenario dict to file in the standard format."""
    lines = [f"Per-scenario results — {dataset_tag}\n"]
    for sk, res in data.items():
        lines.append(f"\nScenario: {sk}")
        lines.append(f"  {'Method':<20} {'Src F1':>8}  {'Tgt F1':>8}  {'Forget':>8}")
        for key in mkeys:
            name = mnames[key]
            if name in res:
                s, t, f = res[name]
                lines.append(f"  {name:<20} {s:>8.4f}  {t:>8.4f}  {f:>+8.4f}")
    Path(path).write_text("\n".join(lines))


# ── Run semi-supervised methods for one scenario ───────────────────────────────

def run_ss_for_scenario(datasets, frozen_model, device):
    """Run DANN†, CDAN†, CoTMix† and return {key: (src_f1, tgt_f1, forget)}."""
    src_train_ld = ue.make_loader(datasets["src_train"], shuffle=True, drop_last=True)
    tgt_train_ld = ue.make_loader(datasets["tgt_train"], shuffle=True, drop_last=True)
    src_test_ld  = ue.make_loader(datasets["src_test"])
    tgt_test_ld  = ue.make_loader(datasets["tgt_test"])
    calib_ds     = ue.calib_split(datasets["tgt_train"])

    f1_src_before = ue.eval_f1_loader(frozen_model, src_test_ld, device)

    results = {}
    for key in SS_KEYS:
        budget = mbudgets[key]
        name   = mnames[key]
        print(f"  [{budget:12s}] {name:<18}", end=" … ", flush=True)
        try:
            if key == "dann_ss":
                r = ue.run_dann_ss(frozen_model, src_train_ld, tgt_train_ld, calib_ds,
                                   src_test_ld, tgt_test_ld, device)
            elif key == "cdan_ss":
                r = ue.run_cdan_ss(frozen_model, src_train_ld, tgt_train_ld, calib_ds,
                                   src_test_ld, tgt_test_ld, device)
            elif key == "cotmix_ss":
                r = ue.run_cotmix_ss(frozen_model, src_train_ld, tgt_train_ld, calib_ds,
                                     src_test_ld, tgt_test_ld, device)
            src_f1, tgt_f1 = r
            forget = f1_src_before - src_f1
            results[key] = (src_f1, tgt_f1, forget)
            print(f"src={src_f1:.4f}  tgt={tgt_f1:.4f}  forget={forget:+.4f}")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback; traceback.print_exc()
            results[key] = (0.0, 0.0, 0.0)

    return results


# ── Regenerate population summary + figure ─────────────────────────────────────

def _make_figure(data, dataset_tag):
    """Reproduce the annotated bar chart from all 15 method results."""
    pop_tgt = {k: [] for k in mkeys}
    pop_fgt = {k: [] for k in mkeys}

    for _, res in data.items():
        for key in mkeys:
            name = mnames[key]
            if name in res:
                _, t, f = res[name]
                pop_tgt[key].append(t)
                pop_fgt[key].append(f)

    GROUPS = [
        ("Lower bound",                     [0],         "#777777"),
        ("UDA Benchmark\n(label-free)",     [1, 2, 3],   "#2471a3"),
        ("Test-time\n(label-free)",         [4],          "#76448a"),
        ("Semi-supervised DA\n(5% labels)", [5, 6, 7],   "#17a589"),
        ("Fine-tuning\n(5% labels)",        [8, 9],       "#d35400"),
        ("Continual Learning\n(5% labels)", [10, 11, 12], "#922b21"),
        ("Ours\n(5% labels)",               [13, 14],     "#1e8449"),
    ]
    BAR_COLORS = [
        "#999999",
        "#3498db", "#5dade2", "#85c1e9",
        "#9b59b6",
        "#1abc9c", "#17a589", "#148f77",
        "#e67e22", "#f0a500",
        "#e74c3c", "#c0392b", "#f1948a",
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
    ax.set_ylim(0, 1.15); ax.set_xlim(-0.6, len(mkeys)-0.4)
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
        f"Unified Evaluation : {dataset_tag}  (5% target labels, learned gate)\n"
        f"TSLANet backbone · 10 AdaTime scenarios · mean ± std",
        fontsize=11, fontweight="bold")

    for ext in ["png", "pdf"]:
        out = OUT_DIR / f"{dataset_tag.lower()}_figure.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"  Saved → {out}")
    plt.close()


def save_population_summary(data, dataset_tag):
    """Write population summary for all 15 methods."""
    pop_src = {k: [] for k in mkeys}
    pop_tgt = {k: [] for k in mkeys}
    pop_fgt = {k: [] for k in mkeys}
    for _, res in data.items():
        for key in mkeys:
            name = mnames[key]
            if name in res:
                s, t, f = res[name]
                pop_src[key].append(s)
                pop_tgt[key].append(t)
                pop_fgt[key].append(f)

    hdr = f"  {'Method':<20} {'Budget':>12}  {'Src F1':>12}  {'Tgt F1':>12}  {'Forgetting':>14}"
    sep = "  " + "-"*74
    lines = [f"POPULATION SUMMARY — {dataset_tag}\n", hdr, sep]
    print(f"\n{'='*72}")
    print(f"POPULATION SUMMARY — {dataset_tag}")
    print(f"{'='*72}")
    print(hdr); print(sep)
    for k in mkeys:
        sm = np.mean(pop_src[k]) if pop_src[k] else 0
        ss = np.std(pop_src[k])  if pop_src[k] else 0
        tm = np.mean(pop_tgt[k]) if pop_tgt[k] else 0
        ts = np.std(pop_tgt[k])  if pop_tgt[k] else 0
        fm = np.mean(pop_fgt[k]) if pop_fgt[k] else 0
        fs = np.std(pop_fgt[k])  if pop_fgt[k] else 0
        row = (f"  {mnames[k]:<20} {mbudgets[k]:>12}  "
               f"{sm:.4f}±{ss:.4f}  {tm:.4f}±{ts:.4f}  {fm:+.4f}±{fs:.4f}")
        print(row); lines.append(row)
    (OUT_DIR / f"{dataset_tag.lower()}_population_summary.txt").write_text("\n".join(lines))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Adding semi-supervised DA methods: {[mnames[k] for k in SS_KEYS]}")
    print(f"FD + HHAR, {len(ue.FD_SCENARIOS) + len(ue.HHAR_SCENARIOS)} scenarios total\n")

    for dataset_tag, scenarios, load_fn, is_hhar, scen_file in [
        ("FD",   ue.FD_SCENARIOS,   ue.load_fd_datasets,   False, OUT_DIR / "fd_per_scenario.txt"),
        ("HHAR", ue.HHAR_SCENARIOS, ue.load_hhar_datasets, True,  OUT_DIR / "hhar_per_scenario.txt"),
    ]:
        print(f"\n{'#'*72}")
        print(f"## {dataset_tag}  ({len(scenarios)} scenarios)")
        print(f"{'#'*72}")

        # Load existing 12-method results
        existing = parse_per_scenario(scen_file)
        print(f"  Loaded {len(existing)} scenarios from {scen_file.name}")

        # Run 3 new methods for each scenario
        for src_id, tgt_id in scenarios:
            datasets    = load_fn(src_id, tgt_id)
            scen_key    = f"{datasets['src_name']}→{datasets['tgt_name']}"
            print(f"\n{'='*60}\n  Scenario: {scen_key}\n{'='*60}")

            frozen_model = ue.load_frozen_model(
                src_id if is_hhar else datasets["src_name"], is_hhar, device)

            new_results = run_ss_for_scenario(datasets, frozen_model, device)

            if scen_key not in existing:
                existing[scen_key] = {}
            for key, (s, t, f) in new_results.items():
                existing[scen_key][mnames[key]] = (s, t, f)

        # Write merged per_scenario file
        write_per_scenario(scen_file, existing, dataset_tag)
        print(f"\n  Updated {scen_file.name} with {len(SS_KEYS)} new methods")

        # Regenerate population summary + figure
        save_population_summary(existing, dataset_tag)
        _make_figure(existing, dataset_tag)
        print(f"  Figure regenerated for {dataset_tag}")

    print("\nDone. All 15-method results saved to", OUT_DIR)


if __name__ == "__main__":
    main()
