"""
run_pact_k4.py  —  Re-run PACT (gate=True, K=4) on all 3 datasets, 10 scenarios each.

Patches the PACT row in existing result txt files and regenerates figures with
the updated PACT numbers (all other method rows are left untouched).

Run:
  CUDA_VISIBLE_DEVICES=0 \\
  LD_PRELOAD=/home/AD/sachith/.conda/envs/entrope/lib/libstdc++.so.6 \\
  /home/AD/sachith/.conda/envs/entrope/bin/python -u run_pact_k4.py
"""

import sys, re, copy
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit

REPO = Path(__file__).parent
sys.path.insert(0, str(REPO / "TSLANet" / "Classification"))

import importlib.util

def _load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Import the unified eval module and override K before any function calls
rue = _load_mod("run_unified_eval", REPO / "run_unified_eval.py")
rue.K = 4          # ← use k=4 clusters instead of k=16
K = 4

OUT_DIR  = REPO / "results" / "unified_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}  |  K={K}  |  Running PACT (gate=True) only")


# ── helpers (thin wrappers around rue's functions, using K=4) ─────────────────

def calib_split(ds):
    return rue.calib_split(ds, rue.CALIB_FRAC)

def make_loader(ds, **kw):
    return rue.make_loader(ds, **kw)

def run_pact_k4(src_model, src_train_ld, calib_ds, src_test_ld, tgt_test_ld, device_):
    """Run PACT with gate=True and K=4."""
    frozen_head = src_model.head
    src_embs, _ = rue.extract_embs(src_model, src_train_ld, device_)
    centroids, src_asgn = rue.kmeans_numpy(src_embs, K)

    calib_embs, calib_lbls = rue.extract_embs(src_model, make_loader(calib_ds), device_)
    adapters = rue._train_pact_adapters(calib_embs, calib_lbls, centroids, frozen_head, device_)

    calib_asgn = rue.assign_concepts(calib_embs, centroids)
    gates      = rue.train_learned_gates(src_embs, src_asgn, calib_embs, calib_asgn, device_)

    @torch.no_grad()
    def gate_fn(embs_np, asgn):
        result = np.zeros(len(embs_np), dtype=bool)
        for c in range(K):
            mask = (asgn == c)
            if not mask.any(): continue
            g = gates[c]
            if g is None: result[mask] = False; continue
            g.eval()
            p = g(torch.from_numpy(embs_np[mask]).to(device_)).argmax(1).cpu().numpy()
            result[mask] = (p == 1)
        return result

    def _f1(ld):
        embs, lbls = rue.extract_embs(src_model, ld, device_)
        return rue.eval_adapter_f1(embs, lbls, centroids, adapters, frozen_head,
                                   device_, gate_fn)

    return _f1(src_test_ld), _f1(tgt_test_ld)


def run_pact_on_dataset(dataset_tag, scenarios, load_fn, dataset_key):
    """Collect PACT results for all scenarios in one dataset sweep."""
    print(f"\n{'#'*60}")
    print(f"## {dataset_tag}  —  PACT K={K}  ({len(scenarios)} scenarios)")
    print(f"{'#'*60}")

    per_scenario = {}   # scen_key → (src_f1, tgt_f1, forget)
    src_bef_map  = {}

    for src_id, tgt_id in scenarios:
        datasets  = load_fn(src_id, tgt_id)
        scen_key  = f"{datasets['src_name']}→{datasets['tgt_name']}"
        print(f"\n  Scenario: {scen_key}")

        src_lookup = src_id if dataset_key in ("hhar", "sleep") else datasets["src_name"]
        frozen_model = rue.load_frozen_model(src_lookup, dataset_key, device)
        bs = 64 if dataset_key == "sleep" else None

        src_train_ld = make_loader(datasets["src_train"], shuffle=True, drop_last=True, batch_size=bs)
        src_test_ld  = make_loader(datasets["src_test"],  batch_size=bs)
        tgt_test_ld  = make_loader(datasets["tgt_test"],  batch_size=bs)
        calib_ds     = calib_split(datasets["tgt_train"])

        f1_src_bef = rue.eval_f1_loader(frozen_model, src_test_ld, device)
        src_bef_map[scen_key] = f1_src_bef

        print(f"  src_before={f1_src_bef:.4f}", end="  ")
        try:
            src_f1, tgt_f1 = run_pact_k4(frozen_model, src_train_ld, calib_ds,
                                          src_test_ld, tgt_test_ld, device)
            forget = f1_src_bef - src_f1
            per_scenario[scen_key] = (src_f1, tgt_f1, forget)
            print(f"PACT  src={src_f1:.4f}  tgt={tgt_f1:.4f}  forget={forget:+.4f}")
        except Exception as e:
            import traceback; traceback.print_exc()
            per_scenario[scen_key] = (0.0, 0.0, 0.0)

    return per_scenario


# ── file patching helpers ──────────────────────────────────────────────────────

def _patch_pact_pop(filepath, src_vals, tgt_vals, fgt_vals):
    """Replace the PACT population-summary row in-place (not PACT (no gate))."""
    if not filepath.exists():
        print(f"  [skip] {filepath.name} not found")
        return
    sm = np.mean(src_vals); ss = np.std(src_vals)
    tm = np.mean(tgt_vals); ts = np.std(tgt_vals)
    fm = np.mean(fgt_vals); fs = np.std(fgt_vals)
    new_row = (f"  {'PACT':<20} {'calibration':>12}  "
               f"{sm:.4f}±{ss:.4f}  {tm:.4f}±{ts:.4f}  {fm:+.4f}±{fs:.4f}")
    txt = filepath.read_text()
    # Match the PACT line but NOT "PACT (no gate)" — use word-boundary lookahead
    txt = re.sub(
        r"  PACT {16,}calibration.*",   # 16+ spaces → "PACT" then long pad
        new_row,
        txt
    )
    filepath.write_text(txt)
    print(f"  Updated population row in {filepath.name}")


def _patch_pact_per_scenario(filepath, per_scenario):
    """Replace per-scenario PACT rows in-place (leaves PACT (no gate) untouched)."""
    if not filepath.exists():
        print(f"  [skip] {filepath.name} not found")
        return
    txt = filepath.read_text()
    lines = txt.splitlines()
    current_scen = None
    out = []
    for line in lines:
        m = re.match(r"Scenario: (.+)", line.strip())
        if m:
            current_scen = m.group(1)
        if (current_scen in per_scenario
                and re.match(r"  PACT\s{14,}", line)
                and "no gate" not in line):
            s, t, f = per_scenario[current_scen]
            line = f"  {'PACT':<20} {s:>8.4f}  {t:>8.4f}  {f:>+8.4f}"
        out.append(line)
    filepath.write_text("\n".join(out))
    print(f"  Updated per-scenario rows in {filepath.name}")


def _patch_combined(filepath, fd_vals, hh_vals, sl_vals):
    """Replace the PACT row in combined_summary.txt."""
    if not filepath.exists():
        print(f"  [skip] {filepath.name} not found")
        return
    fd_t, fd_f = np.mean(fd_vals[1]), np.mean(fd_vals[2])
    hh_t, hh_f = np.mean(hh_vals[1]), np.mean(hh_vals[2])
    sl_t, sl_f = np.mean(sl_vals[1]), np.mean(sl_vals[2])
    new_row = (f"  {'PACT':<20} {fd_t:>9.4f} {fd_f:>+9.4f}  "
               f"{hh_t:>9.4f} {hh_f:>+9.4f}  {sl_t:>9.4f} {sl_f:>+9.4f}")
    txt = filepath.read_text()
    txt = re.sub(r"  PACT {16,}.*", new_row, txt)
    filepath.write_text(txt)
    print(f"  Updated PACT row in {filepath.name}")


def _write_sleep_files(per_scenario):
    """Write sleep-edf_population_summary.txt and sleep-edf_per_scenario.txt."""
    tgt_vals = [v[1] for v in per_scenario.values()]
    src_vals = [v[0] for v in per_scenario.values()]
    fgt_vals = [v[2] for v in per_scenario.values()]

    sm, ss = np.mean(src_vals), np.std(src_vals)
    tm, ts = np.mean(tgt_vals), np.std(tgt_vals)
    fm, fs = np.mean(fgt_vals), np.std(fgt_vals)

    pop_line = (f"  {'PACT':<20} {'calibration':>12}  "
                f"{sm:.4f}±{ss:.4f}  {tm:.4f}±{ts:.4f}  {fm:+.4f}±{fs:.4f}")
    hdr = f"  {'Method':<20} {'Budget':>12}  {'Src F1':>12}  {'Tgt F1':>12}  {'Forgetting':>14}"
    sep = "  " + "-"*74

    pop_path = OUT_DIR / "sleep-edf_population_summary.txt"
    if pop_path.exists():
        _patch_pact_pop(pop_path,
                        [v[0] for v in per_scenario.values()],
                        [v[1] for v in per_scenario.values()],
                        [v[2] for v in per_scenario.values()])
    else:
        lines = [f"POPULATION SUMMARY — Sleep-EDF (PACT K={K} only)\n", hdr, sep, pop_line]
        pop_path.write_text("\n".join(lines))
        print(f"  Created {pop_path.name}")

    per_path = OUT_DIR / "sleep-edf_per_scenario.txt"
    if per_path.exists():
        _patch_pact_per_scenario(per_path, per_scenario)
    else:
        lines = [f"Per-scenario results — Sleep-EDF (PACT K={K} only)\n"]
        for sk, (s, t, f) in per_scenario.items():
            lines.append(f"\nScenario: {sk}")
            lines.append(f"  {'Method':<20} {'Src F1':>8}  {'Tgt F1':>8}  {'Forget':>8}")
            lines.append(f"  {'PACT':<20} {s:>8.4f}  {t:>8.4f}  {f:>+8.4f}")
        per_path.write_text("\n".join(lines))
        print(f"  Created {per_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    torch.manual_seed(rue.SEED); np.random.seed(rue.SEED)

    print(f"\nPACT-only re-run  |  K={K}  |  3 datasets × 10 scenarios each")

    # ── FD ─────────────────────────────────────────────────────────────────────
    fd_res = run_pact_on_dataset("FD", rue.FD_SCENARIOS,
                                 rue.load_fd_datasets, "fd")
    fd_src = [v[0] for v in fd_res.values()]
    fd_tgt = [v[1] for v in fd_res.values()]
    fd_fgt = [v[2] for v in fd_res.values()]

    print(f"\n  FD PACT K={K}: tgt={np.mean(fd_tgt):.4f}±{np.std(fd_tgt):.4f}  "
          f"forget={np.mean(fd_fgt):+.4f}±{np.std(fd_fgt):.4f}")

    _patch_pact_pop(OUT_DIR / "fd_population_summary.txt", fd_src, fd_tgt, fd_fgt)
    _patch_pact_per_scenario(OUT_DIR / "fd_per_scenario.txt", fd_res)

    # ── HHAR ───────────────────────────────────────────────────────────────────
    hh_res = run_pact_on_dataset("HHAR", rue.HHAR_SCENARIOS,
                                 rue.load_hhar_datasets, "hhar")
    hh_src = [v[0] for v in hh_res.values()]
    hh_tgt = [v[1] for v in hh_res.values()]
    hh_fgt = [v[2] for v in hh_res.values()]

    print(f"\n  HHAR PACT K={K}: tgt={np.mean(hh_tgt):.4f}±{np.std(hh_tgt):.4f}  "
          f"forget={np.mean(hh_fgt):+.4f}±{np.std(hh_fgt):.4f}")

    _patch_pact_pop(OUT_DIR / "hhar_population_summary.txt", hh_src, hh_tgt, hh_fgt)
    _patch_pact_per_scenario(OUT_DIR / "hhar_per_scenario.txt", hh_res)

    # ── Sleep-EDF ──────────────────────────────────────────────────────────────
    sl_res = run_pact_on_dataset("Sleep-EDF", rue.SLEEP_SCENARIOS,
                                 rue.load_sleep_datasets, "sleep")
    sl_src = [v[0] for v in sl_res.values()]
    sl_tgt = [v[1] for v in sl_res.values()]
    sl_fgt = [v[2] for v in sl_res.values()]

    print(f"\n  Sleep PACT K={K}: tgt={np.mean(sl_tgt):.4f}±{np.std(sl_tgt):.4f}  "
          f"forget={np.mean(sl_fgt):+.4f}±{np.std(sl_fgt):.4f}")

    _write_sleep_files(sl_res)

    # ── Combined summary patch ──────────────────────────────────────────────────
    _patch_combined(OUT_DIR / "combined_summary.txt",
                    (fd_src, fd_tgt, fd_fgt),
                    (hh_src, hh_tgt, hh_fgt),
                    (sl_src, sl_tgt, sl_fgt))

    # ── Cross-dataset printout ──────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"PACT K={K}  Cross-dataset results")
    print(f"{'='*72}")
    print(f"  {'Dataset':<12}  {'Tgt F1 mean':>12}  {'Tgt F1 std':>11}  "
          f"{'Forgetting':>12}  {'Forget std':>11}")
    print("  " + "-"*62)
    for tag, tgt, fgt in [("FD",        fd_tgt, fd_fgt),
                           ("HHAR",      hh_tgt, hh_fgt),
                           ("Sleep-EDF", sl_tgt, sl_fgt)]:
        print(f"  {tag:<12}  {np.mean(tgt):>12.4f}  {np.std(tgt):>11.4f}  "
              f"{np.mean(fgt):>+12.4f}  {np.std(fgt):>11.4f}")

    print(f"\nOutputs updated in {OUT_DIR}")


if __name__ == "__main__":
    main()
