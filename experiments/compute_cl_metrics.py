"""
compute_cl_metrics.py — Compute GEM-style CL metrics from saved per_scenario files.

Metrics (Lopez-Paz & Ranzato, NeurIPS 2017):
  AA  = (src_f1 + tgt_f1) / 2                          ↑ higher better
  BWT = src_f1_after − src_f1_before                    ↑ 0 ideal; negative = forgetting
  F   = max(0, src_f1_before − src_f1_after)            ↓ 0 ideal; positive = forgetting
  FWT = tgt_f1_nocorr − b_target                        ↑ positive = source repr helps

For N=2 tasks (one source → one target):
  src_f1_before  = src_f1 of NO_ADAPT  (model never changed on source)
  tgt_f1_nocorr  = tgt_f1 of NO_ADAPT  (target F1 with source model, no adaptation)
  b_target       = from results/from_scratch/b_target.txt  (if available, else FWT skipped)

Sources:
  results/unified_eval/{fd,hhar}_per_scenario.txt   — FD + HHAR, up to 15 methods
  results/sleep_unified_eval/sleep_per_scenario.txt — Sleep-EDF, 15 methods
  results/ortcl_fd/per_scenario_table.txt           — ORTCL variants (FD only)
"""

import numpy as np
from pathlib import Path

REPO = Path(__file__).parent

# ── Parser ─────────────────────────────────────────────────────────────────────

def parse_per_scenario(path, name_col_end=22):
    """Returns {scenario_key: {method_name: (src, tgt, forget)}}."""
    data = {}
    text = Path(path).read_text()
    for block in text.split("\nScenario:")[1:]:
        lines = block.strip().splitlines()
        scen_key = lines[0].strip()
        data[scen_key] = {}
        for line in lines[2:]:
            if len(line) < name_col_end + 5: continue
            name = line[2:name_col_end].strip()
            nums = line[name_col_end:].split()
            if len(nums) < 3: continue
            try:
                data[scen_key][name] = (float(nums[0]), float(nums[1]), float(nums[2]))
            except ValueError:
                pass
    return data


def load_b_target(path):
    """Load b_target dict: {scenario_key: float}."""
    if not Path(path).exists():
        return {}
    b = {}
    for line in Path(path).read_text().splitlines():
        if ":" not in line: continue
        k, v = line.split(":", 1)
        try: b[k.strip()] = float(v.strip())
        except ValueError: pass
    return b


# ── Metric computation ─────────────────────────────────────────────────────────

def compute_metrics(scenario_data, b_target_dict=None):
    """
    scenario_data: {scen_key: {method: (src, tgt, forget)}}
    Returns: {method: {"AA": [...], "BWT": [...], "F": [...], "FWT": [...] or None}}
    """
    # Collect all method names
    all_methods = set()
    for res in scenario_data.values():
        all_methods.update(res.keys())

    pop = {m: {"AA": [], "BWT": [], "F": [], "FWT": []} for m in all_methods}

    for scen_key, res in scenario_data.items():
        # Baseline: NO_ADAPT or "No correction"
        baseline_name = None
        for cand in ("NO_ADAPT", "No correction"):
            if cand in res:
                baseline_name = cand
                break
        if baseline_name is None:
            continue

        src_before, tgt_nocorr, _ = res[baseline_name]
        b_tgt = (b_target_dict or {}).get(scen_key, None)

        for method, (src_f1, tgt_f1, forget) in res.items():
            aa  = (src_f1 + tgt_f1) / 2
            bwt = src_f1 - src_before
            f   = max(0.0, src_before - src_f1)
            pop[method]["AA"].append(aa)
            pop[method]["BWT"].append(bwt)
            pop[method]["F"].append(f)
            if b_tgt is not None:
                fwt = tgt_nocorr - b_tgt
                pop[method]["FWT"].append(fwt)

    return pop


def summarise(pop):
    """Returns {method: {metric: (mean, std)}}."""
    out = {}
    for method, metrics in pop.items():
        out[method] = {}
        for m, vals in metrics.items():
            if vals:
                out[method][m] = (np.mean(vals), np.std(vals))
            else:
                out[method][m] = (None, None)
    return out


# ── Ordered method lists ───────────────────────────────────────────────────────

UNIFIED_ORDER = [
    "NO_ADAPT",
    "DANN", "Deep_Coral", "MMDA", "TENT",
    "DANN†", "CDAN†", "CoTMix†",
    "Full FT", "Head FT",
    "EWC", "Src-Distill", "LwF", "MAS",
    "PACT (no gate)", "PACT",
]

ORTCL_ORDER = [
    "No correction",
    "ORTCL (LF)", "ORTCL+HeadFT", "ORTCL+FullFT",
]


# ── Formatting ─────────────────────────────────────────────────────────────────

def fmt(mean, std, digits=3, sign=False):
    if mean is None: return " " * (7 if not sign else 8)
    s = f"{mean:+.{digits}f}" if sign else f"{mean:.{digits}f}"
    return s

def print_table(summary, method_order, title, has_fwt=False, file=None):
    import sys
    f = file or sys.stdout
    hdr = f"  {'Method':<22}  {'AA':>7}  {'BWT':>8}  {'F':>7}"
    if has_fwt: hdr += f"  {'FWT':>7}"
    sep = "  " + "-" * (len(hdr) - 2)
    print(f"\n{title}", file=f)
    print(hdr, file=f); print(sep, file=f)
    for m in method_order:
        if m not in summary: continue
        ms = summary[m]
        aa_m,  aa_s  = ms.get("AA",  (None, None))
        bwt_m, bwt_s = ms.get("BWT", (None, None))
        f_m,   f_s   = ms.get("F",   (None, None))
        fwt_m, fwt_s = ms.get("FWT", (None, None))
        row = (f"  {m:<22}  {fmt(aa_m, aa_s):>7}  "
               f"{fmt(bwt_m, bwt_s, sign=True):>8}  {fmt(f_m, f_s):>7}")
        if has_fwt:
            row += f"  {fmt(fwt_m, fwt_s, sign=True):>7}"
        print(row, file=f)


# ── Main ───────────────────────────────────────────────────────────────────────

B_TARGET_FILE = REPO / "results" / "from_scratch" / "b_target.txt"

def main():
    out_lines = []

    _ortcl = REPO / "results" / "ortcl_tslanet"
    datasets = [
        ("FD",              REPO / "results" / "unified_eval"       / "fd_per_scenario.txt",    UNIFIED_ORDER, 22),
        ("HHAR",            REPO / "results" / "unified_eval"       / "hhar_per_scenario.txt",  UNIFIED_ORDER, 22),
        ("Sleep-EDF",       REPO / "results" / "sleep_unified_eval" / "sleep_per_scenario.txt", UNIFIED_ORDER, 22),
        ("FD (ORTCL)",      _ortcl / "fd_per_scenario.txt",                                     ORTCL_ORDER,   27),
        ("HHAR (ORTCL)",    _ortcl / "hhar_per_scenario.txt",                                   ORTCL_ORDER,   27),
        ("Sleep (ORTCL)",   _ortcl / "sleepedf_per_scenario.txt",                               ORTCL_ORDER,   27),
    ]

    b_target = load_b_target(B_TARGET_FILE)
    has_fwt  = bool(b_target)

    import sys
    for tag, path, order, nce in datasets:
        if not path.exists():
            print(f"  [SKIP] {path} not found", file=sys.stderr)
            continue
        data    = parse_per_scenario(path, name_col_end=nce)
        pop     = compute_metrics(data, b_target if has_fwt else None)
        summary = summarise(pop)
        n_scen  = len(data)
        title = (f"{'='*72}\n"
                 f"CL METRICS — {tag}  ({n_scen} AdaTime scenarios)\n"
                 f"  AA↑  BWT↑(0=perfect)  F↓(0=perfect)"
                 + ("  FWT↑" if has_fwt else "  [FWT: run run_from_scratch.py for b_target]"))
        print_table(summary, order, title, has_fwt=has_fwt)
        out_lines.append(f"\n# {tag}")
        import io
        buf = io.StringIO()
        print_table(summary, order, title, has_fwt=has_fwt, file=buf)
        out_lines.append(buf.getvalue())

    # Save
    out_path = REPO / "results" / "cl_metrics_table.txt"
    out_path.write_text("\n".join(out_lines))
    print(f"\n  Saved → {out_path}")

    if not has_fwt:
        print(f"\n  NOTE: FWT not computed. Run run_from_scratch.py first to generate b_target.")
        print(f"        Expected at: {B_TARGET_FILE}")


if __name__ == "__main__":
    main()
