"""LaTeX table: rows=TSFMs, cols=datasets × {frozen, head-ft, PACT}."""

import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import RESULTS_ROOT

TSFMS = ["moment", "chronos-bolt", "moirai", "sundial", "time-moe"]
TSFM_LABEL = {
    "moment": "MOMENT",
    "chronos-bolt": "Chronos-Bolt",
    "moirai": "Moirai",
    "sundial": "Sundial",
    "time-moe": "Time-MoE",
}
DATASETS = [
    ("fd", "c5", "FD"),
    ("wisdm", "n450", "WISDM"),
    ("hhar", "c5", "HHAR"),
    ("sleep", "c5", "Sleep"),
]
METHODS = ["frozen", "head-ft", "pact"]


def _suffix(ds, tag):
    if tag == "c5":
        return "c5"
    return "n450"


def load_frozen(ds, tsfm, tag):
    suf = _suffix(ds, tag)
    p = RESULTS_ROOT / "controls" / f"{ds}_{tsfm}_adatime_K16_{suf}" / "summary.csv"
    if not p.exists():
        return None, None
    r = pd.read_csv(p)
    row = r[r["control"] == "no-adapt"].iloc[0]
    return float(row["AA"]), float(row["F"])


def load_head(ds, tsfm, tag):
    suf = _suffix(ds, tag)
    p = RESULTS_ROOT / "cl_baselines_tsfm" / f"{ds}_{tsfm}_adatime_head-ft_{suf}" / "summary.csv"
    if not p.exists():
        return None, None
    r = pd.read_csv(p).iloc[0]
    return float(r["AA"]), float(r["F"])


def load_pact(ds, tsfm, tag):
    suf = _suffix(ds, tag)
    root = RESULTS_ROOT / "sequential_tsfm"
    if tag == "n450":
        matches = sorted(root.glob(f"{ds}_{tsfm}_adatime_K16_{suf}_learned"))
    else:
        matches = sorted(root.glob(f"{ds}_{tsfm}_adatime_K16_{suf}*"))
        matches = [m for m in matches if "prototype" not in m.name and "n450" not in m.name]
    if not matches:
        return None, None
    m = pd.read_csv(matches[0] / "metrics.csv").iloc[-1]
    return float(m["AA"]), float(m["F_all"])


def pct_gain(pact, head, higher_better=True):
    if pact is None or head is None or head == 0:
        return None
    if higher_better:
        return 100.0 * (pact - head) / abs(head)
    # F: lower is better — report relative reduction
    if head == 0:
        return None
    return 100.0 * (head - pact) / abs(head)


def fmt_aa_f(aa, f, bold=False):
    if aa is None:
        return "--"
    s = f"{aa:.3f}/{f:.3f}"
    return f"\\textbf{{{s}}}" if bold else s


def fmt_pact(aa, f, daa_pct, df_pct, bold_aa, bold_f):
    if aa is None:
        return "--"
    if bold_aa and bold_f:
        core = f"\\textbf{{{aa:.3f}/{f:.3f}}}"
    elif bold_aa:
        core = f"\\textbf{{{aa:.3f}}}/{f:.3f}"
    elif bold_f:
        core = f"{aa:.3f}/\\textbf{{{f:.3f}}}"
    else:
        core = f"{aa:.3f}/{f:.3f}"
    tags = []
    if daa_pct is not None:
        sign = "+" if daa_pct >= 0 else ""
        tags.append(f"{sign}{daa_pct:.1f}\\%")
    if df_pct is not None:
        tags.append(f"-{df_pct:.0f}\\%F")
    if not tags:
        return core
    return f"\\makecell{{{core}\\\\{{\\scriptsize {', '.join(tags)}}}}}"


def main():
    rows = []
    for tsfm in TSFMS:
        rec = {"tsfm": tsfm}
        for ds, tag, _ in DATASETS:
            frozen = load_frozen(ds, tsfm, tag)
            head = load_head(ds, tsfm, tag)
            pact = load_pact(ds, tsfm, tag)
            rec[(ds, "frozen")] = frozen
            rec[(ds, "head-ft")] = head
            rec[(ds, "pact")] = pact
        rows.append(rec)

    lines = [
        "% Requires: \\usepackage{booktabs,multirow,makecell}",
        "% Protocol: FD/HHAR/Sleep @ 5\\% calib; WISDM @ n450 matched baselines.",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Average accuracy $AA$ / forgetting $F$ across foundation models and datasets. "
        "Bold = best among frozen, head fine-tune, and PACT per row--dataset block. "
        "Small percentages beside PACT: relative $AA$ gain and relative $F$ reduction vs.\\ head-FT.}",
        "\\label{tab:tsfm-dataset-grid}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l" + "ccc" * len(DATASETS) + "}",
        "\\toprule",
    ]

    hdr1 = ["\\multirow{2}{*}{\\textbf{FM}}"]
    hdr2 = [""]
    for _, _, label in DATASETS:
        hdr1 += [f"\\multicolumn{{3}}{{c}}{{\\textbf{{{label}}}}}", "", ""]
        hdr2 += ["Frozen", "Head-FT", "PACT"]
    # fix multicolumn spacing
    hdr1_clean = [hdr1[0]]
    for i, (_, _, label) in enumerate(DATASETS):
        hdr1_clean.append(f"\\multicolumn{{3}}{{c}}{{{label}}}")
    lines.append(" & ".join(hdr1_clean) + " \\\\")
    lines.append(" & ".join(hdr2) + " \\\\")
    lines.append("\\midrule")

    for rec in rows:
        cells = [TSFM_LABEL[rec["tsfm"]]]
        for ds, tag, _ in DATASETS:
            frozen = rec[(ds, "frozen")]
            head = rec[(ds, "head-ft")]
            pact = rec[(ds, "pact")]
            trio = {"frozen": frozen, "head-ft": head, "pact": pact}
            valid_aa = {k: v[0] for k, v in trio.items() if v and v[0] is not None}
            valid_f = {k: v[1] for k, v in trio.items() if v and v[1] is not None}
            best_aa = max(valid_aa, key=valid_aa.get) if valid_aa else None
            best_f = min(valid_f, key=valid_f.get) if valid_f else None

            for m in ["frozen", "head-ft"]:
                pair = trio[m]
                if not pair or pair[0] is None:
                    cells.append("--")
                    continue
                aa, f = pair
                aa_b, f_b = m == best_aa, m == best_f
                if aa_b and f_b:
                    cells.append(f"\\textbf{{{aa:.3f}/{f:.3f}}}")
                elif aa_b:
                    cells.append(f"\\textbf{{{aa:.3f}}}/{f:.3f}")
                elif f_b:
                    cells.append(f"{aa:.3f}/\\textbf{{{f:.3f}}}")
                else:
                    cells.append(f"{aa:.3f}/{f:.3f}")

            paa, pf = pact if pact and pact[0] is not None else (None, None)
            haa, hf = head if head and head[0] is not None else (None, None)
            daa = pct_gain(paa, haa, higher_better=True)
            df_ = pct_gain(pf, hf, higher_better=False) if pf is not None and hf is not None else None
            cells.append(fmt_pact(paa, pf, daa, df_,
                                  paa is not None and best_aa == "pact",
                                  pf is not None and best_f == "pact"))
        lines.append(" & ".join(cells) + " \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}}",
        "\\end{table*}",
    ]

    out = RESULTS_ROOT / "tables" / "tsfm_dataset_grid.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    tex = "\n".join(lines) + "\n"
    out.write_text(tex)
    print(tex)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
