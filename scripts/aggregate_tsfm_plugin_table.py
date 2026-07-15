"""Aggregate TSFM plugin-gain table (Option B) across 5 datasets × 5 TSFMs.

Rows: (dataset, tsfm). Columns: frozen, head-ft, lwf, srcdistill, no-gate, pact
with AA/F and ΔAA/ΔF vs frozen.
"""

from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import RESULTS_ROOT

DATASETS = ["fd", "hhar", "sleep", "ucihar", "wisdm"]
TSFMS = ["moment", "chronos-bolt", "moirai", "sundial", "time-moe"]
METHODS = ["frozen", "head-ft", "lwf", "srcdistill", "no-gate", "pact"]


def _read_aa_f(path, kind):
    if not path.exists():
        return None, None
    if kind == "pact":
        m = pd.read_csv(path).iloc[-1]
        return float(m["AA"]), float(m["F_all"])
    df = pd.read_csv(path)
    if "control" in df.columns:
        row = df[df["control"] == kind].iloc[0]
    else:
        row = df.iloc[0]
    return float(row["AA"]), float(row["F"])


def load_controls(ds, tsfm):
    d = RESULTS_ROOT / "controls" / f"{ds}_{tsfm}_adatime_K16_c5" / "summary.csv"
    out = {}
    if d.exists():
        df = pd.read_csv(d)
        for _, r in df.iterrows():
            out[r["control"]] = (float(r["AA"]), float(r["F"]))
    return out


def load_head(ds, tsfm, method):
    d = RESULTS_ROOT / "cl_baselines_tsfm" / f"{ds}_{tsfm}_adatime_{method}_c5" / "summary.csv"
    return _read_aa_f(d, method)


def load_pact(ds, tsfm):
    root = RESULTS_ROOT / "sequential_tsfm"
    matches = sorted(root.glob(f"{ds}_{tsfm}_adatime_K16_c5*"))
    matches = [m for m in matches if "prototype" not in m.name]
    if not matches:
        return None, None
    return _read_aa_f(matches[0] / "metrics.csv", "pact")


def main():
    rows = []
    for ds in DATASETS:
        for tsfm in TSFMS:
            ctrl = load_controls(ds, tsfm)
            frozen = ctrl.get("no-adapt") or (None, None)
            nogate = ctrl.get("no-gate") or (None, None)
            head = {m: load_head(ds, tsfm, m) for m in ["head-ft", "lwf", "srcdistill"]}
            pact = load_pact(ds, tsfm)

            vals = {"frozen": frozen, "no-gate": nogate, "pact": pact, **head}
            rec = dict(dataset=ds, tsfm=tsfm)
            for m in METHODS:
                aa, f = vals.get(m, (None, None)) or (None, None)
                rec[f"{m}_AA"] = aa
                rec[f"{m}_F"] = f
            if frozen and frozen[0] is not None:
                for m in METHODS[1:]:
                    aa, f = vals.get(m, (None, None)) or (None, None)
                    if aa is not None:
                        rec[f"{m}_dAA"] = aa - frozen[0]
                    if f is not None and frozen[1] is not None:
                        rec[f"{m}_dF"] = f - frozen[1]
            rows.append(rec)

    df = pd.DataFrame(rows)
    out_csv = RESULTS_ROOT / "tables" / "tsfm_plugin_table.csv"
    out_tex = RESULTS_ROOT / "tables" / "tsfm_plugin_table.tex"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    # Compact LaTeX: one block per dataset
    lines = ["% Auto-generated TSFM plugin table (Option B)"]
    for ds in DATASETS:
        sub = df[df["dataset"] == ds]
        lines.append(f"\n% --- {ds.upper()} ---")
        lines.append("\\begin{tabular}{l" + "c" * 6 + "}")
        lines.append("\\toprule")
        lines.append("TSFM & Frozen & Head-FT & LwF & SrcDist & No-gate & PACT \\\\")
        lines.append("\\midrule")
        for _, r in sub.iterrows():
            cells = [r["tsfm"].replace("-", "\\-")]
            for m in METHODS:
                aa = r.get(f"{m}_AA")
                f = r.get(f"{m}_F")
                if pd.isna(aa):
                    cells.append("--")
                else:
                    cells.append(f"{aa:.3f}/{f:.3f}")
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")

    out_tex.write_text("\n".join(lines) + "\n")

    # Console summary
    print("=== TSFM PLUGIN TABLE (AA / F) ===")
    for ds in DATASETS:
        print(f"\n--- {ds} ---")
        sub = df[df["dataset"] == ds]
        hdr = f"{'tsfm':<14}" + "".join(f"{m:>12}" for m in METHODS)
        print(hdr)
        for _, r in sub.iterrows():
            parts = [f"{r['tsfm']:<14}"]
            for m in METHODS:
                aa, f = r.get(f"{m}_AA"), r.get(f"{m}_F")
                parts.append(f"{aa:>5.3f}/{f:<5.3f}" if pd.notna(aa) else f"{'--':>12}")
            print("".join(parts))
    n_ok = df["pact_AA"].notna().sum()
    print(f"\nCompleted rows: {n_ok}/{len(df)}  Saved -> {out_csv}")


if __name__ == "__main__":
    main()
