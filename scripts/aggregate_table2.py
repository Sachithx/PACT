"""Build Table 2 (chain continual comparison) from cl_baselines + cl_baselines_tsfm + PACT."""

import re
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import RESULTS_ROOT

DATASETS = ["fd", "hhar", "sleep"]
PACT_METHODS = {
    ("tslanet", "sequential"): "PACT (ours)",
    ("moment", "sequential_tsfm"): "PACT (ours)",
}


def load_summaries(root, pattern):
    rows = []
    if not root.exists():
        return rows
    for d in sorted(root.glob(pattern)):
        s = d / "summary.csv"
        if not s.exists():
            continue
        r = pd.read_csv(s).iloc[0]
        name = d.name
        ds = name.split("_")[0]
        method = r.get("method", r.get("control", name.split("_")[-1]))
        if "fullft" in name and method != "full-ft":
            method = "full-ft"
        rows.append(dict(dataset=ds, tag=name, method=method,
                         AA=r["AA"], BWT=r.get("BWT", float("nan")),
                         F=r.get("F", r.get("F_all", float("nan"))),
                         RF=r.get("RF", float("nan"))))
    return rows


def load_pact():
    rows = []
    for sub, bb in [("sequential", "tslanet"), ("sequential_tsfm", "moment")]:
        root = RESULTS_ROOT / sub
        for ds in DATASETS:
            tag = f"{ds}_{bb}_adatime_K16_c5"
            matches = list(root.glob(f"{tag}*"))
            if not matches:
                continue
            m = pd.read_csv(matches[0] / "metrics.csv").iloc[-1]
            rows.append(dict(dataset=ds, tag=matches[0].name, method="PACT (ours)",
                             AA=m["AA"], BWT=m["BWT"], F=m["F_all"], RF=m.get("RF", 0)))
    return rows


def main():
    rows = []
    rows += load_summaries(RESULTS_ROOT / "cl_baselines", "*_tslanet_adatime_*")
    rows += load_summaries(RESULTS_ROOT / "cl_baselines_tsfm", "*_moment_adatime_*")
    rows += load_pact()
    # controls: no-adapt / no-gate on moment
    for d in RESULTS_ROOT.glob("controls/*_moment_*"):
        s = d / "summary.csv"
        if not s.exists():
            continue
        ds = d.name.split("_")[0]
        for _, r in pd.read_csv(s).iterrows():
            rows.append(dict(dataset=ds, tag=d.name, method=r["control"],
                             AA=r["AA"], BWT=r["BWT"], F=r["F"], RF=r["RF"]))

    df = pd.DataFrame(rows).drop_duplicates(subset=["dataset", "method"], keep="last")
    df = df.sort_values(["dataset", "method"]).reset_index(drop=True)

    order = ["Frozen (no adapt)", "no-adapt", "head-ft", "lwf", "srcdistill",
             "full-ft", "ewc", "mas", "ORTCL (LF)", "ORTCL+HeadFT",
             "no-gate", "PACT (ours)"]
    df["_o"] = df["method"].map({m: i for i, m in enumerate(order)}).fillna(99)
    df = df.sort_values(["dataset", "_o"]).drop(columns="_o")

    out = RESULTS_ROOT / "tables" / "table2_chain.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print("=== TABLE 2 — Chain continual comparison ===")
    print(f"{'dataset':<7}{'method':<18}{'AA':>8}{'F':>8}{'RF':>8}")
    print("-" * 44)
    for _, r in df.iterrows():
        print(f"{r['dataset']:<7}{r['method']:<18}{r['AA']:>8.4f}{r['F']:>8.4f}{r['RF']:>8.4f}")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
