"""Aggregate the full experimental campaign into paper-ready tables.

Produces, from results/{sequential,sequential_tsfm,controls}:
  1. MAIN     : PACT per (dataset, backbone) at the base config.
  2. GATE     : soft vs hard vs oracle routing (from gate_ablation.csv logic).
  3. CONTROLS : no-adapt / no-gate / full-FT vs PACT (isolates each component).
  4. K        : chart-count sweep.
  5. TEMP     : gate temperature sweep.
  6. BUDGET   : calibration label-budget sweep.
  7. ADAPTER  : identity vs LoRA adapters.
  8. ORDER    : domain-order robustness.

Everything is also written as CSVs under results/tables/.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import RESULTS_ROOT
from src.cl_metrics import compute_cl_metrics

DATASETS = ["fd", "hhar", "sleep"]
PAPER_BACKBONES = [
    "tslanet", "patchtst", "moment", "chronos-bolt", "sundial", "time-moe", "moirai"
]
TABLES = RESULTS_ROOT / "tables"


def parse_tag(name):
    """Return dict(dataset, backbone, order, K, c, t, adapter) from a run dir."""
    ds = next((x for x in DATASETS if name.startswith(x + "_")), None)
    if ds is None:
        return None
    rest = name[len(ds) + 1:].split("_")
    backbone = rest[0]
    toks = rest[1:]
    d = dict(dataset=ds, backbone=backbone, order=None, K=None, c=None,
             t=None, adapter=None)
    order_toks = []
    for tk in toks:
        if re.fullmatch(r"K\d+", tk):
            d["K"] = int(tk[1:])
        elif re.fullmatch(r"c\d+", tk):
            d["c"] = int(tk[1:])
        elif re.fullmatch(r"t[\d.]+", tk):
            d["t"] = float(tk[1:])
        elif tk in ("identity", "lora"):
            d["adapter"] = tk
        elif d["K"] is None:  # still in the order segment
            order_toks.append(tk)
    d["order"] = "_".join(order_toks) if order_toks else "default"
    return d


def metrics_from_matrix(path):
    A = pd.read_csv(path, index_col=0).values
    return compute_cl_metrics(A, A.shape[0] - 1, include_source=True)  # AA,BWT,F,RF,PRF


def collect_runs():
    rows = []
    for sub in ["sequential", "sequential_tsfm"]:
        root = RESULTS_ROOT / sub
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            info = parse_tag(d.name)
            soft = d / "soft_f1_matrix.csv"
            if info is None or not soft.exists():
                continue
            aa, bwt, f, rf, prf = metrics_from_matrix(soft)
            info.update(dict(AA=aa, BWT=bwt, F=f, RF=rf, PRF=prf, tag=d.name))
            rows.append(info)
    return pd.DataFrame(rows)


def show(title, df, cols):
    print(f"\n### {title}")
    if df.empty:
        print("(no runs)")
        return
    disp = df[cols].copy()
    fmt = {c: (lambda v: f"{v:.4f}" if isinstance(v, float) else str(v))
           for c in cols}
    print(disp.to_string(index=False, formatters=fmt))


def _isbase_t(t):
    return pd.isna(t) or t == 0.5


def _isbase_adapter(a):
    return pd.isna(a) or a == "identity"


def is_campaign_run(r):
    """Exclude exploratory gate variants from the paper campaign tables."""
    if r["backbone"] not in PAPER_BACKBONES:
        return False
    tag = r["tag"]
    if r["backbone"] in ("tslanet", "patchtst"):
        pattern = rf"{r['dataset']}_{r['backbone']}_[a-z0-9_]+_K\d+_c\d+_t[\d.]+_(identity|lora)"
    else:
        pattern = rf"{r['dataset']}_{r['backbone']}_[a-z0-9_]+_K\d+_c\d+"
    return re.fullmatch(pattern, tag) is not None


def is_base(r):
    return (r["order"] in ("adatime", "default") and r["K"] == 16 and r["c"] == 5
            and _isbase_t(r["t"]) and _isbase_adapter(r["adapter"]))


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    df = collect_runs()
    df = df[df.apply(is_campaign_run, axis=1)]

    # 1. MAIN (base config)
    main_df = df[df.apply(is_base, axis=1)].sort_values(["dataset", "backbone"])
    show("MAIN — PACT (base config, AdaTime chain)", main_df,
         ["dataset", "backbone", "AA", "BWT", "F", "RF", "PRF"])
    main_df.to_csv(TABLES / "main.csv", index=False)

    # 3. CONTROLS
    ctrl_rows = []
    croot = RESULTS_ROOT / "controls"
    if croot.exists():
        for d in sorted(croot.iterdir()):
            info = parse_tag(d.name.replace("_fullft", "_K0_c0"))
            if info is None or info["backbone"] not in PAPER_BACKBONES:
                continue
            matrices = {
                "no-adapt": d / "noadapt_f1_matrix.csv",
                "no-gate": d / "nogate_f1_matrix.csv",
                "full-ft": d / "fullft_f1_matrix.csv",
            }
            for control, matrix in matrices.items():
                if not matrix.exists():
                    continue
                aa, bwt, f, rf, prf = metrics_from_matrix(matrix)
                ctrl_rows.append(dict(dataset=info["dataset"], backbone=info["backbone"],
                                      method=control, AA=aa, BWT=bwt,
                                      F=f, RF=rf, PRF=prf))
    # add PACT base for reference
    for _, r in main_df.iterrows():
        ctrl_rows.append(dict(dataset=r["dataset"], backbone=r["backbone"],
                              method="PACT (ours)", AA=r["AA"], BWT=r["BWT"],
                              F=r["F"], RF=r["RF"], PRF=r["PRF"]))
    ctrl = pd.DataFrame(ctrl_rows)
    order_key = {"no-adapt": 0, "no-gate": 1, "full-ft": 2, "PACT (ours)": 3}
    if not ctrl.empty:
        ctrl["_o"] = ctrl["method"].map(order_key).fillna(9)
        ctrl = ctrl.sort_values(["dataset", "backbone", "_o"]).drop(columns="_o")
    show("CONTROLS — component ablation (vs PACT)", ctrl,
         ["dataset", "backbone", "method", "AA", "F", "RF"])
    ctrl.to_csv(TABLES / "controls.csv", index=False)

    # 4-8. hyper-parameter sweeps (FD)
    fd = df[df["dataset"] == "fd"]

    def sweep(col, fixed, backbone):
        sub = fd[fd["backbone"] == backbone].copy()
        for k, v in fixed.items():
            sub = sub[sub[k] == v]
        sub = sub[sub["order"].isin(["adatime", "default"])]
        return sub.sort_values(col)

    ksw = sweep("K", dict(c=5, adapter="identity"), "tslanet")
    ksw = ksw[ksw["t"].isin([0.5, None])]
    show("K sweep — TSLANet FD", ksw, ["K", "AA", "F", "RF"])
    ksw.to_csv(TABLES / "sweep_K_tslanet.csv", index=False)

    ksw_m = sweep("K", dict(c=5), "moment")
    show("K sweep — MOMENT FD", ksw_m, ["K", "AA", "F", "RF"])
    ksw_m.to_csv(TABLES / "sweep_K_moment.csv", index=False)

    tsw = sweep("t", dict(c=5, K=16, adapter="identity"), "tslanet")
    show("Temperature sweep — TSLANet FD", tsw, ["t", "AA", "F", "RF"])
    tsw.to_csv(TABLES / "sweep_temp_tslanet.csv", index=False)

    bsw = sweep("c", dict(K=16, adapter="identity"), "tslanet")
    bsw = bsw[bsw["t"].isin([0.5, None])]
    show("Label-budget sweep — TSLANet FD", bsw, ["c", "AA", "F", "RF"])
    bsw.to_csv(TABLES / "sweep_budget_tslanet.csv", index=False)

    bsw_m = sweep("c", dict(K=16), "moment")
    show("Label-budget sweep — MOMENT FD", bsw_m, ["c", "AA", "F", "RF"])
    bsw_m.to_csv(TABLES / "sweep_budget_moment.csv", index=False)

    adp = fd[(fd["backbone"] == "tslanet") & (fd["K"] == 16) & (fd["c"] == 5)
             & (fd["t"].isin([0.5, None])) & (fd["order"].isin(["adatime", "default"]))]
    show("Adapter kind — TSLANet FD", adp.sort_values("adapter"),
         ["adapter", "AA", "F", "RF"])
    adp.to_csv(TABLES / "sweep_adapter_tslanet.csv", index=False)

    ordf = fd[(fd["backbone"] == "tslanet") & (fd["K"] == 16) & (fd["c"] == 5)
              & (fd["t"].isin([0.5, None])) & (fd["adapter"].isin(["identity", None]))]
    show("Domain-order robustness — TSLANet FD", ordf.sort_values("order"),
         ["order", "AA", "F", "RF"])
    ordf.to_csv(TABLES / "sweep_order_tslanet.csv", index=False)

    print(f"\nAll tables saved under {TABLES}")


if __name__ == "__main__":
    main()
