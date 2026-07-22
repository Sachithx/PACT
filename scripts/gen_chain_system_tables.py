"""Aggregate the Sundial-backbone, established-random-chain protocol used
throughout the paper (5 datasets x 4 fixed random chains x best-K x LoRA r=8)
into mean +/- std CSVs and ready-to-paste LaTeX table fragments for the
supplement.

Covers: (1) main baseline comparison, (2) PACT across backbones, (3) the
three ablations (component / LoRA-rank / gate), (4) label-efficiency
(Mantis, K=4).

Outputs CSVs to results/tables/chain_system_*.csv and a LaTeX fragment to
paper/aaai27/tables/chain_system_tables.tex.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path("results")
TABLES_OUT = RESULTS / "tables"
TEX_OUT = Path("paper/aaai27/tables/chain_system_tables.tex")

with open("/tmp/random_chains_fd.json") as f:
    FD_CHAINS = json.load(f)["fd"]
with open("/tmp/random_chains.json") as f:
    OTHER_CHAINS = json.load(f)
with open("/tmp/random_chains_ucihar.json") as f:
    UCIHAR_CHAINS = json.load(f)["ucihar"]
CHAINS = {"fd": FD_CHAINS, "ucihar": UCIHAR_CHAINS, **OTHER_CHAINS}

BEST_K = {"fd": 8, "hhar": 4, "sleep": 1, "wisdm": 4, "ucihar": 4}
CALIB_TAG = {"fd": "c5", "hhar": "c5", "sleep": "c5", "wisdm": "n450", "ucihar": "n100"}
DS_ORDER = ["fd", "hhar", "sleep", "wisdm", "ucihar"]
DS_LABEL = {"fd": "FD", "hhar": "HHAR", "sleep": "Sleep-EDF", "wisdm": "WISDM", "ucihar": "UCI-HAR"}


def cstr(chain):
    return ",".join(map(str, chain))


def mean_std(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def read_summary_metric(path, metric):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if len(df) == 0:
        return None
    return float(df.iloc[-1][metric])


def read_pact_metrics_last(path, metric):
    if not path.exists():
        return None
    df = pd.read_csv(path, header=None,
                     names=["step", "domain", "AA", "BWT", "F_all", "F_tgt", "RF", "PRF"])
    return float(df.iloc[-1][metric])


def read_controls_metric(path, control_name, metric):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    row = df[df["control"] == control_name]
    if row.empty:
        return None
    return float(row.iloc[0][metric])


# ── (1) Main baseline comparison ────────────────────────────────────────────
# category dir, glob suffix (without leading "{ds}_sundial_{chain}_"), file, metric-source
BASELINE_SPECS = [
    ("no-adapt", "controls", None, "controls_noadapt"),
    ("head-ft", "cl_baselines_tsfm", "head-ft", "summary"),
    ("ewc", "cl_baselines_tsfm", "ewc", "summary"),
    ("mas", "cl_baselines_tsfm", "mas", "summary"),
    ("lwf", "cl_baselines_tsfm", "lwf", "summary"),
    ("srcdistill", "cl_baselines_tsfm", "srcdistill", "summary"),
    ("dualcd", "cl_baselines_tsfm", "dualcd", "summary"),
    ("tent", "labelfree_baselines", "tent", "summary"),
    ("mapu-free", "labelfree_baselines", "mapu-free", "summary"),
    ("accup", "labelfree_baselines", "accup", "summary"),
    ("mapu", "mapu_baselines", "mapu", "summary"),
    ("cotta", "chain_baselines", "cotta", "summary"),
    ("eata", "chain_baselines", "eata", "summary"),
    ("sprompts", "chain_baselines", "sprompts", "summary"),
    ("ncm", "ncm_baselines", "ncm", "summary"),
    ("emb-replay-head", "embedding_replay", "emb-replay-head", "metrics_last"),
    ("derpp", "derpp_baselines", "derpp", "summary_derpp"),
    ("no-gate", "controls", None, "controls_nogate"),
    ("pact", "sequential_tsfm", None, "pact"),
]


def find_derpp_dir(ds, chain, tag):
    base = RESULTS / "derpp_baselines"
    matches = sorted(base.glob(f"{ds}_sundial_{cstr(chain)}_derpp_{tag}_b*_s42"))
    return matches[0] if matches else None


def collect_main_comparison():
    rows = []
    for method_label, cat_dir, suffix, kind in BASELINE_SPECS:
        for ds in DS_ORDER:
            tag = CALIB_TAG[ds]
            k = BEST_K[ds]
            aas, prfs = [], []
            for chain in CHAINS[ds]:
                cs = cstr(chain)
                if kind == "controls_noadapt":
                    p = RESULTS / "controls" / f"{ds}_sundial_{cs}_K{k}_{tag}" / "summary.csv"
                    aas.append(read_controls_metric(p, "no-adapt", "AA"))
                    prfs.append(read_controls_metric(p, "no-adapt", "PRF"))
                elif kind == "controls_nogate":
                    p = RESULTS / "controls" / f"{ds}_sundial_{cs}_K{k}_{tag}" / "summary.csv"
                    aas.append(read_controls_metric(p, "no-gate", "AA"))
                    prfs.append(read_controls_metric(p, "no-gate", "PRF"))
                elif kind == "pact":
                    p = (RESULTS / "sequential_tsfm" /
                        f"{ds}_sundial_{cs}_K{k}_{tag}_lora_learned_s42" / "metrics.csv")
                    aas.append(read_pact_metrics_last(p, "AA"))
                    prfs.append(read_pact_metrics_last(p, "PRF"))
                elif kind == "summary_derpp":
                    p = find_derpp_dir(ds, chain, tag)
                    if p is not None:
                        aas.append(read_summary_metric(p / "summary.csv", "AA"))
                        prfs.append(read_summary_metric(p / "summary.csv", "PRF"))
                    else:
                        aas.append(None); prfs.append(None)
                elif kind == "metrics_last":
                    p = (RESULTS / cat_dir / f"{ds}_sundial_{cs}_{suffix}_{tag}_s42" / "metrics.csv")
                    aas.append(read_pact_metrics_last(p, "AA"))
                    prfs.append(read_pact_metrics_last(p, "PRF"))
                else:
                    p = RESULTS / cat_dir / f"{ds}_sundial_{cs}_{suffix}_{tag}_s42" / "summary.csv"
                    aas.append(read_summary_metric(p, "AA"))
                    prfs.append(read_summary_metric(p, "PRF"))
            aa_m, aa_s = mean_std(aas)
            prf_m, prf_s = mean_std(prfs)
            rows.append(dict(method=method_label, dataset=ds, AA_mean=aa_m, AA_std=aa_s,
                            PRF_mean=prf_m, PRF_std=prf_s, n=sum(v is not None for v in aas)))
    return pd.DataFrame(rows)


# ── (2) Backbones ────────────────────────────────────────────────────────────
BACKBONE_K = {
    "tslanet": (16, "t0.5_lora_learned_s42", "sequential"),
    "patchtst": (16, "t0.5_lora_learned_s42", "sequential"),
    "moment": (None, "lora_learned_s42", "sequential_tsfm"),  # per-dataset best-K only
    "sundial": (16, "lora_learned_s42", "sequential_tsfm"),
    "mantis": (16, "lora_learned_s42", "sequential_tsfm"),
    "moirai": (16, "lora_learned_s42", "sequential_tsfm"),
    "unishape": (16, "lora_learned_s42", "sequential_tsfm"),
    "chronos-bolt": (16, "lora_learned_s42", "sequential_tsfm"),
}
BACKBONE_LABEL = {
    "tslanet": "TSLANet", "patchtst": "PatchTST", "moment": "MOMENT (frozen)",
    "sundial": "Sundial (frozen)", "mantis": "Mantis (frozen)",
    "moirai": "Moirai (frozen)", "unishape": "UniShape (frozen)",
    "chronos-bolt": "Chronos-Bolt (frozen)",
}


def collect_backbones():
    rows = []
    for bb, (k, suffix, root) in BACKBONE_K.items():
        for ds in DS_ORDER:
            tag = CALIB_TAG[ds]
            k_use = k if k is not None else BEST_K[ds]
            aas, prfs = [], []
            for chain in CHAINS[ds]:
                cs = cstr(chain)
                p = RESULTS / root / f"{ds}_{bb}_{cs}_K{k_use}_{tag}_{suffix}" / "metrics.csv"
                aas.append(read_pact_metrics_last(p, "AA"))
                prfs.append(read_pact_metrics_last(p, "PRF"))
            aa_m, aa_s = mean_std(aas)
            prf_m, prf_s = mean_std(prfs)
            n = sum(v is not None for v in aas)
            rows.append(dict(backbone=bb, dataset=ds, K=k_use, AA_mean=aa_m, AA_std=aa_s,
                            PRF_mean=prf_m, PRF_std=prf_s, n=n))
    return pd.DataFrame(rows)


# ── (3) Ablations: component, lora-vs-fullrank, gate ────────────────────────
def collect_component_ablation():
    rows = []
    for ds in DS_ORDER:
        tag = CALIB_TAG[ds]
        k = BEST_K[ds]
        for comp, kind in [("pact", "pact"), ("no_gate", "controls_nogate"),
                          ("no_adapters", "controls_noadapt"), ("head_ft", "summary_headft")]:
            aas, prfs = [], []
            for chain in CHAINS[ds]:
                cs = cstr(chain)
                if kind == "pact":
                    p = (RESULTS / "sequential_tsfm" /
                        f"{ds}_sundial_{cs}_K{k}_{tag}_lora_learned_s42" / "metrics.csv")
                    aas.append(read_pact_metrics_last(p, "AA"))
                    prfs.append(read_pact_metrics_last(p, "PRF"))
                elif kind == "controls_nogate":
                    p = RESULTS / "controls" / f"{ds}_sundial_{cs}_K{k}_{tag}" / "summary.csv"
                    aas.append(read_controls_metric(p, "no-gate", "AA"))
                    prfs.append(read_controls_metric(p, "no-gate", "PRF"))
                elif kind == "controls_noadapt":
                    p = RESULTS / "controls" / f"{ds}_sundial_{cs}_K{k}_{tag}" / "summary.csv"
                    aas.append(read_controls_metric(p, "no-adapt", "AA"))
                    prfs.append(read_controls_metric(p, "no-adapt", "PRF"))
                else:
                    p = RESULTS / "cl_baselines_tsfm" / f"{ds}_sundial_{cs}_head-ft_{tag}_s42" / "summary.csv"
                    aas.append(read_summary_metric(p, "AA"))
                    prfs.append(read_summary_metric(p, "PRF"))
            aa_m, aa_s = mean_std(aas)
            prf_m, prf_s = mean_std(prfs)
            rows.append(dict(dataset=ds, component=comp, AA_mean=aa_m, AA_std=aa_s,
                            PRF_mean=prf_m, PRF_std=prf_s, n=len(CHAINS[ds])))
    return pd.DataFrame(rows)


def collect_lora_vs_fullrank():
    rows = []
    for ds in DS_ORDER:
        tag = CALIB_TAG[ds]
        k = BEST_K[ds]
        for rank_label, rank_suffix in [("lora8", ""), ("full768", "_r768")]:
            aas, prfs = [], []
            for chain in CHAINS[ds]:
                cs = cstr(chain)
                p = (RESULTS / "sequential_tsfm" /
                    f"{ds}_sundial_{cs}_K{k}_{tag}_lora_learned{rank_suffix}_s42" / "metrics.csv")
                aas.append(read_pact_metrics_last(p, "AA"))
                prfs.append(read_pact_metrics_last(p, "PRF"))
            aa_m, aa_s = mean_std(aas)
            prf_m, prf_s = mean_std(prfs)
            rows.append(dict(dataset=ds, rank=rank_label, AA_mean=aa_m, AA_std=aa_s,
                            PRF_mean=prf_m, PRF_std=prf_s, n=len(CHAINS[ds])))
    return pd.DataFrame(rows)


def collect_gate_ablation():
    rows = []
    for ds in DS_ORDER:
        tag = CALIB_TAG[ds]
        k = BEST_K[ds]
        for gate in ["soft", "hard", "oracle"]:
            aas = []
            for chain in CHAINS[ds]:
                cs = cstr(chain)
                p = (RESULTS / "sequential_tsfm" /
                    f"{ds}_sundial_{cs}_K{k}_{tag}_lora_learned_s42")
                fmatrix = {"soft": "soft_f1_matrix.csv", "hard": "hard_f1_matrix.csv",
                          "oracle": "oracle_f1_matrix.csv"}[gate]
                fp = p / fmatrix
                if fp.exists():
                    mat = pd.read_csv(fp, index_col=0).to_numpy(dtype=float)
                    t = mat.shape[0] - 1
                    aas.append(float(np.nanmean(mat[t, :t + 1])))
                else:
                    aas.append(None)
            aa_m, aa_s = mean_std(aas)
            rows.append(dict(dataset=ds, gate=gate, AA_mean=aa_m, AA_std=aa_s,
                            n=sum(v is not None for v in aas)))
    return pd.DataFrame(rows)


# ── (4) Label efficiency (Mantis, K=4) ───────────────────────────────────────
PCT_BUDGETS = [1, 5, 10, 20, 50, 100]
CNT_BUDGETS_WISDM = [20, 50, 100, 200, 450]
CNT_BUDGETS_UCIHAR = [20, 50, 100, 200, 400]


def collect_label_efficiency():
    rows = []
    for ds in ["fd", "hhar", "sleep"]:
        for pct in PCT_BUDGETS:
            aas = []
            for chain in CHAINS[ds]:
                cs = cstr(chain)
                p = (RESULTS / "sequential_tsfm" /
                    f"{ds}_mantis_{cs}_K4_c{pct}_lora_learned_s42" / "metrics.csv")
                aas.append(read_pact_metrics_last(p, "AA"))
            aa_m, aa_s = mean_std(aas)
            rows.append(dict(dataset=ds, budget_type="pct", budget=pct, AA_mean=aa_m, AA_std=aa_s,
                            n=sum(v is not None for v in aas)))
    for ds, budgets in [("wisdm", CNT_BUDGETS_WISDM), ("ucihar", CNT_BUDGETS_UCIHAR)]:
        for n_budget in budgets:
            aas = []
            for chain in CHAINS[ds]:
                cs = cstr(chain)
                p = (RESULTS / "sequential_tsfm" /
                    f"{ds}_mantis_{cs}_K4_n{n_budget}_lora_learned_s42" / "metrics.csv")
                aas.append(read_pact_metrics_last(p, "AA"))
            aa_m, aa_s = mean_std(aas)
            rows.append(dict(dataset=ds, budget_type="count", budget=n_budget, AA_mean=aa_m,
                            AA_std=aa_s, n=sum(v is not None for v in aas)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    TABLES_OUT.mkdir(parents=True, exist_ok=True)
    main_df = collect_main_comparison()
    main_df.to_csv(TABLES_OUT / "chain_system_main_comparison.csv", index=False)
    print("=== Main comparison (missing cells) ===")
    print(main_df[main_df["AA_mean"].isna()][["method", "dataset"]])

    bb_df = collect_backbones()
    bb_df.to_csv(TABLES_OUT / "chain_system_backbones.csv", index=False)
    print("\n=== Backbones (missing cells) ===")
    print(bb_df[bb_df["AA_mean"].isna()][["backbone", "dataset"]])

    comp_df = collect_component_ablation()
    comp_df.to_csv(TABLES_OUT / "chain_system_component_ablation.csv", index=False)
    print("\n=== Component ablation (missing cells) ===")
    print(comp_df[comp_df["AA_mean"].isna()][["dataset", "component"]])

    lora_df = collect_lora_vs_fullrank()
    lora_df.to_csv(TABLES_OUT / "chain_system_lora_vs_fullrank.csv", index=False)
    print("\n=== LoRA vs full-rank (missing cells) ===")
    print(lora_df[lora_df["AA_mean"].isna()][["dataset", "rank"]])

    gate_df = collect_gate_ablation()
    gate_df.to_csv(TABLES_OUT / "chain_system_gate_ablation.csv", index=False)
    print("\n=== Gate ablation (missing cells) ===")
    print(gate_df[gate_df["AA_mean"].isna()][["dataset", "gate"]])

    label_df = collect_label_efficiency()
    label_df.to_csv(TABLES_OUT / "chain_system_label_efficiency.csv", index=False)
    print("\n=== Label efficiency (missing cells) ===")
    print(label_df[label_df["AA_mean"].isna()][["dataset", "budget_type", "budget"]])

    print("\nAll CSVs written to results/tables/chain_system_*.csv")
