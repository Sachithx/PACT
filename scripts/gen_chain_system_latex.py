"""Render the chain_system_*.csv tables (produced by gen_chain_system_tables.py)
into LaTeX booktabs tables for the supplement.

Outputs paper/aaai27/tables/chain_system_tables.tex.
"""

from pathlib import Path

import pandas as pd

TABLES = Path("results/tables")
OUT = Path("paper/aaai27/tables/chain_system_tables.tex")

DS_ORDER = ["fd", "hhar", "sleep", "wisdm", "ucihar"]
DS_LABEL = {"fd": "FD", "hhar": "HHAR", "sleep": "Sleep-EDF", "wisdm": "WISDM", "ucihar": "UCI-HAR"}

METHOD_ORDER = ["no-adapt", "head-ft", "ewc", "mas", "lwf", "srcdistill", "dualcd",
               "tent", "mapu-free", "accup", "mapu", "cotta", "eata", "sprompts",
               "ncm", "emb-replay-head", "derpp", "no-gate", "pact"]
METHOD_LABEL = {
    "no-adapt": "Frozen (no adapt)", "head-ft": "Head fine-tune", "ewc": "EWC",
    "mas": "MAS", "lwf": "LwF", "srcdistill": "Source-distill", "dualcd": "DualCD",
    "tent": "TENT", "mapu-free": "MAPU (unsup.)", "accup": "ACCUP (unsup.)",
    "mapu": "MAPU", "cotta": "CoTTA",
    "eata": "EATA", "sprompts": "S-Prompts", "ncm": "NCM",
    "emb-replay-head": "Emb-replay (head)", "derpp": r"DER++",
    "no-gate": "PACT (no gate)", "pact": r"\textbf{PACT (ours)}",
}
METHOD_GROUP_AFTER = {"no-adapt", "dualcd", "sprompts", "derpp"}  # draw \midrule after these

BACKBONE_ORDER = ["tslanet", "patchtst", "moment", "chronos-bolt", "sundial",
                  "mantis", "moirai", "unishape"]
BACKBONE_LABEL = {
    "tslanet": "TSLANet", "patchtst": "PatchTST", "moment": "MOMENT (frozen)",
    "chronos-bolt": "Chronos-Bolt (frozen)", "sundial": r"\textbf{Sundial (frozen, ours)}",
    "mantis": "Mantis (frozen)", "moirai": "Moirai (frozen)", "unishape": "UniShape (frozen)",
}


def fmt(mean, std, digits=3):
    if pd.isna(mean):
        return "---"
    return f"{mean:.{digits}f}$\\pm${std:.{digits}f}"


def gen_main_comparison_table(df):
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Full baseline comparison under the established evaluation "
                "protocol used throughout the paper: Sundial (frozen) backbone, "
                "each dataset's best chart count $K$ "
                "(FD 8, HHAR 4, Sleep-EDF 1, WISDM 4, UCI-HAR 4), LoRA rank-8 transports "
                "where applicable, and 4 fixed random domain-order chains per dataset "
                "(mean$\\pm$std over chains). AMF$_1$$\\uparrow$; PRF$\\downarrow$. "
                "Best AMF$_1$ per dataset in \\textbf{bold}.}")
    lines.append(r"\label{tab:chain-main-comparison}")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{l" + "c" * len(DS_ORDER) + "}")
    lines.append(r"\toprule")
    lines.append("Method & " + " & ".join(DS_LABEL[d] for d in DS_ORDER) + r"\\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{6}{c}{\textit{AMF$_1$ ($\uparrow$)}}\\")
    lines.append(r"\midrule")
    best_ds = {d: df[df["dataset"] == d]["AA_mean"].max() for d in DS_ORDER}
    for m in METHOD_ORDER:
        sub = df[df["method"] == m].set_index("dataset")
        cells = []
        for d in DS_ORDER:
            if d not in sub.index:
                cells.append("---")
                continue
            mean, std = sub.loc[d, "AA_mean"], sub.loc[d, "AA_std"]
            s = fmt(mean, std)
            if pd.notna(mean) and abs(mean - best_ds[d]) < 1e-9:
                s = r"\textbf{" + s + "}"
            cells.append(s)
        lines.append(f"{METHOD_LABEL[m]} & " + " & ".join(cells) + r"\\")
        if m in METHOD_GROUP_AFTER:
            lines.append(r"\midrule")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{6}{c}{\textit{PRF ($\downarrow$)}}\\")
    lines.append(r"\midrule")
    best_ds_prf = {d: df[(df["dataset"] == d) & (df["PRF_mean"].notna())]["PRF_mean"].min()
                  for d in DS_ORDER}
    for m in METHOD_ORDER:
        sub = df[df["method"] == m].set_index("dataset")
        cells = []
        for d in DS_ORDER:
            if d not in sub.index:
                cells.append("---")
                continue
            mean, std = sub.loc[d, "PRF_mean"], sub.loc[d, "PRF_std"]
            s = fmt(mean, std)
            if pd.notna(mean) and m != "no-adapt" and abs(mean - best_ds_prf[d]) < 1e-9:
                s = r"\textbf{" + s + "}"
            cells.append(s)
        lines.append(f"{METHOD_LABEL[m]} & " + " & ".join(cells) + r"\\")
        if m in METHOD_GROUP_AFTER:
            lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def gen_backbones_table(df):
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{PACT across backbones under the established evaluation "
                "protocol (4 fixed random chains per dataset, each dataset's best-$K$; "
                r"Sundial uses $K{=}16$ here for a backbone-controlled comparison, "
                "not each dataset's individually tuned best-$K$ used elsewhere). "
                "AMF$_1$$\\uparrow$ mean$\\pm$std over chains. Best per dataset in "
                "\\textbf{bold}.}")
    lines.append(r"\label{tab:chain-backbones}")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{l" + "c" * len(DS_ORDER) + "}")
    lines.append(r"\toprule")
    lines.append("Backbone & " + " & ".join(DS_LABEL[d] for d in DS_ORDER) + r"\\")
    lines.append(r"\midrule")
    best_ds = {d: df[df["dataset"] == d]["AA_mean"].max() for d in DS_ORDER}
    for bb in BACKBONE_ORDER:
        sub = df[df["backbone"] == bb].set_index("dataset")
        cells = []
        for d in DS_ORDER:
            if d not in sub.index:
                cells.append("---")
                continue
            mean, std = sub.loc[d, "AA_mean"], sub.loc[d, "AA_std"]
            s = fmt(mean, std)
            if pd.notna(mean) and abs(mean - best_ds[d]) < 1e-9:
                s = r"\textbf{" + s + "}"
            cells.append(s)
        lines.append(f"{BACKBONE_LABEL[bb]} & " + " & ".join(cells) + r"\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def gen_component_ablation_table(df):
    order = ["no_adapters", "head_ft", "no_gate", "pact"]
    label = {"no_adapters": "w/o Adapters", "head_ft": "Head fine-tune",
            "no_gate": "w/o Gate", "pact": r"\textbf{PACT}"}
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Component ablation (mean$\pm$std over 4 chains per "
                "dataset), Sundial backbone, each dataset's best-$K$. "
                "AMF$_1$$\\uparrow$; PRF$\\downarrow$. "
                r"w/o Adapters has PRF $=0$ by construction (a frozen head never "
                "updates, so it cannot ``forget'').}")
    lines.append(r"\label{tab:component-ablation-full}")
    lines.append(r"\footnotesize")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{l" + "c" * (2 * len(DS_ORDER)) + "}")
    lines.append(r"\toprule")
    lines.append("& " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{DS_LABEL[d]}}}" for d in DS_ORDER)
                + r"\\")
    cmid = " ".join(f"\\cmidrule(lr){{{2*i+2}-{2*i+3}}}" for i in range(len(DS_ORDER)))
    lines.append(cmid)
    lines.append("Variant & " + " & ".join(["AMF$_1$ & PRF"] * len(DS_ORDER)) + r"\\")
    lines.append(r"\midrule")
    for comp in order:
        sub = df[df["component"] == comp].set_index("dataset")
        cells = []
        for d in DS_ORDER:
            row = sub.loc[d]
            cells.append(fmt(row["AA_mean"], row["AA_std"]))
            cells.append(fmt(row["PRF_mean"], row["PRF_std"]))
        lines.append(f"{label[comp]} & " + " & ".join(cells) + r"\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def gen_lora_table(df):
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{LoRA rank-8 vs.\ full-rank (768) transports at each "
                "dataset's best-$K$ (mean$\\pm$std over 4 chains). AMF$_1$$\\uparrow$.}")
    lines.append(r"\label{tab:lora-full}")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{l" + "c" * len(DS_ORDER) + "}")
    lines.append(r"\toprule")
    lines.append("& " + " & ".join(DS_LABEL[d] for d in DS_ORDER) + r"\\")
    lines.append(r"\midrule")
    for rank, label in [("lora8", r"\textbf{LoRA r=8 (ours)}"), ("full768", "Full rank=768")]:
        sub = df[df["rank"] == rank].set_index("dataset")
        cells = [fmt(sub.loc[d, "AA_mean"], sub.loc[d, "AA_std"]) for d in DS_ORDER]
        lines.append(f"{label} & " + " & ".join(cells) + r"\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def gen_gate_table(df):
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Gate routing ablation at each dataset's best-$K$ "
                "(mean$\\pm$std over 4 chains). AMF$_1$$\\uparrow$; soft is the "
                "deployed default.}")
    lines.append(r"\label{tab:gate-full}")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{l" + "c" * len(DS_ORDER) + "}")
    lines.append(r"\toprule")
    lines.append("& " + " & ".join(DS_LABEL[d] for d in DS_ORDER) + r"\\")
    lines.append(r"\midrule")
    for gate, label in [("soft", r"\textbf{Soft (ours)}"), ("hard", "Hard"), ("oracle", "Oracle")]:
        sub = df[df["gate"] == gate].set_index("dataset")
        cells = [fmt(sub.loc[d, "AA_mean"], sub.loc[d, "AA_std"]) for d in DS_ORDER]
        lines.append(f"{label} & " + " & ".join(cells) + r"\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def gen_label_efficiency_table(df):
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Label-efficiency sweep, Mantis (frozen) backbone, "
                "$K{=}4$ (mean$\\pm$std over 4 chains). FD/HHAR/Sleep-EDF vary "
                "the calibration fraction $\\rho$; WISDM/UCI-HAR vary a fixed "
                "per-domain label count (their classes are too skewed for a "
                "meaningful \\% sweep at low budgets). AMF$_1$$\\uparrow$.}")
    lines.append(r"\label{tab:label-efficiency-full}")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{l ccccccc}")
    lines.append(r"\toprule")
    lines.append(r"Dataset & \multicolumn{6}{c}{Budget} \\")
    lines.append(r"\midrule")
    for ds in ["fd", "hhar", "sleep"]:
        sub = df[(df["dataset"] == ds) & (df["budget_type"] == "pct")].set_index("budget")
        header = " & ".join(f"{b}\\%" for b in sub.index)
        lines.append(r"\multicolumn{7}{l}{\textit{" + DS_LABEL[ds] + r" (\% calibration)}}\\")
        lines.append("& " + header + r"\\")
        cells = [fmt(sub.loc[b, "AA_mean"], sub.loc[b, "AA_std"]) for b in sub.index]
        lines.append("AMF$_1$ & " + " & ".join(cells) + r"\\")
        lines.append(r"\midrule")
    for ds in ["wisdm", "ucihar"]:
        sub = df[(df["dataset"] == ds) & (df["budget_type"] == "count")].set_index("budget")
        header = " & ".join(f"{b}" for b in sub.index)
        lines.append(r"\multicolumn{7}{l}{\textit{" + DS_LABEL[ds] +
                    r" (fixed label count per domain)}}\\")
        lines.append("& " + header + r"\\")
        cells = [fmt(sub.loc[b, "AA_mean"], sub.loc[b, "AA_std"]) for b in sub.index]
        lines.append("AMF$_1$ & " + " & ".join(cells) + r"\\")
        if ds != "ucihar":
            lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def gen_hparam_table():
    n_domains = {"fd": 4, "hhar": 9, "sleep": 9, "wisdm": 9, "ucihar": 9}
    best_k = {"fd": 8, "hhar": 4, "sleep": 1, "wisdm": 4, "ucihar": 4}
    calib_display = {"fd": "5\\% calibration", "hhar": "5\\% calibration",
                     "sleep": "5\\% calibration", "wisdm": "450 labels/domain",
                     "ucihar": "100 labels/domain"}
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Reproducibility summary for the established evaluation "
                "protocol (Sundial backbone unless a figure/table states otherwise). "
                r"All 4 chains per dataset use seed 42; only the domain order differs.}")
    lines.append(r"\label{tab:hparams-chain-system}")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{ll}")
    lines.append(r"\toprule")
    lines.append(r"\multicolumn{2}{l}{\textit{Per-dataset protocol}}\\")
    lines.append(r"\midrule")
    lines.append(r"Dataset & \# domains, best-$K$, label budget\\")
    for d in DS_ORDER:
        k_val = best_k[d]
        lines.append(f"{DS_LABEL[d]} & {n_domains[d]} domains, $K{{=}}{k_val}$, "
                    f"{calib_display[d]}\\\\")
    lines.append(r"Chains per dataset & 4, independently drawn, seed 42\\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{2}{l}{\textit{Charts \& transports}}\\")
    lines.append(r"\midrule")
    lines.append(r"$k$-means init/iters & seed-42 examples, Lloyd's, $\le$100 iters\\")
    lines.append(r"Transport family & LoRA rank $r{=}8$ (default); full-affine ($r{=}p$) ablation\\")
    lines.append(r"Transport init & identity ($W{=}I$, $b{=}0$; LoRA $B{=}0$)\\")
    lines.append(r"Transport training & 200 epochs, AdamW $10^{-3}$, wd $0.01$, batch 64 (TSFM)\\")
    lines.append(r"Identity penalty $\lambda$ & $10^{-3}$\\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{2}{l}{\textit{Gate}}\\")
    lines.append(r"\midrule")
    lines.append(r"Architecture & 2-layer MLP, hidden width 64, ReLU, dropout 0.2\\")
    lines.append(r"Temperature $\tau$ & 0.5\\")
    lines.append(r"Training & 200 epochs, AdamW $10^{-3}$, wd $10^{-4}$, class-balanced\\")
    lines.append(r"Refit trigger & every new domain, on retained src + calib embeddings\\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{2}{l}{\textit{Head \& embeddings}}\\")
    lines.append(r"\midrule")
    lines.append(r"Head fitting & 150 epochs, AdamW $10^{-3}$, wd $10^{-4}$, batch 256\\")
    lines.append(r"Standardization & source-train mean/std, floor $10^{-6}$\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


if __name__ == "__main__":
    main_df = pd.read_csv(TABLES / "chain_system_main_comparison.csv")
    bb_df = pd.read_csv(TABLES / "chain_system_backbones.csv")
    comp_df = pd.read_csv(TABLES / "chain_system_component_ablation.csv")
    lora_df = pd.read_csv(TABLES / "chain_system_lora_vs_fullrank.csv")
    gate_df = pd.read_csv(TABLES / "chain_system_gate_ablation.csv")
    label_df = pd.read_csv(TABLES / "chain_system_label_efficiency.csv")

    parts = [
        "% =====================================================================",
        "% CHAIN-SYSTEM SUPPLEMENT TABLES (Sundial backbone, established random",
        "% chains, mean +/- std over 4 chains per dataset). Auto-generated by",
        "% scripts/gen_chain_system_latex.py from results/tables/chain_system_*.csv",
        "% -- do not hand-edit numbers here, regenerate instead.",
        "% =====================================================================",
        r"\section{Full Result Tables, Established Chain Protocol}",
        r"\label{app:chain-full-tables}",
        "The following tables report every number underlying the main paper's "
        "figures under the evaluation protocol used throughout this paper: 5 "
        "datasets, a frozen TSFM backbone, 4 independently drawn random "
        "domain-order chains per dataset, and each dataset's best chart count "
        r"$K$ (FD 8, HHAR 4, Sleep-EDF 1, WISDM 4, UCI-HAR 4), unless noted "
        "otherwise. All entries are mean$\\pm$std over the 4 chains.",
        "",
        gen_main_comparison_table(main_df),
        "",
        gen_backbones_table(bb_df),
        "",
        gen_component_ablation_table(comp_df),
        "",
        gen_lora_table(lora_df),
        "",
        gen_gate_table(gate_df),
        "",
        gen_label_efficiency_table(label_df),
        "",
        gen_hparam_table(),
        "",
    ]
    OUT.write_text("\n".join(parts))
    print(f"Wrote {OUT}")
