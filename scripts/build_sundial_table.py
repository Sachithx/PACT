"""Build the main sequential domain-incremental table (Sundial backbone) from
whatever chain-protocol results currently exist on disk. Missing/incomplete
runs print as "--". FD/HHAR/Sleep use 5% calibration (c5); WISDM uses the
fixed 450-label budget (n450).

Example:
  python scripts/build_sundial_table.py
"""

import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import RESULTS_ROOT

TSFM = "sundial"
SEED = 42
DATASETS = [("fd", "c5"), ("wisdm", "n450"), ("hhar", "c5"), ("sleep", "c5")]


def _last_row(path):
    p = RESULTS_ROOT / path
    if not p.exists():
        return None
    return pd.read_csv(p).iloc[-1]


def fmt(v):
    if v is None:
        return "--"
    return f"{v:.3f}"


def get(ds, tag, kind):
    """kind selects which family/file to read."""
    if kind == "source-only":
        r = _last_row(f"controls/{ds}_{TSFM}_adatime_K16_{tag}/summary.csv")
        if r is None:
            return None
        row = pd.read_csv(RESULTS_ROOT / f"controls/{ds}_{TSFM}_adatime_K16_{tag}/summary.csv")
        row = row[row["control"] == "no-adapt"]
        if row.empty:
            return None
        r = row.iloc[0]
        return dict(AA=r["AA"], BWT=r["BWT"], F=r["F"])
    if kind == "pact-no-gate":
        row = pd.read_csv(RESULTS_ROOT / f"controls/{ds}_{TSFM}_adatime_K16_{tag}/summary.csv")
        row = row[row["control"] == "no-gate"]
        if row.empty:
            return None
        r = row.iloc[0]
        return dict(AA=r["AA"], BWT=r["BWT"], F=r["F"])
    if kind == "pact":
        import glob
        pattern = str(RESULTS_ROOT / f"sequential_tsfm/{ds}_{TSFM}_adatime_K16_{tag}*")
        matches = [m for m in glob.glob(pattern) if "K1_" not in m]
        matches = sorted(matches, key=len)
        if not matches:
            return None
        r = pd.read_csv(f"{matches[0]}/metrics.csv").iloc[-1]
        return dict(AA=r["AA"], BWT=r["BWT"], F=r["F_all"])
    if kind == "k1":
        r = _last_row(f"sequential_tsfm/{ds}_{TSFM}_adatime_K1_{tag}_lora_learned_s{SEED}/metrics.csv")
        if r is None:
            return None
        return dict(AA=r["AA"], BWT=r["BWT"], F=r["F_all"])
    if kind == "ncm":
        r = _last_row(f"ncm_baselines/{ds}_{TSFM}_ncm_c5_s{SEED}/summary.csv")
        if r is None:
            return None
        return dict(AA=r["AA"], BWT=r["BWT"], F=r["F"])
    if kind in ("head-ft", "lwf", "srcdistill", "ewc", "mas", "dualcd"):
        r = _last_row(f"cl_baselines_tsfm/{ds}_{TSFM}_adatime_{kind}_{tag}/summary.csv")
        if r is None:
            r = _last_row(f"cl_baselines_tsfm/{ds}_{TSFM}_adatime_{kind}_{tag}_s{SEED}/summary.csv")
        if r is None:
            return None
        return dict(AA=r["AA"], BWT=r["BWT"], F=r["F"])
    if kind in ("sprompts", "cotta", "eata"):
        r = _last_row(f"chain_baselines/{ds}_{TSFM}_adatime_{kind}_{tag}_s{SEED}/summary.csv")
        if r is None:
            return None
        return dict(AA=r["AA"], BWT=r["BWT"], F=r["F"])
    if kind == "mapu":
        r = _last_row(f"mapu_baselines/{ds}_{TSFM}_adatime_mapu_{tag}_s{SEED}/summary.csv")
        if r is None:
            return None
        return dict(AA=r["AA"], BWT=r["BWT"], F=r["F"])
    if kind == "emb-replay-head":
        if ds == "wisdm":
            return None
        r = _last_row(f"embedding_replay/{ds}_{TSFM}_emb-replay-head_{tag}_s{SEED}/metrics.csv")
        if r is None:
            return None
        return dict(AA=r["AA"], BWT=r["BWT"], F=r["F"])
    raise ValueError(kind)


ROWS = [
    ("Source-only", "source-only"),
    ("NCM prototypes", "ncm"),
    ("Global linear (K=1)", "k1"),
    ("Head fine-tune", "head-ft"),
    ("LwF", "lwf"),
    ("EWC", "ewc"),
    ("MAS", "mas"),
    ("Source-distill", "srcdistill"),
    ("Embedding-replay head", "emb-replay-head"),
    ("CoTTA (head)", "cotta"),
    ("EATA (head)", "eata"),
    ("MAPU", "mapu"),
    ("S-Prompts", "sprompts"),
    ("DualCD", "dualcd"),
    ("PACT (no gate)", "pact-no-gate"),
    ("PACT (ours)", "pact"),
]


def main():
    table = {}
    for label, kind in ROWS:
        table[label] = {}
        for ds, tag in DATASETS:
            table[label][ds] = get(ds, tag, kind)

    # console
    print(f"{'Method':<24}" + "".join(f"{ds.upper():>28}" for ds, _ in DATASETS))
    for label, _ in ROWS:
        line = f"{label:<24}"
        for ds, _ in DATASETS:
            r = table[label][ds]
            if r is None:
                line += f"{'--':>28}"
            else:
                line += f"{fmt(r['AA']):>8}{fmt(r['BWT']):>10}{fmt(r['F']):>10}"
        print(line)

    print("\n% ---- LaTeX rows ----")
    for label, _ in ROWS:
        cells = []
        for ds, _ in DATASETS:
            r = table[label][ds]
            if r is None:
                cells += ["--", "--", "--"]
            else:
                bwt = r["BWT"]
                sign = r"$+$" if bwt >= 0 else r"$-$"
                cells += [fmt(r["AA"]), f"{sign}{abs(bwt):.3f}", fmt(r["F"])]
        print(f"& {label} & " + " & ".join(cells) + r" \\")


if __name__ == "__main__":
    main()
