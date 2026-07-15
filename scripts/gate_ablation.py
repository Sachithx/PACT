"""Gate ablation: soft-mixture vs hard-gate vs oracle routing.

Every sequential run already saves the soft / hard / oracle F1 retention
matrices. This reads them and reports final-step AA and forgetting for each
routing strategy, so we can quantify (a) the value of soft over hard routing
and (b) the gap to the oracle (perfect routing) upper bound -- the sole source
of forgetting in PACT (backbone/head/past-adapters are all frozen).
"""

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import RESULTS_ROOT
from src.method.sequential import compute_cl_metrics

DATASETS = ["fd", "hhar", "sleep"]


def metrics_from(path):
    A = pd.read_csv(path, index_col=0).values
    T = A.shape[0]
    AA, BWT, F, RF, PRF = compute_cl_metrics(A, T - 1, include_source=True)
    return AA, F, RF


def main():
    rows = []
    for sub in ["sequential", "sequential_tsfm"]:
        root = RESULTS_ROOT / sub
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            fsoft, fhard, forc = (d / f"{k}_f1_matrix.csv"
                                  for k in ("soft", "hard", "oracle"))
            if not (fsoft.exists() and forc.exists()):
                continue
            ds = next((x for x in DATASETS if d.name.startswith(x + "_")), None)
            if ds is None:
                continue
            backbone = d.name[len(ds) + 1:].split("_")[0]
            rec = dict(dataset=ds, backbone=backbone)
            rec["AA_soft"], rec["F_soft"], _ = metrics_from(fsoft)
            if fhard.exists():
                rec["AA_hard"], rec["F_hard"], _ = metrics_from(fhard)
            rec["AA_oracle"], rec["F_oracle"], _ = metrics_from(forc)
            rows.append(rec)

    df = pd.DataFrame(rows).sort_values(["dataset", "backbone"]).reset_index(drop=True)
    if df.empty:
        print("No matrices found.")
        return
    print(f"{'dataset':<7}{'backbone':<13}{'AA_soft':>9}{'AA_hard':>9}"
          f"{'AA_orc':>9}{'soft-orc':>9}{'F_soft':>9}")
    print("-" * 65)
    for _, r in df.iterrows():
        gap = r.get("AA_soft", np.nan) - r.get("AA_oracle", np.nan)
        print(f"{r['dataset']:<7}{r['backbone']:<13}{r.get('AA_soft',float('nan')):>9.4f}"
              f"{r.get('AA_hard',float('nan')):>9.4f}{r.get('AA_oracle',float('nan')):>9.4f}"
              f"{gap:>+9.4f}{r.get('F_soft',float('nan')):>9.4f}")
    out = RESULTS_ROOT / "gate_ablation.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
