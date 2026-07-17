"""DualCD vs PACT on the same domain-incremental chains, across every backbone.

FD/HHAR/Sleep use the 5% calibration budget (c5); WISDM uses the fixed
450-label budget (n450). Two backbone families:

  - Frozen TSFM (moment, chronos-bolt, moirai, sundial, time-moe): both
    methods run through scripts/run_cl_baselines_tsfm.py.
      DualCD: results/cl_baselines_tsfm/{ds}_{tsfm}_adatime_dualcd_{tag}_s{seed}/summary.csv
      PACT:   results/sequential_tsfm/{ds}_{tsfm}_adatime_K16_{tag}*/metrics.csv
              (shortest-named match = the headline run, before later
              multi-seed adapter/gate ablation sweeps added more suffixes)

  - Trainable backbones (tslanet, patchtst): both run through
    scripts/run_cl_baselines.py (full encoder+head fine-tuning). WISDM isn't
    wired up for these backbones in PACT's own harness (no cached source
    checkpoint), so it's skipped here, matching how the other CL baselines
    (ewc/mas/lwf) are evaluated too.
      DualCD: results/cl_baselines/{ds}_{backbone}_adatime_dualcd_c5_s{seed}/summary.csv
      PACT:   results/sequential/{ds}_{backbone}_adatime_K16_c5_t0.5_lora_learned_s{seed}/metrics.csv
              (tslanet; run_sequential.py's own default config), or
              results/sequential/{ds}_patchtst_adatime_K16_c5_t0.5_identity/metrics.csv
              (patchtst; the only config run for this backbone)

Example:
  python scripts/compare_dualcd_pact.py
"""

import glob

import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import RESULTS_ROOT

SEED = 42
TSFM_BACKBONES = ["moment", "chronos-bolt", "moirai", "sundial", "time-moe"]
RUNS = [("fd", "c5"), ("hhar", "c5"), ("sleep", "c5"), ("wisdm", "n450")]


def _read_metrics(path, aa_col, f_col):
    m = pd.read_csv(path).iloc[-1]
    return dict(AA=float(m["AA"]), BWT=float(m.get("BWT", float("nan"))),
                F=float(m[f_col]), RF=float(m["RF"]),
                PRF=float(m.get("PRF", float("nan"))), source=str(path))


def load_pact_tsfm(tsfm, ds, tag):
    pattern = str(RESULTS_ROOT / "sequential_tsfm" / f"{ds}_{tsfm}_adatime_K16_{tag}*")
    matches = sorted(glob.glob(pattern), key=len)
    if not matches:
        return None
    return _read_metrics(f"{matches[0]}/metrics.csv", "AA", "F_all")


def load_dualcd_tsfm(tsfm, ds, tag):
    path = RESULTS_ROOT / "cl_baselines_tsfm" / f"{ds}_{tsfm}_adatime_dualcd_{tag}_s{SEED}" / "summary.csv"
    if not path.exists():
        return None
    return _read_metrics(path, "AA", "F")


def load_pact_trainable(backbone, ds):
    if backbone == "tslanet":
        path = RESULTS_ROOT / "sequential" / f"{ds}_tslanet_adatime_K16_c5_t0.5_lora_learned_s{SEED}" / "metrics.csv"
    else:
        path = RESULTS_ROOT / "sequential" / f"{ds}_patchtst_adatime_K16_c5_t0.5_identity" / "metrics.csv"
    if not path.exists():
        return None
    return _read_metrics(path, "AA", "F_all")


def load_dualcd_trainable(backbone, ds):
    path = RESULTS_ROOT / "cl_baselines" / f"{ds}_{backbone}_adatime_dualcd_c5_s{SEED}" / "summary.csv"
    if not path.exists():
        return None
    return _read_metrics(path, "AA", "F")


def _print_row(ds, tag, method, r):
    if r is None:
        print(f"{ds:<8}{tag:<7}{method:<10}{'--':>8}  (missing)")
        return
    print(f"{ds:<8}{tag:<7}{method:<10}{r['AA']:>8.4f}{r['BWT']:>+9.4f}"
          f"{r['F']:>8.4f}{r['RF']:>8.4f}{r['PRF']:>8.4f}")


def main():
    rows = []
    header = f"{'dataset':<8}{'calib':<7}{'method':<10}{'AA':>8}{'BWT':>9}{'F':>8}{'RF':>8}{'PRF':>8}"

    for backbone in TSFM_BACKBONES:
        print(f"\n========== {backbone} (frozen TSFM) ==========")
        print(header)
        for ds, tag in RUNS:
            pact = load_pact_tsfm(backbone, ds, tag)
            dcd = load_dualcd_tsfm(backbone, ds, tag)
            _print_row(ds, tag, "PACT", pact)
            _print_row(ds, tag, "DualCD", dcd)
            if pact:
                rows.append(dict(backbone=backbone, dataset=ds, calib=tag, method="PACT", **{k: pact[k] for k in ("AA", "BWT", "F", "RF", "PRF")}))
            if dcd:
                rows.append(dict(backbone=backbone, dataset=ds, calib=tag, method="DualCD", **{k: dcd[k] for k in ("AA", "BWT", "F", "RF", "PRF")}))
            if pact and dcd:
                print(f"{'':<8}{'':<7}{'Δ(PACT-DualCD)':<10}{pact['AA']-dcd['AA']:>+8.4f}"
                      f"{'':>9}{pact['F']-dcd['F']:>+8.4f}{pact['RF']-dcd['RF']:>+8.4f}"
                      f"{pact['PRF']-dcd['PRF']:>+8.4f}")
            print()

    for backbone in ["tslanet", "patchtst"]:
        print(f"\n========== {backbone} (trainable, full fine-tune) ==========")
        print(header)
        for ds in ["fd", "hhar", "sleep"]:
            pact = load_pact_trainable(backbone, ds)
            dcd = load_dualcd_trainable(backbone, ds)
            _print_row(ds, "c5", "PACT", pact)
            _print_row(ds, "c5", "DualCD", dcd)
            if pact:
                rows.append(dict(backbone=backbone, dataset=ds, calib="c5", method="PACT", **{k: pact[k] for k in ("AA", "BWT", "F", "RF", "PRF")}))
            if dcd:
                rows.append(dict(backbone=backbone, dataset=ds, calib="c5", method="DualCD", **{k: dcd[k] for k in ("AA", "BWT", "F", "RF", "PRF")}))
            if pact and dcd:
                print(f"{'':<8}{'':<7}{'Δ(PACT-DualCD)':<10}{pact['AA']-dcd['AA']:>+8.4f}"
                      f"{'':>9}{pact['F']-dcd['F']:>+8.4f}{pact['RF']-dcd['RF']:>+8.4f}"
                      f"{pact['PRF']-dcd['PRF']:>+8.4f}")
            print()

    out = RESULTS_ROOT / "tables" / "dualcd_vs_pact.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
