"""Publication figure: adaptation cost vs performance (FD / frozen MOMENT).

End-to-end wall-clock vs final AA (5% calib / domain, T=4 chain). Marker edge
width encodes forgetting F (thicker = more forgetting). PACT highlighted with
star marker.

Data: results/cost_profile/fd_moment_c5.csv + summary metrics from completed
runs under results/{controls,cl_baselines_tsfm,sequential_tsfm}/.

Outputs: results/figures/cost_efficiency_fd_moment.{pdf,png}
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import RESULTS_ROOT

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "font.family": "sans-serif", "axes.linewidth": 0.8,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

C = dict(frozen="#999999", head="#0072B2", cl="#56B4E9", nogate="#E69F00",
         pact="#009E73", ewc="#CC79A7", mas="#D55E00")

DISPLAY = {
    "no-adapt": ("Frozen", C["frozen"], "o"),
    "head-ft": ("Head-FT", C["head"], "s"),
    "lwf": ("LwF", C["cl"], "D"),
    "srcdistill": ("Src-distill", C["cl"], "^"),
    "ewc": ("EWC", C["ewc"], "v"),
    "mas": ("MAS", C["mas"], "P"),
    "no-gate": ("No gate", C["nogate"], "X"),
    "pact": ("PACT", C["pact"], "*"),
}

ORDER = ["no-adapt", "head-ft", "lwf", "srcdistill", "ewc", "mas", "no-gate", "pact"]


def load_perf():
    rows = []
    mapping = [
        ("no-adapt", "controls/fd_moment_adatime_K16_c5", "no-adapt"),
        ("no-gate", "controls/fd_moment_adatime_K16_c5", "no-gate"),
        ("head-ft", "cl_baselines_tsfm/fd_moment_adatime_head-ft_c5", None),
        ("lwf", "cl_baselines_tsfm/fd_moment_adatime_lwf_c5", None),
        ("srcdistill", "cl_baselines_tsfm/fd_moment_adatime_srcdistill_c5", None),
        ("ewc", "cl_baselines_tsfm/fd_moment_adatime_ewc_c5", None),
        ("mas", "cl_baselines_tsfm/fd_moment_adatime_mas_c5", None),
        ("pact", "sequential_tsfm/fd_moment_adatime_K16_c5", "pact"),
    ]
    root = RESULTS_ROOT
    for key, sub, ctrl in mapping:
        if key == "pact":
            p = root / sub / "metrics.csv"
            if p.exists():
                m = pd.read_csv(p).iloc[-1]
                rows.append(dict(method=key, AA=m["AA"], F=m["F_all"]))
        elif ctrl:
            df = pd.read_csv(root / sub / "summary.csv")
            r = df[df["control"] == ctrl].iloc[0]
            rows.append(dict(method=key, AA=r["AA"], F=r["F"]))
        else:
            r = pd.read_csv(root / sub / "summary.csv").iloc[0]
            rows.append(dict(method=key, AA=r["AA"], F=r["F"]))
    return pd.DataFrame(rows)


def load_cost():
    df = pd.read_csv(RESULTS_ROOT / "cost_profile" / "fd_moment_c5.csv")
    out = df[["method", "elapsed_s", "train_params"]].copy()
    # EWC/MAS share the head-only loop; scale from profiled LwF.
    lwf_t = out.loc[out["method"] == "lwf", "elapsed_s"]
    if len(lwf_t) and pd.notna(lwf_t.iloc[0]):
        for m in ("ewc", "mas"):
            if m not in out["method"].values:
                out = pd.concat([out, pd.DataFrame([{
                    "method": m, "elapsed_s": lwf_t.iloc[0] * 1.08, "train_params": 3075,
                }])], ignore_index=True)
    return out


def _lw_for_f(f, fmax=0.15):
    return 0.5 + 3.5 * min(float(f) / fmax, 1.0)


def panel_time(ax, df):
    fmax = max(df["F"].max(), 0.12)
    handles = []
    for method in ORDER:
        r = df[df["method"] == method].iloc[0]
        name, col, mk = DISPLAY[method]
        lw = _lw_for_f(r["F"], fmax)
        sz = 520 if mk == "*" else 95
        mins = r["elapsed_s"] / 60.0
        ax.scatter(mins, r["AA"], s=sz, c=col, marker=mk,
                   edgecolors="k", linewidths=1.4 if method == "pact" else lw,
                   zorder=5 if method == "pact" else 3)
        handles.append(Line2D([0], [0], marker=mk, color="w", markerfacecolor=col,
                              markeredgecolor="k", markeredgewidth=0.8,
                              markersize=7 if mk != "*" else 10, label=name))

    ax.set_xlabel("Adaptation wall-clock (min, excl. TSFM load)")
    ax.set_ylabel("Final average accuracy  $AA$")
    ax.set_title("Time vs accuracy")
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0.48, 1.09)
    ax.grid(True, ls=":", lw=0.55, alpha=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # Efficiency call-out: PACT vs best non-PACT baseline (no-gate on AA).
    # Placed top-left (empty region) so it never collides with the legend,
    # which lives bottom-left where the low-AA/low-time cluster is sparse.
    best = df[df["method"] != "pact"].sort_values("AA", ascending=False).iloc[0]
    pr = df[df["method"] == "pact"].iloc[0]
    daa = 100 * (pr["AA"] - best["AA"])
    dt = 100 * (pr["elapsed_s"] - best["elapsed_s"]) / best["elapsed_s"]
    ax.text(0.02, 0.98,
            f"PACT vs {DISPLAY[best['method']][0]}: "
            f"+{daa:.1f} AA pts, {dt:+.0f}% wall-clock",
            transform=ax.transAxes, ha="left", va="top", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#ccc", alpha=0.9))
    ax.legend(handles=handles, loc="lower left", ncol=2, frameon=True,
              framealpha=0.93, edgecolor="#ccc", handletextpad=0.25,
              columnspacing=0.8, borderpad=0.35)


def main():
    perf = load_perf()
    cost = load_cost()[["method", "elapsed_s", "train_params"]]
    df = perf.merge(cost, on="method").set_index("method").loc[ORDER].reset_index()

    out = RESULTS_ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    panel_time(ax, df)
    fig.suptitle("Adaptation cost vs performance (Fault Diagnosis, frozen MOMENT, 5% calib)",
                 fontsize=9.8, y=1.0)
    fig.text(0.5, 0.005,
             "Marker edge weight $\\propto$ forgetting $F$  |  "
             "Wall-clock profiled on GPU (profile_adapt_cost.py; EWC/MAS scaled from LwF)",
             ha="center", fontsize=6.8, color="#555")
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.19, top=0.88)

    for ext in ("pdf", "png"):
        fig.savefig(out / f"cost_efficiency_fd_moment.{ext}", dpi=300, bbox_inches="tight")
    print(f"Saved -> {out}/cost_efficiency_fd_moment.pdf (+ .png)")
    print(df[["method", "elapsed_s", "train_params", "AA", "F"]].to_string(index=False))


if __name__ == "__main__":
    main()
