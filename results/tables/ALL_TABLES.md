# PACT — Finalized Paper Tables

All numbers rounded to 3 decimals. Source: `results/` (July 2026 runs).

**Protocols**
- **Table 1**: AdaTime pairwise benchmark, TSLANet, averaged over 10 shifts.
- **Tables 2–7**: Domain-incremental chain (AdaTime-derived order), 5% calibration labels/domain unless noted.

---

## Table 1 — Main comparison vs baselines (pairwise AdaTime, TSLANet)

| Method | FD AA | FD F | HHAR AA | HHAR F | Sleep AA | Sleep F |
|---|---|---|---|---|---|---|
| **No adaptation** |
| Source-only | 0.839 | 0.000 | 0.742 | 0.000 | 0.670 | 0.000 |
| **UDA** |
| DANN | 0.865 | 0.027 | 0.762 | 0.012 | 0.676 | 0.021 |
| Deep CORAL | 0.831 | 0.014 | 0.784 | 0.004 | 0.673 | 0.007 |
| MMDA | 0.834 | 0.016 | 0.741 | 0.007 | 0.615 | 0.016 |
| TENT | 0.840 | 0.008 | 0.727 | 0.006 | 0.663 | 0.009 |
| DANN† | 0.886 | 0.060 | 0.951 | 0.010 | 0.706 | 0.018 |
| CDAN† | 0.848 | 0.072 | **0.953** | 0.012 | 0.671 | 0.049 |
| CoTMix† | 0.056 | 0.942 | 0.047 | 0.947 | 0.065 | 0.713 |
| **Fine-tuning** |
| Full FT | 0.810 | 0.291 | 0.806 | 0.337 | 0.650 | 0.179 |
| Head FT | 0.846 | 0.010 | 0.801 | 0.067 | 0.657 | 0.124 |
| **Continual learning** |
| EWC | 0.817 | 0.263 | 0.812 | 0.183 | 0.677 | 0.003 |
| LwF | 0.806 | 0.220 | 0.852 | 0.113 | 0.666 | 0.105 |
| MAS | 0.839 | 0.000 | 0.743 | 0.000 | 0.673 | 0.002 |
| Source-distill | 0.923 | 0.004 | 0.961 | 0.001 | **0.721** | 0.020 |
| **Ours** |
| PACT (no gate) | 0.933 | 0.106 | 0.854 | 0.210 | 0.648 | 0.162 |
| **PACT (ours)** | **0.981** | 0.000 | **0.963** | 0.001 | 0.715 | 0.005 |

† = semi-supervised UDA (additional target labels).

---

## Table 2A — Chain continual baselines (TSLANet, 5% calib)

| Method | FD AA | FD F | HHAR AA | HHAR F | Sleep AA | Sleep F |
|---|---|---|---|---|---|---|
| Full fine-tune | 0.640 | 0.283 | 0.654 | 0.345 | 0.584 | 0.116 |
| Head fine-tune | 0.586 | 0.012 | 0.576 | 0.133 | 0.605 | 0.056 |
| EWC | 0.590 | 0.001 | 0.526 | 0.093 | 0.620 | 0.004 |
| MAS | 0.601 | 0.057 | 0.653 | 0.280 | 0.605 | 0.052 |
| LwF | 0.549 | 0.256 | 0.723 | 0.155 | 0.622 | 0.026 |
| Source-distill | 0.573 | 0.256 | 0.482 | 0.083 | 0.622 | 0.026 |
| ORTCL (LF)‡ | 0.538 | 0.004 | 0.375 | 0.009 | 0.370 | 0.091 |
| ORTCL+HeadFT‡ | 0.582 | 0.011 | 0.563 | 0.139 | 0.458 | 0.093 |
| **PACT (ours)** | **0.978** | 0.002 | **0.886** | 0.007 | **0.663** | 0.013 |

‡ = ORTCL adapted to TSLANet (ICB rotation) on the domain-incremental chain; LF uses no labels, HeadFT uses 5% calib per domain.
---

## Table 2B — Chain continual baselines (frozen MOMENT, 5% calib)

| Method | FD AA | FD F | HHAR AA | HHAR F | Sleep AA | Sleep F |
|---|---|---|---|---|---|---|
| Frozen (no adapt) | 0.535 | 0.000 | 0.525 | 0.000 | 0.635 | 0.000 |
| Head fine-tune | 0.856 | 0.139 | 0.600 | 0.152 | 0.552 | 0.159 |
| LwF (head) | 0.859 | 0.033 | 0.646 | 0.050 | 0.619 | 0.052 |
| Source-distill (head) | 0.749 | 0.028 | 0.597 | 0.070 | 0.626 | 0.048 |
| Adapters, no gate | 0.879 | 0.113 | 0.632 | 0.035 | 0.606 | 0.016 |
| **PACT (ours)** | **0.984** | 0.001 | 0.627 | **0.017** | **0.638** | **0.011** |

---

## Table 3 — PACT across backbones (chain, base config)

| Backbone | FD AA | FD F | HHAR AA | HHAR F | Sleep AA | Sleep F |
|---|---|---|---|---|---|---|
| TSLANet | 0.978 | 0.002 | **0.886** | 0.007 | **0.663** | 0.013 |
| PatchTST | 0.889 | 0.003 | 0.767 | 0.022 | 0.653 | 0.010 |
| MOMENT (frozen) | **0.984** | 0.001 | 0.627 | 0.017 | 0.638 | 0.011 |
| Chronos-Bolt (frozen) | 0.977 | 0.002 | 0.630 | 0.013 | 0.620 | 0.008 |
| Sundial (frozen) | 0.982 | 0.000 | 0.806 | 0.013 | 0.608 | 0.006 |
| Time-MoE (frozen) | 0.943 | 0.006 | 0.757 | 0.017 | 0.519 | 0.015 |
| Moirai (frozen) | 0.972 | 0.006 | 0.666 | 0.010 | 0.657 | 0.014 |

Chronos-T5 was dropped (too slow to run the full HHAR/Sleep-EDF chains) and
replaced by Sundial, Time-MoE, and Moirai — see `scripts/tsfm.py` and
`scripts/campaign_table3_newtsfm.sh`.

---

## Table 4 — Component ablation (FD chain)

| Variant | TSLANet AA | TSLANet F | MOMENT AA | MOMENT F |
|---|---|---|---|---|
| Frozen (no adapt) | 0.574 | 0.000 | 0.535 | 0.000 |
| Full fine-tune | 0.640 | 0.283 | — | — |
| Adapters, no gate | 0.748 | 0.032 | 0.879 | 0.113 |
| **PACT (ours)** | **0.978** | 0.002 | **0.984** | 0.001 |

---

## Table 5 — Gate routing ablation (final AA)

| Dataset | TSLANet Soft | Hard | Oracle | MOMENT Soft | Oracle |
|---|---|---|---|---|---|
| FD | 0.978 | 0.979 | 0.982 | 0.984 | 0.985 |
| HHAR | 0.886 | 0.883 | 0.919 | 0.627 | 0.684 |
| Sleep | 0.663 | 0.658 | 0.664 | 0.638 | 0.648 |

---

## Table 6 — Hyperparameter ablations (FD chain)

**Concepts K (TSLANet):** 4→0.968, 8→0.971, 16→0.978, 32→0.976, 64→0.975 (F ≤ 0.002)

**Concepts K (MOMENT):** 8→0.982, 16→0.984, 32→0.980 (F ≤ 0.004)

**Temperature τ:** 0.1→0.979, 0.3→0.979, 0.5→0.978, 1.0→0.978, 2.0→0.971

**Label budget (TSLANet):** 1%→0.959, 2%→0.968, 5%→0.978, 10%→0.986, 20%→0.990

**Adapter:** Identity→0.978, LoRA→0.971

**Domain order:** 0.967–0.978 across 4 permutations

---

## Table 7 — Label-budget scaling (frozen MOMENT, K=16)

| Calib labels | FD | HHAR | Sleep |
|---|---|---|---|
| 1% | 0.955 | 0.564 | 0.630 |
| 2% | 0.970 | 0.597 | 0.610 |
| 5% | 0.984 | 0.627 | 0.638 |
| 10% | 0.987 | 0.662 | 0.658 |
| 20% | 0.993 | 0.689 | 0.679 |

---

## Files

| File | Contents |
|---|---|
| `results/tables/paper_tables.tex` | Tables 1-7, LaTeX (original in-body version) |
| `results/tables/ALL_TABLES.md` | This document |
| `results/tables/main.csv` | Table 3 raw |
| `results/tables/table2_chain.csv` | Table 2 raw |
| `results/tables/controls.csv` | Table 4 raw |
| `results/gate_ablation.csv` | Table 5 raw |

### Figure-first main text (2026-07 redesign)

Tables 2-7 became figures in the main text; Tables 1-7 all moved to the
appendix (Table 6 also got fixed -- see below). Regenerate the figures with
`python scripts/fig_experiments.py` (conda env `entrope`).

| File | Contents |
|---|---|
| `scripts/fig_experiments.py` | Generates all 4 figures below |
| `results/figures/fig_tradeoff.{pdf,png}` | Fig 1: accuracy-forgetting scatter (Table 2A) |
| `results/figures/fig_backbones.{pdf,png}` | Fig 2: AA/F across 5 backbones (Table 3) |
| `results/figures/fig_ablation.{pdf,png}` | Fig 3: component ablation + gate routing (Tables 4-5) |
| `results/figures/fig_robustness.{pdf,png}` | Fig 4: label budget / K / tau sweeps (Tables 6-7) |
| `results/tables/experiments_section.tex` | `\section{Experiments}` ready to paste, cites the 4 figures |
| `results/tables/appendix_tables.tex` | `\appendix` scaffold, Tables 1-7 (Table 6 replaced by expanded per-row version) |

**Table 6 correction:** TSLANet 2% label-budget AA is **0.967**, not 0.968
as listed above and in the old packed Table 6 in `paper_tables.tex`
(`sweep_budget_tslanet.csv` gives 0.9674585..., which rounds down). Fixed
in `appendix_tables.tex`; the entries above and in `paper_tables.tex` are
left as originally recorded and should be corrected at the same time.
