# PACT: Piecewise Adaptation via Chart Transport

**Forgetting-Free Domain-Incremental Adaptation of Time-Series Foundation Models**

> Submitted to AAAI 2027

## Overview

Pretrained Time-Series Foundation Models (TSFMs) deliver strong representations out of the box, but deployed systems often encounter **operating conditions that shift sequentially** after training. Fine-tuning shared parameters can recover accuracy on a new condition but degrades previously observed ones (catastrophic forgetting), while regularization-based methods trade plasticity for stability.

PACT sidesteps this dilemma entirely. It **never modifies** the pretrained encoder, the classifier head, or any previously learned correction. Instead, it partitions the frozen encoder's embedding space into local regions called *charts*, learns lightweight residual transports for each new condition, and softly routes every test sample through the accumulated transports at inference -- retaining an explicit identity path back to the original representation.

### Key idea

Operating-condition shift is **locally concentrated**: some embedding regions move substantially while others stay stable. A global correction therefore damages regions that need no adaptation. PACT applies corrections only where they are needed:

```
e_hat = sum_d  alpha_d(e, c) * A_{d,c}(e)

alpha = softmax(g_c(e) / T)        (per-chart soft gate)
A_{0,c} = identity                  (source path, always retained)
```

Because the source "adapter" is the identity, source-like samples route back to the untouched embedding (preserving old performance) while shifted samples get their local correction. Only the gate is retrained as domains accumulate; the encoder, head, and prior transports remain frozen.

### Highlights

- **Zero host-level forgetting** -- encoder and classifier head are never updated
- **Few-label calibration** -- adapts from as few as 5% labeled samples per new condition
- **No condition identity at test time** -- soft gating handles mixed-condition streams automatically
- **Backbone-agnostic** -- works with TSLANet, PatchTST, TimesNet, MOMENT, Sundial, Chronos, Moirai, and Time-MoE
- **Margin-based preservation guarantee** -- a derived condition under which the original prediction is provably unchanged

## Results at a Glance

On a bearing-fault diagnosis benchmark with four sequential operating conditions (5% calibration labels, no condition identity at test time):

| Method | Avg Accuracy after 4 conditions |
|---|---|
| **PACT** | **0.984** |
| Head fine-tuning | 0.68 |
| Full fine-tuning | 0.38 |
| EWC / MAS / LwF | 0.54 -- 0.86 |

PACT is the only evaluated method to remain above 98% average accuracy as conditions accumulate. The same retention-adaptation advantage is observed on human-activity recognition (HHAR) and sleep-staging (Sleep-EDF) datasets across multiple TSFM backbones.

## Repository Structure

```
PACT/
├── src/                          # Core library (clean, refactored)
│   ├── models/                   # Backbones, adapters, gate, concepts, factory
│   │   ├── tslanet.py            # TSLANet backbone
│   │   ├── patchtst.py           # PatchTST backbone
│   │   ├── timesnet.py           # TimesNet backbone
│   │   ├── tsfm.py               # Foundation model wrappers (MOMENT, Sundial, etc.)
│   │   ├── adapters.py           # Per-concept adapter banks
│   │   ├── gate.py               # Soft-mixture gating network
│   │   ├── concepts.py           # K-means chart partitioning
│   │   └── factory.py            # Backbone registry and loader
│   ├── data/                     # Dataset registry + AdaTime scenarios
│   │   ├── datasets.py           # FD, HHAR, Sleep-EDF, UCIHAR, WISDM loaders
│   │   └── calib.py              # Calibration set sampling
│   ├── method/                   # PACT algorithm
│   │   ├── pact.py               # Core: adapters + gate training + inference
│   │   ├── sequential.py         # Domain-incremental CL loop + metrics (AA/BWT/F)
│   │   └── dp.py                 # Differential privacy extension (DP-PACT)
│   ├── training.py               # Source-backbone training / checkpoint loading
│   └── utils/                    # Seeding, device selection, masking
├── scripts/                      # Entry points
│   ├── run_sequential.py         # MAIN: sequential domain-incremental CL with PACT
│   ├── train_backbone.py         # Train or load a source backbone
│   ├── run_ablations.sh          # Backbone / LoRA / K / label-budget / temperature sweeps
│   ├── run_cl_baselines.py       # Continual learning baselines (EWC, MAS, LwF, etc.)
│   ├── run_cl_baselines_tsfm.py  # CL baselines on foundation model backbones
│   ├── run_sequential_tsfm.py    # PACT on foundation model backbones
│   └── run_controls.py           # Control experiments
├── experiments/                  # Original experiment drivers (exact paper reproduction)
│   ├── run_unified_eval.py       # PACT vs all baselines (FD + HHAR)
│   ├── run_sleep_unified_eval.py # PACT vs all baselines (Sleep-EDF)
│   ├── compute_cl_metrics.py     # AA / BWT / F / FWT aggregation
│   └── ...                       # Transfer, ablation, and sweep scripts
├── results/                      # Pre-computed experiment results (CSVs + figures)
├── requirements.txt
└── README.md
```

## Datasets

| Dataset | Task | Domains | Classes | Seq Length | Channels |
|---|---|---|---|---|---|
| **FD** | Bearing fault diagnosis | 4 operating conditions | 3 | 5120 | 1 |
| **HHAR** | Human activity recognition | 9 users | 6 | 128 | 3 |
| **Sleep-EDF** | Sleep stage classification | 20 subjects | 5 | 3000 | 1 |
| **UCIHAR** | Human activity recognition | 30 subjects | 6 | 128 | 9 |
| **WISDM** | Human activity recognition | 36 users | 6 | 128 | 3 |

Datasets are not included in this repository due to size. Place them under a `datasets/` directory at the project root (or symlink `dataset` to your data location). See `src/data/datasets.py` for the expected directory layout.

## Installation

```bash
conda create -n pact python=3.10
conda activate pact
pip install -r requirements.txt
```

> **Note:** If you encounter `GLIBCXX` errors with `torch.optim` or `optree`, preload the conda `libstdc++`:
> ```bash
> LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6 python ...
> ```

### Requirements

- Python 3.10+
- PyTorch 2.10+
- scikit-learn, scipy, pandas, matplotlib, einops
- opacus (for DP-PACT ablation only)

## Quick Start

### 1. Train or load a source backbone

```bash
python scripts/train_backbone.py --backbone tslanet --dataset fd --domain 0
```

Pre-trained checkpoints for TSLANet and PatchTST are available for all datasets.

### 2. Run PACT (sequential domain-incremental CL)

```bash
# TSLANet on FD (default order: 0 → 1 → 2 → 3)
python scripts/run_sequential.py --dataset fd

# PatchTST on HHAR
python scripts/run_sequential.py --dataset hhar --backbone patchtst

# Custom domain order and calibration fraction
python scripts/run_sequential.py --dataset fd --domain_order 3,2,1,0 --calib_frac 0.05
```

### 3. Run baselines for comparison

```bash
cd experiments
python run_unified_eval.py            # FD + HHAR
python run_sleep_unified_eval.py      # Sleep-EDF
python compute_cl_metrics.py          # Aggregate AA / BWT / F / FWT
```

### 4. Run ablations

```bash
# Backbone, LoRA, K, label-budget, and temperature sweeps
bash scripts/run_ablations.sh fd

# Differential privacy sweep
cd experiments
python run_sequential_soft_gate_dp.py --dataset fd --sweep_epsilons "0.1,0.5,1,2,5,10,inf"
```

### 5. Foundation model backbones

```bash
# Run PACT with MOMENT, Sundial, Chronos, Moirai, or Time-MoE
python scripts/run_sequential_tsfm.py --backbone moment --dataset fd
```

## Method Details

PACT operates in three stages for each new operating condition:

1. **Chart partitioning** -- K-means clusters the source embeddings into K local regions (charts), computed once
2. **Adapter training** -- For each chart, a lightweight residual adapter is trained on the new condition's calibration set (default 5% labels), optimized through the frozen classifier head
3. **Gate calibration** -- A per-chart soft-mixture gate learns to route test samples across all accumulated adapters, retaining an identity path to the source representation

Key hyperparameters (defaults in `PactConfig`):

| Parameter | Default | Description |
|---|---|---|
| `k` | 16 | Number of charts (K-means clusters) |
| `calib_frac` | 0.05 | Fraction of labeled data for calibration |
| `adapter_ep` | 200 | Adapter training epochs |
| `gate_ep` | 200 | Gate training epochs |
| `temperature` | 0.5 | Softmax temperature for gating |

## Citation

If you use PACT in your research, please cite:

```bibtex
@inproceedings{pact2027,
  title={Forgetting-Free Domain Incremental Adaptation of Time-Series Foundation Models via Piecewise Chart Transport},
  author={Anonymous},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2027}
}
```

## License

This project is released under the MIT License.
