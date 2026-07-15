# EvoConcept

A **gating method + adapter plugin** that lets a *frozen* state-of-the-art
time-series classifier adapt to new domains **sequentially**, retaining
performance on old domains (no catastrophic forgetting) while reaching strong
performance on the new domain.

Everything operates in the frozen backbone's pre-head embedding space, so the
host classifier is **never** modified — forgetting is zero by construction.

## Method in one paragraph

Given a SOTA backbone (TSLANet / PatchTST / TimesNet) trained once on a source
domain and then frozen, we (1) cluster the source embeddings into `K` **concepts**
(K-means); (2) for each new domain, train a small **per-concept adapter bank** on a
tiny labelled calibration slice (default 5% labels), correcting that domain's
embeddings so the frozen head classifies them correctly — and then freeze it; and
(3) train a per-concept **soft-mixture gate** that routes each sample by mixing all
domains' adapter outputs:

```
ê = Σ_d  α_d(e, c) · A_{d,c}(e),   α = softmax(g_c(e) / T),   A_{0,c} = identity (source)
```

The source "adapter" is the identity, so source-like samples are routed back to
the untouched embedding (preserving old performance) while shifted samples get
their correction. Only the gate is retrained as domains accumulate.

## Repository layout

```
evoconcept_clean/
├── src/                      # clean, refactored reference implementation
│   ├── models/               # tslanet, patchtst, timesnet, adapters, gate, concepts, factory
│   ├── data/                 # datasets registry (FD/HHAR/Sleep) + AdaTime SCENARIOS + preprocess/
│   ├── method/               # pact (adapters+gate+inference), sequential (CL loop+metrics), dp
│   ├── training.py           # source-backbone training / loading
│   └── utils/                # seeding, device, masking
├── scripts/                  # clean entry points that use src/
│   ├── train_backbone.py     # train/load a source backbone
│   ├── run_sequential.py     # MAIN: sequential domain-incremental CL with PACT
│   └── run_ablations.sh      # backbone / LoRA / K / label-budget / temperature sweeps
├── experiments/              # faithful ORIGINAL drivers for exact paper reproduction
│   ├── run_unified_eval.py           # PACT vs all baselines (FD + HHAR)
│   ├── run_sleep_unified_eval.py     # PACT vs all baselines (Sleep-EDF)
│   ├── run_semisup_only.py           # semi-supervised DA baselines (DANN†/CDAN†/CoTMix†)
│   ├── run_ortcl_tslanet.py          # ORTCL continual-learning baseline
│   ├── run_sequential_soft_gate*.py  # sequential CL (+ DP ablation)
│   ├── run_sequential_pact_lora.py   # LoRA-adapter ablation
│   ├── run_exp8_rf_seqpact.py        # rehearsal-free ablation
│   ├── compute_cl_metrics.py         # AA/BWT/F/FWT aggregation from result files
│   └── (transfer_*.py, exp7_soft_gate.py, dp_utils.py = shared deps)
├── baselines/  -> ../baselines        # third-party baseline repos (AdaTime, ORTCL, DualCD, …)
├── dataset/    -> ../dataset          # symlink to the data
├── checkpoints/{tslanet,patchtst}     # symlinks to pre-trained backbones
├── requirements.txt
└── README.md
```

`src/` is the clean library you read and extend for the paper. `experiments/`
holds the original driver scripts (kept verbatim so the baseline/ablation
numbers reproduce exactly); they run unchanged thanks to the symlinks above.

## Datasets and evaluation protocol

| Dataset    | Domains | Classes | Seq len | Channels |
|------------|---------|---------|---------|----------|
| FD         | 4 (A–D) | 3       | 5120    | 1        |
| HHAR       | 9 users | 6       | 128     | 3        |
| Sleep-EDF  | 20 subj | 5       | 3000    | 1        |

Two evaluation protocols, both using the **AdaTime** benchmark's selected shifts
(`src.data.SCENARIOS`, the single source of truth — 10 src→tgt cases per dataset):

1. **Baseline comparison** — the pairwise AdaTime scenarios, iterated by
   `experiments/run_unified_eval.py` (PACT vs UDA / TTA / semi-sup DA / CL).
2. **Continual-learning / forgetting** — a multi-domain *ordering* (e.g. FD
   `0→1→2→3`), run by `scripts/run_sequential.py`, producing AA/BWT/F/RF/PRF.

## Environment

```bash
conda create -n entrope python=3.10
conda activate entrope
pip install -r requirements.txt
```

> **Note on `libstdc++`:** this environment needs the conda `libstdc++` preloaded,
> otherwise `torch.optim` / `optree` fail with a `GLIBCXX` error. Prefix commands with:
> ```bash
> LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6 python ...
> ```

## Quick start

```bash
# 0. (optional) train a source backbone — FD/HHAR/Sleep TSLANet & PatchTST are pre-trained
python scripts/train_backbone.py --backbone tslanet --dataset fd --domain 0

# 1. MAIN: sequential domain-incremental continual learning with PACT
python scripts/run_sequential.py --dataset fd                     # TSLANet, FD 0→1→2→3
python scripts/run_sequential.py --dataset hhar --backbone patchtst

# 2. Baseline comparison table (from experiments/, which uses AdaTime scenarios)
cd experiments && python run_unified_eval.py            # FD + HHAR
                  python run_sleep_unified_eval.py      # Sleep-EDF
                  python compute_cl_metrics.py          # AA/BWT/F/FWT table

# 3. Ablations
bash scripts/run_ablations.sh fd                        # backbone/LoRA/K/label-budget/temperature
cd experiments && python run_sequential_soft_gate_dp.py --dataset fd --sweep_epsilons "0.1,0.5,1,2,5,10,inf"
```

## Backbones

`tslanet` and `patchtst` ship with pre-trained checkpoints for all three
datasets. **TimesNet** is a self-contained, trainable scaffold conforming to the
same backbone contract (`extract() -> [B, emb_dim]`), but **no TimesNet
checkpoints ship yet** — train one first with `scripts/train_backbone.py
--backbone timesnet ...`.
