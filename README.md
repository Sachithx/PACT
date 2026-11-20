# PACT: Recurrent Domain Incremental Adaptation of Time-Series Foundation Models

[![arXiv](https://img.shields.io/badge/arXiv-2611.24075-b31b1b?logo=arxiv)](https://arxiv.org/abs/2611.24075)
[![HF Paper](https://img.shields.io/badge/HF%20Paper-2611.24075-FFD21E.svg?logo=huggingface)](https://huggingface.co/papers/2611.24075)
[![HF Dataset](https://img.shields.io/badge/HuggingFace-Datasets-FFD21E.svg?logo=huggingface)](https://huggingface.co/datasets/sachithabey/PACT)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)

Piecewise Adaptation via Chart Transport (PACT) adapts a frozen time-series foundation model as operating conditions arrive one at a time. The encoder and the source classifier stay fixed. Each new condition adds chart-local residual transports, and a learned gate mixes those transports with an explicit identity path. Condition identity is not available at test time.

This repository reproduces the **PACT** results on the frozen Sundial backbone: FD, WISDM, HHAR, UCI-HAR, and Sleep-EDF, four streams each, seed 42.

## 1. Setup

```bash
conda create -n pact python=3.10 -y
conda activate pact
pip install -r requirements.txt
```

Tested with Python 3.10 and a single GPU. Sundial (`thuml/sundial-base-128m`) is loaded with `trust_remote_code`, which needs `transformers==4.40.x` (pinned above). A newer `transformers` builds NaN embeddings for this checkpoint.

If `import torch` raises `GLIBCXX_3.4.xx not found`, preload the conda libstdc++:

```bash
LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6 python run_main.py --dataset fd
```

## 2. Data and the backbone checkpoint

The five processed benchmarks are hosted on the HuggingFace Hub (not in git):
**https://huggingface.co/datasets/sachithabey/PACT**

```bash
python download_data.py                         # all five
python download_data.py --datasets FD Sleep-EDF
```

This writes the paths `run_main.py` expects:

```
dataset/FD/train_{0..3}.pt          dataset/FD/test_{0..3}.pt
dataset/WISDM/train_{0..35}.pt      dataset/WISDM/test_{0..35}.pt
dataset/HHAR/train_{0..8}.pt        dataset/HHAR/test_{0..8}.pt
dataset/UCIHAR/train_{1..30}.pt     dataset/UCIHAR/test_{1..30}.pt
dataset/Sleep-EDF/train_{0..19}.pt  dataset/Sleep-EDF/test_{0..19}.pt
```

Each file is a dict `{"samples": ..., "labels": ...}`.

The frozen encoder checkpoint is **`thuml/sundial-base-128m`**. It is downloaded from the Hub on the first run and then cached by `transformers`. PACT does not ship a second copy of those weights: the transports and the gate are fit by `run_main.py` on the calibration split.

## 3. Reproduce the main results

From the repo root, after the download:

```bash
python run_main.py --dataset fd          # one benchmark, four streams
python run_main.py --dataset all         # FD, WISDM, HHAR, UCI-HAR, Sleep-EDF
python run_main.py --dataset hhar --orders 6,0,8,2,5,1,3,4,7
```

`run_fd.py` is the same FD command (`python run_fd.py`).

One Sundial load is shared only within a stream: each stream refits the source head on that stream's source domain, then walks the remaining domains. Metrics are macro-F1 on the held-out test split of every domain seen so far, with no condition id at inference. After the last domain the script prints

- **AMF1** — mean macro-F1 over domains seen so far
- **RF** — mean relative drop from each earlier domain's peak
- **PRF** — the same drop weighted by `(1 − final)`

The `±` is the standard deviation across the four streams (population std, seed 42). Per-stream matrices go to `results/<dataset>/`.

| Dataset | Charts K | Calibration | AMF1 | RF | PRF |
|---|---:|---|---:|---:|---:|
| FD | 8 | 5% | 0.9575 ± 0.0106 | 0.0021 | 0.0001 |
| WISDM | 4 | 450 labels | 0.8774 ± 0.0197 | 0.0292 | 0.0085 |
| HHAR | 4 | 5% | 0.8367 ± 0.0214 | 0.0210 | 0.0036 |
| UCI-HAR | 4 | 100 labels | 0.7864 ± 0.0144 | 0.0425 | 0.0101 |
| Sleep-EDF | 1 | 5% | 0.6129 ± 0.0316 | 0.0171 | 0.0100 |

Streams (first domain is the source) match the paper:

```
FD         (0, 2, 3, 1)  (2, 0, 1, 3)  (2, 1, 0, 3)  (2, 1, 3, 0)
WISDM      (10, 24, 20, 0, 3, 1, 22, 14, 2)  (12, 23, 7, 9, 5, 15, 31, 28, 10)
           (24, 11, 0, 23, 13, 21, 16, 35, 3)  (30, 3, 0, 31, 20, 24, 19, 13, 17)
HHAR       (1, 6, 4, 7, 8, 5, 0, 2, 3)  (5, 1, 6, 3, 2, 7, 4, 0, 8)
           (5, 8, 0, 3, 2, 6, 4, 1, 7)  (6, 0, 8, 2, 5, 1, 3, 4, 7)
UCI-HAR    (14, 24, 1, 27, 26, 10, 12, 7, 9)  (19, 14, 6, 12, 4, 29, 9, 18, 21)
           (24, 14, 19, 17, 6, 5, 7, 3, 30)  (6, 20, 18, 30, 12, 24, 25, 3, 17)
Sleep-EDF  (0, 5, 18, 9, 3, 10, 6, 1, 14)  (1, 5, 9, 0, 4, 3, 18, 8, 13)
           (15, 6, 17, 5, 8, 9, 18, 19, 2)  (18, 5, 12, 7, 10, 17, 0, 3, 1)
```

Fixed settings (argparse defaults in `run_main.py` / `PactConfig`):

| Setting | Value |
|---|---|
| Backbone | Sundial-base-128m, frozen. Linear head fit on the source domain (150 epochs), then frozen |
| Transports | LoRA rank 8, zero residual at init, 200 epochs, lr 1e-3, penalty 1e-3 on ‖B‖² |
| Gate | per-chart MLP (hidden 64, dropout 0.2), 200 epochs, lr 1e-3, temperature 0.5 |
| Charts | k-means on source train embeddings, 10 restarts, then frozen |
| Mask | a target branch is hidden in a chart with fewer than 5 calibration embeddings |
| Seed | 42 |

Small deviations in the third decimal can occur across GPUs because a few CUDA kernels are non-deterministic. Embedding extraction dominates the runtime; Sleep-EDF and FD are the long series (3000 and 5120 steps) and are split into 512-step windows before mean pooling. Series shorter than 512 are linearly resampled to the Sundial context.

## 4. Repository layout

```
run_main.py          entry point for the five benchmarks
run_fd.py            FD-only shortcut
download_data.py     fetch dataset/ from the HuggingFace Hub
pact/backbone.py     frozen Sundial encoder and the linear source head
pact/method.py       charts, transports, gate, metrics
pact/data.py         loaders and the stratified calibration split
dataset/             downloaded benchmarks (not in git)
results/             per-stream metrics (not in git)
```

`pact/method.py` is the file to read for the method. Its top docstring maps each step onto the paper.

## Citing

If you found this work useful for you, please consider citing it.

```bibtex
@misc{abeywickrama2026pact,
  title={Recurrent Domain Incremental Adaptation of Time-Series Foundation Models},
  author={Sachith Abeywickrama and Peiliang Gong and Emadeldeen Eldele and Min Wu and Chau Yuen},
  year={2026},
  eprint={2611.24075},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2611.24075}
}
```

Released under the Apache License 2.0 (see [LICENSE](LICENSE)).
