# PACT: minimal code to reproduce the FD results

This folder contains only the code needed to reproduce the **FD row of PACT**
in the main comparison table (frozen Sundial backbone, 4 domain orders):

| | AMF1 (↑) | PRF (↓) |
|---|---|---|
| PACT, FD | 0.958 ± 0.011 | 0.000 ± 0.000 |

## Files

```
run_fd.py          entry point: fits the source head, runs PACT on the 4 FD domain orders, reports mean ± std
pact/backbone.py   frozen Sundial encoder (multi-window pooling + source standardization) and linear source head
pact/method.py     PACT: k-means charts, chart-local LoRA transports, per-chart soft gate, soft-mixture inference, CL metrics
pact/data.py       FD dataset loader and the stratified 5% calibration split
```

`pact/method.py` is the file to read for the method. Its top docstring maps each
step onto the paper.

## Setup

```bash
pip install -r requirements.txt
```

Use a GPU. The Sundial checkpoint (`thuml/sundial-base-128m`, ~128M parameters)
is downloaded from the Hugging Face Hub on first use. Sundial needs
`transformers==4.40.x` because it is loaded with `trust_remote_code`.

## Data

We use the preprocessed FD (Paderborn bearing) data from the
[AdaTime benchmark](https://github.com/emadeldeen24/AdaTime): 4 operating
conditions (domains 0–3 = FD-A…FD-D), 3 classes, length 5120, 1 channel.
Put the files in one folder:

```
FD/
  train_0.pt  train_1.pt  train_2.pt  train_3.pt
  test_0.pt   test_1.pt   test_2.pt   test_3.pt
```

Each file is a dict with `samples` (N × 5120) and `labels` (N).

## Reproduce the FD row

```bash
python run_fd.py --data_dir /path/to/FD
```

This runs the 4 domain orders used in the paper (first domain = source):

```
FD-C -> FD-B -> FD-A -> FD-D
FD-C -> FD-B -> FD-D -> FD-A
FD-C -> FD-A -> FD-B -> FD-D
FD-A -> FD-C -> FD-D -> FD-B
```

It prints per-step metrics, then the mean ± std over the 4 orders. Per-order
metrics and macro-F1 retention matrices go to `results/`. A single order can be
run with `--orders 2,1,0,3`.

Settings, all defaults in `run_fd.py` / `PactConfig`:

| Setting | Value |
|---|---|
| Backbone | Sundial-base-128m, frozen; linear head fitted on the source domain, then frozen |
| Charts K | 8 (k-means on source train embeddings, 10 restarts) |
| Transports | LoRA rank 8, identity init, 200 epochs, lr 1e-3, penalty 1e-3 on ‖BA‖² |
| Gate | per-chart MLP (hidden 64), 200 epochs, lr 1e-3, temperature 0.5 |
| Labels | stratified 5% of each target domain's train split |
| Seed | 42 |

AMF1 is the average macro-F1 over all domains after the last domain. PRF is the
performance-weighted relative forgetting, (1 − final) × max(0, peak − final)/peak,
averaged over earlier domains. The ± is the (population) std over the 4 orders.
The source backbone and head are never updated.

## Expected output

Per-order final AMF1 from our runs:

| Order | AMF1 | PRF |
|---|---|---|
| 2,1,0,3 | 0.9614 | 0.00002 |
| 2,1,3,0 | 0.9647 | 0.00003 |
| 2,0,1,3 | 0.9647 | 0.00006 |
| 0,2,3,1 | 0.9393 | 0.00029 |
| **mean ± std** | **0.958 ± 0.011** | **0.0001 ± 0.0001** |

Small deviations (third decimal) can occur on different GPUs or library
versions because of non-deterministic CUDA kernels.
