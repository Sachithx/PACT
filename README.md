# PACT — Prototype-Aware Concept Transformation

A lightweight, backbone-agnostic plugin for post-hoc time-series distribution shift correction.

PACT attaches a bank of K concept-specific adapter networks and learned gate MLPs to a **frozen** pre-trained model. At inference, each input embedding is routed through its nearest concept prototype and the gate decides whether to apply the adapter. The source-domain weights are never modified — PACT adds no forgetting.

---

## Key properties

| Property | Value |
|---|---|
| Parameters added | K × (d² + d) + K × (2 · d\_hidden + 2) |
| Backbone modification | None (frozen) |
| Source forgetting | ~0 (gate bypasses source-like inputs) |
| Label budget | 5% stratified target train |
| Training time | ~2 min on a single GPU (K=4, 5% calib) |

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Quick start

**Step 0 — train a source backbone** (skip if you have pre-trained checkpoints):

```bash
# TSLANet (all 9 HHAR users, ~5 min/user on GPU)
python train_tslanet_hhar.py

# TimesNet (all 9 HHAR users)
python train_timesnet_hhar.py
```

Checkpoints are saved to `checkpoints/tslanet_hhar/` and `checkpoints/timesnet_hhar/`.

```python
import torch
from pact import PACT
from backbone.tslanet import load_tslanet
from datasets import HHARDataset
from torch.utils.data import DataLoader

device = torch.device("cuda")

# 1. Load frozen source backbone
backbone = load_tslanet("checkpoints/tslanet_hhar/tslanet_hhar_u0_ps8_d2_dim128.pt", dataset="hhar", device=device)

# 2. Load data
src_train = HHARDataset(user_id=0, split="train", data_root="data")
src_test  = HHARDataset(user_id=0, split="test",  data_root="data")
tgt_train = HHARDataset(user_id=6, split="train", data_root="data")
tgt_test  = HHARDataset(user_id=6, split="test",  data_root="data")

# 3. Create calibration set (5% of target train, stratified)
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
labels = torch.stack([tgt_train[i][1] for i in range(len(tgt_train))]).numpy()
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.95, random_state=42)
idx, _ = next(sss.split(np.zeros(len(labels)), labels))
calib_ds = torch.utils.data.Subset(tgt_train, idx)

# 4. Create PACT and fit
pact = PACT(backbone=backbone, k=4, emb_dim=128, device=device)
pact.fit(
    src_loader=DataLoader(src_train, batch_size=256),
    calib_loader=DataLoader(calib_ds, batch_size=256),
)

# 5. Evaluate
f1_src = pact.evaluate(DataLoader(src_test, batch_size=256))
f1_tgt = pact.evaluate(DataLoader(tgt_test, batch_size=256))
print(f"Source F1 (retention): {f1_src:.4f}")
print(f"Target F1 (recovery):  {f1_tgt:.4f}")
```

---

## Command-line usage

```bash
# HHAR cross-user: User-a → User-g  (TSLANet backbone)
python run_pact.py \
  --dataset hhar --src 0 --tgt 6 --data_root . \
  --backbone_type tslanet \
  --backbone_ckpt checkpoints/tslanet_hhar/tslanet_hhar_u0_ps8_d2_dim128.pt \
  --k 4 --calib_frac 0.05

# Same scenario with TimesNet backbone
python run_pact.py \
  --dataset hhar --src 0 --tgt 6 --data_root . \
  --backbone_type timesnet \
  --backbone_ckpt checkpoints/timesnet_hhar/timesnet_hhar_u0_dm128_e2.pt \
  --k 4 --calib_frac 0.05

# FD fault diagnosis: FD-A → FD-B
python run_pact.py \
  --dataset fd --src FD-A --tgt FD-B --data_root data \
  --backbone_ckpt checkpoints/tslanet_hhar/tslanet_fd_FD-A_ps8_d2_dim128.pt

# Sleep-EDF: subject 0 → subject 11
python run_pact.py \
  --dataset sleep --src 0 --tgt 11 --data_root data \
  --backbone_ckpt checkpoints/tslanet_sleep/tslanet_sleep_u0_ps8_d2_dim128.pt
```

---

## Bring your own backbone

PACT works with any PyTorch model that satisfies three requirements:

```python
class MyBackbone(nn.Module):
    def extract(self, x: Tensor) -> Tensor:
        """Input batch → embedding [B, emb_dim]"""
        ...

    def forward(self, x: Tensor) -> Tensor:
        """Input batch → logits [B, n_classes]"""
        ...

    @property
    def head(self) -> nn.Linear:
        """The final classification layer (used for adapter supervision)"""
        ...
```

Then:
```python
pact = PACT(backbone=my_frozen_model, k=4, emb_dim=<your_dim>, device=device)
pact.fit(src_loader, calib_loader)
```

---

## Supported datasets

| Dataset | Domains | Classes | Input | Seq len |
|---|---|---|---|---|
| FD (Fault Diagnosis) | FD-A/B/C/D | 3 | 1-ch vibration | 5120 |
| HHAR | User 0–8 | 6 | 3-ch accelerometer | 128 |
| Sleep-EDF | Subject 0–19 | 5 | 1-ch EEG | 3000 |

Data files follow the [AdaTime](https://github.com/emadeldeen24/AdaTime) preprocessing format.

---

## Data split discipline

- **Source backbone**: pretrained on source-domain **train** split, then frozen.
- **PACT concepts**: built from source-train embeddings (k-means).
- **PACT calibration**: 5% stratified slice of **target-domain train**.
- **Evaluation**: source **test** (retention) and target **test** (drift recovery).
- Calibration and test splits are disjoint by construction.

---

## Repository layout

```
PACT/
├── pact/
│   ├── core.py              # Main PACT class, IdentityAdapter, GateMLP
│   └── kmeans.py            # Numpy k-means
├── datasets/
│   ├── fd.py                # FD dataset loader
│   ├── hhar.py              # HHAR dataset loader
│   └── sleep_edf.py         # Sleep-EDF dataset loader
├── backbone/
│   ├── tslanet.py           # TSLANet backbone + checkpoint loader
│   └── timesnet.py          # TimesNet backbone + checkpoint loader
├── Time-Series-Library/     # Cloned TSLib (TimesNet layers)
├── train_tslanet_hhar.py    # Train TSLANet source backbones on HHAR
├── train_timesnet_hhar.py   # Train TimesNet source backbones on HHAR
├── run_pact.py              # End-to-end eval (--backbone_type tslanet|timesnet)
└── requirements.txt
```

---

## Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `k` | 4 | Number of concept prototypes. Recommended: `k ≤ calib_size / 50`. |
| `penalty` | 1e-3 | Identity regularisation (keeps adapters near I). |
| `adapter_epochs` | 200 | Adapter training epochs. |
| `gate_epochs` | 200 | Gate MLP training epochs per concept. |
| `calib_frac` | 0.05 | Fraction of target train used for calibration. |

---

## Citation

```bibtex
@article{pact2025,
  title   = {PACT: Prototype-Aware Concept Transformation for Post-Hoc Distribution Shift Correction},
  author  = {[authors]},
  year    = {2025}
}
```
