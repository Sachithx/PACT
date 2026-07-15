"""Motivation check: is a frozen Time-Series Foundation Model already domain-robust?

Backbone: MOMENT-1-large (AutonLab), used as a FROZEN encoder (embedding mode).
Protocol per AdaTime source->target shift:
  1. Extract frozen-MOMENT embeddings (1024-d) for source-train / source-test /
     target-test.  (MOMENT expects length-512 inputs -> we resample.)
  2. Fit a LINEAR PROBE (logistic regression) on source-train embeddings only
     -- i.e. "load pretrained TSFM, fit on source domain, freeze everything".
  3. Evaluate with NO adaptation:
        - source-test  = in-distribution ceiling for the frozen FM,
        - target-test  = cross-domain generalisation of the frozen FM.

The source->target *drop* answers the key question: if it is large, adaptation
is clearly needed (motivation holds); if it is small, the FM is already robust
and the paper's headline should pivot to forgetting-free continual accumulation.

Reference (TSLANet single-source backbone, FD): source ~0.998, target ~0.444.

Usage:
  python scripts/tsfm_motivation.py --dataset fd
"""

import argparse
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

import _bootstrap  # noqa: F401
from _bootstrap import DATA_ROOT, RESULTS_ROOT

from src.data import get_spec
from src.data.datasets import SCENARIOS
from src.utils import get_device, set_seed

MOMENT_LEN = 512
REF = {"fd": (0.998, 0.444)}  # (source, target) for TSLANet single-source backbone


def load_moment(device):
    from momentfm import MOMENTPipeline
    m = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large", model_kwargs={"task_name": "embedding"})
    m.init()
    m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@torch.no_grad()
def embed_domain(model, spec, domain_id, split, data_root, device, batch=64):
    """Frozen-MOMENT embeddings for one (domain, split). Returns (N,1024), (N,)."""
    ds = spec.get_dataset(domain_id, split, data_root)
    ld = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False)
    embs, ys = [], []
    for x, y in ld:
        x = x.float().to(device)                       # [B, C, L]
        if x.shape[-1] != MOMENT_LEN:                  # resample to MOMENT context
            x = F.interpolate(x, size=MOMENT_LEN, mode="linear", align_corners=False)
        out = model(x_enc=x)
        embs.append(out.embeddings.float().cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(embs), np.concatenate(ys)


def linear_probe(Xtr, ytr, Xte, seed):
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0, multi_class="multinomial",
                             random_state=seed)
    clf.fit(scaler.transform(Xtr), ytr)
    return clf.predict(scaler.transform(Xte))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["fd", "hhar", "sleep"], default="fd")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    spec = get_spec(args.dataset)
    scen = SCENARIOS[args.dataset]

    print(f"Device={device}  backbone=MOMENT-1-large (frozen)  dataset={args.dataset}")
    t0 = time.time()
    model = load_moment(device)
    print(f"MOMENT loaded in {time.time()-t0:.1f}s\n")

    # Cache embeddings per (domain, split) since AdaTime scenarios reuse domains.
    cache = {}
    def get(domain_id, split):
        key = (domain_id, split)
        if key not in cache:
            cache[key] = embed_domain(model, spec, domain_id, split,
                                      DATA_ROOT, device, args.batch)
        return cache[key]

    print(f"{'='*74}\nFROZEN MOMENT + LINEAR PROBE (no adaptation) — {args.dataset.upper()}"
          f"  ({len(scen)} AdaTime shifts)\n{'='*74}")
    print(f"  {'scenario':<16}{'src acc':>9}{'src F1':>9}{'tgt acc':>9}{'tgt F1':>9}{'drop(acc)':>11}")
    print(f"  {'-'*62}")

    rows = []
    for (src, tgt) in scen:
        Xtr, ytr = get(src, "train")
        Xste, yste = get(src, "test")
        Xtte, ytte = get(tgt, "test")

        p_src = linear_probe(Xtr, ytr, Xste, args.seed)
        p_tgt = linear_probe(Xtr, ytr, Xtte, args.seed)

        sa = accuracy_score(yste, p_src); sf = f1_score(yste, p_src, average="macro", zero_division=0)
        ta = accuracy_score(ytte, p_tgt); tf = f1_score(ytte, p_tgt, average="macro", zero_division=0)
        rows.append((sa, sf, ta, tf))
        name = f"{spec.domain_label(src)}->{spec.domain_label(tgt)}"
        print(f"  {name:<16}{sa:>9.3f}{sf:>9.3f}{ta:>9.3f}{tf:>9.3f}{sa-ta:>+11.3f}")

    arr = np.array(rows)
    msa, msf, mta, mtf = arr.mean(0)
    print(f"  {'-'*62}")
    print(f"  {'MEAN':<16}{msa:>9.3f}{msf:>9.3f}{mta:>9.3f}{mtf:>9.3f}{msa-mta:>+11.3f}")

    print(f"\n{'-'*74}")
    print("INTERPRETATION")
    if args.dataset in REF:
        rs, rt = REF[args.dataset]
        print(f"  TSLANet single-source backbone : source={rs:.3f}  target={rt:.3f}  "
              f"drop={rs-rt:+.3f}")
    print(f"  Frozen MOMENT + linear probe   : source={msa:.3f}  target={mta:.3f}  "
          f"drop={msa-mta:+.3f}")
    drop = msa - mta
    if drop >= 0.15:
        print(f"  => Large cross-domain drop ({drop:.3f}). Frozen FM is NOT domain-robust;")
        print(f"     per-domain adaptation is clearly motivated. Headline can stay 'recover")
        print(f"     target accuracy without forgetting'.")
    else:
        print(f"  => Small cross-domain drop ({drop:.3f}). Frozen FM is fairly robust;")
        print(f"     pivot headline to 'forgetting-free continual accumulation over many")
        print(f"     domains without per-domain fine-tuning'.")

    out_dir = RESULTS_ROOT / "tsfm_motivation"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / f"{args.dataset}_moment_linprobe.csv", arr,
               delimiter=",", header="src_acc,src_f1,tgt_acc,tgt_f1", comments="")
    print(f"\nSaved -> {out_dir}/{args.dataset}_moment_linprobe.csv")
    print(f"Elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
