"""Train (or load) a source backbone on one or more domains.

Examples:
  python scripts/train_backbone.py --backbone tslanet --dataset fd --domain 0
  python scripts/train_backbone.py --backbone timesnet --dataset hhar --domain all
  python scripts/train_backbone.py --backbone patchtst --dataset sleep --domain 0,3,5
"""

import argparse

import _bootstrap  # noqa: F401  (sets sys.path)
from _bootstrap import CKPT_ROOT, DATA_ROOT

from src.data import get_spec
from src.models.factory import BACKBONES
from src.training import train_or_load_source
from src.utils import get_device, set_seed


def parse_domains(arg, n_domains):
    if arg == "all":
        return list(range(n_domains))
    return [int(x) for x in arg.split(",")]


def main():
    p = argparse.ArgumentParser(description="Train/load a source backbone.")
    p.add_argument("--backbone", choices=BACKBONES, default="tslanet")
    p.add_argument("--dataset", choices=["fd", "hhar", "sleep"], default="fd")
    p.add_argument("--domain", default="0",
                   help="'all', a single id, or comma-separated ids")
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--no_pretrain", action="store_true",
                   help="Skip TSLANet MAE pretraining")
    p.add_argument("--pretrain_epochs", type=int, default=50)
    p.add_argument("--cls_epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    spec = get_spec(args.dataset)
    domains = parse_domains(args.domain, spec.n_domains)
    print(f"Device={device}  backbone={args.backbone}  dataset={args.dataset}  "
          f"domains={domains}")

    for did in domains:
        train_or_load_source(
            args.backbone, args.dataset, did, CKPT_ROOT, DATA_ROOT, device,
            retrain=args.retrain, pretrain=not args.no_pretrain,
            pretrain_epochs=args.pretrain_epochs, cls_epochs=args.cls_epochs,
            seed=args.seed)


if __name__ == "__main__":
    main()
