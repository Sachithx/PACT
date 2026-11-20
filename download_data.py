"""
download_data.py — fetch the five PACT benchmarks from the HuggingFace Hub.

The processed train/test splits are too large for git, so they live at
    https://huggingface.co/datasets/sachithabey/PACT

The frozen Sundial encoder (thuml/sundial-base-128m) is not in that dataset.
``run_main.py`` downloads it from the Hub on first use.

Usage:
    pip install huggingface_hub
    python download_data.py                         # all five benchmarks
    python download_data.py --datasets FD HHAR      # a subset
"""

import argparse
import os

HF_REPO = "sachithabey/PACT"
HF_REPO_TYPE = "dataset"

DATASETS = ["FD", "WISDM", "HHAR", "UCIHAR", "Sleep-EDF"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=HF_REPO, help="HuggingFace dataset repo id")
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS,
                    help="which benchmarks to fetch (default: all)")
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("Please `pip install huggingface_hub` first.")

    allow = [f"dataset/{name}/*" for name in args.datasets]
    print(f"Downloading {args.datasets} from {args.repo} …")
    snapshot_download(repo_id=args.repo, repo_type=HF_REPO_TYPE,
                      local_dir=".", allow_patterns=allow)
    for name in args.datasets:
        print(f"    -> dataset/{name}/")
    print("Done. You can now run `python run_main.py`.")


if __name__ == "__main__":
    main()
