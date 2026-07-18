"""Profile adaptation cost, inference latency, peak memory, and buffer size.

Saves:
  results/cost_profile/latency_hardware.csv
  results/tables/latency_supplement.tex  (snippet for appendix)

Example:
  python scripts/profile_latency.py --device cuda
"""

import argparse
import json
import platform
import subprocess
import time

import numpy as np
import pandas as pd
import torch

import _bootstrap  # noqa: F401
from _bootstrap import DATA_ROOT, RESULTS_ROOT

from src.data import get_spec
from src.data.calib import calib_indices
from src.data.datasets import adatime_chain
from src.method import PactConfig, run_sequential
from src.method.pact import extract_embs
from src.method.sequential import predict_soft_mixture
from src.models.tsfm import make_tsfm
from src.utils import get_device, set_seed
from run_sequential_tsfm import fit_standardizer_and_head


def gpu_name():
    if not torch.cuda.is_available():
        return platform.processor() or "CPU"
    return torch.cuda.get_device_name(0)


def peak_mem_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return float("nan")


def buffer_mb(p, n_src, n_calib_total):
    return p * (n_src + n_calib_total) * 4 / (1024 ** 2)


def profile_pact(model, spec, order, device, cfg, temperature=0.5):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_load = time.perf_counter()

    src_ld = torch.utils.data.DataLoader(
        spec.get_dataset(order[0], "train", DATA_ROOT), batch_size=cfg.batch)
    src_embs, _ = extract_embs(model, src_ld, device)
    n_src = len(src_embs)
    n_calib = 0
    domain_times = []

    from src.models.concepts import kmeans_numpy
    from src.method.sequential import fit_gates
    from src.method.pact import train_adapters
    from src.models.concepts import assign_concepts

    centroids, src_asgn = kmeans_numpy(src_embs, cfg.k, seed=cfg.seed,
                                       n_init=cfg.kmeans_n_init)
    accumulated, adapters_list = [], []
    soft_gates = None

    for t in range(1, len(order)):
        t_step = time.perf_counter()
        ds = spec.get_dataset(order[t], "train", DATA_ROOT)
        y_all = ds.y.numpy()
        idx = calib_indices(y_all, cfg.calib_frac, cfg.seed)
        n_calib += len(idx)
        calib_ld = torch.utils.data.DataLoader(
            torch.utils.data.Subset(ds, idx), batch_size=cfg.batch)
        calib_embs, calib_lbls = extract_embs(model, calib_ld, device)
        calib_asgn = assign_concepts(calib_embs, centroids)
        accumulated.append((calib_embs, calib_asgn))
        adapters = train_adapters(calib_embs, calib_lbls, centroids, model.head,
                                  device, cfg)
        adapters_list.append(adapters)
        soft_gates = fit_gates(src_embs, src_asgn, accumulated, device, cfg)
        domain_times.append(time.perf_counter() - t_step)

    # inference latency on one test batch
    test_ld = torch.utils.data.DataLoader(
        spec.get_dataset(order[-1], "test", DATA_ROOT), batch_size=1)
    xb, _ = next(iter(test_ld))
    xb = xb.to(device)
    with torch.no_grad():
        emb = model.extract(xb).cpu().numpy()
    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        predict_soft_mixture(emb, centroids, soft_gates, adapters_list,
                             model.head, device, temperature, cfg.k)
        latencies.append((time.perf_counter() - t0) * 1000)

    return dict(
        gpu=gpu_name(),
        calib_per_domain_s=float(np.mean(domain_times)) if domain_times else 0.0,
        infer_ms_per_sample=float(np.mean(latencies)),
        peak_mem_mb=peak_mem_mb(),
        buffer_mb=buffer_mb(model.emb_dim, n_src, n_calib),
        total_after_load_s=time.perf_counter() - t_load,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    spec = get_spec("fd")
    order, _ = adatime_chain("fd")
    batch = 64

    rows = []
    for adapter_kind in ("lora", "identity"):
        set_seed(args.seed)
        model = make_tsfm("moment", spec.num_classes, device=str(device)).to(device)
        fit_standardizer_and_head(model, spec, order[0], device,
                                  150, 1e-3, batch, args.seed)
        cfg = PactConfig(k=16, emb_dim=model.emb_dim, calib_frac=0.05,
                         seed=args.seed, batch=batch, adapter_kind=adapter_kind)
        r = profile_pact(model, spec, order, device, cfg)
        rows.append(dict(dataset="fd", tsfm="moment", adapter=adapter_kind, **r))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    out_dir = RESULTS_ROOT / "cost_profile"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "latency_hardware.csv"
    df.to_csv(csv_path, index=False)

    tex_path = RESULTS_ROOT / "tables" / "latency_supplement.tex"
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tex_path, "w") as f:
        f.write("% auto-generated by scripts/profile_latency.py\n")
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Latency, memory, and embedding-buffer cost on FD with "
                "frozen MOMENT (5\\% calibration, seed 42).}\n")
        f.write("\\label{tab:latency}\n\\small\n\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write("Adapter & Cal./domain (s) & Infer. (ms) & Peak GPU (MB) & Buffer (MB)\\\\\n")
        f.write("\\midrule\n")
        for _, row in df.iterrows():
            f.write(f"{row['adapter']} & {row['calib_per_domain_s']:.1f} & "
                    f"{row['infer_ms_per_sample']:.2f} & "
                    f"{row['peak_mem_mb']:.0f} & {row['buffer_mb']:.1f}\\\\\n")
        f.write(f"\\multicolumn{{5}}{{l}}{{\\footnotesize GPU: {df.iloc[0]['gpu']}}}\\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    print(df.to_string(index=False))
    print(f"\nSaved -> {csv_path}\n       -> {tex_path}")


if __name__ == "__main__":
    main()
