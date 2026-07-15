"""Oracle-routing-gap ablation: learned (MLP, rehearsal) vs. prototype
(parameter-free) gate, isolated to the entangled FD-B/FD-D pair.

By Proposition 1, forgetting can only enter through the gap between the soft
mixture and the oracle (perfect-routing) ceiling: gap_j = A_orc[t,j] - A_soft[t,j]
for a domain j evaluated after t domains have arrived. FD-B and FD-D are the
domains identified as the "hard case" in the earlier shift-locality analysis --
they shift the same charts, so their prototypes may nearly coincide (bad for
the parameter-free gate) while an MLP can still find a thin discriminating
direction (bad for the "no rehearsal needed" story if it wins by a lot).

This script runs the chain source=FD-A, then FD-B, then FD-D (the order that
forces the gate to separate the two entangled domains) under both gate kinds,
and reports the oracle-routing gap per domain at each step -- the number that
actually decides which gate the paper should default to.

Also runs the reverse introduction order (FD-D before FD-B) as a symmetry
check: if the gap is order-dependent, that itself is worth reporting.
"""

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from _bootstrap import CKPT_ROOT, DATA_ROOT, RESULTS_ROOT

from src.data import get_spec
from src.method import PactConfig, run_sequential
from src.models.factory import load_backbone
from src.training import train_or_load_source
from src.utils import get_device, set_seed

DATASET, BACKBONE, SOURCE = "fd", "tslanet", 0
ORDERS = {"B_then_D": [0, 1, 3], "D_then_B": [0, 3, 1]}
GATE_KINDS = ["learned", "prototype"]
K, CALIB_FRAC, TEMPERATURE, SEED = 16, 0.05, 0.5, 42


def main():
    set_seed(SEED)
    device = get_device()
    spec = get_spec(DATASET)
    train_or_load_source(BACKBONE, DATASET, SOURCE, CKPT_ROOT, DATA_ROOT, device)
    model = load_backbone(BACKBONE, spec, CKPT_ROOT, DATASET, SOURCE, device, freeze=True)

    rows = []
    for order_name, order in ORDERS.items():
        for gate_kind in GATE_KINDS:
            cfg = PactConfig(k=K, emb_dim=model.emb_dim, calib_frac=CALIB_FRAC,
                             seed=SEED, gate_kind=gate_kind)
            res = run_sequential(model, spec, order, DATA_ROOT, device, cfg,
                                 temperature=TEMPERATURE, verbose=False)
            labels, A_orc, A_soft, A_hard = res["labels"], res["A_orc"], res["A_soft"], res["A_hard"]
            T = len(labels)
            print(f"\n=== order={order_name}  gate={gate_kind}  "
                 f"labels={labels} ===")
            for t in range(T):
                for j in range(t + 1):
                    orc, soft, hard = A_orc[t, j], A_soft[t, j], A_hard[t, j]
                    gap = orc - soft
                    hard_gap = orc - hard
                    rows.append(dict(order=order_name, gate_kind=gate_kind, step=t,
                                     step_domain=labels[t], eval_domain=labels[j],
                                     A_oracle=orc, A_soft=soft, A_hard=hard,
                                     soft_oracle_gap=gap, hard_oracle_gap=hard_gap))
                    print(f"  step={t} ({labels[t]:<6}) eval={labels[j]:<6} "
                         f"oracle={orc:.4f} soft={soft:.4f} hard={hard:.4f} "
                         f"  gap(orc-soft)={gap:+.4f}")

    df = pd.DataFrame(rows)
    out_dir = RESULTS_ROOT / "gate_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "oracle_routing_gap_fdb_fdd.csv"
    df.to_csv(out_path, index=False)

    # Headline: gap at the FINAL step of each chain, for every domain seen.
    final = df[df.groupby(["order", "gate_kind"])["step"].transform("max") == df["step"]]
    print("\n=== Final-step soft-oracle gap (this is the number that decides forgetting) ===")
    piv = final.pivot_table(index=["order", "eval_domain"], columns="gate_kind",
                            values="soft_oracle_gap")
    print(piv.to_string(float_format=lambda x: f"{x:+.4f}"))

    print("\n=== Mean |gap| across all (step, eval_domain) cells, by gate ===")
    summary = df.groupby("gate_kind")[["soft_oracle_gap", "hard_oracle_gap"]].apply(
        lambda g: g.abs().mean())
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
