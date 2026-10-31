#!/usr/bin/env python3
"""Aggregate named encoder/history-window factor cells from canonical eval JSON."""
import argparse
import json
from pathlib import Path

import numpy as np

from eval_result_utils import load_compare_result, mean_cluster_ci


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs"))
    parser.add_argument("--training-seed", type=int, default=2)
    parser.add_argument("--cell", action="append", required=True,
                        help="label=checkpoint_run_name; repeat for every 2x2 cell")
    parser.add_argument("--out-json", type=Path, default=Path("runs/mos_factor_cells.json"))
    parser.add_argument("--n-boot", type=int, default=20000)
    args = parser.parse_args()
    cells = dict(item.split("=", 1) for item in args.cell)
    if len(cells) != len(args.cell):
        raise ValueError("duplicate factor-cell labels")
    rng = np.random.default_rng(20261031)
    report = {"scope": "from-scratch DiT; open-loop replay", "training_seed": args.training_seed,
              "cells": {}}
    for label, run_name in cells.items():
        report["cells"][label] = {"run_name": run_name, "domains": {}}
        for domain in ("synth", "droid"):
            records = []
            for eval_seed in (range(4) if domain == "synth" else range(1)):
                result = args.run_dir / (
                    f"eval_mos_scratch_mos_best_ckpt5000_{domain}_es{eval_seed}_"
                    f"s{args.training_seed}"
                )
                records.append(load_compare_result(
                    result, f"{args.run_dir}/{run_name}/checkpoint-5000", domain,
                    expected_seed=eval_seed,
                ))
            report["cells"][label]["domains"][domain] = {
                metric: mean_cluster_ci(records, metric, rng, args.n_boot)
                for metric in ("mse_effective", "repeat_divergence")
            }
            report["cells"][label]["domains"][domain]["sources"] = [r["path"] for r in records]
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
