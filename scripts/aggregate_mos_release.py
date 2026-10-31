#!/usr/bin/env python
"""Aggregate pretrained-DiT MoS/Gaussian deployment efficiency and final metrics.

The script validates every result's checkpoint identity and evaluation protocol,
then writes early-checkpoint MSE/repeat curves, final paired cluster-bootstrap
confidence intervals, a machine-readable JSON report, and a CSV summary.

uv run python scripts/aggregate_mos_release.py --run-suffix _v2 --training-seed 2
"""
import argparse
import csv
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval_result_utils import (
    legacy_synth_overrides, load_compare_result, mean_cluster_ci, paired_ratio_ci,
)

STEPS = (200, 500, 1000, 2500, 3500, 5000)
ARMS = {"mos_best": "Semantic start", "mos_best_no_mem": "Gaussian source"}


def result_dir(root, arm, step, domain, es, seed):
    return root / f"eval_mos_release_{arm}_ckpt{step}_{domain}_es{es}_s{seed}"


def mean_metric(record, metric):
    return float(np.mean([float(v[metric]) for v in record["trajs"].values()]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default="runs")
    ap.add_argument("--run-suffix", default="_v2")
    ap.add_argument("--training-seed", type=int, default=2)
    ap.add_argument("--out-prefix", default=None)
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--bootstrap-seed", type=int, default=20261029)
    ap.add_argument("--legacy-synth-equivalence",
                    help="PASS JSON from verify_legacy_synth17d.py; required only for old 22D-container paths")
    args = ap.parse_args()
    root = Path(args.run_dir)
    legacy = legacy_synth_overrides(args.legacy_synth_equivalence)

    def load_result(path, model, domain, **kwargs):
        return load_compare_result(
            path, model, domain,
            protocol_overrides=legacy if domain == "synth" else None, **kwargs,
        )

    prefix = Path(args.out_prefix or root / f"mos_release_comparison_s{args.training_seed}")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.bootstrap_seed)
    report = {
        "scope": "pretrained Action DiT; open-loop held-out replay; not closed-loop success",
        "training_seed": args.training_seed,
        "steps": list(STEPS), "n_bootstrap": args.n_boot,
        "legacy_synth_equivalence": args.legacy_synth_equivalence,
        "ratio_convention": "Gaussian / semantic; values above 1 favor semantic",
        "early": {}, "final": {}, "training_runtime_seconds": {},
    }
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for row, domain in enumerate(("synth", "droid")):
        report["early"][domain] = {}
        for ai, (arm, label) in enumerate(ARMS.items()):
            curve = []
            for step in STEPS:
                path = result_dir(root, arm, step, domain, 0, args.training_seed)
                expected = f"{root}/{arm}{args.run_suffix}/checkpoint-{step}"
                rec = load_result(path, expected, domain, expected_seed=0)
                point = {"step": step, "mse_eff": mean_metric(rec, "mse_effective"),
                         "repeat": mean_metric(rec, "repeat_divergence"),
                         "source": rec["path"]}
                curve.append(point)
                rows.append({"domain": domain, "arm": arm, "step": step, **point})
            report["early"][domain][arm] = curve
            for col, metric in enumerate(("mse_eff", "repeat")):
                ax = axes[row, col]
                ax.plot(STEPS, [p[metric] for p in curve], marker="o", label=label,
                        color=plt.cm.tab10.colors[ai])
                ax.set_xscale("log", base=2)
                ax.set_xticks(STEPS, [str(x) for x in STEPS])
                ax.set_xlabel("Training checkpoint (steps)")
                ax.set_ylabel("mse_eff" if metric == "mse_eff" else r"repeat $\|\Delta\|$")
                ax.set_title(f"{domain}: {metric}")
                ax.grid(alpha=0.25)
                ax.legend()

        final = {}
        final_records = {}
        eval_seeds = range(4) if domain == "synth" else range(1)
        for arm, label in ARMS.items():
            records = []
            for es in eval_seeds:
                path = result_dir(root, arm, 5000, domain, es, args.training_seed)
                expected = f"{root}/{arm}{args.run_suffix}/checkpoint-5000"
                records.append(load_result(path, expected, domain, expected_seed=es))
            final_records[arm] = records
            final[arm] = {
                metric: mean_cluster_ci(records, metric, rng, args.n_boot)
                for metric in ("mse_effective", "repeat_divergence")
            }
        final["paired_gaussian_over_semantic"] = {
            metric: paired_ratio_ci(
                final_records["mos_best_no_mem"], final_records["mos_best"],
                metric, rng, args.n_boot,
            )
            for metric in ("mse_effective", "repeat_divergence")
        }
        report["final"][domain] = final

    fig.suptitle("Official release pair: deployment metrics versus training steps\n"
                 "Pretrained DiT; evaluation seed 0; 10 trajectories; R=2")
    fig.tight_layout()
    fig.savefig(prefix.with_suffix(".png"), dpi=180, bbox_inches="tight")

    for arm in ARMS:
        log = root / f"{arm}{args.run_suffix}_log.txt"
        if log.exists():
            matches = re.findall(r"'train_runtime':\s*([0-9.]+)", log.read_text(encoding="utf-8", errors="replace"))
            if matches:
                report["training_runtime_seconds"][arm] = float(matches[-1])
    prefix.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=("domain", "arm", "step", "mse_eff", "repeat", "source"))
        writer.writeheader()
        writer.writerows(rows)
    for domain, block in report["final"].items():
        print(f"[{domain}]")
        for metric, stat in block["paired_gaussian_over_semantic"].items():
            print(f"  Gaussian/semantic {metric}: {stat['ratio_reference_over_candidate_mean']:.4f} "
                  f"95% CI {stat['ci95']}")
    print(f"Saved {prefix.with_suffix('.png')}, {prefix.with_suffix('.json')}, {prefix.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
