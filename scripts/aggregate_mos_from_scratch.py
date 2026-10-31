#!/usr/bin/env python
"""Aggregate paper from-scratch G/Z final ratios and early-checkpoint curves.

Final paired CIs resample (training_seed, trajectory) clusters. Early curves
use held-out deployment metrics, never training loss. All inputs are checked
against the canonical evaluation protocol by eval_result_utils.
"""
import argparse
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval_result_utils import load_compare_result

ARMS = {"mos_best_no_mem": "Gaussian", "mos_best": "Semantic"}
METRICS = ("mse_effective", "repeat_divergence")
STEPS = (200, 500, 1000, 2500, 5000)


def run_name(arm: str, seed: int) -> str:
    if arm == "mos_best_no_mem":
        return f"db28g_s{seed}"
    return "db28z_s3_soft" if seed == 3 else f"db28z_s{seed}"


def result_dir(root: Path, arm: str, step: int, domain: str, es: int, seed: int) -> Path:
    return root / f"eval_mos_scratch_{arm}_ckpt{step}_{domain}_es{es}_s{seed}"


def load_one(root: Path, arm: str, step: int, domain: str, es: int, seed: int) -> dict:
    expected = f"{root}/{run_name(arm, seed)}/checkpoint-{step}"
    record = load_compare_result(
        result_dir(root, arm, step, domain, es, seed), expected, domain, expected_seed=es
    )
    record["training_seed"] = seed
    return record


def cluster_values(records: list[dict], metric: str) -> np.ndarray:
    grouped = {}
    for record in records:
        for tid, values in record["trajs"].items():
            key = (record["training_seed"], int(tid))
            grouped.setdefault(key, []).append(float(values[metric]))
    return np.asarray([grouped[key] for key in sorted(grouped)], dtype=np.float64)


def mean_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> dict:
    indices = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot = values[indices].mean(axis=(1, 2))
    return {
        "mean": float(values.mean()),
        "ci95": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
        "n_seed_trajectory_clusters": len(values),
        "n_records": int(values.size),
    }


def paired_ratio(reference: list[dict], candidate: list[dict], metric: str,
                 rng: np.random.Generator, n_boot: int) -> dict:
    ref = cluster_values(reference, metric)
    cand = cluster_values(candidate, metric)
    if ref.shape != cand.shape:
        raise ValueError(f"unpaired clusters: {ref.shape} vs {cand.shape}")
    if np.any(cand == 0):
        raise ValueError(f"zero denominator for {metric}")
    ratios = ref / cand
    indices = rng.integers(0, len(ratios), size=(n_boot, len(ratios)))
    boot = ratios[indices].mean(axis=(1, 2))
    return {
        "ratio_gaussian_over_semantic_mean": float(ratios.mean()),
        "ci95": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
        "estimator": "mean(Gaussian_i / Semantic_i)",
        "cluster": "(training_seed, trajectory)",
        "n_seed_trajectory_clusters": len(ratios),
        "n_records": int(ratios.size),
    }


def point(record: dict, step: int) -> dict:
    return {
        "step": step,
        **{
            metric: float(np.mean([float(v[metric]) for v in record["trajs"].values()]))
            for metric in METRICS
        },
        "source": record["path"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs")
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--efficiency-seeds", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--skip-early", action="store_true")
    parser.add_argument("--n-boot", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261026)
    parser.add_argument("--out-prefix", default="runs/mos_paper_main")
    args = parser.parse_args()
    root = Path(args.run_dir)
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.bootstrap_seed)
    report = {
        "scope": "from-scratch Action DiT; open-loop replay; not closed-loop success",
        "training_seeds": args.training_seeds,
        "ratio_convention": "Gaussian / semantic; >1 favors semantic",
        "cluster": "(training_seed, trajectory)",
        "n_bootstrap": args.n_boot,
        "final": {}, "early": {}, "efficiency_thresholds": {},
        "training_runtime_seconds": {},
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for mi, metric in enumerate(METRICS):
        domain_stats = []
        for domain in ("synth", "droid"):
            records = {arm: [] for arm in ARMS}
            for seed in args.training_seeds:
                eval_seeds = range(4) if domain == "synth" else range(1)
                for arm in ARMS:
                    for es in eval_seeds:
                        records[arm].append(load_one(root, arm, 5000, domain, es, seed))
            block = report["final"].setdefault(domain, {"arms": {}, "paired": {}})
            for arm in ARMS:
                block["arms"][arm][metric] = mean_ci(cluster_values(records[arm], metric), rng, args.n_boot)
            block["paired"][metric] = paired_ratio(
                records["mos_best_no_mem"], records["mos_best"], metric, rng, args.n_boot
            )
            domain_stats.append(block["paired"][metric])
        x = np.arange(2)
        means = [s["ratio_gaussian_over_semantic_mean"] for s in domain_stats]
        errors = np.array([
            [s["ratio_gaussian_over_semantic_mean"] - s["ci95"][0] for s in domain_stats],
            [s["ci95"][1] - s["ratio_gaussian_over_semantic_mean"] for s in domain_stats],
        ])
        axes[mi].bar(x, means, yerr=errors, capsize=5, color=["#2878B5", "#169873"])
        axes[mi].axhline(1.0, color="#666666", linestyle="--", linewidth=1)
        axes[mi].set_xticks(x, ["Synthetic", "DROID replay"])
        axes[mi].set_title(metric)
        axes[mi].set_ylabel("Gaussian / semantic paired ratio")
        axes[mi].grid(axis="y", alpha=0.2)
    fig.suptitle("From-scratch DiT: paired final results (95% seed×trajectory cluster CI)")
    fig.tight_layout()
    fig.savefig(prefix.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)

    if not args.skip_early:
        for seed in args.efficiency_seeds:
            report["early"][str(seed)] = {}
            for domain in ("synth", "droid"):
                report["early"][str(seed)][domain] = {}
                for arm in ARMS:
                    curve = [point(load_one(root, arm, step, domain, 0, seed), step) for step in STEPS]
                    report["early"][str(seed)][domain][arm] = curve
            for threshold in (0.055, 0.050):
                key = f"synthetic_mse_eff_le_{threshold:.3f}"
                entry = {"threshold": threshold}
                for arm in ARMS:
                    curve = report["early"][str(seed)]["synth"][arm]
                    reached = next((p["step"] for p in curve if p["mse_effective"] <= threshold), None)
                    entry[arm] = reached
                if all(entry[arm] is not None for arm in ARMS):
                    entry["gaussian_over_semantic_step_ratio"] = (
                        entry["mos_best_no_mem"] / entry["mos_best"]
                    )
                report["efficiency_thresholds"].setdefault(key, {})[str(seed)] = entry

        fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
        colors = {"mos_best_no_mem": "#75808D", "mos_best": "#169873"}
        for row, domain in enumerate(("synth", "droid")):
            for col, metric in enumerate(METRICS):
                ax = axes[row, col]
                for seed in args.efficiency_seeds:
                    for arm, label in ARMS.items():
                        curve = report["early"][str(seed)][domain][arm]
                        ax.plot(STEPS, [p[metric] for p in curve], marker="o",
                                color=colors[arm], linestyle="-" if seed == 2 else "--",
                                label=f"{label}, train seed {seed}")
                ax.set_xscale("log", base=2)
                ax.set_xticks(STEPS, [str(x) for x in STEPS])
                ax.set_title(f"{domain}: {metric}")
                ax.grid(alpha=0.2)
                ax.legend(fontsize=8)
        fig.suptitle("Held-out deployment metrics across training checkpoints")
        fig.tight_layout()
        fig.savefig(prefix.with_name(prefix.name + "_efficiency").with_suffix(".png"),
                    dpi=180, bbox_inches="tight")
        plt.close(fig)

    for seed in set(args.training_seeds + args.efficiency_seeds):
        for arm in ARMS:
            name = run_name(arm, seed)
            log = root / f"{name}_log.txt"
            if log.exists():
                found = re.findall(r"'train_runtime':\s*([0-9.]+)", log.read_text(encoding="utf-8", errors="replace"))
                if found:
                    report["training_runtime_seconds"][name] = float(found[-1])

    prefix.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for domain, block in report["final"].items():
        for metric, stat in block["paired"].items():
            print(f"{domain} {metric}: {stat['ratio_gaussian_over_semantic_mean']:.4f} "
                  f"95% CI {stat['ci95']}")
    print(f"Wrote {prefix.with_suffix('.json')} and {prefix.with_suffix('.png')}")


if __name__ == "__main__":
    main()
