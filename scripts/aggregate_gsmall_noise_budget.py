#!/usr/bin/env python
"""Paired G-small noise-budget analysis against the same-training-seed G and Z.

    uv run python scripts/aggregate_gsmall_noise_budget.py

Synth uses eval seeds 0..3; DROID uses eval seed 0. Every comparison uses
mean(reference_i/candidate_i) and a shared whole-trajectory bootstrap (20k,
rng(0)). Ratios >1 mean the candidate has the lower metric.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval_result_utils import (
    legacy_synth_overrides, load_compare_result, mean_cluster_ci, paired_ratio_ci,
)

ARMS = {
    "G_s2": "mos_best_no_mem",
    "G-small_s2": "db29_gsmall_s2",
    "Z_s2": "mos_best",
}
MODEL_RUNS = {
    "mos_best_no_mem": "db28g_s2",
    "mos_best": "db28z_s2",
    "db29_gsmall_s2": "db29_gsmall_s2",
}


def result_dir(run_dir, arm, domain, es, legacy_layout=False):
    if legacy_layout and arm in ("mos_best_no_mem", "mos_best"):
        if arm == "mos_best_no_mem":
            suffix = f"synth_es{es}" if domain == "synth" else "droid"
            return run_dir / f"eval_db28s2_db28g_s2_{suffix}"
        suffix = f"final_es{es}" if domain == "synth" else "droid"
        return run_dir / f"eval_s2_db28z_s2_{suffix}"
    if arm == "db29_gsmall_s2":
        suffix = f"synth_es{es}" if domain == "synth" else "droid"
        return run_dir / f"eval_s2_{arm}_{suffix}"
    return run_dir / f"eval_mos_scratch_{arm}_ckpt5000_{domain}_es{es}_s2"


def load_arm(run_dir, arm, domain, load_result, legacy_layout=False):
    records = []
    for es in (range(4) if domain == "synth" else range(1)):
        records.append(load_result(
            result_dir(run_dir, arm, domain, es, legacy_layout),
            f"{run_dir}/{MODEL_RUNS[arm]}/checkpoint-5000",
            domain,
            expected_seed=es,
        ))
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default="runs")
    ap.add_argument("--out-prefix", default="runs/db29_gsmall_s2_noise_budget")
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--legacy-run-layout", action="store_true",
                    help="Explicitly read the historical eval_db28s2/eval_s2 result directory layout")
    ap.add_argument("--legacy-synth-equivalence",
                    help="PASS JSON from verify_legacy_synth17d.py; required only for old 22D-container paths")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    legacy = legacy_synth_overrides(args.legacy_synth_equivalence)

    def load_result(path, model, domain, **kwargs):
        return load_compare_result(
            path, model, domain,
            protocol_overrides=legacy if domain == "synth" else None, **kwargs,
        )

    rng = np.random.default_rng(0)
    report = {
        "estimator": "mean(reference_i/candidate_i), trajectory-cluster bootstrap",
        "n_boot": args.n_boot,
        "rng_seed": 0,
        "legacy_run_layout": args.legacy_run_layout,
        "legacy_synth_equivalence": args.legacy_synth_equivalence,
        "scope": "open-loop replay; 10 trajectories; R=2 repeat pairs; no closed-loop success claim",
        "domains": {},
    }
    loaded = {}
    for domain in ("synth", "droid"):
        records = {
            label: load_arm(run_dir, arm, domain, load_result, args.legacy_run_layout)
            for label, arm in ARMS.items()
        }
        loaded[domain] = records
        block = {"eval_seeds": [0, 1, 2, 3] if domain == "synth" else [0], "arms": {}, "comparisons": {}}
        for label, recs in records.items():
            block["arms"][label] = {
                metric: mean_cluster_ci(recs, metric, rng, args.n_boot)
                for metric in ("repeat_divergence", "mse_effective")
            }
            block["arms"][label]["source_files"] = [r["path"] for r in recs]
        for comparison, ref, cand in (
            ("G-small_vs_Z", "Z_s2", "G-small_s2"),
            ("G-small_vs_G", "G_s2", "G-small_s2"),
        ):
            block["comparisons"][comparison] = {"reference": ref, "candidate": cand}
            for metric in ("repeat_divergence", "mse_effective"):
                block["comparisons"][comparison][metric] = paired_ratio_ci(
                    records[ref], records[cand], metric, rng, args.n_boot
                )
        report["domains"][domain] = block
        print(f"\n=== {domain} ===")
        for label, values in block["arms"].items():
            print(label, " ".join(
                f"{metric}={values[metric]['mean']:.6f} CI95={values[metric]['ci95']}"
                for metric in ("repeat_divergence", "mse_effective")
            ))
        for name, values in block["comparisons"].items():
            for metric in ("repeat_divergence", "mse_effective"):
                stat = values[metric]
                print(f"{name} {metric}: {values['reference']}/{values['candidate']}="
                      f"{stat['ratio_reference_over_candidate_mean']:.4f} CI95={stat['ci95']}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    labels = list(ARMS)
    for row, domain in enumerate(("synth", "droid")):
        for col, metric in enumerate(("repeat_divergence", "mse_effective")):
            ax = axes[row, col]
            stats = [report["domains"][domain]["arms"][label][metric] for label in labels]
            means = [s["mean"] for s in stats]
            errors = np.array([
                [s["mean"] - s["ci95"][0] for s in stats],
                [s["ci95"][1] - s["mean"] for s in stats],
            ])
            ax.bar(labels, means, yerr=errors, capsize=4,
                   color=["#7f7f7f", "#bcbd22", "#1f77b4"])
            ax.set_title(f"{domain}: {metric}")
            ax.grid(axis="y", alpha=0.25)
            ax.tick_params(axis="x", rotation=12)
            lines = []
            for name, comparison in report["domains"][domain]["comparisons"].items():
                stat = comparison[metric]
                lines.append(f"{name}: {stat['ratio_reference_over_candidate_mean']:.3f} "
                             f"[{stat['ci95'][0]:.3f},{stat['ci95'][1]:.3f}]")
            ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
                    ha="left", va="top", fontsize=8)
    fig.suptitle("G-small noise-budget control: mean ± trajectory-cluster CI95\n"
                 "Ratio = reference/candidate (>1 favors candidate); open-loop replay, R=2")
    fig.tight_layout()
    out = Path(args.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=180, bbox_inches="tight")
    out.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {out.with_suffix('.png')} and {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
