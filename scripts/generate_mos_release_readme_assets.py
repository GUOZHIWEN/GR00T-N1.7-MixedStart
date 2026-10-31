#!/usr/bin/env python
"""Generate the ModelScope README performance figure from verified release metrics."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--input-json", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "release" / "assets")
    args = parser.parse_args()
    source = args.input_json or args.run_dir / "mos_release_comparison_s2.json"
    output = args.output_dir / "mos_best_performance.png"
    data = json.loads(source.read_text(encoding="utf-8"))["final"]
    domains = ["synth", "droid"]
    domain_labels = ["Synthetic held-out", "DROID replay"]
    # Keep model identities consistent with mos_action_trajectories_synth.png.
    colors = {"mos_best": "#327E9D", "mos_best_no_mem": "#DD9B50"}
    labels = {
        "mos_best": "mos_best (semantic start)",
        "mos_best_no_mem": "mos_best_no_mem (Gaussian)",
    }

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titleweight": "bold",
        "axes.edgecolor": "#B8C0C8",
    })
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.5))
    x = np.arange(2)
    width = 0.32

    for ax, metric, title in zip(
        axes,
        ["mse_effective", "repeat_divergence"],
        ["Effective action MSE  (lower is better)", "Repeat divergence  (lower is better)"],
    ):
        for offset, arm in [(-width / 2, "mos_best"), (width / 2, "mos_best_no_mem")]:
            values = [data[d][arm][metric]["mean"] for d in domains]
            bars = ax.bar(
                x + offset,
                values,
                width,
                label=labels[arm],
                color=colors[arm],
                zorder=3,
            )
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{value:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color="#26313B",
                )

        ymax = max(
            data[d][arm][metric]["mean"] for d in domains for arm in labels
        )
        ax.set_ylim(0, ymax * 1.48)
        for i, domain in enumerate(domains):
            paired = data[domain]["paired_gaussian_over_semantic"][metric]
            ratio = paired["ratio_reference_over_candidate_mean"]
            low, high = paired["ci95"]
            significant = low > 1.0
            note = "significant" if significant else "not significant"
            ax.text(
                x[i],
                ymax * 1.30,
                f"ratio {ratio:.2f}x [{low:.2f}, {high:.2f}]\n{note}",
                ha="center",
                va="center",
                fontsize=10,
                color="#146C67" if significant else "#5D6670",
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "facecolor": "#F2F8F7" if significant else "#F4F5F6",
                    "edgecolor": "#C7D9D6" if significant else "#D8DDE2",
                },
            )
        ax.set_title(title, pad=14)
        ax.set_xticks(x, domain_labels)
        ax.grid(axis="y", alpha=0.22, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Metric value")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(
        "MoS official release pair  |  Pretrained Action DiT fine-tuning",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.025,
        "Bars show metric means. Paired ratios are mos_best_no_mem / mos_best (>1 favors mos_best);\n"
        "brackets show 95% trajectory-cluster bootstrap CIs. Open-loop replay; not closed-loop success.",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#5D6670",
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.76, bottom=0.20, wspace=0.22)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
