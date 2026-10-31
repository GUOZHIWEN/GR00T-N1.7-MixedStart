#!/usr/bin/env python
"""Plot rtc/16 heldout deployment metrics across early checkpoints.

    uv run python scripts/aggregate_f2early.py --include-shift-reference

Requires both rtc arms at 200/500/1000/2500 plus their existing 5000 result.
Optional shift/1 references are read from existing files only; missing reference
curves are listed explicitly and never synthesized. No training-loss comparison.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_result_utils import load_compare_result

STEPS = [200, 500, 1000, 2500, 5000]
MAIN_ARMS = {"MLP+rtc/16": "db28z2b_s2", "Transformer+rtc/16": "db28z2c_s2"}
SHIFT_ARMS = {"MLP+shift/1": "db28z_s2", "Transformer+shift/1": "db28z_s2_tx_shift"}


def result_path(run_dir, arm, step, domain, reference):
    if not reference:
        if step == 5000:
            suffix = "synth_es0" if domain == "synth" else "droid"
        else:
            suffix = f"ckpt{step}_synth_es0" if domain == "synth" else f"ckpt{step}_droid"
        return run_dir / (f"eval_s2_{arm}_{suffix}" if step == 5000 else f"eval_f2early_{arm}_{suffix}")
    if domain == "droid":
        suffix = "droid" if step == 5000 else f"droid_ckpt{step}"
    else:
        suffix = f"final_es0" if step == 5000 else f"ckpt{step}"
    return run_dir / f"eval_s2_{arm}_{suffix}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default="runs")
    ap.add_argument("--include-shift-reference", action="store_true")
    ap.add_argument("--out-prefix", default="runs/f2early_heldout")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    groups = [(MAIN_ARMS, False)]
    if args.include_shift_reference:
        groups.append((SHIFT_ARMS, True))
    report = {
        "scope": "heldout open-loop replay, eval seed 0, 10 trajectories, R=2; not closed-loop success",
        "steps": STEPS,
        "domains": {},
        "missing_optional_references": [],
    }
    for domain in ("synth", "droid"):
        curves = {}
        for arms, reference in groups:
            for label, arm in arms.items():
                points = []
                for step in STEPS:
                    path = result_path(run_dir, arm, step, domain, reference)
                    if not (path / "compare_metrics.json").is_file():
                        if reference:
                            report["missing_optional_references"].append(str(path / "compare_metrics.json"))
                            continue
                        raise FileNotFoundError(f"required rtc result missing: {path / 'compare_metrics.json'}")
                    record = load_compare_result(
                        path, f"runs/{arm}/checkpoint-{step}", domain, expected_seed=0
                    )
                    trajs = record["trajs"]
                    points.append({
                        "step": step,
                        "mse_effective": sum(t["mse_effective"] for t in trajs.values()) / 10,
                        "repeat_divergence": sum(t["repeat_divergence"] for t in trajs.values()) / 10,
                        "per_trajectory": {
                            tid: {k: t[k] for k in ("mse_effective", "repeat_divergence")}
                            for tid, t in trajs.items()
                        },
                        "source": record["path"],
                    })
                if points:
                    curves[label] = points
                    best = min(points, key=lambda p: p["mse_effective"])
                    final = points[-1]
                    print(f"{domain} {label}: best mse_eff={best['mse_effective']:.6f} "
                          f"@ {best['step']}; final={final['mse_effective']:.6f}; "
                          f"final-best={final['mse_effective']-best['mse_effective']:+.6f}; "
                          f"repeat(final)={final['repeat_divergence']:.6f}")
        report["domains"][domain] = curves

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    colors = plt.cm.tab10.colors
    for row, domain in enumerate(("synth", "droid")):
        for col, metric in enumerate(("mse_effective", "repeat_divergence")):
            ax = axes[row, col]
            for i, (label, points) in enumerate(report["domains"][domain].items()):
                x, y = [p["step"] for p in points], [p[metric] for p in points]
                style = "--" if "shift/1" in label else "-"
                if len(points) > 1:
                    ax.plot(x, y, style, marker="o", color=colors[i % 10], label=label)
                else:
                    ax.plot(x, y, "x", markersize=9, color=colors[i % 10], label=f"{label} (final only)")
            ax.set_xscale("log", base=2)
            ax.set_xticks(STEPS, [str(s) for s in STEPS])
            ax.set_title(f"{domain}: {metric}")
            ax.set_xlabel("training checkpoint")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
    fig.suptitle("rtc/16 early-checkpoint heldout deployment curves\n"
                 "open-loop replay; es0; n=10 trajectories; R=2; training loss is not a deployment verdict")
    fig.tight_layout()
    out = Path(args.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=180, bbox_inches="tight")
    out.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {out.with_suffix('.png')} and {out.with_suffix('.json')}")
    if report["missing_optional_references"]:
        print("Optional shift references not found (not imputed):")
        print("\n".join(report["missing_optional_references"]))


if __name__ == "__main__":
    main()
