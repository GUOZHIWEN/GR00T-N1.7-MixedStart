# -*- coding: utf-8 -*-
"""Generate fig_multiseed_ci.png (2x2: two domains x two metrics, mean ± 95% CI).
Data source: runs/multiseed_db7/compare_metrics.json, runs/multiseed_droid/compare_metrics.json
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures" / "fig_multiseed_ci.png"

panels = [
    ("runs/multiseed_db7",   "Synthetic 22D (heldout, n=10 traj x 3 seeds)"),
    ("runs/multiseed_droid", "Real DROID subset (n=10 traj x 3 seeds)"),
]
metrics = [
    ("repeat_divergence", "repeat$\\|\\Delta\\|$ (lower = more deterministic)",
     "ms_repeat_divergence_mean", "ms_repeat_divergence_ci95"),
    ("mse_effective", "mse_eff (lower = better)",
     "ms_mse_effective_mean", "ms_mse_effective_ci95"),
]
ARM_ORDER = ["pure_z0", "no_mem"]
ARM_LABEL = {"pure_z0": "pure_z0 (memory)", "no_mem": "no_mem (gaussian)"}
ARM_COLOR = {"pure_z0": "#1f77b4", "no_mem": "#ff7f0e"}

fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8))

for r, (d, row_title) in enumerate(panels):
    with (ROOT / d / "compare_metrics.json").open(encoding="utf-8") as f:
        j = json.load(f)
    ratio_block = j["multi_seed"]["ratios"]["pure_z0 vs no_mem"]
    for c, (mkey, mlabel, mk_mean, mk_ci) in enumerate(metrics):
        ax = axes[r][c]
        means, cis, colors, labels = [], [], [], []
        for arm in ARM_ORDER:
            s = j["arms"][arm]["summary"]
            means.append(s[mk_mean]); cis.append(s[mk_ci])
            colors.append(ARM_COLOR[arm]); labels.append(ARM_LABEL[arm])
        x = np.arange(2)
        ax.bar(x, means, yerr=cis, capsize=5, color=colors, width=0.55,
               error_kw=dict(ecolor="black", lw=1.4))
        for xi, (m, ci) in enumerate(zip(means, cis)):
            ax.text(xi, m + ci + max(means) * 0.05, f"{m:.4f}", ha="center", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, max(np.array(means) + np.array(cis)) * 1.38)
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(f"{row_title}\n{mlabel}", fontsize=10)
        rb = ratio_block.get(mkey)
        if rb:
            lo, hi = rb["ci95"]
            sig = "CI excludes 1" if (lo > 1 or hi < 1) else "CI contains 1 (n.s.)"
            ax.text(0.5, -0.30,
                    f"ratio no_mem/pure_z0 = {rb['ratio_ref_over_arm_mean']:.2f}x  "
                    f"CI95 [{lo:.2f}, {hi:.2f}]  ({sig})",
                    transform=ax.transAxes, ha="center", fontsize=9, style="italic")

fig.suptitle("Multi-seed paired evaluation: determinism asset vs accuracy parity (mean ± 95% CI)",
             fontsize=11.5, y=0.995)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
fig.savefig(OUT, dpi=160)
print("saved:", OUT)
