"""Plot checkpoint-wise open-loop MSE_eff for G, G-small and Z.

Missing evaluations remain missing; no interpolation or training-loss substitution.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
STEPS = [200, 500, 1000, 2500, 5000]
ARMS = {
    "G: standard Gaussian": ("db28g_s2", "#b4533f", "o"),
    "G-small: 0.04 noise": ("db29_gsmall_s2", "#39739d", "s"),
    "Z: semantic start + 0.04 noise": ("db28z_s2", "#16836b", "^"),
}
DOMAINS = {"Synthetic": "synth", "DROID": "droid"}


def eval_dir(arm: str, checkpoint: int, domain: str) -> Path:
    if arm == "db29_gsmall_s2":
        suffix = f"{domain}_es0" if domain == "synth" else domain
        return RUNS / f"eval_s2_{arm}_{suffix}" if checkpoint == 5000 else RUNS / f"eval_s2_{arm}_{domain}_ckpt{checkpoint}"
    if checkpoint == 5000:
        if domain == "synth":
            return RUNS / ("eval_s2_db28z_s2_final_es0" if arm == "db28z_s2" else "eval_db28s2_db28g_s2_synth_es0")
        return RUNS / f"eval_db28s2_{arm}_droid"
    suffix = f"droid_ckpt{checkpoint}" if domain == "droid" else f"ckpt{checkpoint}"
    return RUNS / f"eval_s2_{arm}_{suffix}"


def read_score(path: Path) -> float | None:
    file = path / "compare_metrics.json"
    if not file.exists():
        return None
    data = json.loads(file.read_text(encoding="utf-8"))
    return float(next(iter(data["arms"].values()))["summary"]["avg_mse_effective"])


def main() -> None:
    values = {
        domain: {
            arm: [read_score(eval_dir(arm, checkpoint, domain)) for checkpoint in STEPS]
            for arm, _, _ in ARMS.values()
        }
        for domain in DOMAINS.values()
    }
    for arm, _, _ in ARMS.values():
        for domain in DOMAINS.values():
            assert values[domain][arm][-1] is not None, (domain, arm, 5000)

    plt.rcParams.update({"font.size": 12, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharex=True, constrained_layout=True)
    for ax, (title, domain) in zip(axes, DOMAINS.items()):
        for label, (arm, color, marker) in ARMS.items():
            series = np.asarray([np.nan if v is None else v for v in values[domain][arm]], dtype=float)
            present = np.isfinite(series)
            # A single available point is drawn as a marker only, never joined across missing checkpoints.
            ax.plot(np.asarray(STEPS)[present], series[present], marker=marker, color=color,
                    linewidth=2.2 if present.sum() > 1 else 0, markersize=7, label=label)
            for x, y in zip(np.asarray(STEPS)[present], series[present]):
                ax.annotate(f"{y:.4f}", (x, y), xytext=(0, 7), textcoords="offset points",
                            ha="center", fontsize=9, color=color)
        ax.set_xscale("log")
        ax.set_xticks(STEPS, [str(x) for x in STEPS])
        ax.set_xlabel("Training checkpoint (optimizer step)")
        ax.set_ylabel("Open-loop MSE_eff (lower is better)")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=10)
    fig.suptitle("Deployment accuracy versus training progress (evaluation seed 0, 10 trajectories)", fontsize=16, fontweight="bold")
    fig.text(0.5, -0.025,
             "Same open-loop replay protocol: 200 steps, execution horizon 16, 4 denoising steps. "
             "G-small checkpoints 200–2500 are pending evaluation; only its measured 5000 endpoint is shown.",
             ha="center", fontsize=10, color="#475569")
    output = RUNS / "db29_deployment_mse_eff_convergence.png"
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)

    report = {
        "protocol": "open-loop replay; evaluation seed 0; 10 trajectories; 200 steps; execution horizon 16; 4 denoising steps",
        "checkpoints": STEPS,
        "values": values,
        "missing": [
            {"domain": domain, "arm": arm, "checkpoint": checkpoint}
            for domain in DOMAINS.values()
            for arm, _, _ in ARMS.values()
            for checkpoint, value in zip(STEPS, values[domain][arm])
            if value is None
        ],
    }
    json_path = RUNS / "db29_deployment_mse_eff_convergence.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output)
    print(json_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
