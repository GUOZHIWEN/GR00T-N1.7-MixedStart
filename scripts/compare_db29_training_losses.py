"""Compare G, G-small and Z training losses on identical step axes."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
ARMS = {
    "G: standard Gaussian": RUNS / "db28g_s2_log.txt",
    "G-small: 0.04 noise": RUNS / "db29_gsmall_s2" / "db29_gsmall_s2_log.txt",
    "Z: semantic start + 0.04 noise": RUNS / "db28z_s2_log.txt",
}
COLORS = {
    "G: standard Gaussian": "#b4533f",
    "G-small: 0.04 noise": "#39739d",
    "Z: semantic start + 0.04 noise": "#16836b",
}


def parse_log(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    losses, flows = [], []
    runtime = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.strip()
        if not (text.startswith("{") and text.endswith("}")):
            continue
        try:
            item = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            continue
        if "loss" in item and "grad_norm" in item:
            losses.append(item)
        if "dbg_flow_loss" in item:
            flows.append(item)
        if "train_runtime" in item:
            runtime = item
    # Total loss is logged every 10 steps; debug flow loss every 5 steps.
    loss_df = pd.DataFrame(losses)
    loss_df.insert(0, "step", (np.arange(len(loss_df)) + 1) * 10)
    flow_df = pd.DataFrame(flows)
    flow_df.insert(0, "step", (np.arange(len(flow_df)) + 1) * 5)
    return loss_df, flow_df, runtime


def window_mean(df: pd.DataFrame, column: str, center: int, half_width: int = 100) -> float:
    selected = df.loc[df["step"].between(center - half_width, center + half_width), column]
    return float(selected.mean())


def main() -> None:
    data = {name: parse_log(path) for name, path in ARMS.items()}
    for name, (loss, flow, _) in data.items():
        assert len(loss) == 500, (name, len(loss))
        assert len(flow) == 1000, (name, len(flow))
        assert loss["step"].iloc[-1] == 5000 and flow["step"].iloc[-1] == 5000

    plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    specs = [
        ("loss", "Reported training loss", "loss", 20),
        ("flow", "Flow-matching loss only (debug)", "dbg_flow_loss", 40),
    ]
    for row, (_, title, column, smooth_window) in enumerate(specs):
        for col, (xmax, subtitle) in enumerate([(5000, "Full 5000 steps"), (500, "Early training")]):
            ax = axes[row, col]
            for name, (loss, flow, _) in data.items():
                df = loss if column == "loss" else flow
                view = df[df["step"] <= xmax]
                smooth = view[column].rolling(smooth_window, center=True, min_periods=1).mean()
                ax.plot(view["step"], view[column], color=COLORS[name], alpha=0.09, linewidth=0.7)
                ax.plot(view["step"], smooth, color=COLORS[name], label=name, linewidth=2.0)
            ax.set_xlim(0, xmax)
            ax.set_yscale("log" if xmax == 5000 else "linear")
            ax.set_xlabel("Optimizer step")
            ax.set_ylabel(column if row == 0 else "MSE")
            ax.set_title(f"{title} — {subtitle}")
            ax.grid(True, which="both", alpha=0.22)
            if row == 0 and col == 0:
                ax.legend(frameon=False, loc="upper right")
    fig.suptitle("Training convergence: Gaussian vs reduced-noise vs semantic source (seed 2)", fontsize=17, fontweight="bold")
    fig.text(0.5, -0.015,
             "Thin lines: raw logs; thick lines: centered 200-step moving average. Z total loss additionally includes its role-anchor term. "
             "Different sources induce different velocity targets, so cross-arm loss magnitudes are not deployment-accuracy rankings.",
             ha="center", fontsize=10, color="#475569")
    output = RUNS / "db29_training_loss_comparison.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "protocol": "seed 2; 5000 optimizer steps; accumulated batch size 8; same mix_droid_synth_train dataset",
        "smoothing": "centered 200-step moving average; statistics below are unsmoothed window means",
        "arms": {},
    }
    for name, (loss, flow, runtime) in data.items():
        summary["arms"][name] = {
            "checkpoint_window_means": {
                str(step): {
                    "loss": window_mean(loss, "loss", step),
                    "dbg_flow_loss": window_mean(flow, "dbg_flow_loss", step),
                }
                for step in [200, 500, 1000, 2500, 5000]
            },
            "final_500_steps": {
                "loss": float(loss.loc[loss["step"] > 4500, "loss"].mean()),
                "dbg_flow_loss": float(flow.loc[flow["step"] > 4500, "dbg_flow_loss"].mean()),
            },
            "runtime": runtime,
        }
    json_path = RUNS / "db29_training_loss_comparison.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(output)
    print(json_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
