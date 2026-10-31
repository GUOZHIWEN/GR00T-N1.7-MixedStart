#!/usr/bin/env python
"""Draw ground truth and both model arms together from saved rollout arrays.

Run ``compare_flow_sources.py --save-trajectory-arrays`` with both checkpoints
first. Selection is the deterministic upper-median paired effective-MSE ratio
by default; --selection max-ratio explicitly requests a clear-gap example.
Fixed action dimensions are used and no model imports or GPU are needed.
A clear-gap figure must not be presented as a typical or aggregate result.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

from eval_result_utils import legacy_synth_overrides, load_compare_result


DOMAIN_TITLES = {
    "synth": "Synthetic held-out",
    "droid": "DROID replay",
}
GT_COLOR = "#000000"
SEMANTIC_COLOR = "#327E9D"
GAUSSIAN_COLOR = "#DD9B50"


def select_representative(semantic_trajs, gaussian_trajs, selection="upper-median"):
    """Select a trajectory by a declared metric rule, never by visual appeal."""
    tids = sorted(map(int, semantic_trajs))
    if tids != list(range(10)) or set(gaussian_trajs) != set(semantic_trajs):
        raise ValueError("trajectory selection requires the same trajectories 0..9")
    if selection not in {"upper-median", "max-ratio"}:
        raise ValueError(f"unsupported selection rule: {selection}")
    ratios = sorted(
        (float(gaussian_trajs[str(tid)]["mse_effective"] /
               semantic_trajs[str(tid)]["mse_effective"]), tid)
        for tid in tids
    )
    index = len(ratios) // 2 if selection == "upper-median" else len(ratios) - 1
    return ratios[index][::-1]


def load_trajectory(run_dir, tid, semantic_label, gaussian_label, semantic_model, gaussian_model,
                    expected_dataset):
    path = Path(run_dir) / f"traj_{tid}_trajectory.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}; rerun the paired evaluation with --save-trajectory-arrays"
        )
    with np.load(path, allow_pickle=False) as data:
        required = {"gt", "predictions", "labels", "model_paths", "trajectory_id",
                    "execution_horizon", "dataset_path", "seed"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"{path}: missing arrays {sorted(missing)}")
        labels = [str(value) for value in data["labels"].tolist()]
        model_paths = [str(value) for value in data["model_paths"].tolist()]
        if len(labels) != len(set(labels)) or len(model_paths) != len(labels):
            raise ValueError(f"{path}: labels and checkpoint paths must align one-to-one")
        expected = {semantic_label: semantic_model, gaussian_label: gaussian_model}
        if set(labels) != set(expected):
            raise ValueError(f"{path}: labels={labels}, expected {sorted(expected)}")
        for label, expected_path in expected.items():
            actual = model_paths[labels.index(label)]
            if Path(actual).as_posix().removeprefix("./") != Path(expected_path).as_posix().removeprefix("./"):
                raise ValueError(f"{path}: {label} is {actual}, not {expected_path}")
        if Path(str(data["dataset_path"])).as_posix().removeprefix("./") != Path(expected_dataset).as_posix().removeprefix("./"):
            raise ValueError(f"{path}: trajectory dataset differs from validated metrics")
        gt = np.asarray(data["gt"], dtype=np.float32)
        predictions = np.asarray(data["predictions"], dtype=np.float32)
        if gt.ndim != 2 or predictions.shape != (len(labels), *gt.shape):
            raise ValueError(f"{path}: invalid GT/prediction shapes {gt.shape}/{predictions.shape}")
        if not np.isfinite(gt).all() or not np.isfinite(predictions).all():
            raise ValueError(f"{path}: non-finite trajectory values")
        if int(data["trajectory_id"]) != tid or str(data["seed"]) != "0":
            raise ValueError(f"{path}: trajectory identity/evaluation seed mismatch")
        if int(data["execution_horizon"]) != 16 or not 0 < gt.shape[0] <= 200:
            raise ValueError(f"{path}: expected at most 200 steps at execution horizon 16")
        return {
            "gt": gt,
            "semantic": predictions[labels.index(semantic_label)],
            "gaussian": predictions[labels.index(gaussian_label)],
            "horizon": int(data["execution_horizon"]),
            "source": str(path),
        }


def load_domain(args):
    overrides = legacy_synth_overrides(args.legacy_synth_equivalence) if args.domain == "synth" else None
    records = {}
    for key, label, model in (
        ("semantic", args.semantic_label, args.semantic_model),
        ("gaussian", args.gaussian_label, args.gaussian_model),
    ):
        records[key] = load_compare_result(
            args.evaluation_dir, model, args.domain, expected_seed=0,
            protocol_overrides=overrides, arm_label=label,
        )
    selection = getattr(args, "selection", "upper-median")
    tid, ratio = select_representative(
        records["semantic"]["trajs"], records["gaussian"]["trajs"], selection=selection,
    )
    trajectory = load_trajectory(
        args.evaluation_dir, tid, args.semantic_label, args.gaussian_label,
        args.semantic_model, args.gaussian_model, records["semantic"]["args"]["dataset_path"],
    )
    return trajectory, records, tid, ratio


def plot_domain(domain, trajectory, records, tid, ratio, dims, output_path,
                semantic_label, gaussian_label, selection="upper-median"):
    """One shared axis per action dimension; legend is outside all data axes."""
    gt = trajectory["gt"]
    if not dims or len(set(dims)) != len(dims) or min(dims) < 0 or max(dims) >= gt.shape[1]:
        raise ValueError(f"--dims must be unique indices in [0, {gt.shape[1] - 1}]")
    semantic_mse = records["semantic"]["trajs"][str(tid)]["mse_effective"]
    gaussian_mse = records["gaussian"]["trajs"][str(tid)]["mse_effective"]
    fig, axes = plt.subplots(len(dims), 1, figsize=(11.7, 2.9 * len(dims) + 1.7), squeeze=False)
    handles = []
    for row, dim in enumerate(dims):
        ax = axes[row, 0]
        lines = [
            ax.plot(gt[:, dim], color=GT_COLOR, lw=2.4, label="Ground truth", zorder=4)[0],
            ax.plot(trajectory["semantic"][:, dim], color=SEMANTIC_COLOR, lw=1.6,
                    label=f"{semantic_label} (semantic)", zorder=3)[0],
            ax.plot(trajectory["gaussian"][:, dim], color=GAUSSIAN_COLOR, lw=1.6,
                    label=f"{gaussian_label} (Gaussian)", zorder=2)[0],
        ]
        if row == 0:
            handles = lines
        ax.set_title(f"Action dim {dim}", fontsize=13, pad=7)
        ax.set_xlabel("Environment step", fontsize=10)
        ax.set_ylabel("Action value", fontsize=10)
        ax.grid(alpha=0.18, linewidth=0.6)
        ax.margins(x=0.015, y=0.13)
        ax.tick_params(labelsize=9)
    suffix = " — clear-gap example" if selection == "max-ratio" else ""
    fig.suptitle(f"{DOMAIN_TITLES[domain]}: action trajectories vs ground truth{suffix}",
                 fontsize=17, y=0.985)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.945),
               ncol=3, frameon=False, fontsize=11, handlelength=2.8, columnspacing=2.4)
    scope = ("Explicit clear-gap example (maximum paired MSE ratio), not a typical result."
             if selection == "max-ratio" else
             "Open-loop replay; one representative, not an aggregate ranking.")
    fig.text(
        0.5, 0.009,
        f"Trajectory {tid} | evaluation seed 0 | {selection} Gaussian/semantic MSE ratio = {ratio:.3f}\n"
        f"Trajectory MSE_eff: semantic = {semantic_mse:.4f}; Gaussian = {gaussian_mse:.4f}. {scope}",
        ha="center", va="bottom", fontsize=10, color="#4A555E",
    )
    fig.tight_layout(rect=(0.025, 0.075, 0.995, 0.895), h_pad=2.1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True,
                        help="Paired evaluation directory containing compare_metrics.json and NPZ arrays")
    parser.add_argument("--semantic-label", default="mos_best")
    parser.add_argument("--gaussian-label", default="mos_best_no_mem")
    parser.add_argument("--semantic-model", required=True)
    parser.add_argument("--gaussian-model", required=True)
    parser.add_argument("--domain", choices=sorted(DOMAIN_TITLES), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="mos_action_trajectories")
    parser.add_argument("--dims", type=int, nargs="+", default=[0, 1, 3, 4])
    parser.add_argument("--selection", choices=("upper-median", "max-ratio"), default="upper-median")
    parser.add_argument("--legacy-synth-equivalence",
                        help="PASS JSON from verify_legacy_synth17d.py; only for audited historical metrics")
    args = parser.parse_args()
    trajectory, records, tid, ratio = load_domain(args)
    output_path = args.output_dir / f"{args.output_prefix}_{args.domain}.png"
    plot_domain(args.domain, trajectory, records, tid, ratio, args.dims, output_path,
                args.semantic_label, args.gaussian_label, selection=args.selection)
    rule = f"{args.selection} Gaussian/semantic effective-MSE ratio"
    provenance = {
        "domain": args.domain, "evaluation_dir": args.evaluation_dir.as_posix(),
        "semantic_label": args.semantic_label, "gaussian_label": args.gaussian_label,
        "semantic_model": args.semantic_model, "gaussian_model": args.gaussian_model,
        "evaluation_seed": 0, "trajectory_id": tid, "representative_rule": rule,
        "gaussian_over_semantic_mse_ratio": ratio, "dims": args.dims,
        "trajectory_mse_effective": {
            args.semantic_label: records["semantic"]["trajs"][str(tid)]["mse_effective"],
            args.gaussian_label: records["gaussian"]["trajs"][str(tid)]["mse_effective"],
        },
        "trajectory_npz": Path(trajectory["source"]).as_posix(), "figure": output_path.as_posix(),
        "scope": "open-loop replay; not closed-loop success or real-robot performance",
        "example_type": "clear-gap, not typical" if args.selection == "max-ratio" else "representative",
    }
    provenance_path = args.output_dir / f"{args.output_prefix}_{args.domain}.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {output_path} | trajectory={tid} | Gaussian/semantic MSE ratio={ratio:.4f}")
    print(f"Wrote {provenance_path}")


if __name__ == "__main__":
    main()
