#!/usr/bin/env python
"""Aggregate zero-noise, fixed-noise, perturbation, and long-horizon controls."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARMS = {"G": "mos_best_no_mem", "Z": "mos_best"}
METRICS = ("mse_effective", "repeat_divergence")


def run_name(arm: str, seed: int) -> str:
    if arm == "mos_best_no_mem":
        return f"db28g_s{seed}"
    return "db28z_s3_soft" if seed == 3 else f"db28z_s{seed}"


def normalized(path: str) -> str:
    return Path(path).as_posix().removeprefix("./")


def load_control(root: Path, arm: str, domain: str, seed: int, suffix: str = "",
                 steps: int = 200, n_traj: int = 10,
                 long_dataset: str = "data/synth_mixture_eval_long") -> dict:
    result = root / f"eval_mos_scratch_{arm}_ckpt5000_{domain}_es0_s{seed}{suffix}"
    path = result / "compare_metrics.json"
    blob = json.loads(path.read_text(encoding="utf-8"))
    args = blob["args"]
    dataset = {
        "synth": "data/synth_mixture_eval_db7_17d",
        "droid": "data/droid_subset_eval",
        "long": long_dataset,
    }[domain]
    expected = {
        "dataset_path": dataset, "seed": "0", "steps": str(steps),
        "execution_horizon": "16", "denoising_steps": "4",
        "history_source": "pred", "fixed_beta_value": "0.0",
        "force_fixed_beta": "True", "fixed_noise_draw": str("fixed_noise" in suffix),
        "fixed_noise_std": "0.0" if "eta0" in suffix else "None",
        "obs_image_noise_std": "8.0" if "perturb" in suffix else "0.0",
        "obs_state_noise_std": "0.02" if "perturb" in suffix else "0.0",
    }
    for key, want in expected.items():
        got = args.get(key)
        if key in {"fixed_noise_std", "obs_image_noise_std", "obs_state_noise_std"} and got is None:
            got = "None" if key == "fixed_noise_std" else "0.0"
        if key == "dataset_path":
            valid = normalized(str(got)) == normalized(want)
        else:
            valid = str(got) == want
        if not valid:
            raise ValueError(f"{path}: {key}={got!r}, expected {want!r}")
    expected_model = f"{root}/{run_name(arm, seed)}/checkpoint-5000"
    model_paths = args["model_paths"]
    if isinstance(model_paths, str):
        import ast
        model_paths = ast.literal_eval(model_paths)
    if len(model_paths) != 1 or normalized(model_paths[0]) != normalized(expected_model):
        raise ValueError(f"{path}: wrong checkpoint {model_paths}, expected {expected_model}")
    if len(blob.get("arms", {})) != 1:
        raise ValueError(f"{path}: expected one arm")
    record = next(iter(blob["arms"].values()))
    if set(record["trajs"]) != {str(i) for i in range(n_traj)}:
        raise ValueError(f"{path}: expected trajectories 0..{n_traj - 1}")
    for values in record["trajs"].values():
        for metric in METRICS:
            if values.get(metric) is None or not np.isfinite(float(values[metric])):
                raise ValueError(f"{path}: invalid {metric}")
    return {"path": str(path), "trajs": record["trajs"]}


def mean(record: dict, metric: str) -> float:
    return float(np.mean([float(v[metric]) for v in record["trajs"].values()]))


def paired(reference: dict, candidate: dict, metric: str, rng: np.random.Generator,
           n_boot: int) -> dict:
    if reference["trajs"].keys() != candidate["trajs"].keys():
        raise ValueError("unpaired trajectory sets")
    tids = sorted(reference["trajs"], key=int)
    reference_values = np.asarray([float(reference["trajs"][tid][metric]) for tid in tids])
    candidate_values = np.asarray([float(candidate["trajs"][tid][metric]) for tid in tids])
    if np.any(candidate_values == 0):
        return {
            "ratio_reference_over_candidate": None,
            "ci95": None,
            "reason": "zero candidate denominator; use absolute metrics instead",
            "n_trajectory_clusters": len(tids),
        }
    ratios = reference_values / candidate_values
    boot = ratios[rng.integers(0, len(ratios), size=(n_boot, len(ratios)))].mean(axis=1)
    return {
        "ratio_reference_over_candidate": float(ratios.mean()),
        "ci95": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
        "n_trajectory_clusters": len(ratios),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs")
    parser.add_argument("--training-seed", type=int, default=2)
    parser.add_argument("--n-boot", type=int, default=20000)
    parser.add_argument("--long-dataset", default="data/synth_mixture_eval_long",
                        help="Exact long-eval dataset path recorded in compare_metrics.json")
    parser.add_argument("--out-prefix", default="runs/mos_paper_controls")
    args = parser.parse_args()
    root = Path(args.run_dir)
    seed = args.training_seed
    rng = np.random.default_rng(20261027)
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    load = lambda arm, domain, suffix="", steps=200, n=10: load_control(
        root, ARMS[arm], domain, seed, suffix, steps, n, args.long_dataset
    )
    records = {
        "G": load("G", "synth"), "G_fixed": load("G", "synth", "_fixed_noise"),
        "G_perturb": load("G", "synth", "_perturb"),
        "Z": load("Z", "synth"), "Z_eta0": load("Z", "synth", "_eta0"),
        "Z_perturb": load("Z", "synth", "_perturb"),
        "Z_perturb_eta0": load("Z", "synth", "_perturb_eta0"),
        "G_droid30": load("G", "droid", "_all30", n=30),
        "Z_droid30": load("Z", "droid", "_all30", n=30),
        "Z_droid30_eta0": load("Z", "droid", "_eta0", n=30),
        "G_long": load("G", "long", steps=400),
        "Z_long_eta0": load("Z", "long", "_eta0", steps=400),
        "G_long_perturb": load("G", "long", "_perturb", steps=400),
        "Z_long_perturb_eta0": load("Z", "long", "_perturb_eta0", steps=400),
    }
    report = {
        "scope": "from-scratch DiT; training seed 2; open-loop controls",
        "records": {key: {"source": rec["path"], **{m: mean(rec, m) for m in METRICS}}
                    for key, rec in records.items()},
        "comparisons": {},
    }
    for name, ref, cand in (
        ("eta004_over_eta0_synth", "Z", "Z_eta0"),
        ("eta004_over_eta0_droid30", "Z_droid30", "Z_droid30_eta0"),
        ("G_over_Z_eta0_long", "G_long", "Z_long_eta0"),
        ("G_over_Z_eta0_perturb", "G_perturb", "Z_perturb_eta0"),
        ("G_over_Z_eta0_long_perturb", "G_long_perturb", "Z_long_perturb_eta0"),
        ("G_fixed_over_G_resampled", "G_fixed", "G"),
    ):
        report["comparisons"][name] = {
            metric: paired(records[ref], records[cand], metric, rng, args.n_boot)
            for metric in METRICS
        }

    points = [
        ("Gaussian, resampled", "G", "#5B6573", "s"),
        ("Gaussian, fixed at deployment", "G_fixed", "#E8843C", "s"),
        ("Semantic, eta = 0.04", "Z", "#2878B5", "o"),
        ("Semantic, eta = 0", "Z_eta0", "#169873", "o"),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    for label, key, color, marker in points:
        ax.scatter(mean(records[key], "repeat_divergence"),
                   mean(records[key], "mse_effective"), s=90, color=color,
                   marker=marker, label=label)
    ax.set_xlabel(r"repeat $\|\Delta\|$ (lower is better)")
    ax.set_ylabel("Effective action MSE (lower is better)")
    ax.set_title("Source content and randomness: synthetic held-out")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(prefix.with_name(prefix.name + "_source_pricing").with_suffix(".png"),
                dpi=180, bbox_inches="tight")
    plt.close(fig)
    prefix.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {prefix.with_suffix('.json')} and source-pricing figure")


if __name__ == "__main__":
    main()
