"""Validated access to compare_flow_sources results and paired bootstrap statistics.

The estimators match the review contract: means over all (trajectory, eval-seed)
records, per-record reference/candidate ratios, and whole-trajectory resampling.
No model or training imports are needed.
"""

import ast
import json
import math
from pathlib import Path

import numpy as np

DOMAIN_PROTOCOL = {
    "synth": {
        "dataset_path": "data/synth_mixture_eval_db7_17d",
        "segment_meta_path": "data/synth_mixture_eval_db7_17d/meta/segment_schedule.jsonl",
        "high_level_task_from_meta": "True",
    },
    "droid": {
        "dataset_path": "data/droid_subset_eval",
        "segment_meta_path": "None",
        "high_level_task_from_meta": "False",
    },
}
COMMON_PROTOCOL = {
    "steps": "200",
    "execution_horizon": "16",
    "history_replan_stride": "None",
    "history_source": "pred",
    "denoising_steps": "4",
    "repeat_probe": "True",
    "fixed_noise_draw": "False",
    "force_fixed_beta": "True",
    "fixed_beta_value": "0.0",
    "obs_image_noise_std": "0.0",
    "obs_state_noise_std": "0.0",
}


def _list_arg(value):
    return ast.literal_eval(value) if isinstance(value, str) else value


def _same_path(actual, expected):
    return Path(actual).as_posix().removeprefix("./") == Path(expected).as_posix().removeprefix("./")


def legacy_synth_overrides(report_path):
    """Read an explicit verify_legacy_synth17d.py PASS report, never a bare alias."""
    if report_path is None:
        return None
    path = Path(report_path)
    report = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "status": "PASS", "legacy_dataset": "data/synth_mixture_eval_db7",
        "canonical_dataset": "data/synth_mixture_eval_db7_17d", "episodes": 10,
        "state_prefix_17d_equal": True, "actions_equal": True,
        "task_indices_equal": True, "video_sha256_equal": True,
        "metadata_equal": True, "evaluation_statistics_source": "checkpoint",
        "dataset_statistics_compared": False,
    }
    for key, expected in required.items():
        if report.get(key) != expected:
            raise ValueError(f"{path}: invalid legacy-equivalence field {key}={report.get(key)!r}")
    return {
        "dataset_path": report["legacy_dataset"],
        "segment_meta_path": report["legacy_dataset"] + "/meta/segment_schedule.jsonl",
    }


def load_compare_result(
    run_dir, expected_model_path, domain, expected_seed=0, protocol_overrides=None,
    arm_label=None,
):
    """Load one arm's JSON, rejecting mixed protocols or mislabeled ckpts.

    ``arm_label`` enables one explicitly named arm from a paired evaluation;
    omitting it preserves the strict historical single-arm behavior.
    ``protocol_overrides`` allows only an explicit protocol change, never a
    blanket bypass of the evaluation contract.
    """
    run_dir = Path(run_dir)
    path = run_dir / "compare_metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}")
    blob = json.loads(path.read_text(encoding="utf-8"))
    args = blob["args"]
    protocol = COMMON_PROTOCOL | DOMAIN_PROTOCOL[domain] | (protocol_overrides or {})
    for key, want in protocol.items():
        got = args.get(key)
        # Pre-E4 JSONs omitted disabled observation-noise fields.
        if key in ("obs_image_noise_std", "obs_state_noise_std") and got in (None, "None"):
            got = "0.0"
        if key in ("dataset_path", "segment_meta_path") and got is not None:
            valid = _same_path(got, want)
        else:
            valid = got == want
        if not valid:
            raise ValueError(f"{path}: protocol {key}={got!r}, expected {want!r}")
    if str(args.get("seed")) != str(expected_seed) or args.get("seeds") != "None":
        raise ValueError(f"{path}: expected one eval seed={expected_seed}, got {args.get('seed')}")
    if _list_arg(args["traj_ids"]) != list(range(10)):
        raise ValueError(f"{path}: trajectory set differs from 0..9")
    model_paths = _list_arg(args["model_paths"])
    arms = blob.get("arms", {})
    if arm_label is None:
        if len(model_paths) != 1 or not _same_path(model_paths[0], expected_model_path):
            raise ValueError(f"{path}: model_paths={model_paths}, expected {expected_model_path}")
        if len(arms) != 1:
            raise ValueError(f"{path}: expected exactly one arm")
        arm = next(iter(arms.values()))
    else:
        labels = _list_arg(args.get("labels"))
        if not isinstance(labels, list) or len(labels) != len(model_paths) or len(set(labels)) != len(labels):
            raise ValueError(f"{path}: paired evaluation requires unique aligned labels")
        if len(model_paths) != 2 or len(set(model_paths)) != 2 or arm_label not in labels:
            raise ValueError(f"{path}: invalid paired model paths or missing label {arm_label}")
        if arm_label not in arms:
            raise ValueError(f"{path}: missing arm {arm_label}")
        arm = arms[arm_label]
        if not _same_path(model_paths[labels.index(arm_label)], expected_model_path):
            raise ValueError(f"{path}: label {arm_label} is not {expected_model_path}")
    if not _same_path(arm["model_path"], expected_model_path):
        raise ValueError(f"{path}: arm checkpoint is {arm['model_path']}, not {expected_model_path}")
    trajs = arm["trajs"]
    if set(trajs) != {str(i) for i in range(10)}:
        raise ValueError(f"{path}: incomplete trajectory keys: {sorted(trajs)}")
    for metric in ("mse_effective", "repeat_divergence"):
        for tid, record in trajs.items():
            value = record.get(metric)
            if value is None or not math.isfinite(float(value)):
                raise ValueError(f"{path}: invalid {metric} at trajectory {tid}")
    return {"path": str(path), "args": args, "trajs": trajs}


def _clusters(records, metric):
    by_traj = {}
    for record in records:
        for tid, values in record["trajs"].items():
            by_traj.setdefault(tid, []).append(float(values[metric]))
    expected_tids = {str(i) for i in range(10)}
    if set(by_traj) != expected_tids:
        raise ValueError("paired records do not cover the same ten trajectories")
    clusters = [by_traj[str(i)] for i in range(10)]
    widths = {len(c) for c in clusters}
    if len(widths) != 1:
        raise ValueError(f"ragged eval-seed counts: {sorted(map(len, clusters))}")
    return np.asarray(clusters, dtype=np.float64)


def _finite_ci(samples):
    samples = np.asarray(samples)
    if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError("non-finite bootstrap distribution")
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def mean_cluster_ci(records, metric, rng, n_boot=20000):
    values = _clusters(records, metric)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=(1, 2))
    lo, hi = _finite_ci(means)
    return {
        "mean": float(values.mean()),
        "ci95": [lo, hi],
        "n_traj_clusters": len(values),
        "n_records": int(values.size),
    }


def paired_ratio_ci(reference, candidate, metric, rng, n_boot=20000):
    """Mean(reference_i / candidate_i), bootstrapped with shared traj indices."""
    ref, cand = _clusters(reference, metric), _clusters(candidate, metric)
    if ref.shape != cand.shape:
        raise ValueError(f"unpaired eval-seed counts: {ref.shape} vs {cand.shape}")
    if np.any(cand == 0):
        raise ValueError(f"zero {metric} denominator: ratio undefined; inspect before aggregation")
    ratios = ref / cand
    idx = rng.integers(0, len(ref), size=(n_boot, len(ref)))
    boot = ratios[idx].mean(axis=(1, 2))
    lo, hi = _finite_ci(boot)
    return {
        "ratio_reference_over_candidate_mean": float(ratios.mean()),
        "ci95": [lo, hi],
        "n_traj_clusters": len(ref),
        "n_records": int(ref.size),
        "per_trajectory_ratio_mean": {
            str(i): float(ratios[i].mean()) for i in range(len(ref))
        },
    }
