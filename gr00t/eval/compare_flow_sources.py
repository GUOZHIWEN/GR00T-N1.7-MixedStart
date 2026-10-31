# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Open-loop comparison of flow-source arms (gaussian / history / memory / mix).

Evaluates several checkpoints on the same trajectories and produces:
  1. Per-trajectory overlay plots: GT vs each arm's predicted actions.
  2. A metrics table (MSE / MAE per arm per trajectory) -> compare_metrics.json.
  3. A repeatability probe (--repeat-probe): each trajectory is rolled out
     twice per arm (policy.reset() before each run), and the mean |run1-run2|
     divergence is reported/plotted. This is the practical "multimodality
     collapse" meter: the hard-anchor memory arm (eta=0) should show ~0
     repeat variance, while noisy arms keep non-zero variance.

Example (three arms on the demo DROID sample):

python gr00t/eval/compare_flow_sources.py \
  --dataset-path demo_data/droid_sample \
  --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
  --model-paths runs/p0_hist/checkpoint-2000 runs/mem_hard/checkpoint-2000 runs/mem_soft/checkpoint-2000 \
  --labels history memory_hard memory_soft \
  --traj-ids 0 --steps 200 --execution-horizon 16 --denoising-steps 4 \
  --out-dir runs/flow_source_compare
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import gc
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import tyro

if TYPE_CHECKING:
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy


logger = logging.getLogger(__name__)


def rollout_trajectory(
    policy: Gr00tPolicy,
    loader: LeRobotEpisodeLoader,
    traj_id: int,
    embodiment_tag: EmbodimentTag,
    action_keys: list[str],
    steps: int,
    execution_horizon: int,
    info_out: list[dict[str, Any]] | None = None,
    high_level_task: str | None = None,
    history_source: str = "pred",
    executed_per_key_out: list[dict[str, np.ndarray]] | None = None,
    fixed_noise_draw: bool = False,
    fixed_noise_seed: int = 12345,
    obs_image_noise_std: float = 0.0,
    obs_state_noise_std: float = 0.0,
    obs_perturb_seed: int = 777,
    rng_offset: int = 0,
) -> np.ndarray:
    """Open-loop rollout; returns predicted actions (T, D_action).

    When ``info_out`` is given, the per-replan info dicts from
    ``policy.get_action`` (e.g. v4.1 ``confidence`` / T1 ``beta_hat``) are
    appended to it. When ``high_level_task`` is given (dual text port), it
    drives the task text while the per-frame dataset instruction is demoted
    to the sub-instruction port.

    ``history_source`` (survey §13.51, E1): ``pred`` keeps the legacy self-fed
    history buffer (open-loop protocol); ``gt`` overwrites the buffer at every
    replan point with the ground-truth action window the robot would actually
    have executed, giving the history channel the same causal consistency the
    memory channel gets from re-encoding fresh observations. This is the
    decisive control for whether the mixed-source gain theorem (Prop. 1/2)
    survives protocol symmetrization.

    ``executed_per_key_out`` (survey §13.51, E2): when given, the per-frame
    EXECUTED actions are additionally appended as ``{bare_key: (D_key,)}``
    dicts (truncated to the same length as the returned array) — exactly what
    scripts/build_relabeled_rollout_dataset.py needs to write the
    ``action_relabeled`` column.

    Observation perturbation (E4, η-necessity terminal test): when
    ``obs_image_noise_std`` / ``obs_state_noise_std`` > 0, i.i.d. Gaussian
    perturbations are injected into every observation BEFORE parsing —
    pixel noise on ``video.*`` (0–255 scale, clipped back to the original
    dtype) and scale-free relative noise on ``state.*``
    (``noise = std · |v| · N(0,1)``). The perturbation RNG is seeded per
    (traj, replan step, pass) as ``obs_perturb_seed + traj_id·100003 +
    step_count·1009 + rng_offset``: identical across arms (paired) but
    independent across repeat passes (``rng_offset`` = run seed for run1,
    run seed + 500000 for run2), mimicking deployments where disturbances
    differ between executions. Perturbation never leaks across passes and
    is applied to BOTH runs, so repeat‖Δ‖ under perturbation stays an
    honest run-to-run measure.
    """
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.utils import parse_observation_gr00t
    from gr00t.eval.open_loop_eval import parse_action_gr00t

    assert history_source in ("pred", "gt"), (
        f"history_source must be 'pred' or 'gt', got {history_source!r}"
    )

    traj = loader[traj_id]
    actual_steps = min(steps, len(traj))

    # No history leakage across trajectories / repeats.
    policy.reset()

    modality_configs = deepcopy(loader.modality_configs)
    modality_configs.pop("action")
    action_horizon = len(loader.modality_configs["action"].delta_indices)
    state_keys = loader.modality_configs["state"].modality_keys

    pred_action_across_time = []
    for step_count in range(0, actual_steps, execution_horizon):
        data_point = extract_step_data(traj, step_count, modality_configs, embodiment_tag)
        obs: dict[str, Any] = {}
        for k, v in data_point.states.items():
            obs[f"state.{k}"] = v
        for k, v in data_point.images.items():
            obs[f"video.{k}"] = np.array(v)
        for language_key in loader.modality_configs["language"].modality_keys:
            obs[language_key] = high_level_task if high_level_task else data_point.text
        if obs_image_noise_std > 0.0 or obs_state_noise_std > 0.0:
            # E4: seeded per (traj, step, pass) — paired across arms,
            # independent across repeat passes. See docstring.
            _prng = np.random.default_rng(
                obs_perturb_seed + traj_id * 100003 + step_count * 1009 + rng_offset
            )
            if obs_state_noise_std > 0.0:
                for _k in [k for k in obs if k.startswith("state.")]:
                    _arr = np.asarray(obs[_k], dtype=np.float32)
                    obs[_k] = _arr + _prng.standard_normal(
                        _arr.shape, dtype=np.float32
                    ) * (obs_state_noise_std * np.abs(_arr))
            if obs_image_noise_std > 0.0:
                for _k in [k for k in obs if k.startswith("video.")]:
                    _img = np.asarray(obs[_k])
                    _noisy = _img.astype(np.float32) + _prng.standard_normal(
                        _img.shape, dtype=np.float32
                    ) * obs_image_noise_std
                    obs[_k] = np.clip(_noisy, 0, 255).astype(_img.dtype)
        parsed_obs = parse_observation_gr00t(obs, loader.modality_configs)
        if high_level_task:
            # Dual text port: episode-level task drives `text`; the per-frame
            # instruction becomes the sub-instruction.
            parsed_obs["sub_instruction"] = [data_point.text]
        if history_source == "gt" and step_count > 0:
            # E1 (§13.51): overwrite the self-fed history buffer with the GT
            # action window [t-stride, t-stride+H) — what the previous replan's
            # chunk would have been under perfect tracking. The episode start
            # keeps the empty buffer (cold start -> Gaussian source), matching
            # the training-time has_action_history=False semantics there.
            win_start = step_count - execution_horizon
            if win_start >= 0:
                ep_len = len(traj)
                win_idx = [min(win_start + j, ep_len - 1) for j in range(action_horizon)]
                gt_chunks = {
                    key: np.stack(
                        [
                            np.asarray(traj[f"action.{key}"].iloc[i], dtype=np.float32)
                            for i in win_idx
                        ],
                        axis=0,
                    )[None]
                    for key in action_keys
                }  # bare key -> (1, H, D_key), the self-fed buffer's layout
                anchor = min(win_start, ep_len - 1)
                gt_states = [
                    {
                        skey: np.asarray(
                            traj[f"state.{skey}"].iloc[anchor], dtype=np.float32
                        )[None]
                        for skey in state_keys
                    }
                ]
                policy.override_history_with(gt_chunks, gt_states)
        if fixed_noise_draw:
            # G-fixed control (survey §13.98): identical source-noise draw at
            # every call AND across repeat passes — determinism by construction.
            torch.manual_seed(fixed_noise_seed + traj_id)
        _action_chunk, _info = policy.get_action(parsed_obs)
        if info_out is not None and _info:
            info_out.append({"step": step_count, **_info})
        action_chunk = parse_action_gr00t(_action_chunk)
        for j in range(execution_horizon):
            per_key = {
                key: np.atleast_1d(np.atleast_1d(action_chunk[f"action.{key}"])[j]).astype(
                    np.float32
                )
                for key in action_keys
            }
            if executed_per_key_out is not None:
                executed_per_key_out.append(per_key)
            concat_pred_action = np.concatenate(
                [per_key[key] for key in action_keys],
                axis=0,
            )
            pred_action_across_time.append(concat_pred_action)
    pred_arr = np.array(pred_action_across_time)[:actual_steps]
    if executed_per_key_out is not None:
        del executed_per_key_out[len(pred_arr) :]
    return pred_arr


def extract_gt(traj, action_keys: list[str], actual_steps: int) -> np.ndarray:
    np_dict = {f"action.{c}": np.vstack([arr for arr in traj[f"action.{c}"]]) for c in action_keys}
    return np.concatenate([np_dict[f"action.{c}"] for c in action_keys], axis=-1)[:actual_steps]


def estimate_temporal_lag(gt: np.ndarray, pred: np.ndarray, max_lag: int = 8) -> list[int]:
    """Per-dim best temporal shift of pred vs gt, scanned over lag in [-max_lag, +max_lag].

    Convention: lag > 0 means pred LAGS gt (pred[lag:] aligns with gt[:-lag]);
    lag < 0 means pred LEADS gt. Returns one best lag per action dim.
    Guards: dims whose best-shift window would be shorter than 16 steps keep lag 0.
    """
    lags: list[int] = []
    for d in range(gt.shape[1]):
        best_lag, best_mse = 0, np.inf
        for lag in range(-max_lag, max_lag + 1):
            if lag > 0:
                g, p = gt[:-lag, d], pred[lag:, d]
            elif lag < 0:
                g, p = gt[-lag:, d], pred[:lag, d]
            else:
                g, p = gt[:, d], pred[:, d]
            if len(g) < 16:
                continue
            m = float(np.mean((g - p) ** 2))
            if m < best_mse:
                best_mse, best_lag = m, lag
        lags.append(int(best_lag))
    return lags


def shifted_mse(gt: np.ndarray, pred: np.ndarray, lag: int) -> float:
    """MSE over the overlap after shifting pred by lag.

    Same sign convention as estimate_temporal_lag (lag>0: pred lags gt).
    Diagnostic only: a large drop vs the unshifted MSE evidences a genuine
    phase shift; a tiny drop means the measured lag is a smoothing artifact.
    """
    if lag > 0:
        g, p = gt[:-lag], pred[lag:]
    elif lag < 0:
        g, p = gt[-lag:], pred[:lag]
    else:
        g, p = gt, pred
    return float(np.mean((g - p) ** 2))


def split_dead_dims(gt: np.ndarray, eps: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Split action dims into (live, dead) by GT per-dim std.

    Dead dims (std < eps) carry no learnable signal — e.g. the rot6d delta
    channels of the synthetic mixture dataset are iid sensor-scale noise
    (~5e-4 raw) that minmax normalization amplifies to the full [-1, 1]
    range, so a checkpoint's output there is unpredictable noise that
    dominates the all-dims MSE and dilutes every arm comparison. Reporting
    effective (live-only) and dead MSE separately keeps the metric honest.
    """
    stds = gt.std(axis=0)
    return np.where(stds >= eps)[0], np.where(stds < eps)[0]


def dim_masked_mse(gt: np.ndarray, pred: np.ndarray, dims: np.ndarray) -> float | None:
    """MSE restricted to the given action dims; None when dims is empty."""
    if dims is None or len(dims) == 0:
        return None
    return float(np.mean((gt[:, dims] - pred[:, dims]) ** 2))


def save_trajectory_arrays(
    gt: np.ndarray,
    preds: dict[str, np.ndarray],
    results: dict[str, Any],
    args: ArgsConfig,
    traj_id: int,
    save_path: Path,
) -> None:
    """Save the exact plotted arrays so paired figures never rely on cropped PNGs."""
    if not preds:
        raise ValueError(f"trajectory {traj_id}: no predictions to save")
    gt = np.asarray(gt, dtype=np.float32)
    if gt.ndim != 2 or not np.isfinite(gt).all():
        raise ValueError(f"trajectory {traj_id}: invalid ground-truth array {gt.shape}")
    labels = list(preds)
    if len(set(labels)) != len(labels):
        raise ValueError("trajectory labels must be unique")
    predictions = []
    model_paths = []
    for label in labels:
        pred = np.asarray(preds[label], dtype=np.float32)
        if pred.shape != gt.shape or not np.isfinite(pred).all():
            raise ValueError(
                f"trajectory {traj_id}, arm {label}: prediction {pred.shape} "
                f"does not match finite GT {gt.shape}"
            )
        predictions.append(pred)
        model_paths.append(results[label]["model_path"])
    np.savez_compressed(
        save_path,
        schema_version=np.asarray(1, dtype=np.int64),
        gt=gt,
        predictions=np.stack(predictions).astype(np.float32, copy=False),
        labels=np.asarray(labels, dtype=np.str_),
        model_paths=np.asarray(model_paths, dtype=np.str_),
        trajectory_id=np.asarray(traj_id, dtype=np.int64),
        dataset_path=np.asarray(args.dataset_path, dtype=np.str_),
        seed=np.asarray("" if args.seed is None else str(args.seed), dtype=np.str_),
        execution_horizon=np.asarray(args.execution_horizon, dtype=np.int64),
        denoising_steps=np.asarray(args.denoising_steps, dtype=np.int64),
    )


def plot_overlay(
    gt: np.ndarray,
    preds: dict[str, np.ndarray],
    traj_id: int,
    execution_horizon: int,
    save_path: Path,
    max_dims: int,
) -> None:
    """GT (black) + one line per arm, one subplot per action dim."""
    from matplotlib import pyplot as plt

    n_dims = min(gt.shape[1], max_dims)
    fig, axes = plt.subplots(nrows=n_dims, ncols=1, figsize=(9, 3.2 * n_dims), squeeze=False)
    colors = plt.cm.tab10.colors
    for d in range(n_dims):
        ax = axes[d][0]
        ax.plot(gt[:, d], color="black", linewidth=2.0, label="gt action")
        for i, (label, pred) in enumerate(preds.items()):
            ax.plot(pred[:, d], color=colors[i % 10], alpha=0.85, linewidth=1.2, label=label)
        for j in range(0, len(gt), execution_horizon):
            ax.plot(j, gt[j, d], "ro", markersize=3)
        ax.set_title(f"Action dim {d}")
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Trajectory {traj_id}: GT vs flow-source arms", fontsize=13)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_repeatability(
    repeat_div: dict[str, float],
    traj_id: int,
    save_path: Path,
) -> None:
    """Bar chart of mean |run1 - run2| per arm (multimodality-collapse meter)."""
    from matplotlib import pyplot as plt

    labels = list(repeat_div.keys())
    values = [repeat_div[k] for k in labels]
    fig, ax = plt.subplots(figsize=(max(6, 2 * len(labels)), 4))
    bars = ax.bar(labels, values, color=plt.cm.tab10.colors[: len(labels)])
    ax.set_ylabel("mean |run1 - run2| (unnormalized action units)")
    ax.set_title(f"Trajectory {traj_id}: repeat divergence (0 = deterministic/collapsed)")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_repeat_overlay(
    gt: np.ndarray,
    arm_runs: dict[str, tuple[np.ndarray, np.ndarray, float]],
    traj_id: int,
    dims: list[int],
    save_path: Path,
) -> None:
    """Bundle-tightness view of determinism: GT + run1 (solid) + run2 (dashed)
    per arm (rows) over selected action dims (cols). A deterministic arm's two
    runs collapse onto one curve; a stochastic arm's bundle visibly fans out.
    ``arm_runs`` maps label -> (run1, run2, repeat_divergence)."""
    from matplotlib import pyplot as plt

    labels = list(arm_runs.keys())
    n_rows, n_cols = len(labels), len(dims)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 2.9 * n_rows), squeeze=False)
    colors = plt.cm.tab10.colors
    for r, label in enumerate(labels):
        run1, run2, div = arm_runs[label]
        for c, d in enumerate(dims):
            ax = axes[r][c]
            ax.plot(gt[:, d], color="black", linewidth=1.8, alpha=0.7,
                    label="gt" if (r == 0 and c == 0) else None)
            ax.plot(run1[:, d], color=colors[r % 10], linewidth=1.4,
                    label="run1" if (r == 0 and c == 0) else None)
            ax.plot(run2[:, d], color=colors[r % 10], linewidth=1.4, linestyle="--",
                    label="run2" if (r == 0 and c == 0) else None)
            if r == 0:
                ax.set_title(f"action dim {d}", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{label}\nrepeat|Δ|={div:.4f}", fontsize=9)
            ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8, loc="upper right")
    fig.suptitle(
        f"Trajectory {traj_id}: repeat overlay — solid=run1, dashed=run2 "
        f"(tight bundle = deterministic start)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def _cluster_bootstrap_ci95(
    clusters: list[list[float]], rng: np.random.Generator, n_boot: int = 10000
) -> tuple[float, float]:
    """95% CI of the pooled mean via CLUSTER bootstrap (review P1-1): the
    (traj × seed) samples are not i.i.d. — seeds within one trajectory are
    correlated — so whole trajectories are resampled with replacement and all
    of their seed records come along. Returns the (2.5%, 97.5%) band."""
    clusters = [np.asarray(c, dtype=np.float64) for c in clusters if len(c)]
    if len(clusters) < 2:
        v = float(np.concatenate(clusters).mean()) if clusters else 0.0
        return v, v
    sizes = np.array([len(c) for c in clusters])
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    flat = np.concatenate(clusters)
    idx = rng.integers(0, len(clusters), size=(n_boot, len(clusters)))
    stats = np.empty(n_boot)
    for b in range(n_boot):
        stats[b] = np.concatenate(
            [flat[offsets[i]:offsets[i + 1]] for i in idx[b]]
        ).mean()
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def plot_summary_ci(
    table_rows: list[dict[str, Any]],
    save_path: Path,
    ratio_block: dict[str, Any],
    ref_label: str | None,
) -> None:
    """Multi-seed error-bar summary (paper §8 item 9): per-arm bar charts of
    repeat‖Δ‖ and mse_eff with 95% CIs over (traj × seed), plus the paired-
    bootstrap ratio CI vs the reference arm annotated on the bars."""
    from matplotlib import pyplot as plt

    labels = [r["arm"] for r in table_rows]
    colors = plt.cm.tab10.colors[: len(labels)]
    fig, axes = plt.subplots(1, 2, figsize=(max(9, 2.6 * len(labels)), 4.2))
    for ax, prefix, title in (
        (axes[0], "ms_repeat_divergence", "repeat‖Δ‖ (run-to-run divergence)"),
        (axes[1], "ms_mse_effective", "mse_eff (live dims)"),
    ):
        means = [r.get(f"{prefix}_mean", 0.0) for r in table_rows]
        cis = [r.get(f"{prefix}_ci95", 0.0) for r in table_rows]
        bars = ax.bar(labels, means, yerr=cis, capsize=4, color=colors,
                      error_kw={"linewidth": 1.2})
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        for b, m, ci in zip(bars, means, cis):
            ax.text(b.get_x() + b.get_width() / 2, m + ci, f"{m:.4f}",
                    ha="center", va="bottom", fontsize=8)
        ax.tick_params(axis="x", rotation=15)
    if ratio_block and ref_label:
        lines = []
        for pair_name, metrics in ratio_block.items():
            for metric, blk in metrics.items():
                lines.append(
                    f"{metric} [{pair_name}]: {blk['ratio_ref_over_arm_mean']:.2f}× "
                    f"CI95 [{blk['ci95'][0]:.2f}, {blk['ci95'][1]:.2f}]"
                )
        fig.text(0.5, -0.04, "\n".join(lines), ha="center", fontsize=9)
        fig.tight_layout(rect=[0, 0.06, 1, 1])
    else:
        fig.tight_layout()
    fig.suptitle("Multi-seed arm comparison (mean ± 95% CI over traj × seed)",
                 fontsize=12)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


@dataclass
class ArgsConfig:
    dataset_path: str
    """Path to the LeRobot-format dataset."""

    embodiment_tag: str
    """Embodiment tag (name or value, case-insensitive)."""

    model_paths: list[str]
    """Checkpoints to compare (one per flow-source arm)."""

    labels: list[str] | None = None
    """Display labels, same length as model_paths. Default: directory names."""

    traj_ids: list[int] = field(default_factory=lambda: [0])
    steps: int = 200
    execution_horizon: int = 16
    history_replan_stride: int | None = None
    """P0-2/T1 lag fix: rtc history-slice offset used at inference. None
    (default) = execution_horizon, which is the correct alignment for this
    open-loop replan protocol; 0 disables the override (legacy behavior —
    reproduces the lag-aliasing artifact on periodic signals)."""
    history_source: str = "pred"
    """E1 closed-loop protocol probe (survey §13.51): what feeds the policy's
    history buffer at each replan point. ``pred`` (default) = the model's own
    previous prediction — the legacy open-loop protocol where the history
    channel drifts causally. ``gt`` = overwrite the buffer with the
    ground-truth action window the robot would actually have executed, giving
    the history channel the same causal consistency the memory channel gets
    from re-encoding fresh observations. The decisive test of the mixed-source
    gain theorem: if routed-β̂ under ``gt`` beats pure-z₀, the mixture's
    failure under ``pred`` was protocol asymmetry, not content."""
    fixed_noise_draw: bool = False
    """G-fixed control (survey §13.98, theory-doc §1.3/§5.3): reseed the RNG to
    the SAME trajectory-fixed seed right before every policy call, so the
    source-noise draw is identical across calls and across repeat passes —
    determinism achieved by "never resampling", with no encoder involved.
    The decisive control for whether the semantic encoder is NECESSARY for
    low repeat, or merely one way to obtain it. Expect repeat ≡ 0 by
    construction; the metric under test is precision (mse_eff) — a fixed
    Gaussian draw is an arbitrary mode sample, not a conditional-mean anchor."""
    fixed_noise_seed: int = 12345
    """Base seed for --fixed-noise-draw; per-trajectory draw uses seed+traj_id
    (each trajectory = one condition gets its own fixed ε, cf. the theory
    doc's per-condition fixed noise)."""
    denoising_steps: int = 4
    out_dir: str = "runs/flow_source_compare"
    repeat_probe: bool = True
    """Roll each trajectory out twice per arm and report |run1-run2| divergence."""
    repeat_overlay: bool = False
    """With repeat_probe: also write traj_*_repeat_overlay.png (GT + run1 solid +
    run2 dashed per arm over dims 0-2 + gripper) — determinism made visible.
    Plotting only; metrics untouched."""
    max_plot_dims: int = 8
    max_lag: int = 8
    """Temporal-alignment diagnostic: per-dim lag scan in [-max_lag, +max_lag].
    lag>0 means the prediction lags GT; lag<0 means it appears to lead."""
    segment_meta_path: str | None = None
    """Optional segment_schedule.jsonl written by
    scripts/build_synthetic_mixture_dataset.py. Enables stratified metrics
    (segment-head vs segment-mid MSE and β̂ split) — the direct test of the
    mixed-source paper's Propositions 1/2."""
    segment_head_steps: int = 8
    """Steps after a segment start counted as the cold-start 'head' region."""
    dead_dim_eps: float = 1e-3
    """GT per-dim std below this (raw action units) marks a dead, signal-free
    dim (e.g. constant/white-noise rot6d deltas). Metrics are reported both
    all-dims (backward compatible) and effective (live dims only)."""
    high_level_task: str | None = None
    """Dual text port evaluation: when set, this episode-level task drives the
    policy's task text while each frame's dataset instruction is fed as the
    sub-instruction (requires a checkpoint trained with
    sub_instruction_mode concat/replace). None = legacy single-port eval."""
    force_fixed_beta: bool = False
    """Fixed-β control arm: after loading, disable confidence routing so the
    scalar ``memory_start_mix_beta`` drives the mixture instead of β̂. MUST be
    a post-load override — flipping the flag in config.json before load would
    shrink the confidence head (4→2 outputs) and break weight loading."""
    fixed_beta_value: float | None = None
    """Source-simplex completion: post-load override of the scalar
    ``memory_start_mix_beta`` (e.g. 0.0 = pure memory source, 1.0 = pure
    history source). None = keep the checkpoint's stored scalar. Only
    meaningful together with --force-fixed-beta (routing disabled)."""
    fixed_noise_std: float | None = None
    """db23a (survey §13.91): post-load override of ``memory_start_noise_std``
    (deployment-time scalar eta). Needed when training used
    ``memory_start_noise_sampling=uniform`` with the std slot holding the
    sampling MAX (e.g. 0.3): the gate must run at the deployment scalar
    (0.04), not the training max. None = keep the checkpoint's value."""
    obs_image_noise_std: float = 0.0
    """E4 perturbation protocol: i.i.d. Gaussian pixel noise (0–255 scale)
    injected into every video observation before parsing. 0 = off. Seeded
    per (traj, step, repeat pass) — paired across arms, independent across
    passes; see rollout_trajectory docstring."""
    obs_state_noise_std: float = 0.0
    """E4 perturbation protocol: scale-free relative Gaussian noise on state
    entries (noise = std · |v| · N(0,1)). 0 = off. Same seeding contract as
    --obs-image-noise-std."""
    obs_perturb_seed: int = 777
    """Base seed for the E4 perturbation RNG (per-traj/step/pass derivation)."""
    save_trajectory_arrays: bool = False
    """Save each plotted trajectory's GT and aligned arm predictions as a
    compressed, pickle-free NPZ. Enables exact paired figure regeneration."""
    memory_start_inference_sample: bool = False
    """Experiment ③ (LeaP adaptive-σ arm): post-load, draw the memory start
    z₀ ~ N(μ(c), σ(c)²) from the db17 variance head instead of the
    deterministic μ. Only affects checkpoints trained with
    memory_encoder_var_head=True; silently ignored otherwise."""
    memory_source_ablate_zero: bool = False
    """Decisive control for the memory encoder's content value: post-load,
    zero the memory-start z₀ so the (1−β) source component becomes the
    normalized-space constant zero (= dataset mean action). If this arm
    matches the pure-z₀ arm, the encoder's CONTENT is worthless (the win was
    merely 'drop history'); if it is much worse, z₀'s content carries the
    win (survey doc §13.44)."""
    memory_encoder_weights_from: str | None = None
    """Cross-checkpoint encoder swap (review test D, survey §13.102): after
    loading this arm's checkpoint, overwrite ONLY the memory-encoder weights
    with those from the DONOR checkpoint directory given here, keeping the
    DiT field, processor and stats from the arm's own checkpoint. Decisive
    factorization of 'encoder drift' vs 'field degradation' when two
    checkpoints of the SAME architecture disagree (e.g. ckpt-200 encoder +
    ckpt-5000 field). strict=True load: an architecture mismatch (mlp vs
    transformer trunk, different query counts) fails loudly instead of
    silently producing a chimera."""
    high_level_task_from_meta: bool = False
    """Dual text port evaluation with per-episode high-level tasks: read each
    episode's ``high_level_task`` from ``<dataset>/meta/episodes.jsonl``
    (written by the builder's --high-level-task-pool) instead of driving every
    trajectory with the single --high-level-task string. A fixed sentence
    mismatches ~9/10 pool-sampled episodes and acts as pure text noise
    (survey doc §13.17)."""
    seed: int | None = None
    """Reproducibility: before each (arm, trajectory) rollout, seed torch with
    ``seed + traj_id`` so all arms see the SAME initial flow noise per
    trajectory (paired comparison). Repeat probes stay unseeded on purpose —
    their divergence is the measurement."""
    seeds: list[int] | None = None
    """Multi-seed evaluation (paper §8 item 9): when set, overrides --seed and
    repeats every (arm, trajectory) measurement once per seed. Seed 0 builds
    the full entry (plots, β̂ stats, stratification); every seed contributes a
    light scalar record under entry['seed_runs'][str(seed)], and the summary
    gains mean/std/95%-CI plus paired-bootstrap ratio CIs vs --reference-arm."""
    reference_arm: str | None = None
    """Label of the reference arm for the paired ratio CIs (default: first
    label). Ratio convention: ref_metric / arm_metric, paired by (traj, seed)
    — e.g. with reference=no_mem, repeat‖Δ‖ ratio >1 reproduces the paper's
    'baseline/z₀ = N×' convention."""
    state_extra_modality_keys: str = ""
    """db7/A1: comma-separated extra state modality keys (e.g.
    'target_relative_pose') appended to the embodiment's state config for THIS
    process before any Gr00tPolicy/loader is built. Must match the keys the
    checkpoint was trained with (launch_finetune --state-extra-modality-keys);
    the training-side assertion chain (expected_state_dim/data_version) is the
    source of truth for which datasets need it."""


def main(args: ArgsConfig):
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.eval._horizon_contract import PolicyHorizonSpec
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    logging.basicConfig(level=logging.INFO)
    args.embodiment_tag = EmbodimentTag.resolve(args.embodiment_tag)
    if args.state_extra_modality_keys:
        # db7/A1: the process-local state config must be extended BEFORE any
        # Gr00tPolicy/loader is built (training used the same mechanism).
        from gr00t.configs.data.state_extra_keys import (
            apply_state_extra_keys,
            parse_extra_keys,
        )

        apply_state_extra_keys(
            args.embodiment_tag.value, parse_extra_keys(args.state_extra_modality_keys)
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # §13.46 logging rule, now automatic: mirror all INFO+ logs to
    # <out_dir>/log.txt so every evaluation run archives its own log
    # (previously log.txt depended on a manual `|& tee`, cf. v1.57 P1).
    _fh = logging.FileHandler(out_dir / "log.txt", mode="w", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logging.getLogger().addHandler(_fh)

    labels = args.labels or [Path(p).parent.name + "/" + Path(p).name for p in args.model_paths]
    assert len(labels) == len(args.model_paths), "labels must match model_paths length"

    # P0-2/T1 lag fix: align the inference-time history slice with the replan
    # stride (0 disables). See ArgsConfig.history_replan_stride.
    replan_stride = (
        None
        if args.history_replan_stride == 0
        else (args.history_replan_stride or args.execution_horizon)
    )

    # Modality config comes from the first checkpoint (arms share the data config).
    first_policy = Gr00tPolicy(
        embodiment_tag=args.embodiment_tag,
        model_path=args.model_paths[0],
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    modality = first_policy.get_modality_config()
    del first_policy
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    dataset = LeRobotEpisodeLoader(dataset_path=args.dataset_path, modality_configs=modality)
    action_keys = modality["action"].modality_keys
    PolicyHorizonSpec.from_modality_config(modality, n_action_steps=args.execution_horizon)

    results: dict[str, Any] = {}
    # gt per trajectory is arm-independent; cache it.
    gt_cache: dict[int, np.ndarray] = {}
    for traj_id in args.traj_ids:
        traj = dataset[traj_id]
        actual_steps = min(args.steps, len(traj))
        gt_cache[traj_id] = extract_gt(traj, action_keys, actual_steps)

    # Dual text port: per-episode high-level tasks from meta/episodes.jsonl
    # (builder --high-level-task-pool). Takes precedence over the single
    # --high-level-task string.
    hl_from_meta: dict[int, str] = {}
    if args.high_level_task_from_meta:
        meta_path = Path(args.dataset_path) / "meta" / "episodes.jsonl"
        with open(meta_path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("high_level_task"):
                    hl_from_meta[int(rec["episode_index"])] = rec["high_level_task"]
        logger.info(f"high_level_task_from_meta: loaded {len(hl_from_meta)} episode tasks")

    # Optional segment sidecar (synthetic mixture dataset): per-trajectory
    # segment-start steps -> head/mid mask for stratified Prop. 1/2 metrics.
    # A replan step r is a "head" if it is the first replan that observes a new
    # segment's instruction (r = ceil(s/horizon)*horizon) or the episode start.
    # db7 adds two finer splits from the same sidecar:
    #   * teleport vs continuous heads (``teleported`` field, present since
    #     db6) — the decisive Prop. 2 contrast: history is useless at a
    #     teleport, partially informative at a continuous switch;
    #   * approach vs dwell (``approach_end`` field, db7 phase-explicit data
    #     only) — within-segment conditional structure.
    head_masks: dict[int, np.ndarray] = {}
    teleport_head_masks: dict[int, np.ndarray] = {}
    continuous_head_masks: dict[int, np.ndarray] = {}
    approach_masks: dict[int, np.ndarray] = {}
    dwell_masks: dict[int, np.ndarray] = {}
    if args.segment_meta_path:
        segs_by_traj: dict[int, list[dict]] = {}
        with open(args.segment_meta_path) as f:
            for line in f:
                rec = json.loads(line)
                segs_by_traj.setdefault(int(rec["episode_index"]), []).append(rec)
        for traj_id in args.traj_ids:
            if traj_id not in segs_by_traj or traj_id not in gt_cache:
                continue
            n = len(gt_cache[traj_id])
            mask = np.zeros(n, dtype=bool)
            mask[: args.execution_horizon] = True  # episode-start cold start
            tele = np.zeros(n, dtype=bool)
            cont = np.zeros(n, dtype=bool)
            approach = np.zeros(n, dtype=bool)
            dwell = np.zeros(n, dtype=bool)
            for rec in segs_by_traj[traj_id]:
                s = int(rec["start"])
                if "approach_end" in rec:
                    ae = int(rec["approach_end"])
                    approach[s:ae] = True
                    dwell[ae : int(rec["end"])] = True
                if s <= 0:
                    continue
                r = int(np.ceil(s / args.execution_horizon)) * args.execution_horizon
                mask[r : r + args.execution_horizon] = True
                if rec.get("teleported"):
                    tele[r : r + args.execution_horizon] = True
                else:
                    cont[r : r + args.execution_horizon] = True
            head_masks[traj_id] = mask
            if tele.any():
                teleport_head_masks[traj_id] = tele
            if cont.any():
                continuous_head_masks[traj_id] = cont
            if approach.any():
                approach_masks[traj_id] = approach
                dwell_masks[traj_id] = dwell

    for label, model_path in zip(labels, args.model_paths):
        logger.info(f"=== Arm {label}: {model_path} ===")
        policy = Gr00tPolicy(
            embodiment_tag=args.embodiment_tag,
            model_path=model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            history_replan_stride=replan_stride,
        )
        policy.model.action_head.num_inference_timesteps = args.denoising_steps
        if args.force_fixed_beta:
            # Post-load fixed-β control: keep the checkpoint (and its 4-output
            # confidence head) intact, but skip the β̂ readout so the scalar
            # memory_start_mix_beta drives the mixture. Flipping the flag in
            # config.json BEFORE from_pretrained shrinks the head (4→2 outputs)
            # and crashes weight loading — this must stay a post-load override.
            seen: set[int] = set()
            for cfg in (policy.model.config, policy.model.action_head.config):
                if id(cfg) in seen:
                    continue
                seen.add(id(cfg))
                if hasattr(cfg, "confidence_routing_enabled"):
                    cfg.confidence_routing_enabled = False
            logger.info("force_fixed_beta: confidence routing disabled post-load")
        if args.fixed_beta_value is not None:
            # Source-simplex completion: override the scalar mixture β post-load
            # (0.0 = pure memory z₀, 1.0 = pure history h).
            if not 0.0 <= args.fixed_beta_value <= 1.0:
                raise ValueError(
                    f"--fixed-beta-value must be in [0, 1], got {args.fixed_beta_value}"
                )
            seen_beta: set[int] = set()
            for cfg in (policy.model.config, policy.model.action_head.config):
                if id(cfg) in seen_beta:
                    continue
                seen_beta.add(id(cfg))
                if hasattr(cfg, "memory_start_mix_beta"):
                    cfg.memory_start_mix_beta = float(args.fixed_beta_value)
            logger.info(
                "fixed_beta_value: memory_start_mix_beta overridden to %.3f post-load",
                args.fixed_beta_value,
            )
        if args.fixed_noise_std is not None:
            # db23a: deployment-time scalar-eta override. Train may carry the
            # uniform-sampling MAX in memory_start_noise_std (e.g. 0.3); the
            # gate must measure the deployment configuration (e.g. 0.04).
            if args.fixed_noise_std < 0.0:
                raise ValueError(
                    f"--fixed-noise-std must be >= 0, got {args.fixed_noise_std}"
                )
            seen_eta: set[int] = set()
            for cfg in (policy.model.config, policy.model.action_head.config):
                if id(cfg) in seen_eta:
                    continue
                seen_eta.add(id(cfg))
                if hasattr(cfg, "memory_start_noise_std"):
                    cfg.memory_start_noise_std = float(args.fixed_noise_std)
            logger.info(
                "fixed_noise_std: memory_start_noise_std overridden to %.4f post-load",
                args.fixed_noise_std,
            )
        if args.memory_start_inference_sample:
            # Experiment ③: post-load enable of z₀ ~ N(μ, σ²) sampling. Only
            # meaningful for var-head checkpoints (db17); assert loudly so a
            # mistargeted run cannot silently produce a μ arm.
            if not getattr(policy.model.action_head.config, "memory_encoder_var_head", False):
                raise ValueError(
                    "--memory-start-inference-sample requires a checkpoint with "
                    "memory_encoder_var_head=True (db17 lineage); got "
                    f"{policy.model.action_head.config.memory_encoder_var_head!r}"
                )
            seen_s: set[int] = set()
            for cfg in (policy.model.config, policy.model.action_head.config):
                if id(cfg) in seen_s:
                    continue
                seen_s.add(id(cfg))
                if hasattr(cfg, "memory_start_inference_sample"):
                    cfg.memory_start_inference_sample = True
                elif hasattr(cfg, "memory_start_enabled"):
                    # config dataclass predates the flag: attach dynamically
                    cfg.memory_start_inference_sample = True
            logger.info("memory_start_inference_sample: z₀ ~ N(μ(c), σ(c)²) enabled post-load")
        if args.memory_source_ablate_zero:
            # Zero-source control: z₀ := 0 (normalized-space constant = dataset
            # mean action). Isolates the memory encoder's CONTENT value from the
            # mere 'drop history' effect.
            seen_abl: set[int] = set()
            for cfg in (policy.model.config, policy.model.action_head.config):
                if id(cfg) in seen_abl:
                    continue
                seen_abl.add(id(cfg))
                if hasattr(cfg, "memory_source_ablate_zero"):
                    cfg.memory_source_ablate_zero = True
            logger.info("memory_source_ablate_zero: z0 will be zeroed at source mix")
        if args.memory_encoder_weights_from is not None:
            # Cross-checkpoint encoder swap (review test D, survey §13.102):
            # overwrite ONLY the memory-encoder weights with those from the
            # DONOR checkpoint, keeping this arm's DiT field / processor /
            # stats. Decisive factorization of "encoder drift" vs "field
            # degradation" when two checkpoints of the SAME architecture
            # disagree (e.g. donor ckpt-200 encoder + arm ckpt-5000 field).
            from safetensors.torch import load_file as _load_safetensors

            donor_dir = Path(args.memory_encoder_weights_from)
            donor_files = sorted(donor_dir.glob("model-*.safetensors"))
            if not donor_files:
                raise FileNotFoundError(
                    "--memory-encoder-weights-from: no model-*.safetensors under "
                    f"{donor_dir}"
                )
            donor_sd: dict[str, torch.Tensor] = {}
            for _f in donor_files:
                donor_sd.update(_load_safetensors(str(_f)))
            enc_module = getattr(policy.model.action_head, "memory_encoder", None)
            if enc_module is None:
                raise ValueError(
                    "--memory-encoder-weights-from requires an arm checkpoint with "
                    "memory_start_enabled=True (this arm has no memory_encoder)"
                )
            # Checkpoint keys are rooted at Gr00tN1d7: "action_head.memory_encoder.<...>".
            # Probe the prefix instead of assuming, so a wrapped variant fails loudly.
            donor_enc_sd: dict[str, torch.Tensor] = {}
            used_prefix: str | None = None
            for _p in ("action_head.memory_encoder.", "model.action_head.memory_encoder."):
                donor_enc_sd = {
                    k[len(_p):]: v for k, v in donor_sd.items() if k.startswith(_p)
                }
                if donor_enc_sd:
                    used_prefix = _p
                    break
            if not donor_enc_sd:
                raise KeyError(
                    "no memory-encoder keys found in donor checkpoint "
                    f"{donor_dir} (tried 'action_head.memory_encoder.' and "
                    "'model.action_head.memory_encoder.')"
                )
            # strict=True: an architecture mismatch (mlp vs transformer trunk,
            # different query count / d_model) raises instead of silently
            # producing a chimera — that is the intended safety property.
            enc_module.load_state_dict(donor_enc_sd, strict=True)
            logger.info(
                "memory_encoder_weights_from: swapped %d tensors (%.2fM params, "
                "prefix %r) from %s",
                len(donor_enc_sd),
                sum(v.numel() for v in donor_enc_sd.values()) / 1e6,
                used_prefix,
                donor_dir,
            )

        arm: dict[str, Any] = {"model_path": model_path, "trajs": {}}
        for traj_id in args.traj_ids:
            if traj_id >= len(dataset):
                logger.warning(f"Trajectory {traj_id} out of range, skipped")
                continue
            step_infos: list[dict[str, Any]] = []
            # Multi-seed evaluation (paper §8 item 9): with --seeds, every
            # (arm, trajectory) measurement repeats once per seed. The first
            # seed builds the full entry (plots, β̂ stats, stratification);
            # every seed adds a light scalar record for dispersion stats.
            seed_list: list[int | None] = list(args.seeds) if args.seeds else [args.seed]
            seed_runs: dict[str, dict[str, Any]] = {}
            entry: dict[str, Any] = {}
            for seed_idx, seed_i in enumerate(seed_list):
                if seed_i is not None:
                    # Paired seeding: identical initial noise for this
                    # trajectory across arms within one seed (the repeat probe
                    # below stays unseeded on purpose).
                    torch.manual_seed(seed_i + traj_id)
                pred = rollout_trajectory(
                    policy, dataset, traj_id, args.embodiment_tag,
                    action_keys, args.steps, args.execution_horizon,
                    info_out=step_infos if seed_idx == 0 else None,
                    high_level_task=hl_from_meta.get(traj_id, args.high_level_task),
                    history_source=args.history_source,
                    fixed_noise_draw=args.fixed_noise_draw,
                    fixed_noise_seed=args.fixed_noise_seed,
                    obs_image_noise_std=args.obs_image_noise_std,
                    obs_state_noise_std=args.obs_state_noise_std,
                    obs_perturb_seed=args.obs_perturb_seed,
                    rng_offset=seed_i if seed_i is not None else 0,
                )
                gt = gt_cache[traj_id][: len(pred)]
                mse = float(np.mean((gt - pred) ** 2))
                mae = float(np.mean(np.abs(gt - pred)))
                # Per-dim breakdown + dead-dim split: unpredictable constant /
                # white-noise dims (synthetic rot6d deltas) must not dilute the
                # arm comparison — effective MSE is over live dims only.
                mse_per_dim = ((gt - pred) ** 2).mean(axis=0)
                live_dims, dead_dims = split_dead_dims(gt, args.dead_dim_eps)
                lags = estimate_temporal_lag(gt, pred, args.max_lag)
                lag_med = int(np.median(lags))
                mse_eff_val = dim_masked_mse(gt, pred, live_dims)
                div: float | None = None
                if args.repeat_probe:
                    if seed_i is not None:
                        # Strictly independent second realization (paper §5.1):
                        # run2 draws seed+traj_id+500000 explicitly instead of
                        # inheriting run1's RNG state (review P2-2). All other
                        # randomness bits (data, weights) are identical, so the
                        # divergence isolates source-sampling noise. Numbers
                        # stay comparable with the legacy unseeded probe:
                        # noisy arms draw fresh ε either way, deterministic
                        # arms are RNG-insensitive.
                        torch.manual_seed(seed_i + traj_id + 500000)
                    pred2 = rollout_trajectory(
                        policy, dataset, traj_id, args.embodiment_tag,
                        action_keys, args.steps, args.execution_horizon,
                        high_level_task=hl_from_meta.get(traj_id, args.high_level_task),
                        history_source=args.history_source,
                        fixed_noise_draw=args.fixed_noise_draw,
                        fixed_noise_seed=args.fixed_noise_seed,
                        obs_image_noise_std=args.obs_image_noise_std,
                        obs_state_noise_std=args.obs_state_noise_std,
                        obs_perturb_seed=args.obs_perturb_seed,
                        rng_offset=(seed_i if seed_i is not None else 0) + 500000,
                    )
                    div = float(np.mean(np.abs(pred - pred2)))
                # Light per-seed scalar record (the dispersion pool).
                run_rec: dict[str, Any] = {
                    "mse": mse,
                    "mse_effective": mse_eff_val,
                    "repeat_divergence": div,
                }
                for _mask_dict, _tag in (
                    (teleport_head_masks, "mse_teleport_head_eff"),
                    (continuous_head_masks, "mse_continuous_head_eff"),
                    (head_masks, "mse_segment_head_eff"),
                ):
                    _m = _mask_dict.get(traj_id)
                    if _m is None:
                        continue
                    _mm = _m[: len(pred)]
                    if _mm.any():
                        run_rec[_tag] = dim_masked_mse(gt[_mm], pred[_mm], live_dims)
                seed_runs[str(seed_i)] = run_rec

                if seed_idx > 0:
                    logger.info(
                        f"[{label}] traj {traj_id} seed {seed_i}: MSE={mse:.6f} "
                        f"mse_eff={mse_eff_val:.6f}"
                        + (f" repeat|Δ|={div:.6f}" if div is not None else "")
                    )
                    continue

                # ---- seed 0: full entry (plots, β̂ stats, stratification) ----
                entry = {
                    "mse": mse,
                    "mae": mae,
                    "mse_per_dim": [float(v) for v in mse_per_dim],
                    "dead_dims": [int(d) for d in dead_dims],
                    "mse_effective": mse_eff_val,
                    "mse_dead": dim_masked_mse(gt, pred, dead_dims),
                    "temporal_lag_median": lag_med,
                    "temporal_lag_per_dim": lags,
                    "mse_lag_compensated": shifted_mse(gt, pred, lag_med),
                    "pred": pred,
                }
                if dead_dims.size:
                    logger.info(
                        f"[{label}] traj {traj_id}: dead_dims={dead_dims.tolist()} "
                        f"mse_eff={entry['mse_effective']:.6f} "
                        f"mse_dead={entry['mse_dead']:.6f}"
                    )
                # v4.1/T1: confidence readout + routed β̂ stats when the
                # checkpoint provides them (β̂ = weight on the history term).
                beta_vals = [
                    float(np.atleast_1d(si["beta_hat"]).mean())
                    for si in step_infos
                    if "beta_hat" in si
                ]
                conf_rows = [
                    np.atleast_2d(np.asarray(si["confidence"], dtype=np.float64))
                    for si in step_infos
                    if "confidence" in si
                ]
                if beta_vals:
                    entry["beta_hat_mean"] = float(np.mean(beta_vals))
                    entry["beta_hat_std"] = float(np.std(beta_vals))
                if conf_rows:
                    conf_mat = np.concatenate(conf_rows, axis=0)  # (n_replans, n_out)
                    entry["confidence_mean"] = [float(v) for v in conf_mat.mean(axis=0)]
                    logger.info(
                        f"[{label}] traj {traj_id}: confidence_mean="
                        f"{np.round(conf_mat.mean(axis=0), 4).tolist()}"
                        + (f" beta_hat={entry['beta_hat_mean']:.3f}±{entry['beta_hat_std']:.3f}" if beta_vals else "")
                    )
                # Prop. 1/2 stratified metrics (synthetic mixture sidecar):
                # head = first replan window after a segment switch (or episode
                # start) — semantic source should win; mid = smooth
                # continuation — history source suffices.
                head_mask = head_masks.get(traj_id)
                if head_mask is not None and head_mask[: len(pred)].any():
                    h = head_mask[: len(pred)]
                    entry["mse_segment_head"] = float(np.mean((gt[h] - pred[h]) ** 2))
                    entry["mse_segment_mid"] = float(np.mean((gt[~h] - pred[~h]) ** 2))
                    if dead_dims.size:
                        entry["mse_segment_head_eff"] = dim_masked_mse(gt[h], pred[h], live_dims)
                        entry["mse_segment_mid_eff"] = dim_masked_mse(gt[~h], pred[~h], live_dims)
                    logger.info(
                        f"[{label}] traj {traj_id}: MSE head={entry['mse_segment_head']:.6f} "
                        f"mid={entry['mse_segment_mid']:.6f}"
                        + (
                            f" | eff head={entry['mse_segment_head_eff']:.6f} "
                            f"mid={entry['mse_segment_mid_eff']:.6f}"
                            if dead_dims.size
                            else ""
                        )
                    )
                    if beta_vals:
                        beta_head = [
                            float(np.atleast_1d(si["beta_hat"]).mean())
                            for si in step_infos
                            if "beta_hat" in si and si["step"] < len(h) and h[si["step"]]
                        ]
                        beta_mid = [
                            float(np.atleast_1d(si["beta_hat"]).mean())
                            for si in step_infos
                            if "beta_hat" in si and si["step"] < len(h) and not h[si["step"]]
                        ]
                        if beta_head:
                            entry["beta_hat_head_mean"] = float(np.mean(beta_head))
                        if beta_mid:
                            entry["beta_hat_mid_mean"] = float(np.mean(beta_mid))

                # db7 finer stratification: teleport vs continuous heads (the
                # decisive Prop. 2 contrast) and approach vs dwell.
                def _masked_mse_pair(m: np.ndarray | None) -> dict[str, float] | None:
                    if m is None:
                        return None
                    mm = m[: len(pred)]
                    if not mm.any():
                        return None
                    out = {"mse": float(np.mean((gt[mm] - pred[mm]) ** 2))}
                    if dead_dims.size:
                        out["mse_eff"] = dim_masked_mse(gt[mm], pred[mm], live_dims)
                    return out

                for mask_dict, tag in (
                    (teleport_head_masks, "teleport_head"),
                    (continuous_head_masks, "continuous_head"),
                    (approach_masks, "approach"),
                    (dwell_masks, "dwell"),
                ):
                    res = _masked_mse_pair(mask_dict.get(traj_id))
                    if res is None:
                        continue
                    entry[f"mse_{tag}"] = res["mse"]
                    if "mse_eff" in res:
                        entry[f"mse_{tag}_eff"] = res["mse_eff"]
                    if beta_vals and tag in ("teleport_head", "continuous_head"):
                        m_arr = mask_dict[traj_id][: len(pred)]
                        b_vals = [
                            float(np.atleast_1d(si["beta_hat"]).mean())
                            for si in step_infos
                            if "beta_hat" in si and si["step"] < len(m_arr) and m_arr[si["step"]]
                        ]
                        if b_vals:
                            entry[f"beta_hat_{tag}_mean"] = float(np.mean(b_vals))
                if "mse_teleport_head" in entry or "mse_approach" in entry:
                    logger.info(
                        f"[{label}] traj {traj_id}: "
                        + " ".join(
                            f"{k}={entry[k]:.6f}"
                            for k in (
                                "mse_teleport_head_eff",
                                "mse_continuous_head_eff",
                                "mse_approach_eff",
                                "mse_dwell_eff",
                            )
                            if k in entry
                        )
                        + " ".join(
                            f" {k}={entry[k]:.3f}"
                            for k in ("beta_hat_teleport_head_mean", "beta_hat_continuous_head_mean")
                            if k in entry
                        )
                    )

                if args.repeat_probe:
                    entry["repeat_divergence"] = div
                    entry["pred_repeat"] = pred2
                    logger.info(
                        f"[{label}] traj {traj_id}: MSE={mse:.6f} MAE={mae:.6f} "
                        f"repeat|Δ|={div:.6f} lag_med={lag_med:+d}"
                    )
                else:
                    logger.info(
                        f"[{label}] traj {traj_id}: MSE={mse:.6f} MAE={mae:.6f} lag_med={lag_med:+d}"
                    )
            if len(seed_list) > 1:
                entry["seed_runs"] = seed_runs
            arm["trajs"][str(traj_id)] = entry

        results[label] = arm
        del policy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- plots ----
    for traj_id in args.traj_ids:
        if str(traj_id) not in next(iter(results.values()))["trajs"]:
            continue
        preds = {
            label: arm["trajs"][str(traj_id)]["pred"]
            for label, arm in results.items()
            if str(traj_id) in arm["trajs"]
        }
        gt = gt_cache[traj_id][: len(next(iter(preds.values())))]
        if args.save_trajectory_arrays:
            save_trajectory_arrays(
                gt, preds, results, args, traj_id,
                out_dir / f"traj_{traj_id}_trajectory.npz",
            )
        plot_overlay(
            gt, preds, traj_id, args.execution_horizon,
            out_dir / f"traj_{traj_id}_compare.png", args.max_plot_dims,
        )
        if args.repeat_probe:
            rep = {
                label: arm["trajs"][str(traj_id)]["repeat_divergence"]
                for label, arm in results.items()
                if str(traj_id) in arm["trajs"] and "repeat_divergence" in arm["trajs"][str(traj_id)]
            }
            if rep:
                plot_repeatability(rep, traj_id, out_dir / f"traj_{traj_id}_repeatability.png")
            if args.repeat_overlay:
                arm_runs = {
                    label: (
                        arm["trajs"][str(traj_id)]["pred"],
                        arm["trajs"][str(traj_id)]["pred_repeat"],
                        arm["trajs"][str(traj_id)]["repeat_divergence"],
                    )
                    for label, arm in results.items()
                    if str(traj_id) in arm["trajs"]
                    and "pred_repeat" in arm["trajs"][str(traj_id)]
                }
                if arm_runs:
                    overlay_dims = [d for d in (0, 1, 2, 9) if d < gt.shape[1]]
                    plot_repeat_overlay(
                        gt, arm_runs, traj_id, overlay_dims,
                        out_dir / f"traj_{traj_id}_repeat_overlay.png",
                    )

    # ---- metrics table + json ----
    table_rows = []
    for label, arm in results.items():
        mses = [t["mse"] for t in arm["trajs"].values()]
        maes = [t["mae"] for t in arm["trajs"].values()]
        divs = [t.get("repeat_divergence") for t in arm["trajs"].values()]
        divs = [d for d in divs if d is not None]
        lag_meds = [t.get("temporal_lag_median") for t in arm["trajs"].values()]
        lag_meds = [x for x in lag_meds if x is not None]
        mse_comps = [t.get("mse_lag_compensated") for t in arm["trajs"].values()]
        mse_comps = [x for x in mse_comps if x is not None]
        mse_effs = [t.get("mse_effective") for t in arm["trajs"].values()]
        mse_effs = [x for x in mse_effs if x is not None]
        mse_deads = [t.get("mse_dead") for t in arm["trajs"].values()]
        mse_deads = [x for x in mse_deads if x is not None]
        head_effs = [t.get("mse_segment_head_eff") for t in arm["trajs"].values()]
        head_effs = [x for x in head_effs if x is not None]
        mid_effs = [t.get("mse_segment_mid_eff") for t in arm["trajs"].values()]
        mid_effs = [x for x in mid_effs if x is not None]
        row = {
            "arm": label,
            "avg_mse": float(np.mean(mses)) if mses else None,
            "avg_mae": float(np.mean(maes)) if maes else None,
            "avg_repeat_divergence": float(np.mean(divs)) if divs else None,
            "avg_lag_median": float(np.median(lag_meds)) if lag_meds else None,
            "avg_mse_lag_compensated": float(np.mean(mse_comps)) if mse_comps else None,
            "avg_mse_effective": float(np.mean(mse_effs)) if mse_effs else None,
            "avg_mse_dead": float(np.mean(mse_deads)) if mse_deads else None,
            "avg_mse_segment_head_eff": float(np.mean(head_effs)) if head_effs else None,
            "avg_mse_segment_mid_eff": float(np.mean(mid_effs)) if mid_effs else None,
        }
        # db7 finer stratification (None when the sidecar fields are absent)
        for key in (
            "mse_teleport_head_eff",
            "mse_continuous_head_eff",
            "mse_approach_eff",
            "mse_dwell_eff",
            "beta_hat_teleport_head_mean",
            "beta_hat_continuous_head_mean",
        ):
            vals = [t.get(key) for t in arm["trajs"].values()]
            vals = [v for v in vals if v is not None]
            row[f"avg_{key}"] = float(np.mean(vals)) if vals else None
        # Multi-seed dispersion (§8 item 9): pool the light per-(traj, seed)
        # scalar records into mean / std / 95% CI per metric. CI uses CLUSTER
        # bootstrap over trajectories (review P1-1): seeds within one traj are
        # correlated, so the naive 1.96·std/√n would underestimate variance.
        pooled: dict[str, list[float]] = {}
        pooled_by_traj: dict[str, dict[str, list[float]]] = {}
        for tid, t in arm["trajs"].items():
            for rec in (t.get("seed_runs") or {}).values():
                for mk, mv in rec.items():
                    if mv is not None:
                        pooled.setdefault(mk, []).append(float(mv))
                        pooled_by_traj.setdefault(mk, {}).setdefault(tid, []).append(float(mv))
        rng_ms = np.random.default_rng(0)
        for mk, vals in pooled.items():
            arr = np.asarray(vals)
            row[f"ms_{mk}_mean"] = float(arr.mean())
            row[f"ms_{mk}_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            lo, hi = _cluster_bootstrap_ci95(list(pooled_by_traj[mk].values()), rng_ms)
            row[f"ms_{mk}_ci95"] = float(max(hi - lo, 0.0) / 2)  # half-width (± form)
            row[f"ms_{mk}_ci95_band"] = [lo, hi]
            row[f"ms_{mk}_n"] = int(len(arr))
        table_rows.append(row)
        arm["summary"] = {k: v for k, v in row.items() if k != "arm"}

    # Paired-bootstrap ratio CIs vs the reference arm (paired by traj × seed).
    # Convention: ratio = ref_metric / arm_metric (>1 ⇔ arm beats ref on the
    # metric — e.g. repeat‖Δ‖ with ref=no_mem reproduces the paper's N×).
    ref_label = args.reference_arm or (labels[0] if labels else None)
    ratio_block: dict[str, Any] = {}
    if ref_label and ref_label in results:
        ref_trajs = results[ref_label]["trajs"]
        rng = np.random.default_rng(0)
        for label, arm in results.items():
            if label == ref_label:
                continue
            for metric in ("repeat_divergence", "mse_effective", "mse_teleport_head_eff"):
                pairs_by_traj: dict[str, list[float]] = {}
                for tid, t in arm["trajs"].items():
                    rt = ref_trajs.get(tid, {})
                    for seed_key, rec in (t.get("seed_runs") or {}).items():
                        rref = (rt.get("seed_runs") or {}).get(seed_key, {})
                        a_v, r_v = rec.get(metric), rref.get(metric)
                        if a_v is None or r_v is None or a_v == 0.0:
                            continue
                        pairs_by_traj.setdefault(tid, []).append(float(r_v) / float(a_v))
                pairs = [v for vs in pairs_by_traj.values() for v in vs]
                if len(pairs) < 2:
                    continue
                arr = np.asarray(pairs)
                # Cluster bootstrap by trajectory (review P1-1): same-traj
                # seeds are correlated, resample whole trajectories.
                lo, hi = _cluster_bootstrap_ci95(list(pairs_by_traj.values()), rng)
                ratio_block.setdefault(f"{label} vs {ref_label}", {})[metric] = {
                    "ratio_ref_over_arm_mean": float(arr.mean()),
                    "ci95": [lo, hi],
                    "n_pairs": int(len(arr)),
                    "n_traj_clusters": int(len(pairs_by_traj)),
                }
        for pair_name, metrics in ratio_block.items():
            for metric, blk in metrics.items():
                logger.info(
                    f"[multi-seed] {metric} ratio ({pair_name}, ref/arm): "
                    f"{blk['ratio_ref_over_arm_mean']:.3f}× "
                    f"95%CI [{blk['ci95'][0]:.3f}, {blk['ci95'][1]:.3f}] "
                    f"(n={blk['n_pairs']})"
                )

    logger.info("=== Flow-source arm comparison ===")
    header = (
        f"{'arm':<24} {'avg_mse':>10} {'mse_eff':>10} {'mse_dead':>10} {'avg_mae':>10} "
        f"{'repeat|Δ|':>10} {'lag_med':>8} {'mse@lag':>10}"
    )
    logger.info(header)
    for r in table_rows:
        div = f"{r['avg_repeat_divergence']:.6f}" if r["avg_repeat_divergence"] is not None else "-"
        lag = f"{r['avg_lag_median']:+.0f}" if r["avg_lag_median"] is not None else "-"
        mc = f"{r['avg_mse_lag_compensated']:.6f}" if r["avg_mse_lag_compensated"] is not None else "-"
        eff = f"{r['avg_mse_effective']:.6f}" if r["avg_mse_effective"] is not None else "-"
        dead = f"{r['avg_mse_dead']:.6f}" if r["avg_mse_dead"] is not None else "-"
        logger.info(
            f"{r['arm']:<24} {r['avg_mse']:.6f}   {eff:>10} {dead:>10} "
            f"{r['avg_mae']:.6f}   {div:>10} {lag:>8} {mc:>10}"
        )
        if "ms_repeat_divergence_mean" in r:
            logger.info(
                f"{'':>24} multi-seed: repeat|Δ| {r['ms_repeat_divergence_mean']:.4f}"
                f"±{r['ms_repeat_divergence_ci95']:.4f} (95%CI, n={r['ms_repeat_divergence_n']})"
                f" | mse_eff {r['ms_mse_effective_mean']:.4f}"
                f"±{r['ms_mse_effective_ci95']:.4f}"
            )
    if any("ms_repeat_divergence_mean" in r for r in table_rows):
        plot_summary_ci(
            table_rows, out_dir / "summary_multiseed_ci.png", ratio_block, ref_label
        )

    # strip raw arrays before json dump
    for arm in results.values():
        for t in arm["trajs"].values():
            t.pop("pred", None)
            t.pop("pred_repeat", None)
    with open(out_dir / "compare_metrics.json", "w") as f:
        json.dump(
            {
                "args": {k: str(v) for k, v in vars(args).items()},
                "arms": results,
                "multi_seed": {"reference_arm": ref_label, "ratios": ratio_block},
            },
            f,
            indent=2,
        )
    logger.info(f"Wrote plots and metrics to {out_dir}")


if __name__ == "__main__":
    main(tyro.cli(ArgsConfig))
