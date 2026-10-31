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

from copy import deepcopy
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import re
from typing import Any
import warnings

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.utils import parse_observation_gr00t
from gr00t.eval._horizon_contract import PolicyHorizonSpec, migrate_deprecated_action_horizon_argv
from gr00t.policy import BasePolicy
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.server_client import PolicyClient
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tyro


warnings.simplefilter("ignore", category=FutureWarning)

"""
Example commands:

NOTE: provide --model_path to load up the model checkpoint in this script,
        else it will use the default host and port via RobotInferenceClient

"""


def plot_trajectory_results(
    state_joints_across_time: np.ndarray,
    gt_action_across_time: np.ndarray,
    pred_action_across_time: np.ndarray,
    traj_id: int,
    state_keys: list[str],
    action_keys: list[str],
    execution_horizon: int,
    save_plot_path: str,
) -> None:
    """
    Plot and save trajectory results comparing ground truth and predicted actions.

    Args:
        state_joints_across_time: Array of state joints over time
        gt_action_across_time: Ground truth actions over time
        pred_action_across_time: Predicted actions over time
        traj_id: Trajectory ID
        state_keys: List of state modality keys
        action_keys: List of action modality keys
        execution_horizon: Number of predicted-chunk steps executed per inference
        save_plot_path: Path to save the plot
    """
    actual_steps = len(gt_action_across_time)
    action_dim = gt_action_across_time.shape[1]

    indices_to_plot = list(range(action_dim))

    num_plots = len(indices_to_plot)
    if num_plots == 0:
        logging.warning("No valid indices to plot")
        return

    # Always plot and save
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(8, 4 * num_plots))

    # Handle case where there's only one subplot
    if num_plots == 1:
        axes = [axes]

    # Add a global title showing the modality keys
    fig.suptitle(
        f"Trajectory {traj_id} - State: {', '.join(state_keys)} | Action: {', '.join(action_keys)}",
        fontsize=16,
        color="blue",
    )

    for plot_idx, action_idx in enumerate(indices_to_plot):
        ax = axes[plot_idx]

        # The dimensions of state_joints and action are the same
        # only when the robot uses actions directly as joint commands.
        # Therefore, do not plot them if this is not the case.
        if state_joints_across_time.shape == gt_action_across_time.shape:
            ax.plot(state_joints_across_time[:, action_idx], label="state joints")
        ax.plot(gt_action_across_time[:, action_idx], label="gt action")
        ax.plot(pred_action_across_time[:, action_idx], label="pred action")

        # put a dot every ACTION_HORIZON
        for j in range(0, actual_steps, execution_horizon):
            if j == 0:
                ax.plot(
                    j,
                    gt_action_across_time[j, action_idx],
                    "ro",
                    label="inference point",
                )
            else:
                ax.plot(j, gt_action_across_time[j, action_idx], "ro")

        ax.set_title(f"Action {action_idx}")
        ax.legend()

    plt.tight_layout()

    # Create filename with trajectory ID
    Path(save_plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_plot_path)

    plt.close()  # Close the figure to free memory


def parse_action_gr00t(action: dict[str, Any]) -> dict[str, Any]:
    # Unbatch and add prefix
    return {f"action.{key}": action[key][0] for key in action}


def _build_missing_state_fill(
    model_path: str,
    embodiment_tag: EmbodimentTag,
    loader: LeRobotEpisodeLoader,
    required_state_keys: list[str],
    mode: str,
) -> dict[str, np.ndarray]:
    """Build per-key constant fill vectors for STATE keys the checkpoint's
    modality config requires but the evaluated dataset does not carry.

    Used for cross-evaluation (E4): a db7 checkpoint expects
    ``state.target_relative_pose`` (22D state) while real DROID only has the
    default 17D/3-key state. Fill vectors come from the checkpoint's
    ``statistics.json`` so the state-encoder input width still matches:
    ``mean`` fills raw values with the per-dim training mean (after
    normalization the channel reads ~0 — an "average target" constant, the
    minimal-distribution-shift imputation); ``zero`` fills raw zeros.
    """
    probe = loader[0]
    missing = [k for k in required_state_keys if f"state.{k}" not in probe.columns]
    if not missing:
        return {}
    stats_file = Path(model_path) / "statistics.json"
    if not stats_file.exists():
        raise FileNotFoundError(
            f"--state-missing-keys-fill {mode}: checkpoint statistics.json not found at "
            f"{stats_file}; cannot determine fill dimension for missing state keys {missing}."
        )
    with open(stats_file) as f:
        statistics = json.load(f)
    emb_stats = statistics.get(embodiment_tag.value)
    if emb_stats is None:
        if len(statistics) == 1:
            emb_stats = next(iter(statistics.values()))
        else:
            raise KeyError(
                f"Embodiment {embodiment_tag.value!r} not found in {stats_file} top-level "
                f"keys {sorted(statistics.keys())}."
            )
    state_stats = emb_stats.get("state", {})
    fill: dict[str, np.ndarray] = {}
    for key in missing:
        key_stats = state_stats.get(key)
        if key_stats is None:
            raise KeyError(
                f"--state-missing-keys-fill: no statistics for state key {key!r} in "
                f"{stats_file} (available: {sorted(state_stats.keys())})."
            )
        vec = None
        for stat_name in ("mean", "q50"):
            if stat_name in key_stats:
                vec = np.asarray(key_stats[stat_name], dtype=np.float32).reshape(-1)
                break
        if vec is None:
            vec = np.zeros(len(next(iter(key_stats.values()))), dtype=np.float32)
        fill[key] = vec if mode == "mean" else np.zeros_like(vec)
    logging.warning(
        "E4 protocol deviation: filling dataset-missing state keys with %s imputation: "
        "%s — interpret resulting metrics as tracking with a CONSTANT dummy channel "
        "(forgetting and imputation effects are confounded).",
        mode,
        sorted(fill.keys()),
    )
    return fill


def evaluate_single_trajectory(
    policy: BasePolicy,
    loader: LeRobotEpisodeLoader,
    traj_id: int,
    embodiment_tag: EmbodimentTag,
    modality_keys: list[str] | None = None,
    steps=300,
    execution_horizon=16,
    save_plot_path=None,
    high_level_task: str | None = None,
    state_missing_fill: dict[str, np.ndarray] | None = None,
):
    # Ensure steps doesn't exceed trajectory length
    traj = loader[traj_id]
    traj_length = len(traj)
    actual_steps = min(steps, traj_length)
    logging.info(
        f"Using {actual_steps} steps (requested: {steps}, trajectory length: {traj_length})"
    )

    # Clear any action-history buffer (history-conditioned flow matching) so
    # chunks predicted for a previous trajectory cannot leak into this one.
    policy.reset()

    pred_action_across_time = []

    # Extract state and action keys separately and sort for consistent order.
    # Plot-side only: state keys the dataset lacks (E4 cross-eval with a
    # checkpoint trained on extra state channels, e.g. db7's
    # target_relative_pose on real DROID) are dropped from the optional state
    # overlay — the inference path handles them via missing_state_fill, and the
    # overlay is shape-guarded downstream anyway.
    state_keys = [
        k for k in loader.modality_configs["state"].modality_keys if f"state.{k}" in traj.columns
    ]
    dropped_state_keys = [
        k for k in loader.modality_configs["state"].modality_keys if f"state.{k}" not in traj.columns
    ]
    if dropped_state_keys:
        logging.warning(
            "State keys absent from dataset dropped from plot overlay: %s", dropped_state_keys
        )
    action_keys = (
        loader.modality_configs["action"].modality_keys if modality_keys is None else modality_keys
    )

    # Fail fast if the open-loop stride doesn't fit the model's predicted chunk
    # (also rejects a non-contiguous action window, which the linear indexing
    # below would silently mis-execute).
    PolicyHorizonSpec.from_modality_config(
        loader.modality_configs, n_action_steps=execution_horizon
    )

    modality_configs = deepcopy(loader.modality_configs)
    modality_configs.pop("action")
    for step_count in range(0, actual_steps, execution_horizon):
        data_point = extract_step_data(
            traj,
            step_count,
            modality_configs,
            embodiment_tag,
            missing_state_fill=state_missing_fill,
        )
        logging.info(f"inferencing at step: {step_count}")
        obs = {}
        for k, v in data_point.states.items():
            obs[f"state.{k}"] = v  # (T, D)
        for k, v in data_point.images.items():
            obs[f"video.{k}"] = np.array(v)  # (T, H, W, C)
        for language_key in loader.modality_configs["language"].modality_keys:
            obs[language_key] = high_level_task if high_level_task else data_point.text
        parsed_obs = parse_observation_gr00t(obs, loader.modality_configs)
        if high_level_task:
            # Dual text port: episode-level task drives `text`; the per-frame
            # instruction becomes the sub-instruction.
            parsed_obs["sub_instruction"] = [data_point.text]
        _action_chunk, _ = policy.get_action(parsed_obs)
        action_chunk = parse_action_gr00t(_action_chunk)
        for j in range(execution_horizon):
            # NOTE: concat_pred_action = action[f"action.{modality_keys[0]}"][j]
            # the np.atleast_1d is to ensure the action is a 1D array, handle where single value is returned
            concat_pred_action = np.concatenate(
                [
                    np.atleast_1d(np.atleast_1d(action_chunk[f"action.{key}"])[j])
                    for key in action_keys
                ],
                axis=0,
            )
            pred_action_across_time.append(concat_pred_action)

    def extract_state_joints(traj: pd.DataFrame, columns: list[str]):
        np_dict = {}
        for column in columns:
            np_dict[column] = np.vstack([arr for arr in traj[column]])
        return np.concatenate([np_dict[column] for column in columns], axis=-1)

    # plot the joints
    state_joints_across_time = extract_state_joints(traj, [f"state.{key}" for key in state_keys])
    gt_action_across_time = extract_state_joints(traj, [f"action.{key}" for key in action_keys])[
        :actual_steps
    ]
    pred_action_across_time = np.array(pred_action_across_time)[:actual_steps]
    assert gt_action_across_time.shape == pred_action_across_time.shape, (
        f"gt_action: {gt_action_across_time.shape}, pred_action: {pred_action_across_time.shape}"
    )

    # calc MSE and MAE across time
    mse = np.mean((gt_action_across_time - pred_action_across_time) ** 2)
    mae = np.mean(np.abs(gt_action_across_time - pred_action_across_time))
    logging.info(f"Unnormalized Action MSE across single traj: {mse}")
    logging.info(f"Unnormalized Action MAE across single traj: {mae}")

    logging.info(f"state_joints vs time {state_joints_across_time.shape}")
    logging.info(f"gt_action_joints vs time {gt_action_across_time.shape}")
    logging.info(f"pred_action_joints vs time {pred_action_across_time.shape}")

    # Plot trajectory results
    plot_trajectory_results(
        state_joints_across_time=state_joints_across_time,
        gt_action_across_time=gt_action_across_time,
        pred_action_across_time=pred_action_across_time,
        traj_id=traj_id,
        state_keys=state_keys,
        action_keys=action_keys,
        execution_horizon=execution_horizon,
        save_plot_path=save_plot_path or f"/tmp/open_loop_eval/traj_{traj_id}.jpeg",
    )

    return mse, mae


@dataclass
class ArgsConfig:
    """Configuration for evaluating a policy."""

    host: str = "127.0.0.1"
    """Host to connect to."""

    port: int = 5555
    """Port to connect to."""

    steps: int = 200
    """Maximum number of steps to evaluate (will be capped by trajectory length)."""

    traj_ids: list[int] = field(default_factory=lambda: [0])
    """List of trajectory IDs to evaluate."""

    execution_horizon: int = 16
    """How many steps of each predicted action chunk to execute before re-planning
    (must be <= the model's predicted chunk length)."""

    history_replan_stride: int | None = None
    """P0-2/T1 lag fix: rtc history-slice offset used at inference. None
    (default) = execution_horizon, which is the correct alignment for this
    open-loop replan protocol; 0 disables the override (legacy behavior —
    reproduces the lag-aliasing artifact on periodic signals). Only applies to
    local checkpoints trained with flow_source_mode history/memory_history_mix."""

    dataset_path: str = "demo_data/cube_to_bowl_5/"
    """Path to the dataset."""

    embodiment_tag: str = "new_embodiment"
    """Embodiment tag (name or value, case-insensitive). Run with --help to see known tags."""

    model_path: str | None = None
    """Path to the model checkpoint."""

    denoising_steps: int = 4
    """Number of denoising steps to use."""

    save_plot_path: str | None = None
    """Path to save the plot to."""

    modality_keys: list[str] | None = None
    """List of modality keys to plot. If None, plot all keys."""

    high_level_task: str | None = None
    """Dual text port evaluation: when set, this episode-level task drives the
    policy's task text while each frame's dataset instruction is fed as the
    sub-instruction (requires a checkpoint trained with sub_instruction_mode
    concat/replace). None = legacy single-port eval."""
    state_extra_modality_keys: str = ""
    """db7/A1: comma-separated extra state modality keys (e.g.
    'target_relative_pose') appended to the embodiment's state config for THIS
    process before the policy/loader is built (must match training)."""

    state_missing_keys_fill: str = "none"
    """E4 cross-eval: how to handle STATE modality keys required by the
    checkpoint's modality config but absent from the dataset (e.g. a db7
    checkpoint's target_relative_pose vs real DROID's 17D/3-key state).
    'none' (default) keeps the hard KeyError; 'mean' fills the raw channel
    with the checkpoint statistics per-dim mean (recommended: after
    normalization the channel reads ~0, i.e. a constant average target — the
    minimal-distribution-shift imputation); 'zero' fills raw zeros. Metrics
    produced with mean/zero are tracking-with-constant-dummy-channel numbers:
    forgetting and imputation effects are confounded."""


def main(args: ArgsConfig):
    args.embodiment_tag = EmbodimentTag.resolve(args.embodiment_tag)
    if args.state_extra_modality_keys:
        # db7/A1: extend the process-local state config BEFORE policy/loader.
        from gr00t.configs.data.state_extra_keys import (
            apply_state_extra_keys,
            parse_extra_keys,
        )

        apply_state_extra_keys(
            args.embodiment_tag.value, parse_extra_keys(args.state_extra_modality_keys)
        )
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Download model checkpoint if it's an S3 path
    local_model_path = args.model_path

    # Extract global_step and checkpoint directory name from checkpoint path
    global_step = None
    if local_model_path:
        # Search for pattern "checkpoint-{number}" anywhere in the path
        match = re.search(r"checkpoint-(\d+)", local_model_path)
        if match:
            try:
                global_step = int(match.group(1))
                logging.info(f"Extracted global_step {global_step} from checkpoint path")
            except ValueError:
                logging.warning(
                    f"Could not parse step number from checkpoint path: {local_model_path}"
                )
        else:
            logging.warning(f"Could not find checkpoint-<step> pattern in path: {local_model_path}")

    if local_model_path is not None:
        import torch

        # P0-2/T1 lag fix: align the inference-time history slice with the
        # replan stride (0 disables; None default = execution_horizon).
        stride = (
            None
            if args.history_replan_stride == 0
            else (args.history_replan_stride or args.execution_horizon)
        )
        policy = Gr00tPolicy(
            embodiment_tag=args.embodiment_tag,
            model_path=local_model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            history_replan_stride=stride,
        )
        # Apply --denoising-steps: the action head reads num_inference_timesteps
        # at sampling time.
        policy.model.action_head.num_inference_timesteps = args.denoising_steps
        logging.info(f"Using {args.denoising_steps} denoising steps")
    else:
        policy = PolicyClient(host=args.host, port=args.port)
        if args.denoising_steps != ArgsConfig.denoising_steps:
            logging.warning(
                "--denoising-steps=%d is ignored when running against a remote "
                "policy server; set the denoising steps on the server "
                "(run_gr00t_server.py) instead.",
                args.denoising_steps,
            )

    # Get the supported modalities for the policy
    modality = policy.get_modality_config()
    logging.info(f"Current modality config: \n{modality}")

    # Create the dataset
    dataset = LeRobotEpisodeLoader(
        dataset_path=args.dataset_path,
        modality_configs=modality,
    )

    # E4 cross-eval: resolve constant fill vectors for state keys the dataset
    # lacks (e.g. db7 checkpoint's target_relative_pose on real DROID).
    assert args.state_missing_keys_fill in ("none", "zero", "mean"), (
        f"--state-missing-keys-fill must be one of none/zero/mean, got "
        f"{args.state_missing_keys_fill!r}"
    )
    state_missing_fill = None
    if args.state_missing_keys_fill != "none":
        if local_model_path is None:
            raise ValueError(
                "--state-missing-keys-fill requires a local --model-path checkpoint "
                "(statistics.json is read from it)."
            )
        state_missing_fill = _build_missing_state_fill(
            model_path=local_model_path,
            embodiment_tag=args.embodiment_tag,
            loader=dataset,
            required_state_keys=modality["state"].modality_keys,
            mode=args.state_missing_keys_fill,
        )

    logging.info(f"Dataset length: {len(dataset)}")
    logging.info(f"Running evaluation on trajectories: {args.traj_ids}")

    all_mse = []
    all_mae = []

    for traj_id in args.traj_ids:
        if traj_id >= len(dataset):
            logging.warning(f"Trajectory ID {traj_id} is out of range. Skipping.")
            continue

        logging.info(f"Running trajectory: {traj_id}")
        mse, mae = evaluate_single_trajectory(
            policy,
            dataset,
            traj_id,
            args.embodiment_tag,
            args.modality_keys,
            steps=args.steps,
            execution_horizon=args.execution_horizon,
            save_plot_path=args.save_plot_path,
            high_level_task=args.high_level_task,
            state_missing_fill=state_missing_fill,
        )
        logging.info(f"MSE for trajectory {traj_id}: {mse}, MAE: {mae}")
        all_mse.append(mse)
        all_mae.append(mae)

    if all_mse:
        avg_mse = np.mean(np.array(all_mse))
        avg_mae = np.mean(np.array(all_mae))
        logging.info(f"Average MSE across all trajs: {avg_mse}")
        logging.info(f"Average MAE across all trajs: {avg_mae}")
    else:
        logging.info("No valid trajectories were evaluated.")
    logging.info("Done")


if __name__ == "__main__":
    if migrate_deprecated_action_horizon_argv():
        logging.warning("--action-horizon is deprecated; use --execution-horizon.")
    # Parse arguments using tyro
    config = tyro.cli(ArgsConfig)
    main(config)
