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

"""E2 rolling-relabel dataset builder (survey §13.51).

DAgger-style history relabeling for the history-conditioned flow source:

  1. Roll a trained checkpoint out over every episode of an existing dataset
     with the STANDARD open-loop replan protocol (the policy's self-fed
     history buffer is left untouched — it already accumulates the model's own
     predictions, i.e. the drifted history distribution seen at inference).
  2. Record the per-frame EXECUTED actions (the first ``execution_horizon``
     actions of each replan chunk, in physical units).
  3. Mirror the source dataset to ``--output-dir`` (videos are hardlinked,
     not copied) and add one matrix column ``action_relabeled`` — same row
     layout as ``action`` — carrying those executed actions.
  4. Training with ``--use-relabeled-history`` then extracts history windows
     from this column (lerobot_episode_loader slices it into per-key
     ``action_relabeled.<key>`` columns; ShardedSingleStepDataset prefers them
     over the pristine GT ``action`` column). Observations, GT actions, texts
     and all meta files are byte-identical to the source dataset.

The GT action supervision target is UNCHANGED — only the history channel's
input distribution is relabeled, which is precisely the train/inference
distribution match DAgger prescribes.

Resume support: episodes whose target parquet already carries the
``action_relabeled`` column are skipped, so an interrupted run can be
relaunched with the same command.

Example (relabel the 100-episode db7 train set with the db9 checkpoint):

python scripts/build_relabeled_rollout_dataset.py \
  --dataset-path data/synth_mixture_train_db7 \
  --output-dir data/synth_mixture_train_db7_dagger1 \
  --model-path runs/integrated_db9_pz/checkpoint-5000 \
  --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
  --execution-horizon 16 --denoising-steps 4 --seed 0 \
  --state-extra-modality-keys target_relative_pose
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

RELABELED_COLUMN = "action_relabeled"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a rolling-relabeled (DAgger-style history) dataset "
        "by rolling a checkpoint over an existing dataset (survey §13.51 E2)."
    )
    parser.add_argument("--dataset-path", required=True, help="Source LeRobot-format dataset.")
    parser.add_argument("--output-dir", required=True, help="Target dataset directory (created).")
    parser.add_argument("--model-path", required=True, help="Checkpoint used for the rollout.")
    parser.add_argument("--embodiment-tag", required=True, help="Embodiment tag (name or value).")
    parser.add_argument("--episode-ids", type=int, nargs="*", default=None,
                        help="Episodes to relabel. Default: all.")
    parser.add_argument("--execution-horizon", type=int, default=16,
                        help="Replan stride / executed prefix length of each predicted chunk.")
    parser.add_argument("--denoising-steps", type=int, default=4,
                        help="Flow-matching denoising steps at inference.")
    parser.add_argument("--history-replan-stride", type=int, default=None,
                        help="rtc history-slice offset override (default: execution_horizon).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Per-episode torch seed is seed + episode_index "
                        "(paired, reproducible relabeling). -1 disables seeding.")
    parser.add_argument("--state-extra-modality-keys", type=str, default="",
                        help="Comma-separated extra state keys the checkpoint was trained "
                        "with (e.g. 'target_relative_pose'); applied process-locally "
                        "BEFORE the policy/loader are built, same as eval.")
    return parser.parse_args()


def _mirror_tree(src: Path, dst: Path) -> None:
    """Mirror the dataset tree; videos (the dominant bytes) are hardlinked."""
    for path in sorted(src.rglob("*")):
        rel = path.relative_to(src)
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            continue
        try:
            os.link(path, target)
        except OSError:
            shutil.copy2(path, target)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.eval._horizon_contract import PolicyHorizonSpec
    from gr00t.eval.compare_flow_sources import extract_gt, rollout_trajectory
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    src = Path(args.dataset_path).resolve()
    dst = Path(args.output_dir).resolve()
    assert src.is_dir(), f"dataset not found: {src}"
    assert dst != src, "--output-dir must differ from --dataset-path"

    embodiment_tag = EmbodimentTag.resolve(args.embodiment_tag)

    if args.state_extra_modality_keys:
        # Same process-local extension the training/eval harness uses (db7+).
        from gr00t.configs.data.state_extra_keys import (
            apply_state_extra_keys,
            parse_extra_keys,
        )

        apply_state_extra_keys(embodiment_tag.value, parse_extra_keys(args.state_extra_modality_keys))

    # ── policy ──────────────────────────────────────────────────────────
    policy = Gr00tPolicy(
        embodiment_tag=embodiment_tag,
        model_path=args.model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        history_replan_stride=(
            None
            if args.history_replan_stride == 0
            else (args.history_replan_stride or args.execution_horizon)
        ),
    )
    policy.model.action_head.num_inference_timesteps = args.denoising_steps
    modality = policy.get_modality_config()
    action_keys = modality["action"].modality_keys
    PolicyHorizonSpec.from_modality_config(modality, n_action_steps=args.execution_horizon)

    # Dual text port: checkpoints trained with a sub-instruction mode consume
    # the episode-level high_level_task as the driving text (same as eval).
    sub_mode = getattr(policy.model.config, "sub_instruction_mode", "off")
    hl_from_meta: dict[int, str] = {}
    if sub_mode != "off":
        episodes_jsonl = src / "meta" / "episodes.jsonl"
        if episodes_jsonl.exists():
            with open(episodes_jsonl) as f:
                for line in f:
                    rec = json.loads(line)
                    if rec.get("high_level_task"):
                        hl_from_meta[int(rec["episode_index"])] = rec["high_level_task"]
            logger.info(
                f"sub_instruction_mode={sub_mode!r}: loaded {len(hl_from_meta)} "
                "episode-level high-level tasks"
            )

    # ── mirror the dataset tree, then rewrite parquets one by one ──────
    logger.info(f"Mirroring {src} -> {dst} (videos hardlinked)")
    _mirror_tree(src, dst)

    loader = LeRobotEpisodeLoader(dataset_path=str(src), modality_configs=modality)
    episode_ids = args.episode_ids if args.episode_ids is not None else list(range(len(loader)))
    logger.info(f"Relabeling {len(episode_ids)} episodes from {args.model_path}")

    provenance: dict[str, object] = {
        "model_path": args.model_path,
        "dataset_path": str(src),
        "embodiment_tag": str(args.embodiment_tag),
        "execution_horizon": args.execution_horizon,
        "denoising_steps": args.denoising_steps,
        "history_replan_stride": args.history_replan_stride,
        "seed": args.seed,
        "sub_instruction_mode": sub_mode,
        "state_extra_modality_keys": args.state_extra_modality_keys,
        "created": datetime.now(timezone.utc).isoformat(),
        "episodes": {},
    }

    for ep_id in episode_ids:
        parquet_path = dst / "data" / "chunk-000" / f"episode_{ep_id:06d}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"target parquet missing after mirror: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        if RELABELED_COLUMN in df.columns:
            logger.info(f"[ep {ep_id}] already relabeled, skipped (resume)")
            continue

        ep_len = len(df)
        if args.seed is not None and args.seed >= 0:
            # Paired seeding: reproducible initial flow noise per episode.
            torch.manual_seed(args.seed + ep_id)
        executed: list[dict[str, np.ndarray]] = []
        pred = rollout_trajectory(
            policy,
            loader,
            ep_id,
            embodiment_tag,
            action_keys,
            steps=ep_len,
            execution_horizon=args.execution_horizon,
            high_level_task=hl_from_meta.get(ep_id),
            executed_per_key_out=executed,
        )
        assert len(executed) == len(pred) == ep_len, (
            f"ep {ep_id}: executed frames {len(executed)} / pred {len(pred)} != "
            f"episode length {ep_len}"
        )

        # Concatenate per-key vectors in modality-key order -> (T, D_total),
        # exactly the column layout of the source ``action`` matrix column.
        mat = np.stack(
            [np.concatenate([frame[key] for key in action_keys], axis=0) for frame in executed],
            axis=0,
        ).astype(np.float32)

        # Drift diagnostic: how far the model's executed trajectory wandered
        # from GT on THIS episode (the relabel gap the retrain will absorb).
        traj = loader[ep_id]
        gt = extract_gt(traj, action_keys, ep_len)
        mse = float(np.mean((gt - pred) ** 2))
        mae = float(np.mean(np.abs(gt - pred)))
        provenance["episodes"][str(ep_id)] = {
            "length": ep_len,
            "mse_to_gt": mse,
            "mae_to_gt": mae,
        }
        logger.info(f"[ep {ep_id}] T={ep_len} relabel gap: MSE={mse:.6f} MAE={mae:.6f}")

        df[RELABELED_COLUMN] = list(mat)
        df.to_parquet(parquet_path, index=False)

        # Self-verifying write: the loader's joint-group slicing requires the
        # column to round-trip as np.ndarray cells (same storage form as the
        # source ``action`` column). Fail loudly here rather than at training.
        back = pd.read_parquet(parquet_path, columns=[RELABELED_COLUMN])
        cell = back[RELABELED_COLUMN].iloc[0]
        assert isinstance(cell, np.ndarray) and cell.shape == (mat.shape[1],), (
            f"ep {ep_id}: {RELABELED_COLUMN} round-trip broken — cell type "
            f"{type(cell)} shape {getattr(cell, 'shape', None)}, expected "
            f"np.ndarray ({mat.shape[1]},)"
        )

    meta_dir = dst / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    with open(meta_dir / "relabel_provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)
    logger.info(f"Wrote relabeled dataset to {dst} (provenance: meta/relabel_provenance.json)")
    logger.info(
        "Train on it with: launch_finetune.py --dataset-path "
        f"{dst} --use-relabeled-history ... (history windows come from the "
        "model's own rollout; GT action supervision unchanged)"
    )


if __name__ == "__main__":
    main()
