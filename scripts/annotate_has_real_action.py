#!/usr/bin/env python3

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

"""
Annotate a GR00T LeRobot v2 dataset with ``has_real_action`` flags for P0-1
co-training (mixing pseudo-labelled / action-free data with real trajectories).

Two annotation granularities are supported, mirroring the loader's resolution
order (per-frame column wins over the episode-level sidecar):

1. Episode-level sidecar (this script's main job):
   writes ``<dataset>/meta/has_real_action.json``:

       {"default": 1.0, "episodes": {"3": 0.0, "7": 0.0}}

   Episodes not listed fall back to ``default``. Use this to mark whole
   episodes as pseudo-labelled (IDM / LAPA / DreamGen) or real.

2. Per-frame parquet column: if you need frame-level granularity (e.g. only
   part of an episode is pseudo-labelled), add a float ``has_real_action``
   column to the episode parquet files directly; ``LeRobotEpisodeLoader``
   passes it through and it takes precedence over the sidecar.

Training consumes the flags with ``--use-has-real-action`` (see
gr00t/configs/finetune_config.py): real samples keep loss weight 1.0,
pseudo-labelled samples get ``--pseudo-action-loss-weight`` (or 0 with
``--drop-unlabeled-action``).

Usage:
    # Mark episodes 3 and 7 as pseudo-labelled; everything else real:
    python scripts/annotate_has_real_action.py \\
        --dataset-path /path/to/dataset --pseudo-episodes 3 7

    # Mark everything pseudo except episodes 0 and 1 (default flipped to 0.0):
    python scripts/annotate_has_real_action.py \\
        --dataset-path /path/to/dataset --default 0.0 --real-episodes 0 1

    # Mark a range of episodes pseudo-labelled:
    python scripts/annotate_has_real_action.py \\
        --dataset-path /path/to/dataset --pseudo-episodes $(seq 100 199)

    # Remove the sidecar (back to all-real):
    python scripts/annotate_has_real_action.py \\
        --dataset-path /path/to/dataset --clear
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from gr00t.data.dataset.lerobot_episode_loader import (
    LEROBOT_EPISODES_FILENAME,
    LEROBOT_HAS_REAL_ACTION_FILENAME,
    LEROBOT_META_DIR_NAME,
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--dataset-path", required=True, help="Path to the GR00T LeRobot v2 dataset root"
    )
    parser.add_argument(
        "--default",
        type=float,
        default=1.0,
        choices=(0.0, 1.0),
        help="Fallback flag for unlisted episodes (1.0 = real, 0.0 = pseudo)",
    )
    parser.add_argument(
        "--pseudo-episodes",
        type=int,
        nargs="*",
        default=[],
        help="Episode indices to mark as pseudo-labelled (flag 0.0)",
    )
    parser.add_argument(
        "--real-episodes",
        type=int,
        nargs="*",
        default=[],
        help="Episode indices to mark as real (flag 1.0); useful with --default 0.0",
    )
    parser.add_argument(
        "--clear", action="store_true", help="Delete the sidecar file and exit"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    meta_dir = dataset_path / LEROBOT_META_DIR_NAME
    sidecar_path = meta_dir / LEROBOT_HAS_REAL_ACTION_FILENAME

    if not meta_dir.is_dir():
        raise FileNotFoundError(f"Dataset meta dir does not exist: {meta_dir}")

    if args.clear:
        if sidecar_path.exists():
            sidecar_path.unlink()
            logging.info(f"Removed {sidecar_path}")
        else:
            logging.info(f"No sidecar at {sidecar_path}; nothing to do")
        return

    # Validate the requested episode indices against episodes.jsonl.
    episodes_path = meta_dir / LEROBOT_EPISODES_FILENAME
    with open(episodes_path, "r") as f:
        episode_indices = {json.loads(line)["episode_index"] for line in f}

    requested = set(args.pseudo_episodes) | set(args.real_episodes)
    unknown = sorted(requested - episode_indices)
    if unknown:
        raise ValueError(
            f"Episode indices {unknown} not found in {episodes_path} "
            f"(dataset has {len(episode_indices)} episodes)"
        )
    overlap = set(args.pseudo_episodes) & set(args.real_episodes)
    if overlap:
        raise ValueError(
            f"Episodes {sorted(overlap)} are listed as both pseudo and real; pick one"
        )

    episodes: dict[str, float] = {}
    for idx in args.pseudo_episodes:
        episodes[str(idx)] = 0.0
    for idx in args.real_episodes:
        episodes[str(idx)] = 1.0

    # Merge with an existing sidecar so incremental annotation does not clobber
    # previous overrides.
    if sidecar_path.exists():
        with open(sidecar_path, "r") as f:
            existing = json.load(f)
        merged = existing.get("episodes", {})
        merged.update(episodes)
        episodes = merged
        if "default" in existing and args.default == 1.0:
            # Keep the existing default unless the user explicitly flipped it.
            args.default = existing["default"]

    payload = {"default": float(args.default), "episodes": episodes}
    with open(sidecar_path, "w") as f:
        json.dump(payload, f, indent=2)

    n_pseudo = sum(1 for v in episodes.values() if v < 0.5)
    logging.info(
        f"Wrote {sidecar_path}: default={args.default}, "
        f"{len(episodes)} overrides ({n_pseudo} pseudo / {len(episodes) - n_pseudo} real)"
    )
    logging.info(
        "Train with --use-has-real-action (plus --pseudo-action-loss-weight / "
        "--drop-unlabeled-action) to consume these flags."
    )


if __name__ == "__main__":
    main()
