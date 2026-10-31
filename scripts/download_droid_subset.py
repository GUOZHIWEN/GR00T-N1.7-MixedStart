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
Download a mid-size DROID subset (hundreds of episodes) and convert it to
GR00T LeRobot v2.1 format, for real-robot fine-tuning / evaluation (Phase 0).

Differences vs ``scripts/download_droid_sample.py``:
  - Two sources: ModelScope (default, fast in CN) or HuggingFace.
  - Multi-file aware: episodes may span several data/video files; selection
    honors --max-data-files / --max-video-files caps.
  - Produces a train dir AND an eval dir (eval episodes are the TAIL of the
    selected list, never seen by training).
  - Built-in validation against the OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT
    modality config (LeRobotEpisodeLoader + 17D shape checks).

Prerequisites:
    pip install modelscope          # for --source modelscope (default)
    ffmpeg + ffprobe on PATH        # video segment extraction
    uv pip install jsonlines        # if not already installed

Usage:
    python scripts/download_droid_subset.py \
        --num-episodes 300 --num-eval-episodes 30 \
        --max-data-files 4 --max-video-files 8 \
        --cache-dir /path/to/droid_cache \
        --output-dir data/droid_subset_train \
        --eval-output-dir data/droid_subset_eval
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import subprocess
import sys

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.state_action.droid_frame import compute_eef_9d
from gr00t.data.stats import generate_rel_stats, generate_stats
import jsonlines
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


REPO_ID = "lerobot/droid_1.0.1"
# Source-repository HEAD recorded on 2026-10-31. Keeping this explicit makes
# episode selection and converted archives reproducible across later uploads.
DROID_SOURCE_REVISION = "b697b419ca258bf377cbc0c3279e5348fc5c7c6f"
EMBODIMENT_TAG = EmbodimentTag.OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT

# The 2 cameras used by the OXE_DROID model config (exterior_2_left is dropped).
VIDEO_KEYS = [
    "observation.images.exterior_1_left",
    "observation.images.wrist_left",
]

META_FILES = ["meta/info.json", "meta/tasks.parquet"]


# ───────────────────────────── download ─────────────────────────────


def build_file_list(max_data_files: int, max_video_files: int) -> list[str]:
    files = list(META_FILES)
    # Episodes metadata: all files of chunk-000 (glob pattern, covers eps 0-999)
    files.append("meta/episodes/chunk-000/*.parquet")
    for i in range(max_data_files):
        files.append(f"data/chunk-000/file-{i:03d}.parquet")
    for video_key in VIDEO_KEYS:
        for i in range(max_video_files):
            files.append(f"videos/{video_key}/chunk-000/file-{i:03d}.mp4")
    return files


def download_modelscope(files: list[str], cache_dir: Path, revision: str) -> None:
    try:
        from modelscope import snapshot_download
    except ImportError as e:
        raise RuntimeError(
            "modelscope not installed in the project venv. "
            "Run: uv pip install modelscope  (NOT plain pip — that installs into base conda)"
        ) from e

    logger.info(
        f"ModelScope snapshot_download {REPO_ID}@{revision} "
        f"({len(files)} patterns) -> {cache_dir}"
    )
    snapshot_download(
        REPO_ID,
        repo_type="dataset",
        revision=revision,
        allow_patterns=files,
        local_dir=str(cache_dir),
    )


def download_huggingface(files: list[str], cache_dir: Path, revision: str) -> None:
    from huggingface_hub import hf_hub_download

    for fname in files:
        if "*" in fname:
            # HF fallback: only the first episodes-metadata file (enough for a subset)
            fname = "meta/episodes/chunk-000/file-000.parquet"
        logger.info(f"  {fname}@{revision}...")
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            revision=revision,
            filename=fname,
            local_dir=str(cache_dir),
        )


# ───────────────────────── episode selection ─────────────────────────


def load_episodes_metadata(cache_dir: Path) -> pd.DataFrame:
    ep_dir = cache_dir / "meta" / "episodes" / "chunk-000"
    files = sorted(ep_dir.glob("*.parquet"))
    if not files:
        raise RuntimeError(
            f"No episodes metadata under {ep_dir} — download step incomplete?"
        )
    # Read only the columns we actually use (memory guard: the full v3.0
    # episodes table carries extra wide columns we never touch).
    wanted = ["episode_index", "data/file_index", "tasks"]
    for k in VIDEO_KEYS:
        wanted += [
            f"videos/{k}/from_timestamp",
            f"videos/{k}/to_timestamp",
            f"videos/{k}/file_index",
        ]
    frames = []
    for f in files:
        cols = [c for c in wanted if c in pq.read_schema(f).names]
        logger.info(f"  loading {f.name} ({f.stat().st_size / 1e6:.1f} MB, {len(cols)} cols)...")
        frames.append(pq.read_table(f, columns=cols).to_pandas())
    df = pd.concat(frames, ignore_index=True)
    logger.info(f"  episodes metadata: {len(df)} rows from {len(files)} file(s)")
    return df.sort_values("episode_index").reset_index(drop=True)


def video_file_index(row: pd.Series, video_key: str, data_fi: int) -> int:
    """Per-camera video file index; fall back to the data file index if the
    column is absent in this schema revision."""
    col = f"videos/{video_key}/file_index"
    if col in row.index and pd.notna(row[col]):
        return int(row[col])
    return data_fi


def select_episodes(
    episodes_df: pd.DataFrame,
    total_needed: int,
    max_data_files: int,
    max_video_files: int,
) -> list[pd.Series]:
    selected = []
    for _, row in episodes_df.iterrows():
        if len(selected) >= total_needed:
            break
        data_fi = int(row["data/file_index"])
        if data_fi >= max_data_files:
            continue
        vfis = [video_file_index(row, k, data_fi) for k in VIDEO_KEYS]
        if any(v >= max_video_files for v in vfis):
            continue
        selected.append(row)
    if len(selected) < total_needed:
        logger.warning(
            f"Only {len(selected)}/{total_needed} episodes fit within "
            f"--max-data-files {max_data_files} / --max-video-files {max_video_files}. "
            "Raise the caps or lower --num-episodes."
        )
    if not selected:
        raise RuntimeError("No episodes selectable within the given file caps.")
    return selected


# ───────────────────────── v2.1 conversion ─────────────────────────


def _rebuild_17d(ep_df: pd.DataFrame) -> pd.DataFrame:
    """eef_9d (XYZ + rot6d) + gripper + joints -> 17D state/action."""
    ep_df = ep_df.copy()
    for prefix in ["observation.state", "action"]:
        cart_col = f"{prefix}.cartesian_position"
        if cart_col in ep_df.columns:
            cart = np.stack(ep_df[cart_col].values)  # (T, 6)
            ep_df[f"{prefix}.eef_9d"] = list(compute_eef_9d(cart))  # (T, 9)

    for target, cols in [
        ("observation.state", [
            "observation.state.eef_9d",
            "observation.state.gripper_position",
            "observation.state.joint_position",
        ]),
        ("action", [
            "action.eef_9d",
            "action.gripper_position",
            "action.joint_position",
        ]),
    ]:
        parts = [
            np.stack([np.atleast_1d(v) for v in ep_df[c].values])
            for c in cols
            if c in ep_df.columns
        ]
        if parts:
            ep_df[target] = list(np.concatenate(parts, axis=-1))  # (T, 17)
    return ep_df


def _extract_video_segment(
    cache_dir: Path,
    video_key: str,
    video_fi: int,
    from_ts: float,
    to_ts: float,
    out_video: Path,
) -> None:
    source_video = cache_dir / "videos" / video_key / "chunk-000" / f"file-{video_fi:03d}.mp4"
    if not source_video.exists():
        raise RuntimeError(
            f"Missing video file: {source_video}. The episode's {video_key} lives in "
            f"file-{video_fi:03d}.mp4 which was not downloaded — raise --max-video-files."
        )
    duration = to_ts - from_ts
    base = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{from_ts:.6f}",
            "-i", str(source_video), "-t", f"{duration:.6f}"]
    result = subprocess.run(base + ["-c", "copy", str(out_video)], capture_output=True, text=True)
    if result.returncode != 0:  # AV1 etc. -> re-encode
        subprocess.run(
            base + ["-c:v", "libx264", "-crf", "23", "-preset", "fast", str(out_video)],
            check=True,
        )


def _probe_video_shape(video_path: Path) -> list[int]:
    """[H, W, 3] via ffprobe; falls back to the DROID standard 180x320."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=height,width", "-of", "csv=p=0", str(video_path)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        w, h = [int(x) for x in out.split(",")[:2]]
        return [h, w, 3]
    except Exception:
        return [180, 320, 3]


def build_v21_dataset(
    episodes: list[pd.Series],
    cache_dir: Path,
    output_dir: Path,
    tasks_df_reset: pd.DataFrame,
    task_text_col: str,
    fps: int,
) -> None:
    output_dir = Path(output_dir)
    data_chunk_dir = output_dir / "data" / "chunk-000"
    data_chunk_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(exist_ok=True)

    # Group episode indices by data file so each parquet is loaded once.
    by_data_file: dict[int, list[int]] = {}
    for pos, row in enumerate(episodes):
        by_data_file.setdefault(int(row["data/file_index"]), []).append(pos)

    episode_records: list[dict] = []
    task_indices_used: set[int] = set()
    src_rows: list[pd.Series] = []

    for data_fi, positions in sorted(by_data_file.items()):
        data_path = cache_dir / "data" / "chunk-000" / f"file-{data_fi:03d}.parquet"
        if not data_path.exists():
            raise RuntimeError(f"Missing data file: {data_path} — raise --max-data-files.")
        logger.info(
            f"  loading {data_path.name} ({data_path.stat().st_size / 1e6:.0f} MB, "
            f"{len(positions)} episodes to extract)..."
        )
        df = pq.read_table(data_path).to_pandas()
        logger.info(f"  {data_path.name} loaded: {len(df)} frames")
        for pos in positions:
            row = episodes[pos]
            ep_idx = int(row["episode_index"])
            ep_df = df[df["episode_index"] == ep_idx].copy()
            if len(ep_df) == 0:
                logger.warning(f"Episode {ep_idx} has no rows in {data_path.name}, skipping")
                continue
            new_ep_idx = len(episode_records)
            ep_df["episode_index"] = new_ep_idx
            ep_df["index"] = range(len(ep_df))
            ep_df = _rebuild_17d(ep_df)

            if "task_index" in ep_df.columns:
                task_indices_used.update(ep_df["task_index"].unique().tolist())
            lang = (
                str(ep_df["language_instruction"].iloc[0])
                if "language_instruction" in ep_df.columns
                else ""
            )
            logger.info(f"  ep {ep_idx} -> {new_ep_idx}: {len(ep_df)} frames, task={lang[:60]!r}")

            ep_df.to_parquet(data_chunk_dir / f"episode_{new_ep_idx:06d}.parquet", index=False)
            episode_records.append(
                {
                    "episode_index": new_ep_idx,
                    "tasks": list(row["tasks"]) if "tasks" in row.index else [],
                    "length": len(ep_df),
                }
            )
            src_rows.append(row)
        del df  # free before next file

    if not episode_records:
        raise RuntimeError("No episodes could be extracted")

    # ── videos ──
    for video_key in VIDEO_KEYS:
        video_chunk_dir = output_dir / "videos" / "chunk-000" / video_key
        video_chunk_dir.mkdir(parents=True, exist_ok=True)
        for rec, row in zip(episode_records, src_rows):
            data_fi = int(row["data/file_index"])
            vfi = video_file_index(row, video_key, data_fi)
            from_ts = float(row[f"videos/{video_key}/from_timestamp"])
            to_ts = float(row[f"videos/{video_key}/to_timestamp"])
            out_video = video_chunk_dir / f"episode_{rec['episode_index']:06d}.mp4"
            _extract_video_segment(cache_dir, video_key, vfi, from_ts, to_ts, out_video)
        logger.info(f"  {video_key}: {len(episode_records)} segments extracted")

    # ── meta files ──
    with jsonlines.open(meta_dir / "episodes.jsonl", mode="w") as writer:
        for rec in episode_records:
            writer.write(rec)

    with jsonlines.open(meta_dir / "tasks.jsonl", mode="w") as writer:
        for _, row in tasks_df_reset.iterrows():
            tidx = int(row["task_index"])
            if tidx in task_indices_used:
                writer.write({"task_index": tidx, "task": str(row[task_text_col])})

    video_shape = _probe_video_shape(
        output_dir / "videos" / "chunk-000" / VIDEO_KEYS[0] / "episode_000000.mp4"
    )
    info = {
        "codebase_version": "v2.1",
        "robot_type": "droid",
        "total_episodes": len(episode_records),
        "total_frames": sum(r["length"] for r in episode_records),
        "fps": fps,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "chunks_size": 1000,
        "splits": {"train": f"0:{len(episode_records)}"},
        "features": {
            "observation.images.exterior_1_left": {"dtype": "video", "shape": video_shape},
            "observation.images.wrist_left": {"dtype": "video", "shape": video_shape},
            "observation.state": {"dtype": "float32", "shape": [17]},
            "action": {"dtype": "float32", "shape": [17]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    modality = {
        "state": {
            "eef_9d": {"start": 0, "end": 9},
            "gripper_position": {"start": 9, "end": 10},
            "joint_position": {"start": 10, "end": 17},
        },
        "action": {
            "eef_9d": {"start": 0, "end": 9},
            "gripper_position": {"start": 9, "end": 10},
            "joint_position": {"start": 10, "end": 17},
        },
        "video": {
            "exterior_1_left": {"original_key": "observation.images.exterior_1_left"},
            "wrist_left": {"original_key": "observation.images.wrist_left"},
        },
        "annotation": {
            "language.language_instruction": {"original_key": "task_index"},
        },
    }
    with open(meta_dir / "modality.json", "w") as f:
        json.dump(modality, f, indent=2)

    logger.info("  Generating stats.json and relative_stats.json ...")
    generate_stats(output_dir)
    generate_rel_stats(output_dir, EMBODIMENT_TAG)

    logger.info(
        f"Dataset ready: {output_dir}  episodes={len(episode_records)}  "
        f"frames={sum(r['length'] for r in episode_records)}"
    )


# ───────────────────────── validation ─────────────────────────


def validate_dataset(output_dir: Path, expected_episodes: int) -> bool:
    """Check the produced dataset against the OXE_DROID modality contract."""
    output_dir = Path(output_dir)
    cfg = MODALITY_CONFIGS[EMBODIMENT_TAG.value]
    results: list[tuple[str, bool]] = []

    def check(name: str, ok: bool) -> None:
        results.append((name, bool(ok)))

    for rel in ["meta/info.json", "meta/modality.json", "meta/episodes.jsonl", "meta/tasks.jsonl"]:
        check(f"exists {rel}", (output_dir / rel).is_file())
    check("exists meta/stats.json", (output_dir / "meta" / "stats.json").is_file())
    check(
        "exists relative stats",
        any("relative" in p.name for p in (output_dir / "meta").glob("*.json")),
    )

    # 17D shapes from the parquet itself
    ep0_path = output_dir / "data" / "chunk-000" / "episode_000000.parquet"
    if ep0_path.is_file():
        ep0 = pd.read_parquet(ep0_path)
        s = np.asarray(ep0["observation.state"].iloc[0])
        a = np.asarray(ep0["action"].iloc[0])
        check(f"state dim == 17 (got {s.shape[-1]})", s.shape[-1] == 17)
        check(f"action dim == 17 (got {a.shape[-1]})", a.shape[-1] == 17)
    else:
        check("episode_000000.parquet exists", False)

    # modality.json covers the keys the embodiment config expects
    with open(output_dir / "meta" / "modality.json") as f:
        modality = json.load(f)
    for key in cfg["state"].modality_keys:
        check(f"modality.state has {key}", key in modality.get("state", {}))
    for key in cfg["action"].modality_keys:
        check(f"modality.action has {key}", key in modality.get("action", {}))
    check("modality.video has exterior_1_left", "exterior_1_left" in modality.get("video", {}))
    check("modality.video has wrist_left", "wrist_left" in modality.get("video", {}))

    # The real contract test: the training-side loader must accept the dataset.
    try:
        loader = LeRobotEpisodeLoader(output_dir, cfg)
        check(
            f"LeRobotEpisodeLoader episodes == {expected_episodes} (got {len(loader)})",
            len(loader) == expected_episodes,
        )
        df0 = loader[0]
        check("episode 0 loads non-empty", len(df0) > 0)
    except Exception as e:  # noqa: BLE001
        check(f"LeRobotEpisodeLoader accepts dataset ({type(e).__name__}: {e})", False)

    logger.info(f"── validation: {output_dir} ──")
    all_ok = True
    for name, ok in results:
        logger.info(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        all_ok &= ok
    logger.info(f"── validation {'PASSED' if all_ok else 'FAILED'} ──")
    return all_ok


# ───────────────────────── main ─────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download a DROID subset (ModelScope or HF) and convert to GR00T LeRobot v2.1.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", choices=["modelscope", "huggingface"], default="modelscope")
    parser.add_argument("--revision", default=DROID_SOURCE_REVISION,
                        help="immutable source-repository commit")
    parser.add_argument("--num-episodes", type=int, default=300, help="train episodes")
    parser.add_argument("--num-eval-episodes", type=int, default=30,
                        help="eval episodes (tail of the selected list)")
    parser.add_argument("--max-data-files", type=int, default=4)
    parser.add_argument("--max-video-files", type=int, default=8)
    parser.add_argument("--cache-dir", default=".cache/droid_ms_cache")
    parser.add_argument("--output-dir", default="data/droid_subset_train")
    parser.add_argument("--eval-output-dir", default="data/droid_subset_eval")
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    total_needed = args.num_episodes + args.num_eval_episodes

    # 1) download
    files = build_file_list(args.max_data_files, args.max_video_files)
    if args.source == "modelscope":
        download_modelscope(files, cache_dir, args.revision)
    else:
        download_huggingface(files, cache_dir, args.revision)
    logger.info("phase 1/4 download complete")

    # 2) select episodes (data/video file caps honored)
    episodes_df = load_episodes_metadata(cache_dir)
    logger.info("phase 2/4 selecting episodes...")
    selected = select_episodes(
        episodes_df, total_needed, args.max_data_files, args.max_video_files
    )
    train_eps = selected[: args.num_episodes]
    eval_eps = selected[args.num_episodes :]
    logger.info(f"Selected {len(train_eps)} train + {len(eval_eps)} eval episodes")

    # 3) shared metadata
    logger.info("phase 3/4 reading info.json / tasks.parquet ...")
    with open(cache_dir / "meta" / "info.json") as f:
        fps = json.load(f).get("fps", 15)
    tasks_df = pq.read_table(cache_dir / "meta" / "tasks.parquet").to_pandas()
    tasks_df_reset = tasks_df.reset_index()
    task_text_col = tasks_df_reset.columns[0]

    # 4) convert
    ok = True
    if train_eps:
        logger.info(f"phase 4/4 converting train -> {args.output_dir} ...")
        build_v21_dataset(train_eps, cache_dir, Path(args.output_dir),
                          tasks_df_reset, task_text_col, fps)
        if not args.skip_validation:
            ok &= validate_dataset(Path(args.output_dir), len(train_eps))
    if eval_eps:
        logger.info(f"phase 4/4 converting eval -> {args.eval_output_dir} ...")
        build_v21_dataset(eval_eps, cache_dir, Path(args.eval_output_dir),
                          tasks_df_reset, task_text_col, fps)
        if not args.skip_validation:
            ok &= validate_dataset(Path(args.eval_output_dir), len(eval_eps))

    logger.info("Done. Train with embodiment tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT; "
                "do NOT pass --state-extra-modality-keys or --expected-data-version for real DROID.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
