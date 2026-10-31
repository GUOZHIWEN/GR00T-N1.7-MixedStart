#!/usr/bin/env python
"""Export a real decoded dataset observation for examples/mos_inference.py.

The two RGB frames and the current 17D state come from the same LeRobot
episode. The default instruction is the episode-level task used by the
release evaluation; frame-level text is retained in the sidecar JSON.
No actions, random observations, or robot connection are involved.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

CAMERAS = {
    "observation.images.exterior_1_left": "video.exterior_image_1_left",
    "observation.images.wrist_left": "video.wrist_image_left",
}
STATE_KEYS = ("state.eef_9d", "state.gripper_position", "state.joint_position")


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_two_rgb_frames(path, frame_index):
    """Sequentially decode frames [frame_index-1, frame_index], preserving RGB order."""
    if frame_index < 1:
        raise ValueError("--frame-index must be >= 1 so two consecutive frames exist")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    frames = []
    try:
        for index in range(frame_index + 1):
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(f"cannot decode frame {index} from {path}")
            if index >= frame_index - 1:
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if len(frames) != 2 or frames[0].shape != frames[1].shape or frames[0].ndim != 3 or frames[0].shape[2] != 3:
        raise ValueError(f"invalid RGB frame pair in {path}")
    return np.stack(frames).astype(np.uint8, copy=False)


def export_observation(dataset, episode_index, frame_index, task_override=None):
    dataset = Path(dataset)
    info = json.loads((dataset / "meta/info.json").read_text(encoding="utf-8"))
    features = info["features"]
    if features["observation.state"]["shape"] != [17]:
        raise ValueError("the released single-sample example requires a 17D observation.state")
    parquet = dataset / info["data_path"].format(
        episode_chunk=episode_index // info["chunks_size"], episode_index=episode_index,
    )
    frame = pd.read_parquet(parquet)
    if not 0 <= frame_index < len(frame):
        raise ValueError(f"frame {frame_index} outside episode with {len(frame)} rows")
    row = frame.iloc[frame_index]
    if "episode_index" in row and int(row["episode_index"]) != episode_index:
        raise ValueError(f"episode identity mismatch in {parquet}")
    state = np.asarray(row["observation.state"], dtype=np.float32)
    if state.shape != (17,) or not np.isfinite(state).all():
        raise ValueError(f"invalid 17D state at {parquet}:{frame_index}")
    values = {
        "state.eef_9d": state[:9][None, None, :].copy(),
        "state.gripper_position": state[9:10][None, None, :].copy(),
        "state.joint_position": state[10:17][None, None, :].copy(),
    }
    sources = {}
    for source_key, output_key in CAMERAS.items():
        relative_video = info["video_path"].format(
            episode_chunk=episode_index // info["chunks_size"],
            video_key=source_key, episode_index=episode_index,
        )
        values[output_key] = read_two_rgb_frames(dataset / relative_video, frame_index)[None, ...]
        sources[output_key] = {"video": relative_video, "frame_indices": [frame_index - 1, frame_index]}
    tasks = {int(record["task_index"]): record["task"] for record in read_jsonl(dataset / "meta/tasks.jsonl")}
    frame_instruction = tasks[int(row["task_index"])]
    episodes = {int(record["episode_index"]): record for record in read_jsonl(dataset / "meta/episodes.jsonl")}
    high_level_task = episodes.get(episode_index, {}).get("high_level_task")
    task = task_override or high_level_task or frame_instruction
    if not task.strip():
        raise ValueError("empty language instruction")
    provenance = {
        "schema_version": 1,
        "dataset_path": str(dataset), "episode_index": episode_index, "frame_index": frame_index,
        "timestamp": float(row["timestamp"]) if "timestamp" in row else None,
        "task": task, "task_source": "override" if task_override else ("episode.high_level_task" if high_level_task else "frame.task_index"),
        "high_level_task": high_level_task, "frame_instruction": frame_instruction,
        "state_layout": {"eef_9d": [0, 9], "gripper_position": [9, 10], "joint_position": [10, 17]},
        "video_sources": sources, "frame_order": "older frame first, current frame and state second",
        "color_order": "RGB", "units": "unchanged from dataset", "robot_observation": False,
        "note": "Real decoded synthetic evaluation sample, not a real-robot observation; inference only.",
    }
    return values, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--frame-index", type=int, default=16)
    parser.add_argument("--task", help="Override the default episode-level instruction")
    parser.add_argument("--output", type=Path, default=Path("examples/real_observation.npz"))
    args = parser.parse_args()
    values, provenance = export_observation(args.dataset_path, args.episode_index, args.frame_index, args.task)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **values)
    sidecar = args.output.with_suffix(".json")
    sidecar.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {sidecar}")
    print(json.dumps({"task": provenance["task"], "arrays": {key: [str(value.dtype), list(value.shape)] for key, value in values.items()}}, indent=2))


if __name__ == "__main__":
    main()
