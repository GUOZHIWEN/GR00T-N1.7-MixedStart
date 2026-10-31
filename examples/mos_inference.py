#!/usr/bin/env python3
"""Run one real NPZ observation through a local MoS checkpoint.

This example performs inference only. It neither connects to nor controls a robot.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VIDEO_SPECS = {
    "video.exterior_image_1_left": (1, 2),
    "video.wrist_image_left": (1, 2),
}
STATE_SPECS = {
    "state.eef_9d": (1, 1, 9),
    "state.gripper_position": (1, 1, 1),
    "state.joint_position": (1, 1, 7),
}


def load_observation(path: Path, task: str) -> dict[str, dict[str, np.ndarray | list[list[str]]]]:
    """Load and validate the single-sample contract without altering model units."""
    path = path.expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"Observation NPZ not found: {path.resolve()}. Use "
            "examples/real_observation.npz or pass an absolute path to your own observation."
        )
    if not task.strip():
        raise ValueError("--task must be a non-empty instruction")
    observation: dict[str, dict] = {"video": {}, "state": {}, "language": {}}
    with np.load(path, allow_pickle=False) as sample:
        for key, prefix in VIDEO_SPECS.items():
            if key not in sample:
                raise KeyError(f"missing NPZ key: {key}")
            value = sample[key]
            if value.dtype != np.uint8 or value.ndim != 5 or value.shape[:2] != prefix:
                raise ValueError(f"{key}: expected uint8 (1,2,H,W,3), got {value.dtype} {value.shape}")
            if value.shape[-1] != 3 or min(value.shape[2:4]) < 1:
                raise ValueError(f"{key}: invalid image dimensions {value.shape}")
            observation["video"][key.removeprefix("video.")] = value
        for key, shape in STATE_SPECS.items():
            if key not in sample:
                raise KeyError(f"missing NPZ key: {key}")
            value = sample[key]
            if value.dtype != np.float32 or value.shape != shape:
                raise ValueError(f"{key}: expected float32 {shape}, got {value.dtype} {value.shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"{key}: contains NaN or infinity")
            observation["state"][key.removeprefix("state.")] = value
    observation["language"]["annotation.language.language_instruction"] = [[task]]
    return observation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--sample-npz", required=True, type=Path)
    parser.add_argument("--task", default="Perform the task shown in the observation.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--fixed-beta", type=float, default=0.0)
    parser.add_argument("--fixed-noise-std", type=float, default=0.0)
    parser.add_argument("--history-replan-stride", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("runs/mos_inference_action.npz"))
    args = parser.parse_args()

    from gr00t.policy.gr00t_policy import Gr00tPolicy

    observation = load_observation(args.sample_npz, args.task)
    policy = Gr00tPolicy(
        embodiment_tag="OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT",
        model_path=str(args.model_path),
        device=args.device,
        strict=True,
        history_replan_stride=args.history_replan_stride,
        denoising_steps=args.denoising_steps,
        fixed_beta=args.fixed_beta,
        fixed_noise_std=args.fixed_noise_std,
    )
    policy.reset()
    action, info = policy.get_action(observation)
    expected = {"eef_9d": 9, "gripper_position": 1, "joint_position": 7}
    output = {}
    for key, width in expected.items():
        value = np.asarray(action[key])
        if value.shape != (1, 40, width) or value.dtype != np.float32 or not np.isfinite(value).all():
            raise RuntimeError(f"unexpected model output {key}: {value.dtype} {value.shape}")
        output[f"action.{key}"] = value[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    print(json.dumps({
        "model_path": str(args.model_path),
        "output": str(args.output),
        "action_shapes": {key: list(value.shape) for key, value in output.items()},
        "deployment": {
            "beta": args.fixed_beta,
            "eta": args.fixed_noise_std,
            "denoising_steps": args.denoising_steps,
            "history_replan_stride": args.history_replan_stride,
        },
        "info_keys": sorted(info),
        "robot_connected": False,
    }, indent=2))


if __name__ == "__main__":
    main()
