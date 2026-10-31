#!/usr/bin/env python
"""Validate the extracted MoS datasets without importing torch or a model."""
import argparse
import json
from pathlib import Path


def load_manifest(root: Path) -> dict:
    return json.loads((root / "docs" / "DATA_MANIFEST.json").read_text(encoding="utf-8"))


def verify_dataset(path: Path, expected: dict) -> dict:
    info_path = path / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"missing {info_path}; extract the dataset archive first")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    checks = {
        "total_episodes": expected["episodes"],
        "total_frames": expected["frames"],
    }
    for key, want in checks.items():
        if info.get(key) != want:
            raise ValueError(f"{path}: {key}={info.get(key)!r}, expected {want}")
    for feature in ("observation.state", "action"):
        if info["features"][feature]["shape"] != [17]:
            raise ValueError(f"{path}: {feature} must be 17D")
    for video in ("observation.images.exterior_1_left", "observation.images.wrist_left"):
        if video not in info["features"]:
            raise ValueError(f"{path}: missing video feature {video}")
    parquet = sorted((path / "data" / "chunk-000").glob("episode_*.parquet"))
    if len(parquet) != expected["episodes"]:
        raise ValueError(f"{path}: {len(parquet)} parquet episodes, expected {expected['episodes']}")
    for view in ("observation.images.exterior_1_left", "observation.images.wrist_left"):
        videos = sorted((path / "videos" / "chunk-000" / view).glob("episode_*.mp4"))
        if len(videos) != expected["episodes"]:
            raise ValueError(f"{path}/{view}: {len(videos)} videos, expected {expected['episodes']}")
    if expected["dataset_dir"] == "synth_mixture_eval_db7_17d":
        segment = path / "meta" / "segment_schedule.jsonl"
        if not segment.is_file():
            raise FileNotFoundError(f"missing required segment metadata: {segment}")
    return {
        "path": str(path),
        "episodes": info["total_episodes"],
        "frames": info["total_frames"],
        "state_dim": 17,
        "action_dim": 17,
        "parquet_episodes": len(parquet),
        "status": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data_dir = Path(args.data_dir)
    manifest = load_manifest(root)
    reports = []
    for spec in manifest["files"].values():
        reports.append(verify_dataset(data_dir / spec["dataset_dir"], spec))
        print(f"PASS {reports[-1]['path']}: {reports[-1]['episodes']} episodes, "
              f"{reports[-1]['frames']} frames, 17D state/action")
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"datasets": reports}, indent=2), encoding="utf-8")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
