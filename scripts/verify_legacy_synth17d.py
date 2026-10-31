#!/usr/bin/env python3
"""Verify that a legacy 22D synthetic container is identical in the 17D eval inputs.

This is an explicit audit for reaggregating historical release results. It does
not rewrite old JSON files and does not permit arbitrary dataset substitutions.
The action model consumes checkpoint statistics, not dataset meta/stats.json.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def verify(legacy: Path, canonical: Path) -> dict:
    for i in range(10):
        a = pd.read_parquet(legacy / f"data/chunk-000/episode_{i:06d}.parquet")
        b = pd.read_parquet(canonical / f"data/chunk-000/episode_{i:06d}.parquet")
        if len(a) != len(b) or not a["task_index"].equals(b["task_index"]):
            raise ValueError(f"episode {i}: frame count or task indices differ")
        for key in ("observation.state", "action"):
            av = np.stack(a[key].map(np.asarray))
            bv = np.stack(b[key].map(np.asarray))
            if key == "observation.state":
                av = av[:, :17]
            if not np.array_equal(av, bv):
                raise ValueError(f"episode {i}: {key} differs in the model-visible 17D inputs")
        for view in ("observation.images.exterior_1_left", "observation.images.wrist_left"):
            relative = f"videos/chunk-000/{view}/episode_{i:06d}.mp4"
            if hashlib.sha256((legacy / relative).read_bytes()).digest() != hashlib.sha256((canonical / relative).read_bytes()).digest():
                raise ValueError(f"episode {i}: {view} differs")
    for name in ("tasks.jsonl", "episodes.jsonl", "segment_schedule.jsonl"):
        if (legacy / "meta" / name).read_bytes() != (canonical / "meta" / name).read_bytes():
            raise ValueError(f"metadata differs: {name}")
    return {
        "status": "PASS",
        "legacy_dataset": f"data/{legacy.name}",
        "canonical_dataset": f"data/{canonical.name}",
        "legacy_path_checked": legacy.as_posix(),
        "canonical_path_checked": canonical.as_posix(),
        "episodes": 10,
        "state_prefix_17d_equal": True,
        "actions_equal": True,
        "task_indices_equal": True,
        "video_sha256_equal": True,
        "metadata_equal": True,
        "evaluation_statistics_source": "checkpoint",
        "dataset_statistics_compared": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, default=Path("data/synth_mixture_eval_db7_17d"))
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.legacy, args.canonical)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"PASS: model-visible inputs identical; wrote {args.out_json}")


if __name__ == "__main__":
    main()
