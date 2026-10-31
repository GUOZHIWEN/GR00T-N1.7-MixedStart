#!/usr/bin/env python
"""Build a small, real tabular preview from the two held-out MoS archives.

The preview is only for the ModelScope web viewer. It never replaces the four
versioned LeRobot archives used for training or evaluation. No images are copied
into this table, and no trajectory values are synthesized or rescaled.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import pandas as pd


SOURCES = (
    ("synthetic", "synth_mixture_eval_db7_17d.tar.gz"),
    ("DROID", "droid_subset_eval.tar.gz"),
)
DIMENSION = 17
STATE_COLUMNS = [f"state_{i:02d}" for i in range(DIMENSION)]
ACTION_COLUMNS = [f"action_{i:02d}" for i in range(DIMENSION)]
METADATA_COLUMNS = [
    "domain", "source_archive", "episode_index", "frame_index",
    "timestamp", "instruction",
]


def read_jsonl(archive: tarfile.TarFile, name: str) -> list[dict]:
    member = archive.getmember(name)
    if not member.isfile():
        raise ValueError(f"not a regular file: {name}")
    handle = archive.extractfile(member)
    if handle is None:
        raise ValueError(f"cannot read {name}")
    return [json.loads(line) for line in handle.read().decode("utf-8").splitlines()
            if line.strip()]


def read_parquet(archive: tarfile.TarFile, name: str) -> pd.DataFrame:
    member = archive.getmember(name)
    if not member.isfile():
        raise ValueError(f"not a regular file: {name}")
    handle = archive.extractfile(member)
    if handle is None:
        raise ValueError(f"cannot read {name}")
    return pd.read_parquet(io.BytesIO(handle.read()))


def vector_column(df: pd.DataFrame, name: str) -> np.ndarray:
    if name not in df.columns:
        raise ValueError(f"missing {name}")
    matrix = np.stack(df[name].map(lambda x: np.asarray(x, dtype=np.float32)))
    if matrix.shape != (len(df), DIMENSION):
        raise ValueError(f"{name} must have shape ({len(df)}, {DIMENSION}), got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def sample_archive(staging: Path, domain: str, archive_name: str,
                   episodes_per_domain: int, frames_per_episode: int) -> pd.DataFrame:
    dataset_dir = archive_name.removesuffix(".tar.gz")
    with tarfile.open(staging / archive_name, "r:gz") as archive:
        episodes = sorted(
            read_jsonl(archive, f"{dataset_dir}/meta/episodes.jsonl"),
            key=lambda record: record["episode_index"],
        )
        if not episodes:
            raise ValueError(f"no episodes in {archive_name}")
        rows = []
        for episode in episodes[:episodes_per_domain]:
            episode_index = int(episode["episode_index"])
            relative = f"{dataset_dir}/data/chunk-000/episode_{episode_index:06d}.parquet"
            df = read_parquet(archive, relative)
            states = vector_column(df, "observation.state")
            actions = vector_column(df, "action")
            instructions = episode.get("tasks", [])
            if not instructions or not isinstance(instructions[0], str) or not instructions[0].strip():
                raise ValueError(f"missing primary instruction for episode {episode_index} in {archive_name}")
            positions = np.linspace(0, len(df) - 1, num=min(frames_per_episode, len(df)), dtype=int)
            for position in positions:
                frame_index = int(df["frame_index"].iloc[position]) if "frame_index" in df else position
                rows.append({
                    "domain": domain,
                    "source_archive": archive_name,
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    "timestamp": float(df["timestamp"].iloc[position]),
                    "instruction": instructions[0],
                    **{name: value for name, value in zip(STATE_COLUMNS, states[position])},
                    **{name: value for name, value in zip(ACTION_COLUMNS, actions[position])},
                })
    return pd.DataFrame(rows, columns=METADATA_COLUMNS + STATE_COLUMNS + ACTION_COLUMNS)


def validate_preview(path: Path, expected: Path | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = METADATA_COLUMNS + STATE_COLUMNS + ACTION_COLUMNS
    if list(df.columns) != required:
        raise ValueError(f"unexpected preview columns: {list(df.columns)}")
    if df.empty or set(df["domain"]) != {domain for domain, _ in SOURCES}:
        raise ValueError("preview must contain both synthetic and DROID rows")
    if df[METADATA_COLUMNS].isna().any().any():
        raise ValueError("preview metadata contains missing values")
    for columns in (STATE_COLUMNS, ACTION_COLUMNS):
        values = df[columns].to_numpy(dtype=np.float32)
        if values.shape != (len(df), DIMENSION) or not np.isfinite(values).all():
            raise ValueError("preview contains invalid 17D vectors")
    if expected is not None:
        reference = pd.read_parquet(expected)
        pd.testing.assert_frame_equal(df, reference, check_exact=True)
    return df


def build_preview(staging: Path, output: Path, episodes_per_domain: int = 2,
                  frames_per_episode: int = 10) -> pd.DataFrame:
    if episodes_per_domain < 1 or frames_per_episode < 1:
        raise ValueError("episode and frame sample counts must be positive")
    frames = [sample_archive(staging, domain, name, episodes_per_domain, frames_per_episode)
              for domain, name in SOURCES]
    preview = pd.concat(frames, ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    preview.to_parquet(output, index=False)
    return validate_preview(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-dir", type=Path, default=Path("release/dataset_updates"))
    parser.add_argument("--output", type=Path, default=None,
                        help="Default: <staging-dir>/preview/mos_preview.parquet")
    parser.add_argument("--episodes-per-domain", type=int, default=2)
    parser.add_argument("--frames-per-episode", type=int, default=10)
    args = parser.parse_args()
    output = args.output or args.staging_dir / "preview/mos_preview.parquet"
    preview = build_preview(args.staging_dir, output, args.episodes_per_domain,
                            args.frames_per_episode)
    print(f"Wrote {output}: {len(preview)} real frames, {output.stat().st_size} bytes")
    print(preview.groupby("domain").size().to_string())
    print("Columns: metadata + state_00..state_16 + action_00..action_16")


if __name__ == "__main__":
    main()
