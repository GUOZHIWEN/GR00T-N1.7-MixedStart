#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detect repeatability-sensitive (periodic) waveform segments in ANY LeRobot
v2 dataset and optionally rewrite the language instructions to describe them.

Motivation (db11 wave experiment): the memory source's remaining cross-domain
asset is determinism (repeat‖Δ‖ 2–3x lower). The deployment scenario where
that matters is a REPEATABILITY-SENSITIVE WAVEFORM — a periodic motion that
must be reproduced identically every time. The experiment needs (a) such
segments found in real data (DROID) and (b) precise text describing the
waveform, so the memory encoder can in principle encode it.

Detection (pure numpy, torch-free):
  * sliding window over the eef xyz channels (observation.state[:, 0:3]);
  * per window per dim: Hann window + rFFT; a window is PERIODIC when one
    spectral peak carries >= --min-peak-ratio of the non-DC power, the peak
    frequency is in [--min-freq, --max-freq], the window contains
    >= --min-cycles cycles, and the estimated sine amplitude >= --min-amp;
  * overlapping accepted windows are merged into segments; the dominant axis
    is the dim with the largest estimated amplitude.

Rewrite (--rewrite-tasks): episodes whose periodic coverage (union of
segments / episode length) is >= --min-coverage get a NEW task entry whose
text appends an exact waveform description to the original task, e.g.

  "<original task>. Then the end effector oscillates along the x axis at
   about 1.50 hertz with amplitude about 2.0 centimeters for about 4 cycles"

and the covered frames' task_index in the episode parquet is repointed to it.
The original tasks.jsonl is backed up to tasks.jsonl.bak (once).

Decision rule (printed at the end): the DROID wave training arm is VIABLE
only when the dataset yields >= 20 segments across >= 10 episodes; otherwise
the report stands as a scarcity finding and no real-data wave arm is opened.

Works on the synthetic mixture datasets too (validation: it must recover the
injected db11 oscillation frequencies — a detector self-check).

Usage:
  uv run python scripts/annotate_wave_segments.py \
    --dataset-path data/droid_subset_train \
    --rewrite-tasks
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

AXIS_NAMES = ("x", "y", "z")

WAVE_TEXT_TEMPLATE = (
    "Then the end effector oscillates along the {axis} axis at about "
    "{freq:.2f} hertz with amplitude about {amp_cm:.1f} centimeters "
    "for about {cycles:.0f} cycles"
)


def _window_peak(sig: np.ndarray, fps: float) -> tuple[float, float, float]:
    """(peak_freq, peak_power_ratio, sine_amplitude) of a 1-D window.

    Hann window + rFFT. peak_ratio = strongest non-DC bin power / total
    non-DC power. Amplitude = 2|X_k| / sum(w) (Hann coherent-gain corrected).
    """
    sig = sig.astype(np.float64) - np.mean(sig)
    n = len(sig)
    if n < 8 or np.std(sig) < 1e-9:
        return 0.0, 0.0, 0.0
    w = np.hanning(n)
    spec = np.abs(np.fft.rfft(sig * w))
    spec[0] = 0.0  # exclude DC
    power = spec**2
    total = power.sum()
    if total <= 0:
        return 0.0, 0.0, 0.0
    k = int(np.argmax(power))
    # Parabolic peak interpolation (~1/10-bin frequency resolution; the raw
    # fps/n spacing is too coarse for sub-Hz periodic motions).
    k_eff = float(k)
    if 0 < k < len(spec) - 1:
        y0, y1, y2 = spec[k - 1], spec[k], spec[k + 1]
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-12:
            k_eff = float(k + np.clip(0.5 * (y0 - y2) / denom, -1.0, 1.0))
    freq = float(k_eff * fps / n)
    ratio = float(power[k] / total)
    amp = float(2.0 * spec[k] / w.sum())
    return freq, ratio, amp


def detect_wave_segments(
    xyz: np.ndarray,
    fps: float,
    window: int = 50,
    stride: int = 25,
    min_peak_ratio: float = 0.5,
    min_freq: float = 0.3,
    max_freq: float | None = None,
    min_cycles: float = 2.0,
    min_amp: float = 0.005,
) -> list[dict]:
    """Sliding-window periodicity detection on eef xyz (T, 3).

    Returns merged segments: {start, end, axis_index, axis, freq, amp,
    cycles, peak_ratio}.
    """
    T = len(xyz)
    if max_freq is None:
        max_freq = fps / 4.0
    accepted: list[dict] = []
    for s in range(0, max(T - window + 1, 0), stride):
        best = None
        for d in range(3):
            freq, ratio, amp = _window_peak(xyz[s : s + window, d], fps)
            cycles = freq * window / fps
            if (
                ratio >= min_peak_ratio
                and min_freq <= freq <= max_freq
                and cycles >= min_cycles
                and amp >= min_amp
                and (best is None or amp > best["amp"])
            ):
                best = {
                    "start": s,
                    "end": s + window,
                    "axis_index": d,
                    "freq": freq,
                    "amp": amp,
                    "peak_ratio": ratio,
                }
        if best is not None:
            accepted.append(best)

    # Merge overlapping windows that agree on the axis.
    merged: list[dict] = []
    for w in accepted:
        if merged and w["axis_index"] == merged[-1]["axis_index"] and w["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], w["end"])
        else:
            merged.append(dict(w))

    # Re-estimate parameters on the full merged extent.
    segments: list[dict] = []
    for m in merged:
        d = m["axis_index"]
        freq, ratio, amp = _window_peak(xyz[m["start"] : m["end"], d], fps)
        if freq <= 0:
            continue
        segments.append(
            {
                "start": int(m["start"]),
                "end": int(m["end"]),
                "axis_index": int(d),
                "axis": AXIS_NAMES[d],
                "freq": freq,
                "amp": amp,
                "cycles": float(freq * (m["end"] - m["start"]) / fps),
                "peak_ratio": ratio,
            }
        )
    return segments


def _load_fps(dataset: Path) -> float:
    with open(dataset / "meta" / "info.json") as f:
        return float(json.load(f)["fps"])


def _load_tasks(dataset: Path) -> list[dict]:
    tasks = []
    with open(dataset / "meta" / "tasks.jsonl") as f:
        for line in f:
            tasks.append(json.loads(line))
    return tasks


def annotate_dataset(args) -> int:
    dataset = Path(args.dataset_path)
    fps = _load_fps(dataset)
    data_dir = dataset / "data" / "chunk-000"
    files = sorted(data_dir.glob("episode_*.parquet"))
    if args.max_episodes:
        files = files[: args.max_episodes]
    if not files:
        raise RuntimeError(f"no episode parquets under {data_dir}")

    tasks = _load_tasks(dataset)
    task_texts = {int(t["task_index"]): str(t["task"]) for t in tasks}
    next_task_index = max(task_texts) + 1 if task_texts else 0

    out_jsonl = Path(args.out_jsonl) if args.out_jsonl else dataset / "meta" / "wave_segments_detected.jsonl"
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    n_segments_total = 0
    episodes_with_segments: set[int] = set()
    n_rewritten_episodes = 0
    n_rewritten_frames = 0
    new_tasks: list[dict] = []
    schedule_records: list[dict] = []

    with open(out_jsonl, "w") as out_f:
        for fp in files:
            ep_idx = int(fp.stem.split("_")[-1])
            df = pd.read_parquet(fp)
            state = np.stack(df["observation.state"].map(np.asarray).values).astype(np.float64)
            segments = detect_wave_segments(
                state[:, 0:3],
                fps,
                window=args.window,
                stride=args.stride,
                min_peak_ratio=args.min_peak_ratio,
                min_freq=args.min_freq,
                max_freq=args.max_freq,
                min_cycles=args.min_cycles,
                min_amp=args.min_amp,
            )
            if not segments:
                continue
            n_segments_total += len(segments)
            episodes_with_segments.add(ep_idx)
            covered = np.zeros(len(df), dtype=bool)
            for seg in segments:
                covered[seg["start"] : seg["end"]] = True
                out_f.write(json.dumps({"episode_index": ep_idx, **seg}) + "\n")
                # viz-compatible sidecar record (visualize_memory_start's wave
                # block expects segment_schedule.jsonl rows with a "wave"
                # sub-dict; real data has no approach phase -> osc_start=start).
                schedule_records.append(
                    {
                        "episode_index": ep_idx,
                        "start": seg["start"],
                        "end": seg["end"],
                        "kind": "oscillate",
                        "color": "na",
                        "wave": {
                            "axis": seg["axis"],
                            "axis_index": seg["axis_index"],
                            "freq": seg["freq"],
                            "amp": seg["amp"],
                            "cycles": seg["cycles"],
                            "osc_start": seg["start"],
                        },
                    }
                )
            coverage = float(covered.mean())

            if not (args.rewrite_tasks and coverage >= args.min_coverage):
                continue
            # One new task per (episode, dominant segment): the longest
            # segment defines the described waveform.
            dom = max(segments, key=lambda s: s["end"] - s["start"])
            base_text = task_texts.get(int(df["task_index"].iloc[0]), "").rstrip(".")
            # Idempotency: strip a previously appended wave sentence so a
            # re-run never produces "... oscillates ... . Then ... oscillates".
            base_text = base_text.split(". Then the end effector oscillates")[0]
            wave_text = WAVE_TEXT_TEMPLATE.format(
                axis=dom["axis"], freq=dom["freq"],
                amp_cm=dom["amp"] * 100.0, cycles=dom["cycles"],
            )
            new_text = f"{base_text}. {wave_text}"
            new_tasks.append({"task_index": next_task_index, "task": new_text})
            df.loc[covered, "task_index"] = next_task_index
            next_task_index += 1
            df.to_parquet(fp, index=False)
            n_rewritten_episodes += 1
            n_rewritten_frames += int(covered.sum())
            logger.info(
                f"  episode {ep_idx}: coverage={coverage:.2f} -> task #{next_task_index - 1} "
                f"\"{wave_text}\""
            )

    if args.rewrite_tasks and new_tasks:
        tasks_path = dataset / "meta" / "tasks.jsonl"
        backup = dataset / "meta" / "tasks.jsonl.bak"
        if not backup.exists():
            shutil.copy2(tasks_path, backup)
        with open(tasks_path, "a") as f:
            for t in new_tasks:
                f.write(json.dumps(t) + "\n")

    if args.emit_segment_schedule and schedule_records:
        sched_path = dataset / "meta" / "wave_segment_schedule.jsonl"
        with open(sched_path, "w") as f:
            for r in sorted(schedule_records, key=lambda r: (r["episode_index"], r["start"])):
                f.write(json.dumps(r) + "\n")
        logger.info(f"viz-compatible wave segment schedule -> {sched_path}")

    # Summary + decision rule.
    print("\n=== wave segment detection summary ===")
    print(f"dataset: {dataset}")
    print(f"episodes scanned: {len(files)}; with segments: {len(episodes_with_segments)}")
    print(f"segments detected: {n_segments_total}")
    if args.rewrite_tasks:
        print(f"episodes rewritten: {n_rewritten_episodes}; frames repointed: {n_rewritten_frames}")
        print(f"new tasks appended: {len(new_tasks)}")
    viable = n_segments_total >= 20 and len(episodes_with_segments) >= 10
    print(
        "decision: DROID/real wave arm "
        + ("VIABLE (>=20 segments across >=10 episodes)" if viable else
           "NOT viable — scarcity finding only, do not open a training arm")
    )
    print(f"report: {out_jsonl}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--out-jsonl", default=None)
    p.add_argument("--window", type=int, default=50)
    p.add_argument("--stride", type=int, default=25)
    p.add_argument("--min-peak-ratio", type=float, default=0.5)
    p.add_argument("--min-freq", type=float, default=0.3)
    p.add_argument("--max-freq", type=float, default=None, help="default: fps/4")
    p.add_argument("--min-cycles", type=float, default=2.0)
    p.add_argument("--min-amp", type=float, default=0.005, help="metres")
    p.add_argument("--min-coverage", type=float, default=0.3)
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument(
        "--rewrite-tasks",
        action="store_true",
        help="Repoint covered frames to new wave-describing tasks (backs up tasks.jsonl once).",
    )
    p.add_argument(
        "--emit-segment-schedule",
        action="store_true",
        help=(
            "Also write meta/wave_segment_schedule.jsonl in the viz-compatible "
            "format (visualize_memory_start --segment-meta-path), with "
            "wave.osc_start = segment start (real data has no approach phase)."
        ),
    )
    args = p.parse_args()
    return annotate_dataset(args)


if __name__ == "__main__":
    sys.exit(main())
