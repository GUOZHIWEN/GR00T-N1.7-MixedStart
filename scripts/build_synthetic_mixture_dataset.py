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
Synthetic multi-segment mixture dataset builder for the T1 mixed-source
experiments (paper "基于混合源的Δ Flow动作生成方法" §7).

Why synthetic: the 3-episode DROID demo (727 steps) is too small to verify
Propositions 1–3. This generator produces episodes whose action signal is
*smooth within a sub-task segment* (history source provably sufficient,
Prop. 1) and *discontinuous at segment switches* (semantic source strictly
better, Prop. 2: the previous chunk cannot know the new target; the language
instruction names it). Per-frame ``task_index`` lets the instruction change
inside an episode — the in-episode cold start the paper's theory is about.

Two splits are produced:
  * train: segments drawn from combo pool A (default: point/grasp ×
    {red, green, blue} blocks);
  * eval (--eval-output-dir): combos NEVER trained — the yellow block is
    held out entirely, and "grasp blue" is held out — so the eval set tests
    semantic generalisation of the memory encoder / router, not memorisation.

Output layout = GR00T LeRobot v2 (identical to demo_data/droid_sample):
  data/chunk-000/episode_XXXXXX.parquet  (observation.state/action, task_index)
  videos/chunk-000/<cam>/episode_XXXXXX.mp4  (180x320 @ fps)
  meta/{info.json, modality.json, tasks.jsonl, episodes.jsonl}
  meta/segment_schedule.jsonl  (per-episode segment bounds for stratified eval)
  meta/stats.json + meta/relative_stats.json  (unless --skip-stats)
  meta/data_fingerprint.json   (dataset identity for train-time assertions)

ACTION CONVENTION (GR00T RELATIVE contract): the stored action column holds
ABSOLUTE next-step poses — action[t] = pose[t+1] (xyz + rot6d + gripper +
joints). The training pipeline itself converts them to relative actions
against the current state (StateActionProcessor._convert_to_relative_action).
Do NOT store per-step state diffs here (2026-10-16 root cause, see §13.13 of
the post-training review doc).

db7 data-version features (《高质量训练与评估数据生成方案》§4, all OFF by
default so db6-era regeneration stays byte-identical):
  * --relative-pose-state  A1: append 5 state dims (dx,dy,dz,sin_dyaw,cos_dyaw)
    = target − eef per frame (state 17→22D; action stays 17D). modality.json
    gains a ``target_relative_pose`` state key. Treats the probe-verified
    readout failure (survey §13.30 verdict B).
  * --phase-in-instruction  gripper phase explicit: per-frame sub-instruction
    switches approach→hold at the 60% boundary ("grasp the blue block" →
    "hold the blue block"); segment_meta gains ``approach_end``. Treats the
    conditional-mean gripper averaging (root cause ①, §13.22/§13.24).
  * --hl-conditional  high-level task becomes *predictive*: the episode HL
    category (grasp_all / point_all / mixed) constrains segment sampling,
    with paraphrase families for text diversity (fixes the decorative-HL
    token-noise problem, §13.15.2 verdict 4).
  * --randomize-zones + --coord-in-text  A2, PAIRED ONLY (§13.27): per-
    episode block layout (train = inner region, eval = disjoint outer ring,
    §方案 P2-1) + target coordinates in the instruction text. Kills the
    fixed-ZONES color→pose lookup shortcut (generalization hole).
  * --train-pool-as-eval  E0 in-distribution eval split (TRAIN_POOL, new
    seed) written to --eval-output-dir only (train output skipped).
  * --data-version TAG  records the version into info.json and writes
    meta/data_fingerprint.json (episodes/frames/dims/content hash) for the
    launch-time assertion chain (mtime-free, review P1-2).

Usage:
  uv run python scripts/build_synthetic_mixture_dataset.py \
    --output-dir data/synth_mixture_train --num-episodes 40 \
    --eval-output-dir data/synth_mixture_eval --num-eval-episodes 10

  # db7 full version (all four data fixes at once, one-shot):
  uv run python scripts/build_synthetic_mixture_dataset.py \
    --output-dir data/synth_mixture_train_db7 --num-episodes 100 \
    --eval-output-dir data/synth_mixture_eval_db7 --num-eval-episodes 10 \
    --high-level-task-pool --hl-conditional \
    --relative-pose-state --phase-in-instruction \
    --randomize-zones --coord-in-text --data-version db7

  # local schema/logic test without parquet/video deps:
  uv run python scripts/build_synthetic_mixture_dataset.py --dry-run \
    --relative-pose-state --phase-in-instruction --hl-conditional \
    --randomize-zones --coord-in-text
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FPS_DEFAULT = 15
IMG_H, IMG_W = 180, 320
STATE_DIM = 17  # base layout: eef_9d(9) + gripper(1) + joint(7)
ACTION_DIM = STATE_DIM  # action is ALWAYS the 17D base (RELATIVE contract)
# Optional state-only target_relative_pose: (dx,dy,dz,sin_dyaw,cos_dyaw).
# dyaw is encoded as a (sin, cos) pair — the same reason rot6d encodes yaw as
# trig pairs: a raw shortest-arc angle is DISCONTINUOUS at ±π (verify hotfix
# #2: teleport heads with the target almost directly behind made the raw
# channel flip between +π/−π under sensor noise, G7 dyaw_max_step ≈ 2π).
REL_TARGET_DIM = 5

# Dataset-level static scene: 4 colored blocks at fixed workspace zones; zones
# 1 and 4 stay empty so vision must resolve color, not slot index.
COLORS = {
    "red": (220, 40, 40),
    "green": (40, 180, 60),
    "blue": (50, 90, 230),
    "yellow": (230, 210, 40),
}
ZONES = [(-0.40, -0.25), (0.00, -0.25), (0.40, -0.25), (-0.40, 0.25), (0.00, 0.25), (0.40, 0.25)]
BLOCK_ZONE = {"red": 0, "green": 2, "blue": 3, "yellow": 5}
DEFAULT_LAYOUT = {color: ZONES[zone] for color, zone in BLOCK_ZONE.items()}

# A2 layout sampling regions (metres). Train = inner rectangle; eval = outer
# ring disjoint from it (review P2-1: a bare min-distance < block spacing is
# not a coherent disjointness definition — use a spatial partition).
LAYOUT_INNER = (-0.35, 0.35, -0.22, 0.22)  # (xmin, xmax, ymin, ymax)
LAYOUT_OUTER = (-0.50, 0.50, -0.35, 0.35)  # outer bound; ring = OUTER \ INNER
LAYOUT_MIN_SEP = 0.15  # min pairwise block distance

# Combo pools: (kind, color, wave). wave=None for point/grasp; for
# "oscillate" (db11 --oscillation-segments) it carries the waveform params
# {"axis": "x"|"y", "freq": Hz, "amp": metres, "cycles": float} — the
# instruction text is rendered from these params, so text <-> waveform is a
# bijection (repeatability-sensitive: the same instruction ALWAYS yields the
# same waveform, Var(y|c) ~ 0 up to sensor noise).
# yellow never trained; grasp-blue never trained.
TRAIN_POOL = [(k, c, None) for k in ("point", "grasp") for c in ("red", "green", "blue")]
TRAIN_POOL.remove(("grasp", "blue", None))
EVAL_POOL = [("point", "yellow", None), ("grasp", "yellow", None), ("grasp", "blue", None)]

# db11 oscillation grids (held-out generalization split): train and eval use
# DISJOINT frequency and amplitude values, so the eval split measures wave
# parameter generalization rather than memorization. Frequencies are chosen
# so that with the default steps_per_segment=50 @ fps=15 the oscillation
# window (40 frames = 8/3 s) contains an exact integer number of cycles
# (cycles = freq * 40/15).
OSC_FREQS_TRAIN = (0.75, 1.5, 2.25)  # 2 / 4 / 6 cycles per 40-frame window
OSC_FREQS_EVAL = (1.125, 1.875)  # 3 / 5 cycles (held out)
OSC_AMPS_TRAIN = (0.02, 0.04)  # metres
OSC_AMPS_EVAL = (0.03, 0.05)  # held out
OSC_AXES = ("x", "y")


def oscillation_pool(
    freqs: tuple[float, ...],
    amps: tuple[float, ...],
    colors: tuple[str, ...],
    steps_per_segment: int,
    fps: int,
) -> list[tuple[str, str, dict]]:
    """(kind, color, wave) combos for the oscillation segment kind. ``cycles``
    is derived from the same steps/fps the generator uses, keeping the
    instruction text exactly faithful to the produced waveform."""
    n_osc = steps_per_segment - max(2, int(0.2 * steps_per_segment))
    pool = []
    for color in colors:
        for axis in OSC_AXES:
            for freq in freqs:
                for amp in amps:
                    pool.append((
                        "oscillate",
                        color,
                        {
                            "axis": axis,
                            "freq": float(freq),
                            "amp": float(amp),
                            "cycles": float(freq) * n_osc / fps,
                        },
                    ))
    return pool

INSTRUCTION = {
    "point": "point at the {color} block",
    "grasp": "grasp the {color} block",
}
# Phase-aware sub-instruction pair (approach, hold). The approach text carries
# the optional coordinate suffix; the hold text stays coordinate-free.
INSTRUCTION_PHASE = {
    "point": ("point at the {color} block", "stay at the {color} block"),
    "grasp": ("grasp the {color} block", "hold the {color} block"),
}
# db11 oscillation instruction: a complete, exact waveform specification
# (axis / center block / frequency / amplitude / cycle count) — the "strong
# text conditioning" carrier for the repeatability-sensitive waveform
# experiment. The wave parameters appear NOWHERE else (not in vision, not in
# state channels), so the memory encoder can only encode the waveform by
# reading this text.
OSC_INSTRUCTION = (
    "oscillate back and forth along the {axis} axis around the {color} block "
    "at {freq:g} hertz with amplitude {amp:.2f} meters for {cycles:g} cycles"
)

# High-level task pool (dual text port): episode-level task descriptions at a
# higher abstraction level than the per-segment sub-instructions. Sampled per
# episode when --high-level-task-pool is set. Diversity matters here: a fixed
# single sentence carries no information beyond the sub-instructions and acted
# as pure token noise in the _v2 evaluation (survey doc §13.15.2, verdict 4).
HIGH_LEVEL_TASK_POOL = [
    "clean up the blocks on the table",
    "sort the blocks by color",
    "tidy up the workspace",
    "arrange the blocks neatly",
    "organize the colored blocks",
    "gather the blocks into order",
    "clear the table of blocks",
    "put every block in its place",
    "straighten up the block area",
    "take care of the scattered blocks",
]

# HL-conditional scheduling (--hl-conditional): the episode high-level task
# becomes *predictive* of the segment distribution. Category → allowed segment
# kinds + a paraphrase family (text diversity within the category). Diagnostic:
# a linear probe HL-text → episode grasp-fraction should go from R²≈0 (the
# decorative pool) to R²≈1 (survey §13.31 route item / 方案 §4.3).
HL_SCHEDULE = {
    "grasp_all": {
        "texts": [
            "grasp every block on the table",
            "pick up all the blocks one by one",
            "grab each block in turn",
            "lift all the blocks off the table",
        ],
        "kinds": ("grasp",),
    },
    "point_all": {
        "texts": [
            "point at each block in turn",
            "indicate all the blocks one after another",
            "gesture toward every block",
            "take turns pointing at the blocks",
        ],
        "kinds": ("point",),
    },
    "mixed": {
        "texts": HIGH_LEVEL_TASK_POOL,
        "kinds": ("point", "grasp"),
    },
}


@dataclass(frozen=True)
class SegmentSpec:
    kind: str  # "point" | "grasp" | "oscillate"
    color: str  # target block color
    wave: dict | None = None  # oscillation params (db11); None otherwise

    @property
    def text(self) -> str:
        if self.kind == "oscillate":
            assert self.wave is not None, "oscillate segment requires wave params"
            return OSC_INSTRUCTION.format(color=self.color, **self.wave)
        return INSTRUCTION[self.kind].format(color=self.color)


def _segment_texts(
    spec: SegmentSpec,
    target_xy: tuple[float, float],
    phase_in_instruction: bool,
    coord_in_text: bool,
) -> tuple[str, str | None]:
    """Build (approach_text, hold_text|None) for one segment.

    Coordinate suffix goes only on the approach text; hold text stays
    coordinate-free (the target is already reached by then).
    """
    coord = f" at ({target_xy[0]:.2f}, {target_xy[1]:.2f})" if coord_in_text else ""
    if spec.kind == "oscillate":
        # Single text for the whole segment: the waveform has no
        # approach/hold phase split (no approach_end in the sidecar either).
        return spec.text + coord, None
    if phase_in_instruction:
        ap, ho = INSTRUCTION_PHASE[spec.kind]
        return ap.format(color=spec.color) + coord, ho.format(color=spec.color)
    return INSTRUCTION[spec.kind].format(color=spec.color) + coord, None


# ---------------------------------------------------------------------------
# Trajectory generation (pure numpy — unit-testable without parquet/video)
# ---------------------------------------------------------------------------


def _min_jerk(s: np.ndarray) -> float | np.ndarray:
    """Minimum-jerk blend 10s^3 - 15s^4 + 6s^5, s in [0, 1]."""
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def _shortest_arc(d: np.ndarray | float) -> np.ndarray | float:
    """Wrap an angle difference to (-pi, pi]."""
    return (d + np.pi) % (2 * np.pi) - np.pi


def sample_layout(
    rng: np.random.Generator, region: str = "inner"
) -> dict[str, tuple[float, float]]:
    """Sample a per-episode block layout (A2). ``region='inner'`` draws all
    blocks inside LAYOUT_INNER (train); ``region='outer'`` draws in the
    disjoint ring LAYOUT_OUTER \\ LAYOUT_INNER (E3 position-heldout eval).
    Pairwise block separation >= LAYOUT_MIN_SEP; up to 500 rejection rounds.
    """
    xmin_i, xmax_i, ymin_i, ymax_i = LAYOUT_INNER
    xmin_o, xmax_o, ymin_o, ymax_o = LAYOUT_OUTER

    def _draw() -> np.ndarray:
        for _ in range(200):
            x = rng.uniform(xmin_o, xmax_o)
            y = rng.uniform(ymin_o, ymax_o)
            inside_inner = xmin_i <= x <= xmax_i and ymin_i <= y <= ymax_i
            if (region == "inner") == inside_inner:
                return np.array([x, y])
        raise RuntimeError(f"layout sampling failed for region={region}")

    colors = list(COLORS)
    for _ in range(500):
        pts = [_draw() for _ in colors]
        ok = all(
            np.linalg.norm(pts[i] - pts[j]) >= LAYOUT_MIN_SEP
            for i in range(len(pts))
            for j in range(i + 1, len(pts))
        )
        if ok:
            return {c: (float(p[0]), float(p[1])) for c, p in zip(colors, pts)}
    raise RuntimeError("layout sampling failed to satisfy min separation")


def generate_trajectory(
    schedule: list[SegmentSpec],
    steps_per_segment: int,
    rng: np.random.Generator,
    teleport_prob: float = 0.5,
    hover_height: float = 0.15,
    noise_std: float = 0.002,
    layout: dict[str, tuple[float, float]] | None = None,
    phase_in_instruction: bool = False,
    coord_in_text: bool = False,
    relative_pose_state: bool = False,
    fps: int = FPS_DEFAULT,
) -> dict[str, np.ndarray | list[int]]:
    """Generate one episode.

    Returns states (T, 17|22), actions (T, 17), task_ids (T,),
    segment_starts. Within a segment the eef follows a minimum-jerk path to
    the target block (C1-smooth velocity: history source sufficient). At a
    switch, with probability ``teleport_prob`` the eef restarts from a random
    distant position (hard discontinuity: history source useless), otherwise
    it keeps continuous position but an abrupt velocity/direction change
    (soft discontinuity). Both are cold starts in the sense of Prop. 2.

    ``layout``: per-episode {color: (x, y)} block map (A2); default None =
    the fixed ZONES scene (db6 behaviour).
    ``phase_in_instruction``: per-frame task text switches approach→hold at
    the 60% segment boundary; segment_meta gains ``approach_end``.
    ``coord_in_text``: approach text carries " at (x, y)" of the target.
    ``relative_pose_state``: append (dx,dy,dz,sin_dyaw,cos_dyaw)=target−eef
    to the state (17→22D). Pure observation channels — the 17D action
    contract is untouched.
    """
    if layout is None:
        layout = DEFAULT_LAYOUT
    # Fixed smooth random map xyz -> 7 arm joints (dataset-level constant).
    joint_map = rng.normal(0.0, 1.0, size=(3, 7))
    joint_map /= np.linalg.norm(joint_map, axis=0, keepdims=True)

    n_seg = len(schedule)
    T = n_seg * steps_per_segment
    eef = np.zeros((T + 1, 3), dtype=np.float64)
    grip = np.zeros(T + 1, dtype=np.float64)
    task_ids = np.zeros(T, dtype=np.int64)
    segment_starts: list[int] = []
    # task registry: one task_index per unique instruction text, in order.
    task_texts: list[str] = []
    segment_meta: list[dict] = []
    # Per-frame segment targets (for A1 relative-pose channels).
    target_seq = np.zeros((T + 1, 3), dtype=np.float64)
    yaw_target_seq = np.zeros(T + 1, dtype=np.float64)

    pos = np.array([0.0, 0.0, hover_height])  # home
    yaw = np.zeros(T + 1, dtype=np.float64)  # eef heading angle about world z
    prev_yaw = 0.0
    t_global = 0
    for seg_i, spec in enumerate(schedule):
        zone_xy = np.array(layout[spec.color])
        target = np.array([zone_xy[0], zone_xy[1], 0.02 if spec.kind == "grasp" else hover_height])
        start = pos.copy()
        # A repeated target is only possible via the sample_schedule fallback
        # (kind-degenerate HL-conditional pool). Force a distant restart so the
        # repeated segment still opens with a cold-start discontinuity instead
        # of degenerating into 50 stationary frames at the previous target.
        # db6 schedules never repeat a target -> behaviour unchanged there.
        same_target = seg_i > 0 and np.allclose(target, pos)
        if seg_i > 0 and (same_target or rng.random() < teleport_prob):
            start = np.array(
                [rng.uniform(-0.45, 0.45), rng.uniform(-0.3, 0.3), hover_height]
            )
        text_ap, text_ho = _segment_texts(spec, layout[spec.color], phase_in_instruction, coord_in_text)
        if text_ap not in task_texts:
            task_texts.append(text_ap)
        tid_ap = task_texts.index(text_ap)
        tid_ho = None
        if text_ho is not None:
            if text_ho not in task_texts:
                task_texts.append(text_ho)
            tid_ho = task_texts.index(text_ho)

        steps = steps_per_segment
        # 60% minimum-jerk approach + 40% dwell at the target: the dwell makes
        # mid-segment windows stationary (history gap ≈ 0), sharpening the
        # contrast against the switch transient that Prop. 2 is about.
        # db11 oscillation segments: 20% min-jerk approach to the block center
        # + 80% phase-locked sinusoid (ramp-in envelope, initial phase 0), so
        # the instruction text DETERMINES the waveform (Var(y|c) ~ 0).
        approach = max(2, int((0.2 if spec.kind == "oscillate" else 0.6) * steps))
        dwell = steps - approach
        s = np.linspace(0.0, 1.0, approach, endpoint=False)
        blend = _min_jerk(s)
        if spec.kind == "oscillate":
            w = spec.wave
            axis_idx = 0 if w["axis"] == "x" else 1
            tau = np.arange(dwell, dtype=np.float64) / fps
            env = np.clip(tau / 0.4, 0.0, 1.0)  # 0.4 s amplitude ramp-in
            dwell_path = np.tile(target[None, :], (dwell, 1))
            dwell_path[:, axis_idx] += (
                w["amp"] * np.sin(2.0 * np.pi * w["freq"] * tau) * env
            )
        else:
            dwell_path = np.tile(target[None, :], (dwell, 1))
        seg_path = np.concatenate(
            [
                start[None, :] + (target - start)[None, :] * blend[:, None],
                dwell_path,
            ],
            axis=0,
        )
        seg_path += rng.normal(0.0, noise_std, size=seg_path.shape)  # mild sensor noise
        eef[t_global : t_global + steps] = seg_path
        target_seq[t_global : t_global + steps] = target[None, :]
        # Directional yaw: the eef heading turns to face the approach
        # direction (observable from state/video, hence learnable). Within a
        # segment it follows the same min-jerk blend (history source
        # sufficient, Prop. 1); at a switch the new approach direction jumps
        # (semantic source necessary, Prop. 2) — so the rot6d channels carry
        # the same learnable structure as xyz instead of being dead constant
        # dims (2026-10-16 requirement: no zero-signal action channels).
        direction = target[:2] - start[:2]
        if np.linalg.norm(direction) > 1e-6:
            yaw_target = float(np.arctan2(direction[1], direction[0]))
        else:
            yaw_target = prev_yaw
        dyaw = _shortest_arc(yaw_target - prev_yaw)  # shortest arc
        yaw[t_global : t_global + steps] = (
            np.concatenate([prev_yaw + dyaw * blend, np.full(dwell, yaw_target)])
            + rng.normal(0.0, noise_std, size=steps)
        )
        yaw_target_seq[t_global : t_global + steps] = yaw_target
        prev_yaw = yaw_target
        if seg_i > 0:
            # Boundary carry (verify hotfix #2): the first frame of a new
            # segment INHERITS the previous segment's final gripper value, so
            # the last "hold" frame's next-step action no longer commands an
            # unexplained release (1-in-20 dwell frames mislabelled, var
            # 0.05*0.95 = 0.0475 — exactly the G3 failure, present in db6 data
            # too). The release then happens on the FIRST transition of the
            # new segment, under the NEW instruction ("grasp/point at the new
            # block" -> opening first is semantically correct), keeping the
            # conditional map consistent: hold -> always 1.
            grip[t_global] = grip[t_global - 1]
            if spec.kind == "grasp":
                grip[t_global + 1 : t_global + approach] = _min_jerk(s)[1:]
                grip[t_global + approach : t_global + steps] = 1.0
            else:
                grip[t_global + 1 : t_global + steps] = 0.0
        elif spec.kind == "grasp":
            # gripper closes smoothly over the approach, holds during dwell
            grip[t_global : t_global + approach] = _min_jerk(s)
            grip[t_global + approach : t_global + steps] = 1.0
        else:
            grip[t_global : t_global + steps] = 0.0
        if tid_ho is None:
            task_ids[t_global : t_global + steps] = tid_ap
        else:
            task_ids[t_global : t_global + approach] = tid_ap
            task_ids[t_global + approach : t_global + steps] = tid_ho
        segment_starts.append(t_global)
        meta_rec = {
            "segment_index": seg_i,
            "start": int(t_global),
            "end": int(t_global + steps),
            "kind": spec.kind,
            "color": spec.color,
            "task_index": int(tid_ap),
            "teleported": bool(seg_i > 0 and not np.allclose(start, pos)),
        }
        if phase_in_instruction and text_ho is not None:
            meta_rec["approach_end"] = int(t_global + approach)
        if spec.kind == "oscillate":
            # Waveform ground truth for the spectral-fidelity evaluation
            # (visualize_memory_start wave block / annotate_wave_segments).
            meta_rec["wave"] = {
                "axis": spec.wave["axis"],
                "axis_index": int(axis_idx),
                "freq": float(spec.wave["freq"]),
                "amp": float(spec.wave["amp"]),
                "cycles": float(spec.wave["freq"]) * dwell / fps,
                "osc_start": int(t_global + approach),
            }
        if layout is not DEFAULT_LAYOUT or coord_in_text:
            meta_rec["target_xy"] = [float(zone_xy[0]), float(zone_xy[1])]
            # Noise-free segment start: the yaw_target bearing is computed
            # from THIS start, so the verifier must recompute from it too —
            # the observed first frame carries sensor noise (0.002), and for
            # short approaches (|direction| ~ 0.04) recomputing the bearing
            # from it amplifies to ~0.05 rad (unit-smoke seeds 4/6/19/20).
            meta_rec["start_xy"] = [float(start[0]), float(start[1])]
        segment_meta.append(meta_rec)
        pos = target
        t_global += steps
    eef[T] = pos  # terminal state
    grip[T] = grip[T - 1]
    yaw[T] = prev_yaw
    target_seq[T] = target_seq[T - 1]
    yaw_target_seq[T] = prev_yaw

    # Assemble 17D state: eef_9d = xyz + rot6d, gripper, joints =
    # tanh(xyz @ joint_map). rot6d encodes Rz(yaw) — yaw tracks the segment
    # approach direction (see the segment loop above) — with row1 =
    # [cosθ, −sinθ, 0], row2 = [sinθ, cosθ, 0]; the heading is also rendered
    # into the videos so the signal is visually observable.
    # NOTE (2026-10-16): an earlier version added a cumsum random-walk drift
    # (5e-4/step) to rot6d. Its per-step diff is iid white noise — an
    # unpredictable target that minmax normalization amplifies to the full
    # [-1, 1] range, which diluted the training loss and dominated the
    # fair-eval MSE (6 of 17 dims, ~78% of total). Removed.
    cos_t, sin_t = np.cos(yaw), np.sin(yaw)
    zeros = np.zeros(T + 1, dtype=np.float64)
    rot6d = np.stack([cos_t, -sin_t, zeros, sin_t, cos_t, zeros], axis=1)
    joints = np.tanh(eef @ joint_map)
    state_blocks = [eef, rot6d, grip[:, None], joints]
    if relative_pose_state:
        # A1 target_relative_pose: (dx,dy,dz) = target − eef, plus the heading
        # error as a (sin, cos) pair — a raw shortest-arc dyaw is
        # discontinuous at ±π (targets almost directly behind after a
        # teleport made the channel flip under sensor noise, verify hotfix
        # #2); trig pairs are wrap-free, matching the rot6d convention.
        # State-only channels — never enter the action column (RELATIVE
        # contract untouched).
        rel_xyz = target_seq - eef
        rel_yaw = _shortest_arc(yaw_target_seq - yaw)
        state_blocks.append(
            np.concatenate(
                [rel_xyz, np.sin(rel_yaw)[:, None], np.cos(rel_yaw)[:, None]], axis=1
            )
        )
    states = np.concatenate(state_blocks, axis=1)  # (T+1, 17|22)

    # GR00T RELATIVE-action contract (StateActionProcessor._convert_to_relative_action:
    # "Convert absolute action to relative action using reference state"):
    # the stored action column must hold ABSOLUTE next-step poses; the pipeline
    # itself computes relative actions against the current state. An earlier
    # version stored per-step state diffs, which corrupted training on this
    # embodiment (2026-10-16 three-arm non-convergence root cause):
    #   * delta-rot6d noise (~5e-4) was re-interpreted as absolute rotation
    #     matrix rows; row-normalizing near-zero vectors yields random unit
    #     vectors -> rot6d targets became uniform sphere noise (std exactly
    #     1/sqrt(3), see relative_stats.json), unlearnable;
    #   * delta-xyz (near origin) was read as absolute positions, bending the
    #     relative targets toward R_t^T(-p_t) ("vector to origin") instead of
    #     the intended motion.
    # Gripper stays as the next absolute value (its ActionConfig is ABSOLUTE,
    # so no conversion is applied either way).
    actions = np.zeros((T, ACTION_DIM), dtype=np.float64)
    actions[:, :9] = states[1:, :9]  # absolute next EEF pose (xyz + rot6d)
    actions[:, 9] = states[1:, 9]  # absolute next gripper
    actions[:, 10:17] = states[1:, 10:17]  # absolute next joint positions

    return {
        "states": states[:T].astype(np.float32),  # (T, 17|22)
        "actions": actions.astype(np.float32),  # (T, 17)
        "task_ids": task_ids,
        "task_texts": task_texts,
        "segment_starts": segment_starts,
        "segment_meta": segment_meta,
        "layout": layout,
    }


def sample_schedule(
    rng: np.random.Generator,
    pool: list[tuple[str, str]],
    n_segments: int,
    allowed_kinds: tuple[str, ...] | None = None,
) -> list[SegmentSpec]:
    """Random segment sequence; consecutive segments never repeat a combo (a
    repeat would not be a cold start). ``allowed_kinds`` restricts the segment
    types (HL-conditional scheduling)."""
    specs: list[SegmentSpec] = []
    prev = None
    for _ in range(n_segments):
        choices = [
            c for c in pool
            if c != prev and (allowed_kinds is None or c[0] in allowed_kinds)
        ]
        if not choices:
            # Kind-degenerate pool under HL-conditional scheduling (e.g.
            # EVAL_POOL has a single point combo, so a point_all episode runs
            # out of non-repeating choices at segment 2): keep the
            # allowed_kinds constraint (HL semantics is the hard requirement)
            # and relax only the no-consecutive-repeat preference.
            # generate_trajectory forces a teleport restart on a repeated
            # target, so the repeated segment still has a cold start.
            choices = [
                c for c in pool
                if allowed_kinds is None or c[0] in allowed_kinds
            ]
        kind, color, wave = choices[rng.integers(len(choices))]
        specs.append(SegmentSpec(kind=kind, color=color, wave=wave))
        prev = (kind, color, wave)
    return specs


# ---------------------------------------------------------------------------
# Programmatic rendering (180x320 RGB, two camera views)
# ---------------------------------------------------------------------------


def _world_to_px(xy: np.ndarray) -> tuple[int, int]:
    """Orthographic top-down map: workspace [-0.55,0.55]x[-0.4,0.4] -> pixels."""
    u = int(np.clip((xy[0] + 0.55) / 1.1, 0, 1) * (IMG_W - 1))
    v = int(np.clip((xy[1] + 0.4) / 0.8, 0, 1) * (IMG_H - 1))
    return u, v


def render_frame(
    eef_xy: np.ndarray,
    target_color: str,
    camera: str,
    rng: np.random.Generator,
    yaw: float = 0.0,
    layout: dict[str, tuple[float, float]] | None = None,
) -> np.ndarray:
    if layout is None:
        layout = DEFAULT_LAYOUT
    img = np.full((IMG_H, IMG_W, 3), 96, dtype=np.uint8)  # tabletop grey
    # zone grid lines (fixed reference grid, drawn in both layout modes)
    for zx, zy in ZONES:
        u, v = _world_to_px(np.array([zx, zy]))
        img[max(0, v - 1) : v + 2, max(0, u - 1) : u + 2] = (70, 70, 70)
    # blocks (per-episode layout)
    for color, (bx, by) in layout.items():
        u, v = _world_to_px(np.array([bx, by]))
        img[v - 6 : v + 6, u - 6 : u + 6] = COLORS[color]
        if color == target_color:  # highlight current target with a white ring
            img[v - 9 : v - 6, u - 9 : u + 9] = 255
            img[v + 6 : v + 9, u - 9 : u + 9] = 255
            img[v - 9 : v + 9, u - 9 : u - 6] = 255
            img[v - 9 : v + 9, u + 6 : u + 9] = 255
    # eef marker + heading line (world x -> u, world y -> v, no flip): makes
    # the rot6d yaw signal visually observable in both camera views
    u, v = _world_to_px(eef_xy)
    img[v - 3 : v + 3, u - 3 : u + 3] = (255, 255, 255)
    for s in np.linspace(0.0, 1.0, 10):
        uu = int(np.clip(u + round(float(np.cos(yaw)) * 12 * s), 0, IMG_W - 1))
        vv = int(np.clip(v + round(float(np.sin(yaw)) * 12 * s), 0, IMG_H - 1))
        img[vv, uu] = (255, 255, 160)
    if camera == "wrist_left":
        # egocentric: crop 90x160 around eef, upscale 2x, add jitter
        jx, jy = rng.integers(-4, 5), rng.integers(-4, 5)
        cx, cy = np.clip(u + jx, 80, IMG_W - 80), np.clip(v + jy, 45, IMG_H - 45)
        crop = img[cy - 45 : cy + 45, cx - 80 : cx + 80]
        img = np.repeat(np.repeat(crop, 2, axis=0), 2, axis=1)
    img += rng.integers(-3, 4, size=img.shape).astype(np.uint8)  # sensor noise
    return img


# ---------------------------------------------------------------------------
# Serialization (LeRobot v2, mirrors scripts/download_droid_sample.py)
# ---------------------------------------------------------------------------


def _episode_meta_record(
    ep_idx: int, ep: dict, length: int, high_level_task: str | None
) -> dict:
    """Build one episodes.jsonl record; ``high_level_task`` is only attached
    when provided (dual text port / sub-instruction training). ``layout`` is
    attached whenever the episode carries one (A2 randomization audit)."""
    meta = {"episode_index": ep_idx, "tasks": ep["task_texts"], "length": length}
    if high_level_task:
        meta["high_level_task"] = high_level_task
    if ep.get("layout") is not None:
        meta["layout"] = {c: [float(x), float(y)] for c, (x, y) in ep["layout"].items()}
    return meta


def _write_data_fingerprint(
    meta_dir: Path,
    data_version: str | None,
    episodes_meta: list[dict],
    state_dim: int,
    action_dim: int,
    data_dir: Path,
) -> None:
    """Dataset identity for the launch-time assertion chain (review P1-2:
    mtime is unreliable across copies/rebuilds — fingerprint the CONTENT).
    Hash covers the meta files + per-parquet byte sizes (cheap, deterministic
    for identical generator inputs)."""
    payload_files = ["info.json", "modality.json", "tasks.jsonl", "episodes.jsonl", "segment_schedule.jsonl"]
    h = hashlib.sha256()
    for name in payload_files:
        p = meta_dir / name
        h.update(name.encode())
        h.update(p.read_bytes())
    for pq in sorted(data_dir.glob("episode_*.parquet")):
        h.update(pq.name.encode())
        h.update(str(pq.stat().st_size).encode())
    fp = {
        "data_version": data_version,
        "num_episodes": len(episodes_meta),
        "total_frames": int(sum(e["length"] for e in episodes_meta)),
        "state_dim": int(state_dim),
        "action_dim": int(action_dim),
        "sha256": "sha256:" + h.hexdigest(),
    }
    with open(meta_dir / "data_fingerprint.json", "w") as f:
        json.dump(fp, f, indent=2)


def write_dataset(
    output_dir: Path,
    episodes: list[dict],
    fps: int,
    seed: int,
    skip_stats: bool = False,
    high_level_task: str | None = None,
    high_level_task_pool: bool = False,
    high_level_texts: list[str | None] | None = None,
    data_version: str | None = None,
) -> None:
    import cv2
    import pandas as pd

    data_dir = output_dir / "data" / "chunk-000"
    meta_dir = output_dir / "meta"
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed + 999)
    # Independent stream for per-episode high-level task sampling.
    hl_rng = np.random.default_rng(seed + 12345)

    state_dim = int(episodes[0]["states"].shape[1])
    action_dim = int(episodes[0]["actions"].shape[1])

    all_task_texts: list[str] = []
    episodes_meta = []
    schedule_records = []

    for ep_idx, ep in enumerate(episodes):
        T = len(ep["task_ids"])
        # remap local task ids to dataset-global ids
        global_ids = []
        for text in ep["task_texts"]:
            if text not in all_task_texts:
                all_task_texts.append(text)
            global_ids.append(all_task_texts.index(text))
        remap = {i: g for i, g in enumerate(global_ids)}
        task_ids = np.array([remap[t] for t in ep["task_ids"]], dtype=np.int64)

        df = pd.DataFrame(
            {
                "episode_index": np.full(T, ep_idx, dtype=np.int64),
                "index": np.arange(T, dtype=np.int64),
                "timestamp": np.arange(T, dtype=np.float64) / fps,
                "task_index": task_ids,
                "observation.state": list(ep["states"]),
                "action": list(ep["actions"]),
            }
        )
        df.to_parquet(data_dir / f"episode_{ep_idx:06d}.parquet", index=False)

        # videos
        for cam in ("exterior_1_left", "wrist_left"):
            cam_dir = output_dir / "videos" / "chunk-000" / f"observation.images.{cam}"
            cam_dir.mkdir(parents=True, exist_ok=True)
            vw = cv2.VideoWriter(
                str(cam_dir / f"episode_{ep_idx:06d}.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (IMG_W, IMG_H),
            )
            for t in range(T):
                target_color = ep["segment_meta"][
                    np.searchsorted([m["start"] for m in ep["segment_meta"]], t, side="right") - 1
                ]["color"]
                st = ep["states"][t]
                # recover yaw from state rot6d: row1.x = st[3] = cosθ,
                # row2.x = st[6] = sinθ
                frame = render_frame(
                    st[:2], target_color, cam, rng,
                    yaw=float(np.arctan2(st[6], st[3])), layout=ep.get("layout"),
                )
                vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            vw.release()

        if high_level_texts is not None:
            ep_high_level = high_level_texts[ep_idx]
        else:
            ep_high_level = (
                str(hl_rng.choice(HIGH_LEVEL_TASK_POOL))
                if high_level_task_pool
                else high_level_task
            )
        episodes_meta.append(_episode_meta_record(ep_idx, ep, T, ep_high_level))
        for m in ep["segment_meta"]:
            schedule_records.append(
                {"episode_index": ep_idx, "task_text": ep["task_texts"][m["task_index"]], **m}
            )

    # meta files
    def _write_jsonl(path: Path, records: list[dict]) -> None:
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    _write_jsonl(meta_dir / "episodes.jsonl", episodes_meta)
    _write_jsonl(
        meta_dir / "tasks.jsonl",
        [{"task_index": i, "task": t} for i, t in enumerate(all_task_texts)],
    )
    _write_jsonl(meta_dir / "segment_schedule.jsonl", schedule_records)

    info = {
        "codebase_version": "v2.1",
        "robot_type": "droid",
        "total_episodes": len(episodes),
        "total_frames": sum(e["length"] for e in episodes_meta),
        "fps": fps,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "chunks_size": 1000,
        "splits": {"train": f"0:{len(episodes)}"},
        "features": {
            "observation.images.exterior_1_left": {"dtype": "video", "shape": [IMG_H, IMG_W, 3]},
            "observation.images.wrist_left": {"dtype": "video", "shape": [IMG_H, IMG_W, 3]},
            "observation.state": {"dtype": "float32", "shape": [state_dim]},
            "action": {"dtype": "float32", "shape": [action_dim]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }
    if data_version is not None:
        info["data_version"] = data_version
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
        "annotation": {"language.language_instruction": {"original_key": "task_index"}},
    }
    if state_dim == STATE_DIM + REL_TARGET_DIM:
        modality["state"]["target_relative_pose"] = {
            "start": 17,
            "end": 17 + REL_TARGET_DIM,
        }
    with open(meta_dir / "modality.json", "w") as f:
        json.dump(modality, f, indent=2)

    _write_data_fingerprint(meta_dir, data_version, episodes_meta, state_dim, action_dim, data_dir)

    if not skip_stats:
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.data.stats import generate_rel_stats, generate_stats

        if state_dim == STATE_DIM + REL_TARGET_DIM:
            # A1: the training pipeline resolves state keys from the hardcoded
            # MODALITY_CONFIGS (embodiment_configs.py L33-36), NOT from the
            # dataset modality.json (review P0-1). Patch the process-local
            # config so any stats/validity consumer in this process sees the
            # 4th state key. Training/eval processes must pass
            # --state-extra-modality-keys target_relative_pose for the same
            # effect (gr00t/configs/data/state_extra_keys.py).
            from gr00t.configs.data.state_extra_keys import apply_state_extra_keys

            apply_state_extra_keys(
                EmbodimentTag.OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT.value,
                ["target_relative_pose"],
            )

        generate_stats(output_dir)
        generate_rel_stats(output_dir, EmbodimentTag.OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT)

    logger.info(
        "Wrote %d episodes (%d frames) to %s", len(episodes), info["total_frames"], output_dir
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", default="data/synth_mixture_train")
    parser.add_argument("--num-episodes", type=int, default=40)
    parser.add_argument("--eval-output-dir", default="data/synth_mixture_eval")
    parser.add_argument("--num-eval-episodes", type=int, default=10)
    parser.add_argument("--segments-per-episode", type=int, default=4)
    parser.add_argument("--steps-per-segment", type=int, default=50)
    parser.add_argument("--teleport-prob", type=float, default=0.5)
    parser.add_argument("--fps", type=int, default=FPS_DEFAULT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-stats", action="store_true")
    parser.add_argument(
        "--high-level-task",
        default=None,
        help=(
            "Optional episode-level task text written as ``high_level_task`` into "
            "episodes.jsonl (dual text port / sub-instruction training). The "
            "per-frame segment texts then serve as step-level sub-instructions."
        ),
    )
    parser.add_argument(
        "--high-level-task-pool",
        action="store_true",
        help=(
            "Sample the episode-level high_level_task per episode from "
            "HIGH_LEVEL_TASK_POOL instead of using one fixed sentence. "
            "Mutually exclusive with --high-level-task."
        ),
    )
    # ---- db7 feature flags (default OFF = db6-identical output) ----
    parser.add_argument(
        "--relative-pose-state",
        action="store_true",
        help="A1: append target_relative_pose (dx,dy,dz,sin_dyaw,cos_dyaw) to the state (17->22D).",
    )
    parser.add_argument(
        "--phase-in-instruction",
        action="store_true",
        help="Per-frame sub-instruction switches approach->hold at the 60%% segment boundary.",
    )
    parser.add_argument(
        "--hl-conditional",
        action="store_true",
        help="High-level task category constrains segment sampling (grasp_all/point_all/mixed).",
    )
    parser.add_argument(
        "--randomize-zones",
        action="store_true",
        help="A2: per-episode block layout (train=inner region, eval=disjoint outer ring).",
    )
    parser.add_argument(
        "--coord-in-text",
        action="store_true",
        help="A2 (PAIRED with --randomize-zones): target coordinates in the instruction text.",
    )
    parser.add_argument(
        "--train-pool-as-eval",
        action="store_true",
        help=(
            "E0 mode: build ONLY the eval split (from TRAIN_POOL with a new seed) "
            "into --eval-output-dir; the train output is skipped entirely."
        ),
    )
    parser.add_argument(
        "--data-version",
        default=None,
        help="Version tag recorded into info.json + meta/data_fingerprint.json.",
    )
    parser.add_argument(
        "--oscillation-segments",
        action="store_true",
        help=(
            "db11: add repeatability-sensitive OSCILLATION segments to the pools "
            "(axis x/y, DISJOINT train/eval freq and amp grids); the instruction "
            "text carries the exact waveform spec (axis/freq/amp/cycles)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate trajectories and print diagnostics only (no files).",
    )
    args = parser.parse_args()

    if args.high_level_task_pool and args.high_level_task is not None:
        parser.error("--high-level-task-pool is mutually exclusive with --high-level-task")
    if args.hl_conditional and args.high_level_task is not None:
        parser.error("--hl-conditional is mutually exclusive with --high-level-task")
    if args.coord_in_text and not args.randomize_zones:
        # Pairing rule (§13.27): coordinates in text are redundant under fixed
        # zones and randomization without coordinates explodes Var(y|c).
        parser.error("--coord-in-text requires --randomize-zones (A2 must be paired)")
    if args.train_pool_as_eval and Path(args.output_dir) == Path(args.eval_output_dir):
        parser.error("--train-pool-as-eval: --output-dir must differ from --eval-output-dir")

    rng = np.random.default_rng(args.seed)

    # db11: extend pools with oscillation combos (disjoint train/eval grids)
    # and register the HL category so --hl-conditional can schedule them.
    train_pool, eval_pool = list(TRAIN_POOL), list(EVAL_POOL)
    if args.oscillation_segments:
        train_pool += oscillation_pool(
            OSC_FREQS_TRAIN, OSC_AMPS_TRAIN, ("red", "green", "blue"),
            args.steps_per_segment, args.fps,
        )
        eval_pool += oscillation_pool(
            OSC_FREQS_EVAL, OSC_AMPS_EVAL, tuple(COLORS),
            args.steps_per_segment, args.fps,
        )
        HL_SCHEDULE["oscillate_all"] = {
            "texts": [
                "demonstrate the oscillation patterns",
                "perform the periodic wave motions",
                "show the repetitive oscillation routines",
            ],
            "kinds": ("oscillate",),
        }
        HL_SCHEDULE["mixed"]["kinds"] = ("point", "grasp", "oscillate")

    def _build(pool, n, layout_region):
        episodes = []
        hl_texts: list[str | None] = []
        for _ in range(n):
            hl_text = None
            allowed = None
            if args.hl_conditional:
                cat = list(HL_SCHEDULE)[rng.integers(len(HL_SCHEDULE))]
                hl_text = str(rng.choice(HL_SCHEDULE[cat]["texts"]))
                allowed = HL_SCHEDULE[cat]["kinds"]
            schedule = sample_schedule(rng, pool, args.segments_per_episode, allowed_kinds=allowed)
            layout = (
                sample_layout(rng, region=layout_region) if args.randomize_zones else None
            )
            episodes.append(
                generate_trajectory(
                    schedule,
                    args.steps_per_segment,
                    rng,
                    teleport_prob=args.teleport_prob,
                    layout=layout,
                    phase_in_instruction=args.phase_in_instruction,
                    coord_in_text=args.coord_in_text,
                    relative_pose_state=args.relative_pose_state,
                    fps=args.fps,
                )
            )
            hl_texts.append(hl_text)
        return episodes, hl_texts

    if args.train_pool_as_eval:
        train_eps, train_hl = [], []
        eval_eps, eval_hl = _build(
            train_pool, args.num_eval_episodes,
            layout_region="inner",  # E0 = in-distribution (same region as train)
        )
    else:
        train_eps, train_hl = _build(
            train_pool, args.num_episodes, layout_region="inner"
        )
        eval_eps, eval_hl = _build(
            eval_pool, args.num_eval_episodes,
            layout_region="outer" if args.randomize_zones else "inner",
        )

    if args.dry_run:
        # Design-contract sanity checks (the decisive Prop. 1/2 evidence is the
        # eval-time stratified MSE + β̂ routing, not a dataset scalar). Note:
        # actions follow the GR00T RELATIVE contract (absolute next poses), so
        # smoothness is checked on per-step STATE displacement, not actions:
        #   1. within-segment smoothness: per-step displacement is small and
        #      low-acceleration (history source sufficient, Prop. 1);
        #   2. switch discontinuity: consecutive segments' approach directions
        #      are near-random (~90 deg), so the previous chunk cannot predict
        #      the new target (semantic source necessary, Prop. 2).
        for name, eps, hls in (("train", train_eps, train_hl), ("eval", eval_eps, eval_hl)):
            if not eps:
                continue
            step_norms, accels, switch_angles, n_teleport = [], [], [], 0
            for e in eps:
                v = np.diff(e["states"][:, :3], axis=0)  # per-step displacement
                step_norms.append(np.linalg.norm(v, axis=-1).mean())
                accels.append(np.linalg.norm(np.diff(v, axis=0), axis=-1).mean())
                metas = e["segment_meta"]
                for i in range(1, len(metas)):
                    s0, s1 = metas[i - 1], metas[i]
                    tgt0 = s0.get("target_xy") or ZONES[BLOCK_ZONE[s0["color"]]]
                    tgt1 = s1.get("target_xy") or ZONES[BLOCK_ZONE[s1["color"]]]
                    z0 = np.array(tgt0)
                    z1 = np.array(tgt1)
                    d0 = z0 - np.array([0.0, 0.0]) if i == 1 else z0 - np.array(
                        metas[i - 2].get("target_xy") or ZONES[BLOCK_ZONE[metas[i - 2]["color"]]]
                    ) if i > 1 else z0
                    d1 = z1 - z0
                    if s1["teleported"]:
                        n_teleport += 1
                    if np.linalg.norm(d0) > 1e-6 and np.linalg.norm(d1) > 1e-6:
                        cos = float(np.dot(d0, d1) / (np.linalg.norm(d0) * np.linalg.norm(d1)))
                        switch_angles.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
            line = (
                f"[dry-run] {name}: {len(eps)} eps x {len(eps[0]['task_ids'])} steps | "
                f"mean |Δp|={np.mean(step_norms):.4f} mean |Δv|={np.mean(accels):.4f} | "
                f"switch dir-change={np.mean(switch_angles):.0f} deg (n={len(switch_angles)}, "
                f"teleports={n_teleport}) | "
                f"rot6d ch-std={np.mean([e['states'][:, 3:9].std(axis=0).mean() for e in eps]):.3f} "
                f"(>0: yaw signal present)"
            )
            if args.relative_pose_state:
                rel = np.concatenate([e["states"][:, 17:22] for e in eps], axis=0)
                dyaw = np.arctan2(rel[:, 3], rel[:, 4])  # sin, cos -> angle
                line += (
                    f" | rel_pose range dx[{rel[:,0].min():.2f},{rel[:,0].max():.2f}] "
                    f"dyaw[{dyaw.min():.2f},{dyaw.max():.2f}] "
                    f"sincos-norm-dev[{np.abs(np.linalg.norm(rel[:,3:5],axis=1)-1).max():.1e}]"
                )
            if args.phase_in_instruction:
                line += f" | phase texts={sum(len(e['task_texts']) for e in eps)} (2x segments)"
            if args.randomize_zones:
                stds = [
                    np.std([e["layout"][c][ax_i] for e in eps])
                    for c in COLORS for ax_i in (0, 1)
                ]
                line += f" | layout per-axis std={np.mean(stds):.3f} (>0.03: randomized)"
            if args.hl_conditional:
                fracs = {}
                for hl, e in zip(hls, eps):
                    g = np.mean([m["kind"] == "grasp" for m in e["segment_meta"]])
                    fracs.setdefault((hl or "")[:12], []).append(g)
                between = np.std([np.mean(v) for v in fracs.values()]) if len(fracs) > 1 else 0.0
                line += f" | HL grasp-frac spread between cats={between:.2f} (>0.3: predictive)"
            if args.oscillation_segments:
                # Spectral self-check: the injected oscillation must show a
                # clean rFFT peak AT the text-stated frequency (bijection).
                osc_recs = [
                    m for e in eps for m in e["segment_meta"] if m["kind"] == "oscillate"
                ]
                freq_errs = []
                for e in eps:
                    for m in e["segment_meta"]:
                        if m["kind"] != "oscillate":
                            continue
                        w = m["wave"]
                        sig = e["states"][w["osc_start"] : m["end"], w["axis_index"]].astype(np.float64)
                        sig = sig - sig.mean()
                        mag = np.abs(np.fft.rfft(sig))
                        freqs = np.fft.rfftfreq(len(sig), d=1.0 / args.fps)
                        peak = freqs[int(np.argmax(mag[1:])) + 1]
                        freq_errs.append(abs(peak - w["freq"]) / w["freq"])
                med = float(np.median(freq_errs)) if freq_errs else float("nan")
                line += (
                    f" | osc segs={len(osc_recs)} freq rel-err med={med:.3f} "
                    "(<0.15: text<->waveform bijection OK)"
                )
            print(line)
        return

    def _write(eps, hl_texts, out_dir, hl_pool_flag, write_seed):
        if not eps:
            return
        write_dataset(
            Path(out_dir), eps, args.fps, write_seed, args.skip_stats,
            high_level_task=args.high_level_task,
            high_level_task_pool=hl_pool_flag,
            high_level_texts=hl_texts if args.hl_conditional else None,
            data_version=args.data_version,
        )

    if not args.train_pool_as_eval:
        _write(train_eps, train_hl, args.output_dir, args.high_level_task_pool and not args.hl_conditional, args.seed)
    if args.num_eval_episodes > 0:
        # eval split keeps the original seed+1 write-stream (render noise /
        # high-level sampling) so db6-era regeneration stays byte-identical.
        _write(eval_eps, eval_hl, args.eval_output_dir, args.high_level_task_pool and not args.hl_conditional, args.seed + 1)


if __name__ == "__main__":
    main()
