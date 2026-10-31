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

"""GT-action visual verifier + db7 quality gates for the synthetic mixture
dataset (torch-free).

Numeric self-checks (all gates must PASS before any training run):
  * G1 abs-convention error   max|A[:-1,:9] - S[1:,:9]|   (~0: the GR00T
    RELATIVE contract stores ABSOLUTE next-step poses);
  * G1 diff guard             max|A[:-1,:9] - diff(S)[:,:9]| (>> 0: catches
    accidentally stored per-step state diffs);
  * G2 rot6d row-norm deviation (~0: valid rotation rows);
  * G2 dead dims (std < 1e-3) — none expected (structural rot2/rot5 exempt;
    gripper additionally exempt in all-point episodes where identically-0 is
    the correct conditional value — verify hotfix #2);
  * G3 conditional determinism: gripper dwell variance < 1e-3 and approach
    monotonicity corr > 0.9 per grasp segment (phase-explicit data);
  * G5 schema: state 22D / action 17D and modality.json carries
    target_relative_pose (only when the dataset is built 22D);
  * G6 relative-pose consistency: state[:,17:20] ≈ target − eef and
    state[:,20:22] ≈ (sin, cos) of the heading error (recomputed from
    segment_meta target_xy + kind height + noise-free start_xy bearing),
    max err < 0.02;
  * G7 rel-yaw sanity (wrap-free sincos pair, hotfix #2): row norm ≈ 1
    (dev < 1e-3) and within-segment 2-vector step jump < 0.6;
  * G8 phase-text consistency: frames before/after approach_end carry
    approach/hold sub-instructions respectively;
  * G9 coordinate text ↔ layout: parsed " at (x, y)" matches segment_meta
    target_xy within 0.02;
  * G10 randomization active + spatial partition: per-color target std
    > 0.03, and layouts sit entirely in the inner (train) or outer (eval)
    region (--expect-layout);
  * G11 data fingerprint: meta/data_fingerprint.json exists and agrees with
    info.json (episodes / frames / dims).

Plots per episode (visual dataset validation):
  1. all 17 GT action dims with segment boundaries + approach/dwell bands;
  2. xy top-down trajectory over the (per-episode) block layout;
  3. target_relative_pose channels vs recomputed expectation (22D only);
  4. gripper vs segment progress by kind (phase check, visual);
  5. dataset-level layout scatter (randomization at a glance);
  6. decoded video frame grid (white ring on the right block, heading line).

Older (db6, 17D/21D, no sidecar fields) datasets: db7-only gates report
"n/a" and never fail.

Usage:
  uv run python scripts/verify_synthetic_gt_actions.py \
    --dataset-path data/synth_mixture_train_db7 --episode-ids 0 1 2 3 \
    --expect-layout train --expect-state-dim 22 \
    --out-dir runs/verify_db7
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_synthetic_mixture_dataset import (  # noqa: E402
    BLOCK_ZONE,
    COLORS,
    DEFAULT_LAYOUT,
    LAYOUT_INNER,
    LAYOUT_OUTER,
    ZONES,
)

def _plt():
    """Lazy matplotlib import so the numeric checks (and unit tests) run
    without plotting deps."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


DIM_LABELS = (
    ["x", "y", "z"]
    + [f"rot{i}" for i in range(6)]
    + ["grip"]
    + [f"j{i}" for i in range(7)]
)
REL_LABELS = ["dx", "dy", "dz", "sin_dyaw", "cos_dyaw"]
ACTION_DIM = 17
STATE_DIM_DB7 = 22
# rot2 (dim 5) and rot5 (dim 8) are the z-components of the two rot6d rows;
# for the planar Rz(yaw) rotation used by the builder they are identically 0
# BY DESIGN — structural constants, not dead signal channels.
STRUCTURAL_ZERO_DIMS = (5, 8)
# In an all-point episode the gripper is identically 0 as the CORRECT
# conditional value (point tasks never close it) — that is a conditionally
# constant channel, not a dead one (verify hotfix #2: point_all episodes
# failed the dead-dim gate).
GRIP_DIM = 9
HOLD_PREFIXES = ("hold ", "stay ")
_COORD_RE = re.compile(r"at \((-?\d+\.\d+), (-?\d+\.\d+)\)")


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _load_parquet(path: Path):
    """pandas+pyarrow on the server venv; duckdb fallback for minimal envs."""
    import pandas as pd

    try:
        return pd.read_parquet(path)
    except ImportError:
        import duckdb

        return duckdb.sql(f"SELECT * FROM read_parquet('{path.as_posix()}')").df()


def _load_segment_meta(path: Path | None) -> dict[int, list[dict]]:
    if path is None or not path.exists():
        return {}
    by_ep: dict[int, list[dict]] = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            by_ep.setdefault(r["episode_index"], []).append(r)
    for v in by_ep.values():
        v.sort(key=lambda r: r["start"])
    return by_ep


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _load_tasks(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[int(r["task_index"])] = r["task"]
    return out


def _stack_column(df, name: str) -> np.ndarray:
    return np.stack([np.asarray(x, dtype=np.float32) for x in df[name]])


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def check_episode(
    states: np.ndarray, actions: np.ndarray, metas: list[dict] | None = None
) -> dict:
    """G1/G2 numeric design-contract checks for one episode (action side).

    ``metas`` (optional) enables the conditional-constant exemption: in an
    all-point episode the gripper is identically 0 as the correct conditional
    value, not a dead dim (verify hotfix #2)."""
    abs_err = float(np.abs(actions[:-1, :9] - states[1:, :9]).max())
    diff_guard = float(np.abs(actions[:-1, :9] - np.diff(states[:, :9], axis=0)).max())
    row1, row2 = actions[:, 3:6], actions[:, 6:9]
    row_dev = float(
        max(
            np.abs(np.linalg.norm(row1, axis=-1) - 1.0).max(),
            np.abs(np.linalg.norm(row2, axis=-1) - 1.0).max(),
        )
    )
    # Gripper ≡ 0 is the CORRECT conditional value in any grasp-free episode
    # (all-point db6/db7, oscillate_all db11), not a dead dim.
    all_nongrasp = bool(metas) and all(m.get("kind") != "grasp" for m in metas)
    dead_all = [d for d in range(ACTION_DIM) if actions[:, d].std() < 1e-3]
    structural = [DIM_LABELS[d] for d in dead_all if d in STRUCTURAL_ZERO_DIMS]
    conditional = [
        DIM_LABELS[d] for d in dead_all if d == GRIP_DIM and all_nongrasp
    ]
    dead = [
        DIM_LABELS[d]
        for d in dead_all
        if d not in STRUCTURAL_ZERO_DIMS and not (d == GRIP_DIM and all_nongrasp)
    ]
    passed = abs_err < 1e-4 and diff_guard > 1e-2 and row_dev < 1e-3 and not dead
    return {
        "abs_convention_max_err": abs_err,
        "diff_guard_max": diff_guard,
        "rot6d_row_norm_max_dev": row_dev,
        "structural_zero_dims": structural,
        "conditional_zero_dims": conditional,
        "dead_dims": dead,
        "pass": bool(passed),
    }


def check_rel_pose(states: np.ndarray, metas: list[dict]) -> dict:
    """G6/G7 target_relative_pose checks (22D datasets with target_xy).

    Channels: dx,dy,dz = target − eef; (sin_dyaw, cos_dyaw) = heading error as
    a wrap-free trig pair (hotfix #2). G6 = recompute-and-compare; G7 = trig
    row-norm ≈ 1 and within-segment step continuity of the 2-vector."""
    if states.shape[1] != STATE_DIM_DB7 or not any("target_xy" in m for m in metas):
        return {"status": "n/a", "pass": True}
    errs, sc_errs, norm_dev, sc_jump = [], [], 0.0, 0.0
    for m in metas:
        if "target_xy" not in m:
            continue
        seg = states[m["start"] : m["end"]].astype(np.float64)
        tz = 0.02 if m["kind"] == "grasp" else 0.15
        tgt = np.array([m["target_xy"][0], m["target_xy"][1], tz])
        errs.append(np.abs(seg[:, 17:20] - (tgt[None, :] - seg[:, :3])).max())
        # expected heading error: the segment's approach-direction yaw target
        # (constant per segment) minus the actual per-frame yaw. The bearing
        # is recomputed from the noise-free ``start_xy`` stored by the
        # builder — recomputing from the observed first frame amplifies the
        # 0.002 sensor noise to ~0.05 rad on short approaches (unit-smoke
        # seeds 4/6/19/20). Fallback for older metas: observed first frame.
        yaw = np.arctan2(seg[:, 6], seg[:, 3])
        origin = (
            np.array(m["start_xy"], dtype=np.float64)
            if "start_xy" in m
            else seg[0, :2]
        )
        direction = tgt[:2] - origin
        if np.linalg.norm(direction) <= 1e-6:
            continue  # degenerate (target at start): builder keeps prev_yaw
        yaw_tgt = float(np.arctan2(direction[1], direction[0]))
        dyaw = (yaw_tgt - yaw + np.pi) % (2 * np.pi) - np.pi
        expect_sc = np.stack([np.sin(dyaw), np.cos(dyaw)], axis=1)
        sc_errs.append(np.abs(seg[:, 20:22] - expect_sc).max())
        norm_dev = max(
            norm_dev,
            float(np.abs(np.linalg.norm(seg[:, 20:22], axis=1) - 1.0).max()),
        )
        if len(seg) > 1:
            sc_jump = max(
                sc_jump, float(np.linalg.norm(np.diff(seg[:, 20:22], axis=0), axis=1).max())
            )
    err_max = float(max(errs)) if errs else 0.0
    sc_err_max = float(max(sc_errs)) if sc_errs else 0.0
    passed = err_max < 0.02 and sc_err_max < 0.02 and norm_dev < 1e-3 and sc_jump < 0.6
    return {
        "status": "checked",
        "rel_pose_max_err": err_max,
        "rel_sincos_max_err": sc_err_max,
        "sincos_norm_max_dev": norm_dev,
        "sincos_max_step": sc_jump,
        "pass": bool(passed),
    }


def check_phase_text(task_ids: np.ndarray, metas: list[dict], task_map: dict[int, str]) -> dict:
    """G8 phase-text consistency (approach vs hold around approach_end)."""
    if not metas or not any("approach_end" in m for m in metas) or not task_map:
        return {"status": "n/a", "pass": True}
    if not any(t.startswith(HOLD_PREFIXES) for t in task_map.values()):
        return {"status": "n/a", "pass": True}
    mismatches = 0
    total = 0
    for m in metas:
        if "approach_end" not in m:
            continue
        for t in range(m["start"], m["end"]):
            text = task_map.get(int(task_ids[t]), "")
            is_hold = text.startswith(HOLD_PREFIXES)
            expect_hold = t >= m["approach_end"]
            mismatches += int(is_hold != expect_hold)
            total += 1
    rate = mismatches / max(total, 1)
    return {
        "status": "checked",
        "phase_mismatch_rate": rate,
        "pass": bool(total > 0 and rate == 0.0),
    }


def check_coord_text(metas: list[dict]) -> dict:
    """G9 coordinate text ↔ layout consistency."""
    checked = 0
    worst = 0.0
    for m in metas:
        text = m.get("task_text") or ""
        found = _COORD_RE.search(text)
        if not found or "target_xy" not in m:
            continue
        checked += 1
        px, py = float(found.group(1)), float(found.group(2))
        worst = max(
            worst,
            abs(px - m["target_xy"][0]),
            abs(py - m["target_xy"][1]),
        )
    if checked == 0:
        return {"status": "n/a", "pass": True}
    return {
        "status": "checked",
        "coord_text_max_err": float(worst),
        "pass": bool(worst < 0.02),
    }


def check_gripper_phase(actions: np.ndarray, metas: list[dict]) -> dict:
    """G3 conditional determinism for the gripper channel."""
    if not metas:
        return {"status": "n/a", "pass": True}
    dwell_var_max, corr_min, point_max = 0.0, 1.0, 0.0
    n_grasp = 0
    for m in metas:
        grip = actions[m["start"] : m["end"], 9].astype(np.float64)
        approach_end = m.get("approach_end", m["start"] + int(0.6 * len(grip)))
        ap = grip[: approach_end - m["start"]]
        dw = grip[approach_end - m["start"] :]
        if m["kind"] == "grasp":
            n_grasp += 1
            if len(dw):
                dwell_var_max = max(dwell_var_max, float(dw.var()))
            if len(ap) > 2:
                corr = np.corrcoef(ap, np.arange(len(ap)))[0, 1]
                corr_min = min(corr_min, float(corr))
        else:
            point_max = max(point_max, float(np.abs(grip).max()))
    if n_grasp == 0:
        return {"status": "n/a", "pass": True}
    passed = dwell_var_max < 1e-3 and corr_min > 0.9 and point_max < 1e-3
    return {
        "status": "checked",
        "gripper_dwell_var_max": dwell_var_max,
        "gripper_approach_corr_min": corr_min,
        "gripper_point_absmax": point_max,
        "pass": bool(passed),
    }


def check_dataset_level(
    dataset: Path,
    seg_meta: dict[int, list[dict]],
    expect_layout: str,
    expect_state_dim: int | None,
    actual_state_dim: int,
) -> dict:
    """G5/G10/G11 dataset-level gates (schema, layout partition, fingerprint)."""
    out: dict = {}
    modality = _load_json(dataset / "meta" / "modality.json")
    info = _load_json(dataset / "meta" / "info.json")
    # G5 schema
    if expect_state_dim is not None:
        keys = sorted(modality.get("state", {}))
        need_rel = expect_state_dim == STATE_DIM_DB7
        ok = (
            actual_state_dim == expect_state_dim
            and ("target_relative_pose" in keys) == need_rel
            and info.get("features", {}).get("observation.state", {}).get("shape", [None])[0]
            == expect_state_dim
        )
        out["schema"] = {
            "status": "checked",
            "state_keys": keys,
            "state_dim": actual_state_dim,
            "pass": bool(ok),
        }
    else:
        out["schema"] = {"status": "n/a", "pass": True}
    # G10 layout randomization + partition
    all_targets = [
        (m["color"], m["target_xy"])
        for metas in seg_meta.values()
        for m in metas
        if "target_xy" in m
    ]
    if all_targets:
        std = float(
            np.mean([np.std([xy[i] for _, xy in all_targets]) for i in (0, 1)])
        )
        xi, xa, yi, ya = LAYOUT_INNER
        xo, xao, yo, yao = LAYOUT_OUTER
        inside = [
            (xi - 1e-6 <= x <= xa + 1e-6) and (yi - 1e-6 <= y <= ya + 1e-6)
            for _, (x, y) in all_targets
        ]
        all_inner, all_outer = all(inside), not any(inside)
        region_ok = (
            (expect_layout == "train" and all_inner)
            or (expect_layout == "eval" and all_outer)
            or expect_layout == "none"
        )
        out["layout"] = {
            "status": "checked",
            "target_std": std,
            "all_inner": all_inner,
            "all_outer": all_outer,
            "pass": bool(std > 0.03 and region_ok),
        }
    else:
        out["layout"] = {"status": "n/a", "pass": True}
    # G11 fingerprint
    fp = _load_json(dataset / "meta" / "data_fingerprint.json")
    if fp:
        ok = (
            fp.get("num_episodes") == info.get("total_episodes")
            and fp.get("total_frames") == info.get("total_frames")
            and fp.get("state_dim") == actual_state_dim
        )
        out["fingerprint"] = {"status": "checked", **fp, "pass": bool(ok)}
    else:
        # fingerprint is required only when the dataset declares a version
        need = "data_version" in info
        out["fingerprint"] = {
            "status": "missing" if need else "n/a",
            "pass": not need,
        }
    out["pass"] = all(v["pass"] for v in out.values())
    return out


# ---------------------------------------------------------------------------
# Episode metadata loader (per-episode block layout, A2)
# ---------------------------------------------------------------------------


def _load_layouts(path: Path) -> dict[int, dict[str, tuple[float, float]]]:
    """episodes.jsonl -> {episode_index: {color: (x, y)}} when layouts are
    recorded (randomized-layout datasets)."""
    out: dict[int, dict[str, tuple[float, float]]] = {}
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if "layout" in r:
                out[int(r["episode_index"])] = {
                    c: (float(xy[0]), float(xy[1])) for c, xy in r["layout"].items()
                }
    return out


# ---------------------------------------------------------------------------
# Plotting (visual dataset validation — lazy matplotlib)
# ---------------------------------------------------------------------------


def plot_action_dims(
    actions: np.ndarray,
    metas: list[dict],
    ep_idx: int,
    out_path: Path,
) -> None:
    """All 17 GT action dims; segment boundaries (red=teleport, grey=soft);
    approach/dwell bands when approach_end is recorded."""
    plt = _plt()
    nrows, ncols = 6, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 20), sharex=True)
    for d in range(nrows * ncols):
        ax = axes[d // ncols][d % ncols]
        if d >= ACTION_DIM:
            ax.axis("off")
            continue
        ax.plot(actions[:, d], lw=0.9, color="tab:blue")
        for m in metas:
            color = "red" if m.get("teleported") else "grey"
            ax.axvline(m["start"], color=color, ls="--", lw=0.8, alpha=0.8)
            if "approach_end" in m:
                ax.axvspan(m["start"], m["approach_end"], color="tab:orange", alpha=0.08)
                ax.axvspan(m["approach_end"], m["end"], color="tab:green", alpha=0.08)
                ax.axvline(m["approach_end"], color="tab:green", ls=":", lw=0.7)
        ax.set_title(f"dim {d}: {DIM_LABELS[d]}", fontsize=9)
        ax.tick_params(labelsize=7)
    axes[0][0].set_ylabel("abs next pose")
    fig.suptitle(
        f"ep {ep_idx}: GT action dims | red=teleport boundary, grey=soft switch | "
        "orange band=approach, green band=dwell",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_xy_topdown(
    states: np.ndarray,
    metas: list[dict],
    layout: dict[str, tuple[float, float]] | None,
    ep_idx: int,
    out_path: Path,
) -> None:
    """Top-down xy trajectory over the block layout; teleports marked."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 7))
    blocks = layout or {
        m["color"]: tuple(m["target_xy"]) for m in metas if "target_xy" in m
    } or DEFAULT_LAYOUT
    for color, (bx, by) in blocks.items():
        ax.scatter([bx], [by], s=320, marker="s", c=[color], edgecolors="k", zorder=3)
        ax.annotate(color, (bx, by), textcoords="offset points", xytext=(8, 8), fontsize=8)
    xy = states[:, :2]
    sc = ax.scatter(
        xy[:, 0], xy[:, 1], c=np.arange(len(xy)), cmap="viridis", s=6, zorder=2
    )
    for m in metas:
        if m.get("teleported"):
            ax.plot(
                [xy[m["start"] - 1, 0], xy[m["start"], 0]],
                [xy[m["start"] - 1, 1], xy[m["start"], 1]],
                color="red", lw=1.4, ls="--", zorder=4,
            )
    fig.colorbar(sc, ax=ax, label="frame index")
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.45, 0.45)
    ax.set_aspect("equal")
    ax.set_title(f"ep {ep_idx}: eef xy trajectory over block layout (red dash = teleport)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_rel_pose(
    states: np.ndarray,
    metas: list[dict],
    ep_idx: int,
    out_path: Path,
) -> str:
    """target_relative_pose channels (solid) vs recomputed expectation (dashed).

    5 channels (22D datasets): dx, dy, dz, sin_dyaw, cos_dyaw. The trig-pair
    expectation mirrors check_rel_pose / the builder: the segment's
    approach-direction yaw target is computed once from the segment START
    (constant per segment), not per-frame."""
    if states.shape[1] != STATE_DIM_DB7:
        return "n/a"
    plt = _plt()
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
    for k in range(5):
        ax = axes[k]
        ax.plot(states[:, 17 + k], lw=1.0, color="tab:blue", label="state channel")
        # recompute expectation per segment
        expect = np.full(len(states), np.nan)
        for m in metas:
            if "target_xy" not in m:
                continue
            seg = states[m["start"] : m["end"]].astype(np.float64)
            tz = 0.02 if m["kind"] == "grasp" else 0.15
            tgt = np.array([m["target_xy"][0], m["target_xy"][1], tz])
            if k < 3:
                expect[m["start"] : m["end"]] = (tgt[None, :] - seg[:, :3])[:, k]
            else:
                origin = (
                    np.array(m["start_xy"], dtype=np.float64)
                    if "start_xy" in m
                    else seg[0, :2]
                )
                direction = tgt[:2] - origin
                if np.linalg.norm(direction) <= 1e-6:
                    continue  # degenerate: builder keeps prev_yaw, no clean expectation
                yaw = np.arctan2(seg[:, 6], seg[:, 3])
                yaw_tgt = float(np.arctan2(direction[1], direction[0]))
                dyaw = (yaw_tgt - yaw + np.pi) % (2 * np.pi) - np.pi
                expect[m["start"] : m["end"]] = (
                    np.sin(dyaw) if k == 3 else np.cos(dyaw)
                )
        ax.plot(expect, lw=1.0, ls="--", color="tab:red", label="recomputed")
        for m in metas:
            ax.axvline(m["start"], color="grey", ls="--", lw=0.7)
        ax.set_ylabel(REL_LABELS[k])
        ax.legend(fontsize=7, loc="upper right")
    axes[0].set_title(f"ep {ep_idx}: target_relative_pose vs recomputed expectation")
    axes[-1].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return "plotted"


def plot_gripper_phase(
    actions: np.ndarray,
    metas: list[dict],
    ep_idx: int,
    out_path: Path,
) -> None:
    """Gripper vs normalized in-segment progress, colored by kind: the visual
    conditional-determinism check (grasp closes over approach, holds in dwell;
    point keeps it open)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 6))
    for m in metas:
        grip = actions[m["start"] : m["end"], 9]
        prog = np.linspace(0.0, 1.0, len(grip))
        color = "tab:red" if m["kind"] == "grasp" else "tab:blue"
        ax.plot(prog, grip, color=color, lw=0.8, alpha=0.6)
    ax.axvline(0.6, color="grey", ls=":", lw=1.0)
    ax.text(0.61, 0.05, "approach|dwell", fontsize=8, color="grey")
    ax.set_xlabel("normalized segment progress")
    ax.set_ylabel("gripper action")
    ax.set_title(f"ep {ep_idx}: gripper phase profile (red=grasp, blue=point)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_layout_scatter(
    all_targets: list[tuple[str, tuple[float, float]]],
    expect_layout: str,
    out_path: Path,
) -> str:
    """Dataset-level scatter of segment targets; inner/outer partition boxes."""
    if not all_targets:
        return "n/a"
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 7))
    for color, (x, y) in all_targets:
        ax.scatter([x], [y], c=[color], s=60, edgecolors="k", alpha=0.8)
    xi, xa, yi, ya = LAYOUT_INNER
    xo, xao, yo, yao = LAYOUT_OUTER
    for (x0, x1, y0, y1), ls, lab in (
        ((xi, xa, yi, ya), "-", "inner (train)"),
        ((xo, xao, yo, yao), "--", "outer bound (eval ring)"),
    ):
        ax.plot(
            [x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
            color="k", ls=ls, lw=1.2, label=lab,
        )
    ax.legend(fontsize=8)
    ax.set_xlim(xo - 0.06, xao + 0.06)
    ax.set_ylim(yo - 0.06, yao + 0.06)
    ax.set_aspect("equal")
    ax.set_title(f"segment targets over episodes (expect_layout={expect_layout})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return "plotted"


def plot_frame_grid(
    dataset: Path,
    ep_idx: int,
    metas: list[dict],
    out_path: Path,
) -> str:
    """Decoded video frames at segment start / mid-approach for both cameras:
    white ring must sit on the segment's target block, heading line visible."""
    try:
        import cv2  # noqa: F401
    except ImportError:
        return "skipped (cv2 unavailable)"
    cams = ("exterior_1_left", "wrist_left")
    frames: dict[str, list[np.ndarray]] = {c: [] for c in cams}
    picks: list[int] = []
    for m in metas[:4]:  # up to 4 segments: start+2 and mid-approach
        mid = m["start"] + max(1, (m.get("approach_end", m["end"]) - m["start"]) // 2)
        picks.extend([min(m["start"] + 2, m["end"] - 1), min(mid, m["end"] - 1)])
    import cv2

    for cam in cams:
        video = dataset / "videos" / "chunk-000" / f"observation.images.{cam}" / f"episode_{ep_idx:06d}.mp4"
        if not video.exists():
            return f"skipped (missing {video.name})"
        cap = cv2.VideoCapture(str(video))
        got = {}
        t = 0
        wanted = set(picks)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if t in wanted:
                got[t] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t += 1
        cap.release()
        for p in picks:
            if p in got:
                frames[cam].append(got[p])
    if not frames[cams[0]]:
        return "skipped (no frames decoded)"
    plt = _plt()
    nrows = len(picks)
    fig, axes = plt.subplots(nrows, len(cams), figsize=(10, 2.2 * nrows))
    for r in range(nrows):
        for c, cam in enumerate(cams):
            ax = axes[r][c] if nrows > 1 else axes[c]
            idx = r * len(cams) + 0  # frames[cam] is ordered by picks
            if r < len(frames[cam]):
                ax.imshow(frames[cam][r])
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(cam, fontsize=9)
            ax.set_ylabel(f"t={picks[r]}", fontsize=8)
    fig.suptitle(f"ep {ep_idx}: decoded frames (segment start / mid-approach)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return "plotted"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _gate_line(tag: str, result: dict) -> str:
    status = result.get("status", "checked")
    if status == "n/a":
        return f"    {tag:<18} n/a"
    verdict = "PASS" if result["pass"] else "FAIL"
    detail = {
        k: (f"{v:.3e}" if isinstance(v, float) else v)
        for k, v in result.items()
        if k not in ("status", "pass")
    }
    return f"    {tag:<18} {verdict} | {detail}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--episode-ids", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument(
        "--segment-meta-path",
        default=None,
        help="Defaults to <dataset>/meta/segment_schedule.jsonl.",
    )
    parser.add_argument("--out-dir", default="runs/verify_synth_data")
    parser.add_argument(
        "--expect-layout",
        choices=["train", "eval", "none"],
        default="none",
        help="G10 partition expectation: train=all targets inner, eval=all outer.",
    )
    parser.add_argument(
        "--expect-state-dim",
        type=int,
        default=None,
        help="G5 schema expectation (17 or 21). Omit to skip the schema gate.",
    )
    args = parser.parse_args()

    dataset = Path(args.dataset_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seg_path = (
        Path(args.segment_meta_path)
        if args.segment_meta_path
        else dataset / "meta" / "segment_schedule.jsonl"
    )
    seg_meta = _load_segment_meta(seg_path)
    task_map = _load_tasks(dataset / "meta" / "tasks.jsonl")
    layouts = _load_layouts(dataset / "meta" / "episodes.jsonl")

    summary: dict = {"dataset": str(dataset), "episodes": {}, "plots": {}}
    all_pass = True
    actual_state_dim = 0

    for ep_idx in args.episode_ids:
        pq = dataset / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
        if not pq.exists():
            print(f"[ep {ep_idx}] MISSING parquet {pq}")
            summary["episodes"][ep_idx] = {"pass": False, "error": "missing parquet"}
            all_pass = False
            continue
        df = _load_parquet(pq)
        states = _stack_column(df, "observation.state").astype(np.float64)
        actions = _stack_column(df, "action").astype(np.float64)
        task_ids = np.asarray(df["task_index"], dtype=np.int64)
        actual_state_dim = states.shape[1]
        metas = seg_meta.get(ep_idx, [])

        gates = {
            "G1/G2_contract": check_episode(states, actions, metas),
            "G6/G7_rel_pose": check_rel_pose(states, metas),
            "G8_phase_text": check_phase_text(task_ids, metas, task_map),
            "G9_coord_text": check_coord_text(metas),
            "G3_gripper_phase": check_gripper_phase(actions, metas),
        }
        ep_pass = all(g["pass"] for g in gates.values())
        all_pass &= ep_pass
        print(f"[ep {ep_idx}] {'PASS' if ep_pass else 'FAIL'} (state_dim={actual_state_dim})")
        for tag, res in gates.items():
            print(_gate_line(tag, res))
        summary["episodes"][ep_idx] = {"pass": bool(ep_pass), **gates}

        # per-episode plots
        plot_action_dims(actions, metas, ep_idx, out_dir / f"ep{ep_idx:03d}_action_dims.png")
        plot_xy_topdown(
            states, metas, layouts.get(ep_idx), ep_idx, out_dir / f"ep{ep_idx:03d}_xy_topdown.png"
        )
        rel_status = plot_rel_pose(states, metas, ep_idx, out_dir / f"ep{ep_idx:03d}_rel_pose.png")
        plot_gripper_phase(actions, metas, ep_idx, out_dir / f"ep{ep_idx:03d}_gripper_phase.png")
        frame_status = plot_frame_grid(
            dataset, ep_idx, metas, out_dir / f"ep{ep_idx:03d}_frames.png"
        )
        summary["plots"][ep_idx] = {"rel_pose": rel_status, "frames": frame_status}

    # dataset-level gates + layout scatter
    ds = check_dataset_level(
        dataset, seg_meta, args.expect_layout, args.expect_state_dim, actual_state_dim
    )
    all_pass &= ds["pass"]
    print("[dataset]")
    for tag, res in ds.items():
        if tag == "pass":
            continue
        print(_gate_line(f"G_{tag}", res))
    all_targets = [
        (m["color"], tuple(m["target_xy"]))
        for metas in seg_meta.values()
        for m in metas
        if "target_xy" in m
    ]
    scatter_status = plot_layout_scatter(
        all_targets, args.expect_layout, out_dir / "dataset_layout_scatter.png"
    )
    summary["dataset_level"] = ds
    summary["plots"]["dataset_layout_scatter"] = scatter_status
    summary["pass"] = bool(all_pass)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    n_ep = len(args.episode_ids)
    n_pass = sum(1 for e in summary["episodes"].values() if e.get("pass"))
    print(f"\n{n_pass}/{n_ep} episodes passed; dataset-level {'PASS' if ds['pass'] else 'FAIL'} -> {out_dir}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
