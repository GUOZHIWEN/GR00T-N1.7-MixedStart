#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
shape_corr ceiling estimator (survey doc §13.57, db10 verdict calibration).

A deterministic z0 = f(c) can achieve at most

    shape_corr_max = sqrt( Var(E[Δy | c]) / Var(Δy) )

per action dim — the square root of the increment variance explained by the
condition. This script estimates that ceiling on the synthetic mixture
dataset (db6/db7-style), approximating the model's condition c with the
per-frame key

    (segment kind, target color, phase, dist bin, yaw bin, crossing flag)

where dist/yaw come from the target_relative_pose state channels (the same
channels the model actually sees), phase from approach_end, and the crossing
flag marks increments that step INTO a new segment (teleport jumps).

Interpretation discipline:
  - ceiling << 1 means the condition itself does not determine the increment
    (variance floor is irreducible for ANY point anchor — no anchor redesign
    can beat it; improvements must come from the conditioning side);
  - baseline shape_corr << ceiling means the anchor/readout is leaving
    attainable shape on the table (db10's differential anchor has room).

Honesty guards:
  - Law-of-total-variance check: between + within ≈ total (printed).
  - Shuffle control (N permutations): small-group bias inflates between-var;
    the adjusted estimate subtracts the shuffle mean (ANOVA-style).
  - This is a POOLED increment-level estimate; the viz metric computes
    per-window Pearson over H-1 increments. The two coincide when increment
    distributions are stationary across the window — approximation, stated.

Pure CPU. Usage:

    uv run python scripts/estimate_shape_corr_ceiling.py \
      --dataset-path data/synth_mixture_eval_db7 \
      --out-json runs/shape_ceiling_db7_eval.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

STATE_REL_SLICE = slice(17, 22)  # target_relative_pose: rel_xyz(3) + sin + cos
DEAD_DIM_VAR_EPS = 1e-6  # dead_dim_eps=0.001 convention, squared


def _sincos_order(state_rel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Auto-detect (sin, cos) slot order via the unit-norm invariant."""
    a, b = state_rel[:, 3], state_rel[:, 4]
    if abs(float(np.median(a**2 + b**2)) - 1.0) > 0.05:
        logger.warning(f"sin²+cos² median={float(np.median(a**2 + b**2)):.4f} — rel-pose layout unexpected")
    return a, b  # db7 hotfix#2: (sin_dyaw, cos_dyaw) at dims 20, 21


def load_frames(dataset_path: Path, schedule_path: Path, max_episodes: int | None):
    """Returns per-increment arrays: dY (N,17), and key columns."""
    segments: dict[int, list[dict]] = {}
    with open(schedule_path) as f:
        for line in f:
            rec = json.loads(line)
            segments.setdefault(int(rec["episode_index"]), []).append(rec)
    for v in segments.values():
        v.sort(key=lambda r: r["start"])

    rows = []
    data_dir = dataset_path / "data" / "chunk-000"
    files = sorted(data_dir.glob("episode_*.parquet"))
    if max_episodes:
        files = files[:max_episodes]
    for fp in files:
        ep_idx = int(fp.stem.split("_")[-1])
        segs = segments.get(ep_idx)
        if not segs:
            continue
        df = pd.read_parquet(fp)
        action = np.stack(df["action"].map(np.asarray).values).astype(np.float64)
        state = np.stack(df["observation.state"].map(np.asarray).values).astype(np.float64)
        T = len(df)
        starts = [s["start"] for s in segs]
        start_set = set(starts)
        sin_arr, cos_arr = _sincos_order(state[:, STATE_REL_SLICE])
        rel = state[:, 17:20]
        dist = np.linalg.norm(rel, axis=1)
        yaw = np.arctan2(sin_arr, cos_arr)
        for t in range(T - 1):
            seg = segs[np.searchsorted(starts, t, side="right") - 1]
            phase = "na"
            if "approach_end" in seg:
                phase = "approach" if t < seg["approach_end"] else "dwell"
            rows.append(
                (
                    ep_idx,
                    seg["kind"],
                    seg["color"],
                    phase,
                    float(dist[t]),
                    float(yaw[t]),
                    1 if (t + 1) in start_set else 0,  # increment crosses INTO a new segment
                    # db11: wave-aware condition key (params + phase quadrant)
                    _wave_key(seg, state, t) if "wave" in seg else "",
                    action[t + 1] - action[t],
                )
            )
    if not rows:
        raise RuntimeError("no increments collected — check dataset/schedule paths")
    meta = pd.DataFrame(
        [r[:8] for r in rows],
        columns=["ep", "kind", "color", "phase", "dist", "yaw", "crossing", "wave"],
    )
    dY = np.stack([r[8] for r in rows])  # (N, 17)
    return dY, meta


def _wave_key(seg: dict, state: np.ndarray, t: int) -> str:
    """db11 oscillation condition key: wave-parameter signature + phase
    quadrant (pos-rel-to-center sign x velocity sign along the wave axis).

    The wave params split increments by text-determined waveform; the phase
    quadrant is needed because the oscillation flips the increment direction
    every half period — without it the key lumps opposite-velocity increments
    together and the ceiling is spuriously low (the model reads the phase
    from the state, so the proxy condition must contain it too)."""
    w = seg["wave"]
    a = int(w["axis_index"])
    if a < 2 and "target_xy" in seg:
        center = float(seg["target_xy"][a])
    else:
        center = float(np.mean(state[w["osc_start"] : seg["end"], a]))
    pos_rel = state[t, a] - center
    vel = state[t + 1, a] - state[t, a]
    # 8-sector phase key: (pos sign, vel sign, |pos| beyond half amplitude) —
    # 4 quadrants still lump rising/falling magnitudes; the half-amplitude bit
    # separates the slow extremum regions from the fast zero-crossing regions.
    return (
        f"{w['axis']}{w['freq']:g}hz{w['amp']:.2f}m"
        f"|q{int(pos_rel > 0)}{int(vel > 0)}m{int(abs(pos_rel) > 0.5 * w['amp'])}"
    )


def between_group_r2(dy: np.ndarray, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-dim (between-var, within-var, total-var) of dy grouped by integer keys."""
    n, d = dy.shape
    total = dy.var(axis=0)
    between = np.zeros(d)
    within = np.zeros(d)
    for k in np.unique(keys):
        g = dy[keys == k]
        w = len(g) / n
        mu = g.mean(axis=0)
        between += w * (mu - dy.mean(axis=0)) ** 2
        within += w * g.var(axis=0)
    return between, within, total


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--segment-meta-path", default=None,
                   help="default: <dataset>/meta/segment_schedule.jsonl")
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--dist-bins", type=int, default=8)
    p.add_argument("--yaw-bins", type=int, default=8)
    p.add_argument("--shuffle-reps", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-json", default=None)
    args = p.parse_args()

    dataset_path = Path(args.dataset_path)
    schedule_path = Path(args.segment_meta_path) if args.segment_meta_path else dataset_path / "meta" / "segment_schedule.jsonl"

    dY, meta = load_frames(dataset_path, schedule_path, args.max_episodes)
    n, D = dY.shape
    logger.info(f"collected {n} increments ({D} dims) from {meta['ep'].nunique()} episodes")

    # quantile bins (deduped edges; constant channels collapse to one bin)
    def qbin(x: np.ndarray, nb: int) -> np.ndarray:
        edges = np.unique(np.quantile(x, np.linspace(0, 1, nb + 1)))
        if len(edges) <= 2:
            return np.zeros_like(x, dtype=int)
        return np.digitize(x, edges[1:-1])

    meta["dist_bin"] = qbin(meta["dist"].values, args.dist_bins)
    meta["yaw_bin"] = qbin(meta["yaw"].values, args.yaw_bins)
    key_str = (
        meta["kind"] + "|" + meta["color"] + "|" + meta["phase"] + "|"
        + meta["dist_bin"].astype(str) + "|" + meta["yaw_bin"].astype(str)
        + "|" + meta["crossing"].astype(str) + "|" + meta["wave"]
    )
    keys, key_names = pd.factorize(key_str)
    logger.info(f"condition keys: {len(key_names)} distinct, median occupancy "
                f"{int(np.median(np.bincount(keys)))} increments")

    between, within, total = between_group_r2(dY, keys)
    # law of total variance sanity
    rel_err = float(np.abs((between + within) - total).sum() / (total.sum() + 1e-12))
    logger.info(f"total-variance decomposition check: rel err = {rel_err:.2e} (should be < 1e-8)")

    live = total > DEAD_DIM_VAR_EPS
    r2_raw = np.where(live, between / np.maximum(total, 1e-12), np.nan)

    # shuffle control: small-group bias estimate
    rng = np.random.RandomState(args.seed)
    r2_shuf = np.zeros((args.shuffle_reps, D))
    for i in range(args.shuffle_reps):
        b_k, _, _ = between_group_r2(dY, rng.permutation(keys))
        r2_shuf[i] = np.where(live, b_k / np.maximum(total, 1e-12), np.nan)
    r2_shuf_mean = np.nanmean(r2_shuf, axis=0)
    r2_adj = np.clip(r2_raw - r2_shuf_mean, 0.0, 1.0)

    ceiling_raw = np.sqrt(np.clip(r2_raw, 0, 1))
    ceiling_adj = np.sqrt(r2_adj)

    # per-kind / per-phase breakdowns (same machinery on subsets)
    def subset_ceiling(mask: np.ndarray):
        b, w, t = between_group_r2(dY[mask], keys[mask])
        lv = t > DEAD_DIM_VAR_EPS
        r2 = np.where(lv, b / np.maximum(t, 1e-12), np.nan)
        return float(np.nanmean(np.sqrt(np.clip(r2, 0, 1))[lv]))

    breakdowns = {
        "kind=grasp": subset_ceiling((meta["kind"] == "grasp").values),
        "kind=point": subset_ceiling((meta["kind"] == "point").values),
        "phase=approach": subset_ceiling((meta["phase"] == "approach").values),
        "phase=dwell": subset_ceiling((meta["phase"] == "dwell").values),
        "crossing=1": subset_ceiling((meta["crossing"] == 1).values),
        "crossing=0": subset_ceiling((meta["crossing"] == 0).values),
    }
    if (meta["kind"] == "oscillate").any():
        # db11: oscillation increments should sit NEAR the ceiling (~1) once
        # the wave-parameter signature splits the condition keys — the text
        # fully determines the waveform.
        breakdowns["kind=oscillate"] = subset_ceiling((meta["kind"] == "oscillate").values)

    report = {
        "dataset": str(dataset_path),
        "n_increments": n,
        "n_keys": len(key_names),
        "variance_check_rel_err": rel_err,
        "live_dims": [int(i) for i in np.where(live)[0]],
        "per_dim": {
            str(d): {
                "total_var": float(total[d]),
                "r2_raw": None if not live[d] else float(r2_raw[d]),
                "r2_shuffle_mean": None if not live[d] else float(r2_shuf_mean[d]),
                "ceiling_raw": None if not live[d] else float(ceiling_raw[d]),
                "ceiling_adj": None if not live[d] else float(ceiling_adj[d]),
            }
            for d in range(D)
        },
        "mean_ceiling_raw_live": float(np.nanmean(ceiling_raw[live])),
        "mean_ceiling_adj_live": float(np.nanmean(ceiling_adj[live])),
        "breakdowns_raw": breakdowns,
        "params": {
            "dist_bins": args.dist_bins, "yaw_bins": args.yaw_bins,
            "shuffle_reps": args.shuffle_reps, "seed": args.seed,
        },
    }

    # stdout table
    print("\n=== shape_corr ceiling estimate (per live dim) ===")
    print(f"{'dim':>4} {'total_var':>10} {'R²_raw':>8} {'R²_shuf':>8} {'ceil_raw':>9} {'ceil_adj':>9}")
    for d in range(D):
        if not live[d]:
            print(f"{d:>4} {total[d]:>10.2e}   (dead)")
            continue
        print(f"{d:>4} {total[d]:>10.4f} {r2_raw[d]:>8.3f} {r2_shuf_mean[d]:>8.3f} "
              f"{ceiling_raw[d]:>9.3f} {ceiling_adj[d]:>9.3f}")
    print(f"\nmean ceiling (live dims): raw={report['mean_ceiling_raw_live']:.3f} "
          f"adj={report['mean_ceiling_adj_live']:.3f}")
    print("breakdowns (raw):", json.dumps({k: round(v, 3) for k, v in breakdowns.items()}))
    print("reference: db7 baseline z0 shape_corr ≈ 0.00 (viz_db7_baseline_shapecorr)")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"written: {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
