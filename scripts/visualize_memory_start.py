# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visualize WHAT the action memory encoder actually encodes (review Q2 / B2).

UVM confirmed its visual memory encoder by decoding z0 back into an image and
seeing the recalled frame. This script is the action-space analogue: it runs
an open-loop rollout, reads the raw memory-encoder output ``z0`` that
Gr00tPolicy now surfaces via ``info["memory_start"]`` (normalized space),
decodes it to raw action units with the same ``processor.decode_action`` path
the policy uses for ``action_pred``, and compares it against

  * y    — the ground-truth future chunk  gt[t : t + action_horizon]
  * h    — the (stride-aligned) previous predicted chunk the history source
           would use
  * pred — the policy's own denoised chunk

Outputs per trajectory:
  1. ``traj{i}_step{s}_z0.png`` — per-dim curves (GT vs z0 vs history vs pred)
     at episode start, segment-head replans and sampled mid replans, titled
     with the active instruction text.
  2. ``traj{i}_summary_bars.png`` — per-dim NMSE (MSE / GT per-dim variance)
     of z0 and history vs GT, split segment-head / segment-mid when
     ``--segment-meta-path`` is given.
  3. ``boundary_analysis.png`` — instruction-change probe: at every segment
     boundary, cosine similarity between the *jump* of z0 (resp. history)
     across the boundary and the jump of GT. Positive cosine = the source
     moves toward the new segment's target -> the encoder carries semantic,
     instruction-conditioned content (the B2 falsifiable test).
  4. ``memory_start_metrics.json`` — all numbers behind the plots.

Example (soft-anchor arm on the synthetic mixture dataset):

python scripts/visualize_memory_start.py \
  --dataset-path data/synth_mixture_eval \
  --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
  --model-path runs/mem_tx_soft_v2/checkpoint-5000 \
  --traj-ids 0 1 --steps 200 --execution-horizon 16 --denoising-steps 4 \
  --segment-meta-path data/synth_mixture_eval/meta/segment_schedule.jsonl \
  --out-dir runs/memory_start_viz
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rollout with z0 capture
# ---------------------------------------------------------------------------


def _to_2d(x: np.ndarray) -> np.ndarray:
    """(T, d) or batched (1, T, d) -> (T, d). Key-convention note:
    parse_action_gr00t output keys carry the "action." prefix, while
    processor.decode_action returns BARE modality keys — callers must pick
    the right key form (the smoke_db3 viz KeyError:'eef_9d' bug)."""
    a = np.asarray(x, dtype=np.float32)
    return a.reshape(-1, a.shape[-1]) if a.ndim >= 2 else np.atleast_2d(a)


def rollout_capture_z0(
    policy,
    loader,
    traj_id: int,
    embodiment_tag,
    action_keys: list[str],
    steps: int,
    execution_horizon: int,
    history_stride: int,
    high_level_task: str | None = None,
) -> list[dict[str, Any]]:
    """Open-loop rollout capturing z0 / history / pred per replan step.

    Returns a list of per-replan records with keys:
      step, instruction, z0 (T, D) raw or None, pred (T, D) raw,
      hist (T, D) raw (NaN-padded tail) or None, beta_hat / confidence if any.
    """
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.utils import parse_observation_gr00t
    from gr00t.eval.open_loop_eval import parse_action_gr00t

    traj = loader[traj_id]
    actual_steps = min(steps, len(traj))

    modality_configs = deepcopy(loader.modality_configs)
    modality_configs.pop("action")

    policy.reset()
    records: list[dict[str, Any]] = []
    prev_chunk: dict[str, np.ndarray] | None = None
    for step_count in range(0, actual_steps, execution_horizon):
        data_point = extract_step_data(traj, step_count, modality_configs, embodiment_tag)
        obs: dict[str, Any] = {}
        for k, v in data_point.states.items():
            obs[f"state.{k}"] = v
        for k, v in data_point.images.items():
            obs[f"video.{k}"] = np.array(v)
        for language_key in loader.modality_configs["language"].modality_keys:
            obs[language_key] = high_level_task if high_level_task else data_point.text
        parsed_obs = parse_observation_gr00t(obs, loader.modality_configs)
        if high_level_task:
            parsed_obs["sub_instruction"] = [data_point.text]
        action_chunk, info = policy.get_action(parsed_obs)
        pred = parse_action_gr00t(action_chunk)  # dict "action.<key>" -> (T, d)

        rec: dict[str, Any] = {
            "step": step_count,
            "instruction": data_point.text,
            "pred": np.concatenate(
                [_to_2d(pred[f"action.{k}"]) for k in action_keys], axis=-1
            ),
        }
        # z0: normalized (1, T, D) -> decode to raw units through the same
        # processor path the policy uses for action_pred. decode_action
        # returns BARE keys with a leading batch dim -> _to_2d strips it.
        z0 = info.get("memory_start") if info else None
        if z0 is not None:
            batched_states = {
                k: obs[f"state.{k}"][None]
                for k in loader.modality_configs["state"].modality_keys
            }
            z0_dec = policy.processor.decode_action(
                np.asarray(z0, dtype=np.float32), embodiment_tag, batched_states
            )
            rec["z0"] = np.concatenate(
                [_to_2d(z0_dec[k]) for k in action_keys], axis=-1
            )
        else:
            rec["z0"] = None
        for key in ("beta_hat", "confidence"):
            if info and key in info:
                rec[key] = np.asarray(info[key]).tolist()

        # History the model USED at this step = chunk predicted one stride ago,
        # sliced with the rtc offset (h[j] = prev[j + stride]); NaN-pad tail.
        # prev_chunk keys carry the "action." prefix (from parse_action_gr00t).
        if prev_chunk is not None:
            h = np.concatenate(
                [_to_2d(prev_chunk[f"action.{k}"]) for k in action_keys], axis=-1
            )
            horizon = h.shape[0]
            aligned = np.full_like(h, np.nan)
            if history_stride < horizon:
                aligned[: horizon - history_stride] = h[history_stride:]
            rec["hist"] = aligned
        else:
            rec["hist"] = None

        prev_chunk = {k: np.asarray(v, dtype=np.float32) for k, v in pred.items()}
        records.append(rec)
    return records


def gt_future_window(gt: np.ndarray, start: int, horizon: int) -> np.ndarray:
    """gt[start : start+horizon], tail padded by repeating the last row."""
    end = min(start + horizon, len(gt))
    win = gt[start:end]
    if len(win) < horizon:
        win = np.concatenate(
            [win, np.repeat(win[-1:], horizon - len(win), axis=0)], axis=0
        )
    return win


def role_consistent_target(
    rec: dict[str, Any], gt_win: np.ndarray, default_beta: float | None
) -> np.ndarray | None:
    """L1′ companion view (survey doc §13.24 root cause ⑦): in a mixed-source
    arm, z0's *role-correct* reference is (y − βh)/(1−β), not y. Comparing z0
    against y mixes role-internal error (the real problem) with role-external
    offset (a definition artifact).

    β priority: per-step ``rec["beta_hat"]`` (direct_beta arms) →
    ``default_beta`` (fixed-β arms, from the checkpoint config) → None (no
    reference available, returns None). Rows where hist is NaN (episode start
    / NaN-padded tail) fall back to y; β is clipped to 0.99 against the
    division singularity. Returns None when z0 is missing or no β is known.
    """
    if rec.get("z0") is None:
        return None
    beta = rec.get("beta_hat")
    if beta is not None:
        beta = float(np.asarray(beta).flatten()[0])
    elif default_beta is not None:
        beta = float(default_beta)
    else:
        return None
    hist = rec.get("hist")
    if hist is None or beta <= 0.0:
        return gt_win.copy()
    beta = min(beta, 0.99)
    role = (gt_win - beta * np.nan_to_num(hist, nan=0.0)) / (1.0 - beta)
    return np.where(np.isnan(hist), gt_win, role)


def source_gap(
    rec: dict[str, Any], gt_win: np.ndarray, default_beta: float | None
) -> np.ndarray | None:
    """Root cause ⑦ CORRECTED metric: the MIXED source's per-dim gap to y,
    ``‖βh+(1−β)z₀ − y‖²`` — the SAME quantity the training anchor optimises
    (``mse(source, y)``, multiplicative form, §13.26). This is z0's true
    role-internal quality *as a source*: no 1/(1−β) blow-up, and it does NOT
    collapse to a pure function of hist. (The earlier ``z0role =
    mse(role_tgt, y)`` equaled ``β²/(1−β)²·mse(h, y)`` — identical to hist at
    β=0.5 — and therefore measured nothing about z0. This is the fix.)

    β priority: ``rec["beta_hat"]`` → ``default_beta`` → None. Hist-NaN rows
    use ``source ≈ (1−β)z₀`` (the zero-mean η-noise term is dropped). Returns
    the per-dim MSE (squared gap averaged over the horizon, shape (D,)) —
    already averaged, do NOT re-square. NOTE: must be 1-D per-dim; an (T, D)
    return crashes _collect_err -> bar (2-D height array), see the 2026-10-18
    hotfix in survey doc §13.28.
    """
    if rec.get("z0") is None:
        return None
    beta = rec.get("beta_hat")
    if beta is not None:
        beta = float(np.asarray(beta).flatten()[0])
    elif default_beta is not None:
        beta = float(default_beta)
    else:
        return None
    z0 = rec["z0"]
    hist = rec.get("hist")
    if hist is None or beta <= 0.0:
        gap = z0 - gt_win  # β=0: pure memory source, s = z0
    else:
        has_hist = np.isfinite(hist)
        hist_clean = np.nan_to_num(hist, nan=0.0)
        src = beta * hist_clean + (1.0 - beta) * z0
        src = np.where(has_hist, src, (1.0 - beta) * z0)
        gap = src - gt_win
    return np.mean(gap ** 2, axis=0)


def probe_yaw_readout(
    records: list[dict[str, Any]],
    gt: np.ndarray,
    horizon: int,
    yaw_dims: tuple[int, ...] = (3, 4, 6, 7),
    ridge: float = 1.0,
) -> dict[str, Any] | None:
    """B1 linear readout probe (survey doc §13.27): can z0 read out the target
    bearing? PER-TIMESTEP ridge regression z0[t, :] (D,) -> GT yaw dims[t]
    — rows are timesteps, so n = N_steps × horizon (≈480) >> D and the probe
    is well-posed (a flattened-chunk probe would be n≪d and report ≈0 even
    for a perfect z0). Train on the first 2/3 of replan steps, R² on the
    held-out last 1/3 (split by step, not by row — no temporal leakage).

    Interpretation: R²(yaw) ≈ 0 = readout failure — the bearing information
    IS in the condition (white ring in the egocentric crop) but the encoder
    did not extract it (§13.24 yaw laziness) → fix readout (A1 relative-pose
    token), not the data. R²(yaw) ≈ 1 = readout fine, the z0 error on yaw is
    true conditional variance → fix Var (A2 explicit factors). Diagnostic,
    not a benchmark.
    """
    xs, ys = [], []
    for rec in records:
        if rec.get("z0") is None:
            continue
        z0 = rec["z0"]  # (T, D)
        y_win = gt_future_window(gt, rec["step"], horizon)  # (T, D)
        yaw = [i for i in yaw_dims if i < z0.shape[1]]
        if not yaw:
            return None
        xs.append(z0)
        ys.append(y_win[:, yaw])
    if len(xs) < 4:
        return None
    n_tr = max(2, int(round(len(xs) * 2 / 3)))
    Xtr = np.concatenate(xs[:n_tr], axis=0)  # (n_tr*T, D)
    Ytr = np.concatenate(ys[:n_tr], axis=0)  # (n_tr*T, |yaw|)
    Xte = np.concatenate(xs[n_tr:], axis=0)
    Yte = np.concatenate(ys[n_tr:], axis=0)
    xm, ym = Xtr.mean(0, keepdims=True), Ytr.mean(0, keepdims=True)
    Xc, Yc = Xtr - xm, Ytr - ym
    # Normal form (D × D) — D is the action dim (~132), cheap and stable.
    W = np.linalg.solve(Xc.T @ Xc + ridge * np.eye(Xc.shape[1]), Xc.T @ Yc)
    pred = (Xte - xm) @ W + ym
    ss_res = ((pred - Yte) ** 2).sum(axis=0)
    ss_tot = ((Yte - Yte.mean(0, keepdims=True)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    out: dict[str, Any] = {
        "n_train_steps": int(n_tr),
        "n_test_steps": int(len(xs) - n_tr),
        "n_rows_test": int(len(Yte)),
        "r2_per_yaw_dim": r2.tolist(),
        "r2_yaw_mean": float(np.mean(r2)),
        "yaw_dims": yaw,
    }
    # P1 (§13.28) counter-probe: when the regression targets include the xy
    # dims (called with yaw_dims=(0,1,2)), reconstruct the bearing through the
    # NONLINEAR chain atan2(ŷ_y, ŷ_x) and score it against the GT bearing.
    # A high value with a low z0→yaw linear R² proves "nonlinear embedding"
    # (the bearing IS in z0, as xy), not readout failure. Circular-safe R²
    # on (cos, sin) of the bearing angle.
    if 0 in yaw and 1 in yaw:
        i0, i1 = yaw.index(0), yaw.index(1)
        pred_b = np.arctan2(pred[:, i1], pred[:, i0])
        true_b = np.arctan2(Yte[:, i1], Yte[:, i0])
        ss_res_b = float(
            ((np.cos(pred_b) - np.cos(true_b)) ** 2
             + (np.sin(pred_b) - np.sin(true_b)) ** 2).sum()
        )
        cb, sb = np.cos(true_b), np.sin(true_b)
        ss_tot_b = float(
            ((cb - cb.mean()) ** 2 + (sb - sb.mean()) ** 2).sum()
        )
        out["r2_bearing_from_xy"] = 1.0 - ss_res_b / max(ss_tot_b, 1e-12)
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def masked_mse(a: np.ndarray, b: np.ndarray) -> float:
    """MSE over finite entries of a (NaN = padding)."""
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return float("nan")
    return float(np.mean((a[m] - b[m]) ** 2))


def per_dim_mse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    m = np.isfinite(a) & np.isfinite(b)
    out = np.full(a.shape[1], np.nan)
    for d in range(a.shape[1]):
        md = m[:, d]
        if md.any():
            out[d] = float(np.mean((a[md, d] - b[md, d]) ** 2))
    return out


def shape_corr_per_dim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """db10 (survey §13.57): per-dim Pearson correlation of FIRST DIFFERENCES
    along the horizon — corr(Δa, Δb). Measures waveform SHAPE following
    (slope direction/timing), which point-value MSE cannot see: a flat-line
    conditional mean scores NaN/0 here even when its level MSE looks fine.
    NaN for (near-)constant dims where the correlation is undefined; callers
    aggregate over the live-dim mask. Expects (H, D) inputs."""
    da = np.diff(a, axis=0)
    db = np.diff(b, axis=0)
    ma = np.isfinite(da)
    mb = np.isfinite(db)
    ok_rows = ma & mb
    out = np.full(a.shape[1], np.nan)
    for d in range(a.shape[1]):
        m = ok_rows[:, d]
        if m.sum() < 3:
            continue
        x, y = da[m, d], db[m, d]
        sx, sy = x.std(), y.std()
        if sx > 1e-8 and sy > 1e-8:
            out[d] = float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))
    return out


def _dominant_freq(x: np.ndarray, fps: float) -> tuple[float, float]:
    """(peak freq, sine amplitude) of a 1-D signal via Hann-windowed rFFT.

    NaN rows are dropped first. Returns (0, 0) for degenerate input."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 8:
        return 0.0, 0.0
    x = x - x.mean()
    if x.std() < 1e-9:
        return 0.0, 0.0
    w = np.hanning(n)
    spec = np.abs(np.fft.rfft(x * w))
    spec[0] = 0.0  # exclude DC
    k = int(np.argmax(spec))
    # Parabolic peak interpolation: the raw bin spacing fps/n (~0.4 Hz on a
    # 40-frame oscillation window) is too coarse for the 0.75 Hz train grid —
    # refine the peak location to ~1/10 bin before comparing against the
    # text-stated frequency.
    k_eff = float(k)
    if 0 < k < len(spec) - 1:
        y0, y1, y2 = spec[k - 1], spec[k], spec[k + 1]
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-12:
            k_eff = float(k + np.clip(0.5 * (y0 - y2) / denom, -1.0, 1.0))
    return float(k_eff * fps / n), float(2.0 * spec[k] / w.sum())


def _probe_r2(x: np.ndarray, y: np.ndarray) -> float | None:
    """Squared Pearson correlation (text-parameter readout probe): how much of
    the source-estimated wave parameter tracks the TEXT-stated value."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or x[m].std() < 1e-12 or y[m].std() < 1e-12:
        return None
    r = float(np.corrcoef(x[m], y[m])[0, 1])
    return r * r


def wave_segment_entry(
    records: list[dict[str, Any]],
    seg: dict[str, Any],
    gt: np.ndarray,
    horizon: int,
    fps: float,
) -> dict[str, Any] | None:
    """db11: spectral fidelity of z0/pred/hist vs GT on ONE oscillation
    segment. Concatenates the leading rows of every replan chunk inside the
    oscillation window (aligned with GT future rows), then per source:
    dominant frequency (+rel err vs the TEXT-stated freq), sine amplitude,
    RMS ratio, and first-difference shape corr on the wave axis."""
    w = seg["wave"]
    axd = int(w["axis_index"])
    src_rows: dict[str, list[np.ndarray]] = {"z0": [], "pred": [], "hist": []}
    gt_rows: list[np.ndarray] = []
    for r in records:
        if r["step"] < w["osc_start"] or r["step"] >= seg["end"]:
            continue
        take = int(min(8, seg["end"] - r["step"], horizon))
        if take < 4:
            continue
        gt_w = gt_future_window(gt, r["step"], horizon)[:take]
        gt_rows.append(gt_w)
        for k in src_rows:
            v = r[k]
            src_rows[k].append(v[:take] if v is not None else np.full_like(gt_w, np.nan))
    if not gt_rows or sum(len(g) for g in gt_rows) < 24:
        return None
    gtc = np.concatenate(gt_rows)
    entry: dict[str, Any] = {
        "segment_start": int(seg["start"]),
        "freq_gt": float(w["freq"]),
        "amp_gt": float(w["amp"]),
        "axis_index": axd,
    }
    for k, rows in src_rows.items():
        src = np.concatenate(rows)
        f_est, a_est = _dominant_freq(src[:, axd], fps)
        finite = np.isfinite(src[:, axd]) & np.isfinite(gtc[:, axd])
        rms_ratio = (
            float(np.nanstd(src[:, axd]) / max(np.nanstd(gtc[:, axd]), 1e-12))
            if finite.sum() >= 8
            else float("nan")
        )
        sc = shape_corr_per_dim(src, gtc)[axd]
        entry[k] = {
            "freq_est": f_est,
            "freq_rel_err": abs(f_est - w["freq"]) / w["freq"] if f_est > 0 else float("nan"),
            "amp_est": a_est,
            "rms_ratio": rms_ratio,
            "shape_corr_axis": None if not np.isfinite(sc) else float(sc),
        }
    return entry


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    fa, fb = a.flatten(), b.flatten()
    m = np.isfinite(fa) & np.isfinite(fb)
    if not m.any():
        return float("nan")
    fa, fb = fa[m], fb[m]
    denom = np.linalg.norm(fa) * np.linalg.norm(fb)
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(fa, fb) / denom)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_replan_point(
    rec: dict[str, Any],
    gt_win: np.ndarray,
    dim_idx: list[int],
    dim_names: list[str],
    horizon: int,
    save_path: Path,
    tag: str = "",
):
    n = len(dim_idx)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 2.2 * nrows), sharex=True)
    axes = np.atleast_1d(axes).flatten()
    t = np.arange(horizon)
    for ax_i, d in enumerate(dim_idx):
        ax = axes[ax_i]
        ax.plot(t, gt_win[:, d], "k-", lw=1.8, label="GT future")
        if rec["z0"] is not None:
            ax.plot(t, rec["z0"][:, d], "r-", lw=1.4, label="z0 (memory)")
        if rec.get("role_tgt") is not None:
            ax.plot(
                t, rec["role_tgt"][:, d], "m:", lw=1.3,
                label="role target (y-βh)/(1-β)",
            )
        if rec["hist"] is not None:
            ax.plot(t, rec["hist"][:, d], "b--", lw=1.2, label="history (aligned)")
        ax.plot(t, rec["pred"][:, d], "g-", lw=1.0, alpha=0.7, label="policy pred")
        mse_z = masked_mse(rec["z0"][:, d], gt_win[:, d]) if rec["z0"] is not None else float("nan")
        mse_h = masked_mse(rec["hist"][:, d], gt_win[:, d]) if rec["hist"] is not None else float("nan")
        mse_r = (
            masked_mse(rec["z0"][:, d], rec["role_tgt"][:, d])
            if (rec["z0"] is not None and rec.get("role_tgt") is not None)
            else float("nan")
        )
        ax.set_title(
            f"{dim_names[d]} | z0 {mse_z:.4f} vs hist {mse_h:.4f} vs role {mse_r:.4f}",
            fontsize=9,
        )
        ax.grid(alpha=0.3)
        if ax_i == 0:
            ax.legend(fontsize=8, loc="best")
    for ax_i in range(n, len(axes)):
        axes[ax_i].axis("off")
    instr = rec["instruction"]
    if len(instr) > 90:
        instr = instr[:87] + "..."
    beta = rec.get("beta_hat")
    beta_str = f" | beta_hat={beta[0]:.2f}" if beta else ""
    fig.suptitle(f"{tag}step {rec['step']}{beta_str} | \"{instr}\"", fontsize=10)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


def plot_summary_bars(
    nmse: dict[str, np.ndarray],
    dim_names: list[int],
    save_path: Path,
    title: str,
):
    """nmse: {group: per-dim NMSE array} with group in {z0_head, hist_head, z0_mid, hist_mid}."""
    groups = list(nmse.keys())
    n_dims = len(dim_names)
    x = np.arange(n_dims)
    width = 0.8 / max(len(groups), 1)
    fig, ax = plt.subplots(figsize=(max(8, n_dims * 0.9), 4.5))
    colors = {"z0_head": "#d62728", "hist_head": "#1f77b4",
              "z0_mid": "#ff9896", "hist_mid": "#aec7e8"}
    for gi, g in enumerate(groups):
        vals = nmse[g]
        ax.bar(x + gi * width, vals, width, label=g, color=colors.get(g))
    ax.set_xticks(x + width * (len(groups) - 1) / 2)
    ax.set_xticklabels([str(d) for d in dim_names], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("NMSE vs GT future (MSE / GT per-dim var)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@dataclass
class ArgsConfig:
    dataset_path: str
    embodiment_tag: str
    model_path: str
    """Checkpoint trained with a memory encoder (z0 surfaced via info)."""
    traj_ids: list[int] = field(default_factory=lambda: [0])
    steps: int = 200
    execution_horizon: int = 16
    history_replan_stride: int | None = None
    """None = execution_horizon (correct open-loop alignment); 0 disables."""
    denoising_steps: int = 4
    segment_meta_path: str | None = None
    """segment_schedule.jsonl from the synthetic mixture builder (optional)."""
    segment_head_steps: int = 8
    out_dir: str = "runs/memory_start_viz"
    max_plot_dims: int = 8
    """Plot the top-variance action dims (up to this many)."""
    max_plots_per_traj: int = 6
    """Curve-plot budget per trajectory (episode start + segment heads first)."""
    high_level_task: str | None = None
    high_level_task_from_meta: bool = False
    seed: int | None = None
    fixed_beta: float | None = None
    """Explicit source-mix beta for role-target diagnostics (e.g. 0.3)."""
    fixed_noise_std: float | None = None
    """Optional semantic-source eta override; does not alter Gaussian arms."""
    probe_yaw: bool = False
    """B1 readout probe (survey doc §13.27): ridge-regress z0 -> GT yaw dims
    (held-out steps), R² in metrics json. R² ≈ 0 = readout failure (info in
    the white ring but not extracted); R² ≈ 1 = readout fine, residual error
    is true conditional variance."""
    state_extra_modality_keys: str = ""
    """db7/A1: comma-separated extra state modality keys (e.g.
    'target_relative_pose') appended to the embodiment's state config for THIS
    process before the policy/loader is built (must match training)."""


def main(args: ArgsConfig):
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.eval.compare_flow_sources import extract_gt
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    logging.basicConfig(level=logging.INFO)
    args.embodiment_tag = EmbodimentTag.resolve(args.embodiment_tag)
    if args.state_extra_modality_keys:
        # db7/A1: extend the process-local state config BEFORE policy/loader.
        from gr00t.configs.data.state_extra_keys import (
            apply_state_extra_keys,
            parse_extra_keys,
        )

        apply_state_extra_keys(
            args.embodiment_tag.value, parse_extra_keys(args.state_extra_modality_keys)
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    policy = Gr00tPolicy(
        embodiment_tag=args.embodiment_tag,
        model_path=args.model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        history_replan_stride=(
            None
            if args.history_replan_stride == 0
            else (args.history_replan_stride or args.execution_horizon)
        ),
        denoising_steps=args.denoising_steps,
        fixed_beta=args.fixed_beta,
        fixed_noise_std=args.fixed_noise_std,
    )
    policy.model.action_head.num_inference_timesteps = args.denoising_steps
    history_stride = (
        args.execution_horizon
        if args.history_replan_stride is None
        else args.history_replan_stride
    )

    modality = policy.get_modality_config()
    dataset = LeRobotEpisodeLoader(dataset_path=args.dataset_path, modality_configs=modality)
    action_keys = modality["action"].modality_keys

    # Per-episode high-level tasks (dual text port), same convention as
    # compare_flow_sources.
    hl_from_meta: dict[int, str] = {}
    if args.high_level_task_from_meta:
        with open(Path(args.dataset_path) / "meta" / "episodes.jsonl") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("high_level_task"):
                    hl_from_meta[int(rec["episode_index"])] = rec["high_level_task"]

    # Dataset fps (db11 wave spectral metrics); synthetic builders default 15.
    fps_ds = 15.0
    info_path = Path(args.dataset_path) / "meta" / "info.json"
    if info_path.exists():
        with open(info_path) as f:
            fps_ds = float(json.load(f).get("fps", fps_ds))

    # Segment sidecar: starts per trajectory (for head/mid split + boundaries).
    seg_by_traj: dict[int, list[dict[str, Any]]] = {}
    if args.segment_meta_path:
        with open(args.segment_meta_path) as f:
            for line in f:
                rec = json.loads(line)
                seg_by_traj.setdefault(int(rec["episode_index"]), []).append(rec)

    all_metrics: dict[str, Any] = {
        "model_path": args.model_path,
        "protocol": {
            "dataset_path": args.dataset_path,
            "segment_meta_path": args.segment_meta_path,
            "steps": args.steps,
            "execution_horizon": args.execution_horizon,
            "history_replan_stride": history_stride,
            "denoising_steps": args.denoising_steps,
            "high_level_task_from_meta": args.high_level_task_from_meta,
            "seed": args.seed,
            "fixed_beta": args.fixed_beta,
            "fixed_noise_std": args.fixed_noise_std,
        },
        "trajs": {},
    }
    horizon = None
    for traj_id in args.traj_ids:
        if traj_id >= len(dataset):
            logger.warning(f"Trajectory {traj_id} out of range, skipped")
            continue
        traj = dataset[traj_id]
        actual_steps = min(args.steps, len(traj))
        gt = extract_gt(traj, action_keys, actual_steps)
        if args.seed is not None:
            torch.manual_seed(args.seed + traj_id)
        records = rollout_capture_z0(
            policy, dataset, traj_id, args.embodiment_tag, action_keys,
            args.steps, args.execution_horizon, history_stride,
            high_level_task=hl_from_meta.get(traj_id, args.high_level_task),
        )
        if not records:
            continue
        horizon = records[0]["pred"].shape[0]
        n_dims = records[0]["pred"].shape[1]
        if records[0]["z0"] is None:
            logger.warning(
                "info['memory_start'] missing — checkpoint has no memory encoder "
                "or memory_start_enabled=False; z0 curves/metrics will be skipped."
            )

        # Top-variance dims drive the curve plots.
        dim_var = np.nanvar(gt, axis=0)
        dim_idx = list(np.argsort(-dim_var)[: min(args.max_plot_dims, n_dims)])
        dim_idx.sort()
        dim_names = [f"dim{d}" for d in range(n_dims)]

        # Head mask over replan points (episode start + segment heads).
        starts = [s["start"] for s in seg_by_traj.get(traj_id, [])]
        head_steps: set[int] = {0}
        for s in starts:
            if s <= 0:
                continue
            r = int(np.ceil(s / args.execution_horizon)) * args.execution_horizon
            head_steps.add(r)

        # Which replan points get curve plots: episode start, segment heads,
        # then evenly spaced mids up to the budget.
        # L1′ companion (root cause ⑦): per-replan role-consistent reference
        # (y − βh)/(1−β). β comes from rec["beta_hat"] when the checkpoint
        # routes confidence (direct_beta), else from the checkpoint's fixed
        # memory_start_mix_beta; gaussian/pure-history runs get None (skipped).
        default_beta = getattr(
            getattr(getattr(policy, "model", None), "config", None),
            "memory_start_mix_beta",
            None,
        )
        for rec in records:
            gt_w = gt_future_window(gt, rec["step"], horizon)
            rec["role_tgt"] = role_consistent_target(rec, gt_w, default_beta)
            rec["source_gap"] = source_gap(rec, gt_w, default_beta)

        recs_by_step = {r["step"]: r for r in records}
        plot_steps = sorted(head_steps & recs_by_step.keys())
        mid_candidates = [r["step"] for r in records if r["step"] not in head_steps]
        budget = max(0, args.max_plots_per_traj - len(plot_steps))
        if budget and mid_candidates:
            pick = np.linspace(0, len(mid_candidates) - 1, min(budget, len(mid_candidates)))
            plot_steps += [mid_candidates[int(i)] for i in pick]
        for s in sorted(set(plot_steps)):
            rec = recs_by_step[s]
            gt_win = gt_future_window(gt, s, horizon)
            tag = "[SEGMENT HEAD] " if s in head_steps and s > 0 else ""
            plot_replan_point(
                rec, gt_win, dim_idx, dim_names, horizon,
                out_dir / f"traj{traj_id}_step{s:04d}_z0.png", tag=tag,
            )

        # Metrics: per-dim MSE vs GT future, split head/mid.
        def _collect(which: str, steps_set: set[int] | None):
            vals = []
            for r in records:
                if r[which] is None:
                    continue
                if steps_set is not None and r["step"] not in steps_set:
                    continue
                if steps_set is None and r["step"] in head_steps:
                    continue
                vals.append(per_dim_mse(r[which], gt_future_window(gt, r["step"], horizon)))
            return np.nanmean(np.stack(vals), axis=0) if vals else np.full(n_dims, np.nan)

        def _collect_err(which: str, steps_set: set[int] | None):
            """Average a field that is ALREADY a per-dim squared error
            (e.g. ``source_gap``), instead of re-computing it against GT.
            Defensive: a 2-D (T, D) entry is reduced over T first — a raw
            2-D stack once crashed plot_summary_bars (§13.28 hotfix)."""
            vals = []
            for r in records:
                if r[which] is None:
                    continue
                if steps_set is not None and r["step"] not in steps_set:
                    continue
                if steps_set is None and r["step"] in head_steps:
                    continue
                v = np.asarray(r[which], dtype=np.float64)
                vals.append(v.mean(axis=0) if v.ndim == 2 else v)
            return np.nanmean(np.stack(vals), axis=0) if vals else np.full(n_dims, np.nan)

        def _collect_shape(which: str, steps_set: set[int] | None):
            """db10 (§13.57): mean per-dim shape correlation corr(Δ·, Δy)
            over records — waveform shape following, the differential-anchor
            arm's primary endpoint."""
            vals = []
            for r in records:
                if r[which] is None:
                    continue
                if steps_set is not None and r["step"] not in steps_set:
                    continue
                if steps_set is None and r["step"] in head_steps:
                    continue
                vals.append(shape_corr_per_dim(r[which], gt_future_window(gt, r["step"], horizon)))
            return np.nanmean(np.stack(vals), axis=0) if vals else np.full(n_dims, np.nan)

        mid_set = None  # None = "not head"
        mse = {
            "z0_head": _collect("z0", head_steps),
            "z0_mid": _collect("z0", mid_set),
            "hist_head": _collect("hist", head_steps),
            "hist_mid": _collect("hist", mid_set),
            "pred_head": _collect("pred", head_steps),
            "pred_mid": _collect("pred", mid_set),
            # Source gap ‖βh+(1−β)z0 − y‖² (corrected ⑦): z0's role-internal
            # quality as a source, same quantity the training anchor minimises.
            "z0role_head": _collect_err("source_gap", head_steps),
            "z0role_mid": _collect_err("source_gap", mid_set),
        }
        # db10: shape following for z0/hist, with pred as the readout-side
        # reference (shape recovered after denoising).
        shape = {
            "z0_head": _collect_shape("z0", head_steps),
            "z0_mid": _collect_shape("z0", mid_set),
            "hist_head": _collect_shape("hist", head_steps),
            "hist_mid": _collect_shape("hist", mid_set),
            "pred_head": _collect_shape("pred", head_steps),
            "pred_mid": _collect_shape("pred", mid_set),
        }
        # NMSE: normalize per-dim MSE by per-dim GT variance so dims are
        # comparable; dead dims (var < 1e-3^2) excluded from the mean.
        live = dim_var > 1e-6
        nmse = {k: np.where(live, v / np.maximum(dim_var, 1e-12), np.nan) for k, v in mse.items()}
        plot_summary_bars(
            {
                k: nmse[k]
                for k in (
                    "z0_head", "hist_head", "z0role_head",
                    "z0_mid", "hist_mid", "z0role_mid",
                )
            },
            list(range(n_dims)),
            out_dir / f"traj{traj_id}_summary_bars.png",
            f"Traj {traj_id}: z0 vs history vs role-target, NMSE (head/mid split)",
        )

        # Boundary analysis: does z0 jump toward the NEW segment's GT chunk?
        boundaries = []
        for s in sorted(starts):
            if s <= 0:
                continue
            r_after = int(np.ceil(s / args.execution_horizon)) * args.execution_horizon
            r_before = r_after - args.execution_horizon
            if r_after not in recs_by_step or r_before not in recs_by_step:
                continue
            y_after = gt_future_window(gt, r_after, horizon)
            y_before = gt_future_window(gt, r_before, horizon)
            gt_jump = y_after - y_before
            entry: dict[str, Any] = {"boundary_start": s, "replan_after": r_after}
            if recs_by_step[r_after]["z0"] is not None and recs_by_step[r_before]["z0"] is not None:
                entry["cos_z0_jump_gt"] = cos_sim(
                    recs_by_step[r_after]["z0"] - recs_by_step[r_before]["z0"], gt_jump
                )
            if recs_by_step[r_after]["hist"] is not None and recs_by_step[r_before]["hist"] is not None:
                entry["cos_hist_jump_gt"] = cos_sim(
                    recs_by_step[r_after]["hist"] - recs_by_step[r_before]["hist"], gt_jump
                )
            boundaries.append(entry)
        if boundaries:
            fig, ax = plt.subplots(figsize=(max(6, len(boundaries)), 4))
            xs = np.arange(len(boundaries))
            width = 0.35
            cos_z = [b.get("cos_z0_jump_gt", float("nan")) for b in boundaries]
            cos_h = [b.get("cos_hist_jump_gt", float("nan")) for b in boundaries]
            ax.bar(xs - width / 2, cos_z, width, label="z0 jump vs GT jump", color="#d62728")
            ax.bar(xs + width / 2, cos_h, width, label="history jump vs GT jump", color="#1f77b4")
            ax.axhline(0, color="k", lw=0.8)
            ax.set_xticks(xs)
            ax.set_xticklabels([f"b{b['boundary_start']}" for b in boundaries], fontsize=8)
            ax.set_ylabel("cosine similarity")
            ax.set_title(f"Traj {traj_id}: instruction-change probe (higher = source tracks new segment)")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3, axis="y")
            fig.tight_layout()
            fig.savefig(out_dir / f"traj{traj_id}_boundary_analysis.png", dpi=130)
            plt.close(fig)

        all_metrics["trajs"][str(traj_id)] = {
            "per_dim_mse": {k: v.tolist() for k, v in mse.items()},
            "per_dim_nmse": {k: v.tolist() for k, v in nmse.items()},
            "mean_nmse_live_dims": {
                k: float(np.nanmean(v[live])) for k, v in nmse.items()
            },
            "per_dim_shape_corr": {k: v.tolist() for k, v in shape.items()},
            "mean_shape_corr_live_dims": {
                k: float(np.nanmean(v[live])) for k, v in shape.items()
            },
            "boundaries": boundaries,
        }
        # ---- db11 wave block: spectral fidelity on oscillation segments ----
        wave_rows = []
        for s in seg_by_traj.get(traj_id, []):
            if "wave" not in s:
                continue
            entry = wave_segment_entry(records, s, gt, horizon, fps_ds)
            if entry is not None:
                wave_rows.append(entry)
        if wave_rows:
            all_metrics["trajs"][str(traj_id)]["wave_segments"] = wave_rows
            logger.info(
                f"  traj{traj_id} wave: "
                + "; ".join(
                    f"seg@{e['segment_start']} f*={e['freq_gt']:g}Hz "
                    f"z0relerr={e['z0']['freq_rel_err']:.2f} "
                    f"z0sc={e['z0']['shape_corr_axis']} "
                    f"predrelerr={e['pred']['freq_rel_err']:.2f}"
                    for e in wave_rows
                )
            )
        logger.info(
            f"  traj{traj_id} shape_corr(live): "
            + " ".join(f"{k}={float(np.nanmean(v[live])):+.3f}" for k, v in shape.items())
        )
        if args.probe_yaw:
            probe = probe_yaw_readout(records, gt, horizon)
            all_metrics["trajs"][str(traj_id)]["probe_yaw_readout"] = probe
            # P1 (§13.28) counter-probe: z0 -> GT xyz dims + atan2 bearing
            # reconstruction, separating "true readout failure" from
            # "nonlinear embedding" (bearing present as xy, not as rot6d).
            probe_xyz = probe_yaw_readout(records, gt, horizon, yaw_dims=(0, 1, 2))
            all_metrics["trajs"][str(traj_id)]["probe_xyz_readout"] = probe_xyz
            if probe and probe["r2_yaw_mean"] is not None:
                logger.info(
                    f"traj {traj_id}: yaw readout probe R²(yaw)="
                    f"{probe['r2_yaw_mean']:.3f} "
                    f"(per-dim {[f'{v:.2f}' for v in probe['r2_per_yaw_dim']]}, "
                    f"n_test={probe['n_test_steps']} steps)"
                )
            if probe_xyz:
                logger.info(
                    f"traj {traj_id}: counter-probe R²(xyz)="
                    f"{probe_xyz['r2_yaw_mean']:.3f} "
                    f"R²(bearing|xy)={probe_xyz.get('r2_bearing_from_xy', float('nan')):.3f} "
                    "— xyz high + bearing high + yaw low ⇒ nonlinear embedding "
                    "(readout OK); xyz low ⇒ true readout failure"
                )
        logger.info(
            f"traj {traj_id}: mean NMSE (live dims) "
            + ", ".join(
                f"{k}={all_metrics['trajs'][str(traj_id)]['mean_nmse_live_dims'][k]:.4f}"
                for k in (
                    "z0_head", "hist_head", "z0role_head",
                    "z0_mid", "hist_mid", "z0role_mid",
                )
            )
        )

    # ---- db11 wave aggregate + text-parameter readout probe (cross-traj) ----
    wave_all = [
        e
        for tm in all_metrics["trajs"].values()
        for e in tm.get("wave_segments", [])
    ]
    if wave_all:
        agg: dict[str, Any] = {"n_segments": len(wave_all)}
        for k in ("z0", "pred", "hist"):
            rel = [e[k]["freq_rel_err"] for e in wave_all if np.isfinite(e[k]["freq_rel_err"])]
            rr = [e[k]["rms_ratio"] for e in wave_all if np.isfinite(e[k]["rms_ratio"])]
            sc = [e[k]["shape_corr_axis"] for e in wave_all if e[k]["shape_corr_axis"] is not None]
            agg[k] = {
                "freq_rel_err_median": float(np.median(rel)) if rel else None,
                "rms_ratio_mean": float(np.mean(rr)) if rr else None,
                "shape_corr_axis_mean": float(np.mean(sc)) if sc else None,
            }
        # probe-wave: does the source's ESTIMATED wave parameter track the
        # TEXT-stated (GT) value across segments? R² ≈ 1 = the encoder read
        # the wave spec out of the instruction (text-causal evidence).
        f_gt = np.array([e["freq_gt"] for e in wave_all])
        a_gt = np.array([e["amp_gt"] for e in wave_all])
        for k in ("z0", "pred"):
            f_est = np.array([e[k]["freq_est"] for e in wave_all])
            a_est = np.array([e[k]["amp_est"] for e in wave_all])
            agg[f"probe_r2_freq_{k}"] = _probe_r2(f_gt, f_est)
            agg[f"probe_r2_amp_{k}"] = _probe_r2(a_gt, a_est)
        all_metrics["wave"] = agg
        logger.info("wave aggregate: " + json.dumps(agg))

    with open(out_dir / "memory_start_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"All outputs -> {out_dir}")


if __name__ == "__main__":
    main(tyro.cli(ArgsConfig))
