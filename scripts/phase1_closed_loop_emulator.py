#!/usr/bin/env python
"""Phase 1 / L1: closed-loop EMULATION stress suite (no hardware required).

Survey context (§13.57–§13.62): every verdict so far is open-loop MSE — the
model's action chunk is scored against the recorded trajectory, but the world
never RESPONDS to the model's actions. Phase 1 closes that gap in three tiers;
this script is Tier L1, the hardware-free tier. It instruments the rolling
replan loop (pred-fed history, the same causal structure a deployed controller
sees) and measures the five failure axes open-loop MSE cannot see:

  M1  error compounding   per-replan-cycle MSE series + linear slope.
                          Open-loop MSE is the mean of this curve; the SLOPE
                          is what kills a deployed policy (drift accumulation
                          under self-fed history).
  M2  replan-stride/latency stress
                          rerun at several execution horizons (control-rate
                          proxy). A real DROID stack runs ~15 Hz with
                          100–300 ms inference latency; stride sweep tells the
                          latency budget before hardware time is burned.
  M3  perturbation recovery probe
                          at --perturb-step, inject a position jump Δ into the
                          eef state the policy observes (simulated push/slip).
                          The replay world cannot respond, but the policy's
                          INTENT is measurable: cosine alignment between the
                          commanded position delta and the return direction
                          (−Δ), replans-to-align (settling), and output
                          absmax (divergence detection).
  M4  chunk-boundary discontinuity
                          ‖a_last(chunk k) − a_first(chunk k+1)‖ at replan
                          boundaries — the smoothness metric that decides
                          whether a real motor can track the command stream.
                          Reported alongside repeat‖Δ‖ (determinism asset).
  M5  goal-arrival proxy  integrate predicted position deltas from the
                          episode start state → predicted eef path; distance
                          of the predicted END pose to the recorded end pose,
                          with success rates at tolerance thresholds. This is
                          the open-loop-integrable predecessor of the L3
                          binary success metric.

Arms are the Phase-0 checkpoints (integrated_droid β=0 / routed, droid_no_mem,
optionally a synthetic arm). The policy loading + post-load β override pattern
is lifted verbatim from gr00t/eval/compare_flow_sources.py (P2-1 discipline:
no config.json surgery, post-load setattr only).

Usage (single arm, full suite):

  uv run python scripts/phase1_closed_loop_emulator.py \
    --dataset-path data/droid_subset_eval \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
    --model-paths runs/integrated_droid/checkpoint-5000 --labels droid_pure_z0 \
    --traj-ids 0 1 2 3 4 5 6 7 8 9 --steps 200 --execution-horizon 16 \
    --denoising-steps 4 --seed 0 \
    --force-fixed-beta --fixed-beta-value 0.0 \
    --stride-list 16 --perturb-step 96 --perturb-delta 0.05 \
    --out-dir runs/phase1_droid_pure_z0 |& tee runs/phase1_droid_pure_z0_log.txt

NOTE on units: actions/states pass through the checkpoint's own processor
(unnormalized on output), so position deltas are in the dataset's native
units (DROID: meters). Success thresholds default to 2/5/10 (cm if DROID).
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# instrumented rollout (extends compare_flow_sources.rollout_trajectory)
# ---------------------------------------------------------------------------

def instrumented_rollout(
    policy,
    loader,
    traj_id: int,
    embodiment_tag,
    action_keys: list[str],
    steps: int,
    execution_horizon: int,
    pos_key: str,
    pos_dims: slice,
    perturb_step: int | None = None,
    perturb_delta: float = 0.0,
    high_level_task: str | None = None,
) -> dict[str, Any]:
    """Rolling replan with pred-fed history; returns per-replan records.

    Differences from compare_flow_sources.rollout_trajectory (by design):
      * keeps the per-replan chunk boundaries (M4 needs chunk seams);
      * optionally perturbs the observed eef position by +Δ at ONE replan
        step (M3) — the observation is what the controller would see after a
        push; the GT trajectory is unchanged (replay cannot respond);
      * records per-replan GT window so M1 can score each cycle separately.
    """
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.utils import parse_observation_gr00t
    from gr00t.eval.open_loop_eval import parse_action_gr00t

    traj = loader[traj_id]
    actual_steps = min(steps, len(traj))
    policy.reset()

    modality_configs = deepcopy(loader.modality_configs)
    modality_configs.pop("action")

    replans: list[dict[str, Any]] = []
    pred_all: list[np.ndarray] = []
    gt_pos_traj = np.stack(
        [np.asarray(traj[f"state.{pos_key}"].iloc[i], dtype=np.float32)[pos_dims]
         for i in range(actual_steps)]
    )  # (T, 3) recorded eef positions (for M3/M5)

    for step_count in range(0, actual_steps, execution_horizon):
        data_point = extract_step_data(traj, step_count, modality_configs, embodiment_tag)
        obs: dict[str, Any] = {}
        for k, v in data_point.states.items():
            obs[f"state.{k}"] = np.array(v)
        for k, v in data_point.images.items():
            obs[f"video.{k}"] = np.array(v)
        for language_key in loader.modality_configs["language"].modality_keys:
            obs[language_key] = high_level_task if high_level_task else data_point.text

        injected = False
        if perturb_step is not None and step_count == perturb_step:
            # M3: one-shot sensory perturbation — the policy sees the eef at
            # pos+Δ. Its recovery intent should point along −Δ.
            sk = f"state.{pos_key}"
            obs[sk] = obs[sk].copy()
            obs[sk][pos_dims] = obs[sk][pos_dims] + np.float32(perturb_delta)
            injected = True

        parsed_obs = parse_observation_gr00t(obs, loader.modality_configs)
        if high_level_task:
            parsed_obs["sub_instruction"] = [data_point.text]
        _chunk, _info = policy.get_action(parsed_obs)
        action_chunk = parse_action_gr00t(_chunk)
        chunk = np.stack(
            [
                np.concatenate(
                    [
                        np.atleast_1d(np.atleast_1d(action_chunk[f"action.{k}"])[j])
                        for k in action_keys
                    ]
                ).astype(np.float32)
                for j in range(execution_horizon)
            ]
        )  # (H, D_action)
        n_exec = min(execution_horizon, actual_steps - step_count)
        pred_all.append(chunk[:n_exec])
        replans.append(
            {
                "step": step_count,
                "chunk": chunk,
                "n_exec": n_exec,
                "injected": injected,
                "gt_window_pos": gt_pos_traj[step_count : step_count + n_exec],
                "beta_hat": (_info or {}).get("beta_hat"),
            }
        )
    pred_arr = np.concatenate(pred_all, axis=0)[:actual_steps]
    return {"pred": pred_arr, "replans": replans, "gt_pos_traj": gt_pos_traj}


# ---------------------------------------------------------------------------
# metric modules
# ---------------------------------------------------------------------------

def m1_compounding(replans, gt, pos_dims_action, eps=1e-3):
    """Per-replan-cycle MSE (live dims) + least-squares slope across cycles."""
    live = np.std(gt, axis=0) > eps
    per_cycle = []
    for rec in replans:
        s, n = rec["step"], rec["n_exec"]
        w_gt = gt[s : s + n]
        diff = (rec["chunk"][:n] - w_gt)[:, live]
        per_cycle.append(float(np.mean(diff**2)))
    x = np.arange(len(per_cycle), dtype=np.float64)
    slope = float(np.polyfit(x, per_cycle, 1)[0]) if len(per_cycle) > 2 else 0.0
    return {"per_cycle_mse": per_cycle, "slope_per_cycle": slope}


def m3_perturbation(replans, perturb_delta, pos_dims_action, align_thresh=0.3):
    """Recovery intent after the injected jump.

    align_cos: cosine between the mean commanded position delta of the first
    post-injection chunk and the return direction −Δ̂ (constant vector here, so
    Δ̂ = +1 per axis). settle_cycles: replans until align stays ≥ thresh.
    out_absmax: max |action| over the 3 cycles post injection (divergence).
    """
    inj_idx = next((i for i, r in enumerate(replans) if r["injected"]), None)
    if inj_idx is None:
        return None
    ret_dir = -np.sign(perturb_delta) * np.ones(3, dtype=np.float64)
    ret_dir = ret_dir / np.linalg.norm(ret_dir)
    aligns = []
    absmax = 0.0
    for rec in replans[inj_idx : inj_idx + 6]:
        cmd = rec["chunk"][:, pos_dims_action].mean(axis=0).astype(np.float64)
        n = np.linalg.norm(cmd)
        aligns.append(float(cmd @ ret_dir / n) if n > 1e-8 else 0.0)
        absmax = max(absmax, float(np.abs(rec["chunk"]).max()))
    settle = None
    for i, a in enumerate(aligns):
        if a >= align_thresh:
            settle = i
            break
    return {
        "align_cos_per_cycle": aligns,
        "settle_cycles": settle,
        "out_absmax_post_injection": absmax,
        "delta": perturb_delta,
    }


def m4_boundary_jump(replans):
    """Mean ‖a_last(k) − a_first(k+1)‖ over replan seams (smoothness)."""
    jumps = []
    for a, b in zip(replans[:-1], replans[1:]):
        jumps.append(
            float(np.linalg.norm(b["chunk"][0] - a["chunk"][a["n_exec"] - 1]))
        )
    return {"boundary_jump_mean": float(np.mean(jumps)) if jumps else None,
            "boundary_jump_max": float(np.max(jumps)) if jumps else None}


def m5_goal_arrival(pred, replans, pos_dims_action, start_pos, end_pos, thresholds,
                    pos_scale: float = 1.0):
    """Integrate predicted position deltas → predicted end pose; success rates.

    pos_scale converts raw action-delta units into state-position units
    (dataset-level constant calibrated on GT, see calibrate_pos_scale) —
    without it the raw integration overshoots by ~300x on DROID/synthetic
    (actions and state positions live in different units; db18 M6 discovery).
    """
    path = np.concatenate(
        [start_pos[None],
         start_pos[None] + np.cumsum(pred[:, pos_dims_action] * pos_scale, axis=0)]
    )
    pred_end = path[-1]
    dist = float(np.linalg.norm(pred_end - end_pos))
    gt_end = end_pos
    gt_dist = float(np.linalg.norm(gt_end - start_pos))
    return {
        "pred_end_dist": dist,
        "gt_travel": gt_dist,
        "pos_scale": pos_scale,
        **{f"success@{t}": bool(dist < t) for t in thresholds},
    }


def calibrate_pos_scale(loader, traj_ids, action_keys, pos_key, pos_dims, steps,
                        eps=1e-8):
    """GT-only unit calibration: median over trajs of
    ||state end − start|| / ||sum of gt action deltas||.

    Pure unit conversion (no model output involved): the same constant is
    applied to every arm's predicted path before thresholding.
    """
    from gr00t.eval.compare_flow_sources import extract_gt

    ratios: list[float] = []
    for tid in traj_ids:
        traj = loader[tid]
        n = min(steps, len(traj))
        gt = extract_gt(traj, action_keys, n)
        disp_a = float(np.linalg.norm(gt[:, pos_dims].sum(axis=0)))
        p0 = np.asarray(traj[f"state.{pos_key}"].iloc[0], dtype=np.float32)[pos_dims]
        p1 = np.asarray(traj[f"state.{pos_key}"].iloc[n - 1], dtype=np.float32)[pos_dims]
        disp_s = float(np.linalg.norm(p1 - p0))
        if disp_a > eps and disp_s > eps:
            ratios.append(disp_s / disp_a)
    return float(np.median(ratios)) if ratios else 1.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@dataclass
class ArgsConfig:
    dataset_path: str
    embodiment_tag: str
    model_paths: list[str]
    labels: list[str] | None = None
    traj_ids: list[int] = field(default_factory=lambda: [0, 1, 2])
    steps: int = 200
    execution_horizon: int = 16
    stride_list: list[int] = field(default_factory=lambda: [16])
    """M2: control-rate stress. 16 ≈ 1.07 s@15 Hz (current recipe); add
    8/24/32 for the latency budget curve (each extra stride re-rolls all
    trajectories, ~linear runtime)."""
    denoising_steps: int = 4
    seed: int | None = 0
    high_level_task: str | None = None
    high_level_task_from_meta: bool = False
    state_extra_modality_keys: str = ""
    force_fixed_beta: bool = False
    fixed_beta_value: float | None = None
    perturb_step: int | None = 96
    """M3: replan step at which the eef position observation is jumped by
    +perturb_delta (None disables the probe)."""
    perturb_delta: float = 0.05
    pos_key: str = "eef_9d"
    """State/action key whose first 3 dims are eef position (DROID & the
    synthetic builder both use eef_9d: pos3 + rot6d)."""
    success_thresholds: list[float] = field(default_factory=lambda: [0.02, 0.05, 0.10])
    success_repeats: int = 1
    """M6 (experiment ②, survey §8 item 2): multi-solution success probe —
    K PERTURBED replans per trajectory, each with a distinct seed, turning the
    single binary M5 draw into a per-traj success PROBABILITY plus a
    repeat-consensus measure (do all K repeats land on the same outcome?).
    This is where the repeat‖Δ‖ determinism asset is cashed into a
    success-rate difference. 1 = legacy single-rollout behaviour."""
    out_dir: str = "runs/phase1_emulator"


def main(args: ArgsConfig):
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.eval._horizon_contract import PolicyHorizonSpec
    from gr00t.eval.compare_flow_sources import extract_gt
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    logging.basicConfig(level=logging.INFO)
    args.embodiment_tag = EmbodimentTag.resolve(args.embodiment_tag)
    if args.state_extra_modality_keys:
        from gr00t.configs.data.state_extra_keys import (
            apply_state_extra_keys,
            parse_extra_keys,
        )

        apply_state_extra_keys(
            args.embodiment_tag.value, parse_extra_keys(args.state_extra_modality_keys)
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # §13.46 logging rule: mirror INFO+ logs to <out_dir>/log.txt.
    _fh = logging.FileHandler(out_dir / "log.txt", mode="w", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logging.getLogger().addHandler(_fh)

    labels = args.labels or [
        Path(p).parent.name + "/" + Path(p).name for p in args.model_paths
    ]
    assert len(labels) == len(args.model_paths)

    first_policy = Gr00tPolicy(
        embodiment_tag=args.embodiment_tag,
        model_path=args.model_paths[0],
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    modality = first_policy.get_modality_config()
    del first_policy
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    dataset = LeRobotEpisodeLoader(dataset_path=args.dataset_path, modality_configs=modality)
    action_keys = modality["action"].modality_keys
    assert args.pos_key in action_keys, f"{args.pos_key} not in action keys {action_keys}"
    # position dims inside the CONCATENATED action vector (eef_9d is the first
    # action key in both DROID and the synthetic builder: pos3 + rot6d)
    if action_keys[0] != args.pos_key:
        raise AssertionError(
            f"pos_key must be the first action key (got {action_keys[0]!r}); "
            "extend pos_dims_action computation before using M3/M5"
        )
    pos_dims_action = slice(0, 3)

    # db18 M6 discovery: raw action deltas and state positions are in
    # different units (~300x apart), so M5 success was all-zero in every
    # prior phase1 run. Calibrate ONE dataset-level constant on GT (no model
    # output involved) and apply it to every arm's predicted path.
    pos_scale = calibrate_pos_scale(
        dataset, args.traj_ids, action_keys, args.pos_key, pos_dims_action,
        args.steps,
    )
    logger.info(f"M5/M6 action→state unit calibration: pos_scale={pos_scale:.6f}")

    hl_from_meta: dict[int, str] = {}
    if args.high_level_task_from_meta:
        with open(Path(args.dataset_path) / "meta" / "episodes.jsonl") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("high_level_task"):
                    hl_from_meta[int(rec["episode_index"])] = rec["high_level_task"]

    results: dict[str, Any] = {"args": vars(args), "pos_scale": pos_scale, "arms": {}}
    for label, model_path in zip(labels, args.model_paths):
        logger.info(f"=== Arm {label}: {model_path} ===")
        policy = Gr00tPolicy(
            embodiment_tag=args.embodiment_tag,
            model_path=model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        policy.model.action_head.num_inference_timesteps = args.denoising_steps
        if args.force_fixed_beta:
            seen: set[int] = set()
            for cfg in (policy.model.config, policy.model.action_head.config):
                if id(cfg) in seen:
                    continue
                seen.add(id(cfg))
                if hasattr(cfg, "confidence_routing_enabled"):
                    cfg.confidence_routing_enabled = False
            logger.info("force_fixed_beta: confidence routing disabled post-load")
        if args.fixed_beta_value is not None:
            seen_b: set[int] = set()
            for cfg in (policy.model.config, policy.model.action_head.config):
                if id(cfg) in seen_b:
                    continue
                seen_b.add(id(cfg))
                if hasattr(cfg, "memory_start_mix_beta"):
                    cfg.memory_start_mix_beta = float(args.fixed_beta_value)
            logger.info("fixed_beta_value overridden to %.3f", args.fixed_beta_value)

        arm: dict[str, Any] = {"model_path": model_path, "strides": {}}
        for stride in args.stride_list:
            PolicyHorizonSpec.from_modality_config(modality, n_action_steps=stride)
            stride_rec: dict[str, Any] = {"trajs": {}}
            for traj_id in args.traj_ids:
                if traj_id >= len(dataset):
                    continue
                traj = dataset[traj_id]
                actual_steps = min(args.steps, len(traj))
                gt = extract_gt(traj, action_keys, actual_steps)
                if args.seed is not None:
                    torch.manual_seed(args.seed + traj_id)
                out = instrumented_rollout(
                    policy, dataset, traj_id, args.embodiment_tag, action_keys,
                    args.steps, stride, args.pos_key, pos_dims_action,
                    perturb_step=(
                        args.perturb_step if stride == args.execution_horizon else None
                    ),
                    perturb_delta=args.perturb_delta,
                    high_level_task=hl_from_meta.get(traj_id, args.high_level_task),
                )
                pred, replans = out["pred"], out["replans"]
                # determinism probe: rerun once, mean |run1 − run2|
                if args.seed is not None:
                    torch.manual_seed(args.seed + traj_id + 500000)
                out2 = instrumented_rollout(
                    policy, dataset, traj_id, args.embodiment_tag, action_keys,
                    args.steps, stride, args.pos_key, pos_dims_action,
                    perturb_step=None,
                    high_level_task=hl_from_meta.get(traj_id, args.high_level_task),
                )
                n = min(len(pred), len(out2["pred"]))
                repeat_div = float(np.abs(pred[:n] - out2["pred"][:n]).mean())

                live = np.std(gt, axis=0) > 1e-3
                mse_eff = float(np.mean((pred - gt)[:, live] ** 2))
                rec = {
                    "mse_eff": mse_eff,
                    "repeat_divergence": repeat_div,
                    "M1": m1_compounding(replans, gt, pos_dims_action),
                    "M3": m3_perturbation(replans, args.perturb_delta, pos_dims_action)
                    if stride == args.execution_horizon
                    else None,
                    "M4": m4_boundary_jump(replans),
                    "M5": m5_goal_arrival(
                        pred,
                        replans,
                        pos_dims_action,
                        start_pos=out["gt_pos_traj"][0],
                        end_pos=out["gt_pos_traj"][-1],
                        thresholds=args.success_thresholds,
                        pos_scale=pos_scale,
                    ),
                }
                # M6: multi-repeat success under perturbation (experiment ②).
                # K rollouts, each perturbed at --perturb-step with a distinct
                # seed; per-traj success probability + consensus + pairwise
                # divergence replace the single binary M5 draw.
                if args.success_repeats >= 2 and stride == args.execution_horizon:
                    preds_k: list[np.ndarray] = []
                    ends_k: list[float] = []
                    succ_k: dict[float, list[float]] = {
                        t: [] for t in args.success_thresholds
                    }
                    settle_k: list[float] = []
                    for k in range(args.success_repeats):
                        if args.seed is not None:
                            torch.manual_seed(args.seed + traj_id * 1000 + k)
                        outk = instrumented_rollout(
                            policy, dataset, traj_id, args.embodiment_tag,
                            action_keys, args.steps, stride, args.pos_key,
                            pos_dims_action,
                            perturb_step=args.perturb_step,
                            perturb_delta=args.perturb_delta,
                            high_level_task=hl_from_meta.get(
                                traj_id, args.high_level_task
                            ),
                        )
                        preds_k.append(outk["pred"])
                        m5k = m5_goal_arrival(
                            outk["pred"], outk["replans"], pos_dims_action,
                            start_pos=outk["gt_pos_traj"][0],
                            end_pos=outk["gt_pos_traj"][-1],
                            thresholds=args.success_thresholds,
                            pos_scale=pos_scale,
                        )
                        ends_k.append(m5k["pred_end_dist"])
                        for t in args.success_thresholds:
                            succ_k[t].append(float(m5k[f"success@{t}"]))
                        m3k = m3_perturbation(
                            outk["replans"], args.perturb_delta, pos_dims_action
                        )
                        if m3k is not None and m3k["settle_cycles"] is not None:
                            settle_k.append(m3k["settle_cycles"])
                    nn = min(len(p) for p in preds_k)
                    pair = [
                        float(np.abs(preds_k[i][:nn] - preds_k[j][:nn]).mean())
                        for i in range(len(preds_k))
                        for j in range(i + 1, len(preds_k))
                    ]
                    rec["M6"] = {
                        "n_repeats": args.success_repeats,
                        "per_repeat_end_dist": ends_k,
                        "pairwise_repeat_div": float(np.mean(pair)),
                        "settle_cycles": settle_k,
                        **{
                            f"success_prob@{t}": succ_k[t]
                            for t in args.success_thresholds
                        },
                        **{
                            f"consensus@{t}": bool(
                                min(succ_k[t]) == max(succ_k[t])
                            )
                            for t in args.success_thresholds
                        },
                    }
                stride_rec["trajs"][str(traj_id)] = rec
                logger.info(
                    f"  [{label} s{stride} traj{traj_id}] mse_eff={mse_eff:.4f} "
                    f"slope={rec['M1']['slope_per_cycle']:.5f}/cyc "
                    f"jump={rec['M4']['boundary_jump_mean']:.4f} "
                    f"end_dist={rec['M5']['pred_end_dist']:.4f} rep={repeat_div:.4f}"
                    + (
                        f" align={rec['M3']['align_cos_per_cycle']}"
                        if rec["M3"]
                        else ""
                    )
                )
            # stride summary
            trs = list(stride_rec["trajs"].values())
            if trs:
                stride_rec["summary"] = {
                    "mse_eff": float(np.mean([t["mse_eff"] for t in trs])),
                    "compounding_slope": float(
                        np.mean([t["M1"]["slope_per_cycle"] for t in trs])
                    ),
                    "boundary_jump_mean": float(
                        np.nanmean(
                            [t["M4"]["boundary_jump_mean"] for t in trs]
                        )
                    ),
                    "repeat_divergence": float(np.mean([t["repeat_divergence"] for t in trs])),
                    "end_dist_median": float(
                        np.median([t["M5"]["pred_end_dist"] for t in trs])
                    ),
                    **{
                        f"success_rate@{t}": float(
                            np.mean([tr["M5"][f"success@{t}"] for tr in trs])
                        )
                        for t in args.success_thresholds
                    },
                }
                aligns = [t["M3"] for t in trs if t["M3"]]
                if aligns:
                    stride_rec["summary"]["M3_align_mean"] = float(
                        np.nanmean([a["align_cos_per_cycle"][0] for a in aligns])
                    )
                    settles = [a["settle_cycles"] for a in aligns if a["settle_cycles"] is not None]
                    stride_rec["summary"]["M3_settle_mean"] = (
                        float(np.mean(settles)) if settles else None
                    )
                # M6 aggregation: pooled success rate over (traj × repeat)
                # cells with Wilson 95% CI, per-traj consensus fraction
                # (determinism cashed as outcome consistency), and the
                # Flageat-style joint lower bound success − α·repeat‖Δ‖.
                m6s = [t.get("M6") for t in trs if t.get("M6")]
                if m6s:
                    z = 1.96
                    for t in args.success_thresholds:
                        cells = [s for m in m6s for s in m[f"success_prob@{t}"]]
                        n_cells = len(cells)
                        rate = float(np.mean(cells))
                        den = 1 + z * z / n_cells
                        ctr = rate + z * z / (2 * n_cells)
                        half = z * float(
                            np.sqrt(
                                rate * (1 - rate) / n_cells
                                + z * z / (4 * n_cells * n_cells)
                            )
                        )
                        stride_rec["summary"][f"m6_success_rate@{t}"] = rate
                        stride_rec["summary"][f"m6_success_wilson95@{t}"] = [
                            round((ctr - half) / den, 4),
                            round((ctr + half) / den, 4),
                        ]
                        stride_rec["summary"][f"m6_consistency@{t}"] = float(
                            np.mean([m[f"consensus@{t}"] for m in m6s])
                        )
                    rep6 = float(np.mean([m["pairwise_repeat_div"] for m in m6s]))
                    stride_rec["summary"]["m6_pairwise_repeat_div"] = rep6
                    settles6 = [s for m in m6s for s in m["settle_cycles"]]
                    stride_rec["summary"]["m6_settle_mean"] = (
                        float(np.mean(settles6)) if settles6 else None
                    )
                    for alpha in (1.0, 5.0, 10.0):
                        stride_rec["summary"][f"m6_lcb_a{alpha:g}@0.05"] = float(
                            stride_rec["summary"]["m6_success_rate@0.05"] - alpha * rep6
                        )
            arm["strides"][str(stride)] = stride_rec
            logger.info(f"[{label} stride {stride}] summary: {stride_rec.get('summary')}")
        results["arms"][label] = arm
        del policy
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def _json_default(o):
        # numpy 标量转 float；枚举/路径等其余对象转 str（vars(args) 中的
        # embodiment_tag 在 main() 里被 resolve 成了 EmbodimentTag 枚举）。
        if isinstance(o, (int, float, bool)):
            return o
        try:
            return float(o)
        except (TypeError, ValueError):
            return str(o)

    with open(out_dir / "phase1_metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    logger.info(f"written: {out_dir / 'phase1_metrics.json'}")
    return results


if __name__ == "__main__":
    main(tyro.cli(ArgsConfig))
