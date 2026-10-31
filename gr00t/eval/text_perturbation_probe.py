"""Text-perturbation probe (survey §13.99-D, 2026-10-25).

Decisive zero-training audit for "the conditioning channel is TOO strong":
the synthetic benchmark feeds the next segment's target coordinates through
BOTH ``state.target_relative_pose`` AND the coordinate-bearing instruction
text, so the text channel may be nominally dead (short-circuited by the state
channel). This probe measures it directly.

Protocol (paired, per arm, per trajectory):
  * ``ref``      — correct text (high-level task from meta + per-frame
                   sub-instruction), torch seed = seed+traj_id;
  * ``floor``    — ref repeated with seed+traj_id+500000: the run-to-run
                   source-sampling noise floor (repeat‖Δ‖ of this arm);
  * perturbed passes (same seed as ref):
      - ``sub_swap``  — per-frame instruction replaced by the NEXT segment's
                        task_text (grammatical, wrong color+coordinates);
      - ``hl_swap``   — high-level task replaced by another trajectory's;
      - ``sub_blank`` — per-frame instruction replaced by a content-free
                        string ("continue the current motion").

Metrics per (arm, mode): effect = mean|pred_pert − pred_ref| (all dims and
live dims), z₀ shift = mean|z₀_pert − z₀_ref| (normalized source space),
and the ratio effect/floor. **Reading**: ratio ≈ 1 → the perturbation is
indistinguishable from the arm's own sampling noise → text nominally dead;
ratio ≫ 1 → text channel alive. z₀ shift near 0 while text changes means the
memory encoder itself does not read the instruction either.

Example:

  uv run python gr00t/eval/text_perturbation_probe.py \
    --dataset-path data/synth_mixture_eval_db7 \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
    --model-paths runs/integrated_db7/checkpoint-5000 \
                  runs/integrated_db21_yanchor/checkpoint-5000 \
                  runs/ablate_no_mem/checkpoint-5000 \
    --labels db7 db21 no_mem \
    --traj-ids 0 1 2 3 4 5 6 7 8 9 --steps 200 --execution-horizon 16 \
    --denoising-steps 4 \
    --segment-meta-path data/synth_mixture_eval_db7/meta/segment_schedule.jsonl \
    --state-extra-modality-keys target_relative_pose \
    --high-level-task-from-meta --seed 0 \
    --out-dir runs/probe_text_perturb
"""

from __future__ import annotations

import gc
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import tyro

logger = logging.getLogger(__name__)

BLANK_TEXT = "continue the current motion"
MODE_COLORS = {"sub_swap": "#d62728", "hl_swap": "#ff7f0e", "sub_blank": "#9467bd"}


@dataclass
class ArgsConfig:
    dataset_path: str = "data/synth_mixture_eval_db7"
    embodiment_tag: str = "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"
    model_paths: list[str] = field(default_factory=list)
    labels: list[str] | None = None
    traj_ids: list[int] = field(default_factory=lambda: list(range(10)))
    steps: int = 200
    execution_horizon: int = 16
    denoising_steps: int = 4
    segment_meta_path: str | None = None
    """Required for sub_swap (next-segment instruction source)."""
    state_extra_modality_keys: str | None = None
    high_level_task_from_meta: bool = False
    seed: int = 0
    perturb_modes: list[str] = field(
        default_factory=lambda: ["sub_swap", "hl_swap", "sub_blank"]
    )
    max_plot_dims: int = 8
    out_dir: str = "runs/probe_text_perturb"


def rollout_with_text(
    policy,
    loader,
    traj_id: int,
    embodiment_tag,
    action_keys: list[str],
    steps: int,
    execution_horizon: int,
    high_level_task: str | None,
    sub_text_fn: Callable[[int, str], str] | None = None,
    z0_out: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Open-loop rollout with optional per-frame sub-instruction override.

    Mirrors compare_flow_sources.rollout_trajectory (teacher-forced obs,
    self-fed history buffer) but lets the probe rewrite the text channel.
    z0_out collects the raw memory-encoder output (normalized space) per
    replan when the checkpoint surfaces it.
    """
    from copy import deepcopy

    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.utils import parse_observation_gr00t
    from gr00t.eval.open_loop_eval import parse_action_gr00t

    traj = loader[traj_id]
    actual_steps = min(steps, len(traj))
    policy.reset()

    modality_configs = deepcopy(loader.modality_configs)
    modality_configs.pop("action")

    pred_action_across_time = []
    for step_count in range(0, actual_steps, execution_horizon):
        data_point = extract_step_data(traj, step_count, modality_configs, embodiment_tag)
        obs: dict[str, Any] = {}
        for k, v in data_point.states.items():
            obs[f"state.{k}"] = v
        for k, v in data_point.images.items():
            obs[f"video.{k}"] = np.array(v)
        frame_text = data_point.text
        if sub_text_fn is not None:
            frame_text = sub_text_fn(step_count, frame_text)
        for language_key in loader.modality_configs["language"].modality_keys:
            obs[language_key] = high_level_task if high_level_task else frame_text
        parsed_obs = parse_observation_gr00t(obs, loader.modality_configs)
        if high_level_task:
            parsed_obs["sub_instruction"] = [frame_text]
        action_chunk_raw, info = policy.get_action(parsed_obs)
        if z0_out is not None and info and "memory_start" in info:
            z0_out.append(np.asarray(info["memory_start"], dtype=np.float32)[0])
        action_chunk = parse_action_gr00t(action_chunk_raw)
        for j in range(execution_horizon):
            per_key = {
                key: np.atleast_1d(np.atleast_1d(action_chunk[f"action.{key}"])[j]).astype(
                    np.float32
                )
                for key in action_keys
            }
            pred_action_across_time.append(
                np.concatenate([per_key[key] for key in action_keys], axis=0)
            )
    return np.array(pred_action_across_time)[:actual_steps]


def main(args: ArgsConfig) -> None:
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.eval.compare_flow_sources import extract_gt, split_dead_dims
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
    _fh = logging.FileHandler(out_dir / "log.txt", mode="w", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logging.getLogger().addHandler(_fh)

    labels = args.labels or [
        Path(p).parent.name + "/" + Path(p).name for p in args.model_paths
    ]
    assert len(labels) == len(args.model_paths)

    # Segment sidecar -> per-traj ordered segment list (sub_swap source).
    segs_by_traj: dict[int, list[dict]] = {}
    if args.segment_meta_path:
        with open(args.segment_meta_path) as f:
            for line in f:
                rec = json.loads(line)
                segs_by_traj.setdefault(int(rec["episode_index"]), []).append(rec)
        for v in segs_by_traj.values():
            v.sort(key=lambda r: int(r["start"]))
    if "sub_swap" in args.perturb_modes:
        assert segs_by_traj, "sub_swap requires --segment-meta-path"

    hl_from_meta: dict[int, str] = {}
    if args.high_level_task_from_meta:
        with open(Path(args.dataset_path) / "meta" / "episodes.jsonl") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("high_level_task"):
                    hl_from_meta[int(rec["episode_index"])] = rec["high_level_task"]
    hl_pool = sorted(set(hl_from_meta.values()))

    first_policy = Gr00tPolicy(
        embodiment_tag=args.embodiment_tag,
        model_path=args.model_paths[0],
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    modality = first_policy.get_modality_config()
    del first_policy
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dataset = LeRobotEpisodeLoader(dataset_path=args.dataset_path, modality_configs=modality)
    action_keys = modality["action"].modality_keys

    def make_sub_fn(mode: str, traj_id: int):
        if mode == "sub_swap":
            segs = segs_by_traj[traj_id]

            def sub_swap_fn(step: int, text: str) -> str:
                idx = None
                for i, rec in enumerate(segs):
                    if int(rec["start"]) <= step < int(rec["end"]):
                        idx = i
                        break
                if idx is None:
                    return text
                return str(segs[(idx + 1) % len(segs)]["task_text"])

            return sub_swap_fn
        if mode == "sub_blank":
            return lambda step, text: BLANK_TEXT
        return None  # hl_swap does not touch the sub-instruction

    results: dict[str, Any] = {"args": vars(args).copy(), "arms": {}}
    results["args"]["embodiment_tag"] = str(args.embodiment_tag)

    for label, model_path in zip(labels, args.model_paths):
        logger.info(f"=== Arm {label}: {model_path} ===")
        policy = Gr00tPolicy(
            embodiment_tag=args.embodiment_tag,
            model_path=model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        policy.model.action_head.num_inference_timesteps = args.denoising_steps

        arm: dict[str, Any] = {"model_path": model_path, "trajs": {}}
        for traj_id in args.traj_ids:
            if traj_id >= len(dataset):
                continue
            hl = hl_from_meta.get(traj_id)

            torch.manual_seed(args.seed + traj_id)
            z0_ref: list[np.ndarray] = []
            pred_ref = rollout_with_text(
                policy, dataset, traj_id, args.embodiment_tag, action_keys,
                args.steps, args.execution_horizon, hl, None, z0_ref,
            )
            # Noise floor: same text, fresh source draw.
            torch.manual_seed(args.seed + traj_id + 500000)
            pred_floor = rollout_with_text(
                policy, dataset, traj_id, args.embodiment_tag, action_keys,
                args.steps, args.execution_horizon, hl, None, None,
            )
            floor = float(np.mean(np.abs(pred_ref - pred_floor)))

            gt = extract_gt(dataset[traj_id], action_keys, len(pred_ref))
            live_dims, _ = split_dead_dims(gt, 1e-3)

            entry: dict[str, Any] = {"floor": floor, "modes": {}}
            for mode in args.perturb_modes:
                if mode == "hl_swap":
                    if not hl or len(hl_pool) < 2:
                        continue
                    i0 = hl_pool.index(hl)
                    hl_pert = hl_pool[(i0 + 3) % len(hl_pool)]
                    if hl_pert == hl:
                        hl_pert = hl_pool[(i0 + 1) % len(hl_pool)]
                    sub_fn = None
                else:
                    hl_pert = hl
                    sub_fn = make_sub_fn(mode, traj_id)
                torch.manual_seed(args.seed + traj_id)
                z0_pert: list[np.ndarray] = []
                pred_pert = rollout_with_text(
                    policy, dataset, traj_id, args.embodiment_tag, action_keys,
                    args.steps, args.execution_horizon, hl_pert, sub_fn, z0_pert,
                )
                delta = np.abs(pred_pert - pred_ref)  # (T, D)
                effect = float(delta.mean())
                effect_eff = (
                    float(delta[:, live_dims].mean()) if len(live_dims) else None
                )
                z0_delta = None
                if z0_ref and z0_pert and len(z0_ref) == len(z0_pert):
                    z0_delta = float(
                        np.mean(
                            np.abs(np.stack(z0_ref) - np.stack(z0_pert))
                        )
                    )
                entry["modes"][mode] = {
                    "effect": effect,
                    "effect_eff": effect_eff,
                    "z0_delta": z0_delta,
                    "ratio_to_floor": (effect / floor) if floor > 0 else None,
                    "ratio_eff_to_floor": (effect_eff / floor)
                    if (floor > 0 and effect_eff is not None)
                    else None,
                }
                np.save(
                    out_dir / f"pred_{label}_traj{traj_id}_{mode}.npy", pred_pert
                )
            np.save(out_dir / f"pred_{label}_traj{traj_id}_ref.npy", pred_ref)
            arm["trajs"][str(traj_id)] = entry
            logger.info(
                f"  traj {traj_id}: floor={floor:.5f} "
                + " ".join(
                    f"{m}={e['effect']:.5f}(×{e['ratio_to_floor']:.2f})"
                    if e["ratio_to_floor"]
                    else f"{m}={e['effect']:.5f}"
                    for m, e in entry["modes"].items()
                )
            )
        results["arms"][label] = arm
        del policy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregate summary.
    summary: dict[str, Any] = {}
    for label, arm in results["arms"].items():
        floors = [t["floor"] for t in arm["trajs"].values()]
        s: dict[str, Any] = {"floor": float(np.mean(floors)), "modes": {}}
        for mode in args.perturb_modes:
            es, rs, zs = [], [], []
            for t in arm["trajs"].values():
                m = t["modes"].get(mode)
                if not m:
                    continue
                es.append(m["effect"])
                if m["ratio_to_floor"]:
                    rs.append(m["ratio_to_floor"])
                if m["z0_delta"] is not None:
                    zs.append(m["z0_delta"])
            if es:
                s["modes"][mode] = {
                    "effect_mean": float(np.mean(es)),
                    "effect_std": float(np.std(es)),
                    "ratio_mean": float(np.mean(rs)) if rs else None,
                    "z0_delta_mean": float(np.mean(zs)) if zs else None,
                }
        summary[label] = s
    results["summary"] = summary
    with open(out_dir / "text_perturb_metrics.json", "w") as f:
        json.dump(results, f, indent=1)
    logger.info(f"metrics -> {out_dir / 'text_perturb_metrics.json'}")

    _render(out_dir, results, args, labels)


def _render(out_dir: Path, results: dict, args: ArgsConfig, labels: list[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Summary bar: effect/floor ratio per arm per mode (line at 1 = noise floor).
    modes = [m for m in args.perturb_modes]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    width = 0.8 / max(len(modes), 1)
    xs = np.arange(len(labels))
    for mi, mode in enumerate(modes):
        vals = []
        for lab in labels:
            m = results["summary"][lab]["modes"].get(mode)
            vals.append(m["ratio_mean"] if m and m["ratio_mean"] else 0.0)
        ax.bar(
            xs + mi * width,
            vals,
            width * 0.9,
            label=mode,
            color=MODE_COLORS.get(mode),
            alpha=0.85,
        )
    ax.axhline(1.0, color="black", ls="--", lw=1.2, label="noise floor (ratio=1)")
    ax.set_xticks(xs + width * (len(modes) - 1) / 2, labels)
    ax.set_ylabel("mean|Δ(text perturb)| / repeat floor")
    ax.set_title(
        "Text-channel liveness: ratio ≈1 ⇒ text nominally dead "
        "(state channel short-circuits it)"
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "text_perturb_summary.png", dpi=150)
    plt.close(fig)
    logger.info(f"summary figure -> {out_dir / 'text_perturb_summary.png'}")


if __name__ == "__main__":
    main(tyro.cli(ArgsConfig))
