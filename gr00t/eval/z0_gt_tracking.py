"""z₀-vs-GT tracking probe (survey §13.95 follow-up, 2026-10-23).

For each checkpoint, runs an open-loop pass over the eval trajectories and
captures the raw memory-encoder output z₀ — surfaced by Gr00tPolicy as
``info["memory_start"]`` (normalized space) at every policy call — then
unnormalizes it through the SAME ``processor.decode_action`` path the policy
uses for its action output, and renders z₀ against the ground-truth action
trajectory for ALL arms on shared panels:

  * ``z0track_traj{i}.png``  — per-dim time series of z₀'s first action
    (chunk index j=0) vs the GT action curve, all arms overlaid.
  * ``z0chunks_traj{i}.png`` — full-horizon z₀ chunk vs GT chunk at snapshot
    steps (episode start + segment starts from the sidecar meta).
  * ``z0_summary.png``       — per-arm bar chart of first-action MSE and
    full-chunk MSE (mean ± std across trajectories).

Why this is cheap and seed-free: z₀ = MemoryEncoder(c) depends ONLY on the
(teacher-forced) observation c — never on the denoising steps, the deployed
β, or the self-fed history buffer. One DiT step therefore suffices and the
curves are exact (deterministic encoder, no sampling).

Cross-architecture use (db20 / MoT lineage): the MoT checkpoint loads only
under the MoT repo's gr00t package. Run the capture stage there with the same
``--out-dir`` (caches are self-describing npz files), then run ``--plot-only``
once — the plot stage needs no checkpoint and no dataset, only the caches.

Example (main repo, 5 DiT arms):

  uv run python gr00t/eval/z0_gt_tracking.py \
    --dataset-path data/synth_mixture_eval_db7 \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
    --model-paths runs/integrated_db7/checkpoint-5000 \
                  runs/integrated_db21_yanchor/checkpoint-5000 \
                  runs/integrated_db22_purez0/checkpoint-5000 \
                  runs/integrated_db23b_beta03/checkpoint-5000 \
                  runs/integrated_db26_beta03_yanchor/checkpoint-5000 \
    --labels db7 db21 db22 db23b db26 \
    --traj-ids 0 1 2 3 4 5 6 7 8 9 --steps 200 \
    --segment-meta-path data/synth_mixture_eval_db7/meta/segment_schedule.jsonl \
    --state-extra-modality-keys target_relative_pose \
    --high-level-task-from-meta \
    --out-dir runs/viz_z0_tracking

  # MoT repo (db20), same out-dir, then plot everything:
  uv run python gr00t/eval/z0_gt_tracking.py ... --model-paths runs/integrated_db20_mot16/checkpoint-5000 --labels db20 ...
  uv run python gr00t/eval/z0_gt_tracking.py --plot-only --out-dir runs/viz_z0_tracking
"""

from __future__ import annotations

import gc
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro

logger = logging.getLogger(__name__)

# Fixed arm palette (up to 8 arms); GT is always black.
ARM_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
]


@dataclass
class ArgsConfig:
    dataset_path: str = "data/synth_mixture_eval_db7"
    """LeRobot eval dataset (also the GT source in capture mode)."""
    embodiment_tag: str = "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"
    model_paths: list[str] = field(default_factory=list)
    """Checkpoints to probe (capture mode). Empty with --plot-only."""
    labels: list[str] | None = None
    traj_ids: list[int] = field(default_factory=lambda: list(range(10)))
    steps: int = 200
    """Max steps per trajectory."""
    sample_stride: int = 2
    """Capture z₀ every N steps (z₀ changes only when the observation does)."""
    denoising_steps: int = 1
    """z₀ is computed BEFORE denoising; 1 step is enough (capture-only speed)."""
    state_extra_modality_keys: str | None = None
    high_level_task_from_meta: bool = False
    segment_meta_path: str | None = None
    """Optional segment sidecar for boundary markers + chunk snapshot steps."""
    max_chunk_snapshots: int = 4
    """Snapshot steps per trajectory for the full-horizon chunk figure."""
    max_chunk_dims: int = 4
    """Dims shown in the chunk-snapshot figure."""
    plot_only: bool = False
    """Skip capture; render figures from existing z0cache_*.npz in out_dir."""
    out_dir: str = "runs/viz_z0_tracking"


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


def _sanitize(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label)


def capture_arm(
    policy,
    loader,
    traj_ids: list[int],
    args: ArgsConfig,
    action_keys: list[str],
    hl_from_meta: dict[int, str],
    snap_steps_by_traj: dict[int, list[int]],
) -> dict[str, np.ndarray]:
    """Run the open-loop z₀ capture for one arm; returns the npz payload."""
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.utils import parse_observation_gr00t

    modality_configs = {k: v for k, v in loader.modality_configs.items() if k != "action"}
    action_horizon = len(loader.modality_configs["action"].delta_indices)
    payload: dict[str, Any] = {}

    for traj_id in traj_ids:
        if traj_id >= len(loader):
            logger.warning(f"Trajectory {traj_id} out of range, skipped")
            continue
        traj = loader[traj_id]
        actual_steps = min(args.steps, len(traj))
        policy.reset()

        sample_ts: list[int] = []
        z0_first_rows: list[np.ndarray] = []
        snap_chunks: dict[int, np.ndarray] = {}
        snap_set = set(snap_steps_by_traj.get(traj_id, []))

        for t in range(0, actual_steps, args.sample_stride):
            data_point = extract_step_data(traj, t, modality_configs, args.embodiment_tag)
            obs: dict[str, Any] = {}
            for k, v in data_point.states.items():
                obs[f"state.{k}"] = v
            for k, v in data_point.images.items():
                obs[f"video.{k}"] = np.array(v)
            high_level_task = hl_from_meta.get(traj_id)
            for language_key in loader.modality_configs["language"].modality_keys:
                obs[language_key] = high_level_task if high_level_task else data_point.text
            parsed_obs = parse_observation_gr00t(obs, loader.modality_configs)
            if high_level_task:
                parsed_obs["sub_instruction"] = [data_point.text]

            _action_chunk, info = policy.get_action(parsed_obs)
            if "memory_start" not in info:
                raise KeyError(
                    "policy.get_action info lacks 'memory_start' — this checkpoint "
                    "does not surface z₀ (memory encoder absent or old policy code)"
                )
            z0_norm = np.asarray(info["memory_start"], dtype=np.float32)  # (1, H, D)
            assert z0_norm.ndim == 3 and z0_norm.shape[0] == 1, (
                f"unexpected memory_start shape {z0_norm.shape}"
            )

            # Unnormalize through the identical path the policy uses for its
            # action output (relative-action conversion included).
            batched_states = {
                k: np.asarray(v, dtype=np.float32)[None]
                for k, v in data_point.states.items()
            }
            dec = policy.processor.decode_action(
                z0_norm, args.embodiment_tag, batched_states
            )
            z0_chunk = np.concatenate(
                [np.asarray(dec[key], dtype=np.float32)[0] for key in action_keys],
                axis=-1,
            )  # (H, D_total), physical units — comparable to extract_gt output

            sample_ts.append(t)
            z0_first_rows.append(z0_chunk[0])
            # Nearest sampled step to each requested snapshot step.
            for s in list(snap_set):
                if abs(t - s) < args.sample_stride and s not in snap_chunks:
                    snap_chunks[s] = z0_chunk

        payload[f"sample_ts_traj{traj_id}"] = np.asarray(sample_ts, dtype=np.int64)
        payload[f"z0first_traj{traj_id}"] = np.stack(z0_first_rows)  # (N, D)
        if snap_chunks:
            ss = sorted(snap_chunks)
            payload[f"snap_steps_traj{traj_id}"] = np.asarray(ss, dtype=np.int64)
            payload[f"snap_chunks_traj{traj_id}"] = np.stack(
                [snap_chunks[s] for s in ss]
            )  # (S, H, D)
        logger.info(
            f"  traj {traj_id}: captured {len(sample_ts)} z₀ samples "
            f"({len(snap_chunks)} chunk snapshots), H={action_horizon}"
        )
    return payload


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def render_figures(out_dir: Path, caches: dict[str, dict[str, Any]], traj_ids: list[int],
                   max_chunk_dims: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(caches)
    colors = {lab: ARM_COLORS[i % len(ARM_COLORS)] for i, lab in enumerate(labels)}

    # Arm-independent payload pieces come from the first cache.
    first = caches[labels[0]]
    dim_labels = [s.decode() if isinstance(s, bytes) else str(s) for s in first["dim_labels"]]
    n_dims = len(dim_labels)

    def gt_of(traj_id: int) -> np.ndarray:
        return first[f"gt_traj{traj_id}"]  # (T, D) physical units

    def seg_of(traj_id: int) -> np.ndarray:
        k = f"seg_starts_traj{traj_id}"
        return first[k] if k in first else np.asarray([], dtype=np.int64)

    # ---- per-arm scalar metrics (for legends + summary) ----
    first_mse: dict[str, list[float]] = {lab: [] for lab in labels}
    chunk_mse: dict[str, list[float]] = {lab: [] for lab in labels}
    for lab in labels:
        c = caches[lab]
        for traj_id in traj_ids:
            gt = gt_of(traj_id)
            ts = c[f"sample_ts_traj{traj_id}"]
            z0f = c[f"z0first_traj{traj_id}"]  # (N, D)
            gt_rows = gt[np.minimum(ts, len(gt) - 1)]
            first_mse[lab].append(float(np.mean((z0f - gt_rows) ** 2)))
            sk, sc = f"snap_steps_traj{traj_id}", f"snap_chunks_traj{traj_id}"
            if sk in c:
                H = c[sc].shape[1]
                for s, ch in zip(c[sk], c[sc]):
                    gt_chunk = gt[min(int(s), len(gt) - 1):][:H]
                    if len(gt_chunk) < H:
                        pad = np.repeat(gt_chunk[-1:], H - len(gt_chunk), axis=0)
                        gt_chunk = np.concatenate([gt_chunk, pad], axis=0)
                    chunk_mse[lab].append(float(np.mean((ch - gt_chunk) ** 2)))

    # ---- figure 1: per-dim time series, all arms overlaid ----
    ncols = 3
    nrows = int(np.ceil(n_dims / ncols))
    for traj_id in traj_ids:
        gt = gt_of(traj_id)
        segs = seg_of(traj_id)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5.2 * ncols, 2.4 * nrows), sharex=True
        )
        axes = np.atleast_1d(axes).ravel()
        for d in range(n_dims):
            ax = axes[d]
            ax.plot(gt[:, d], color="black", lw=1.8, alpha=0.9, label="GT")
            for lab in labels:
                c = caches[lab]
                ts = c[f"sample_ts_traj{traj_id}"]
                z0f = c[f"z0first_traj{traj_id}"]
                ax.plot(ts, z0f[:, d], color=colors[lab], lw=1.0, alpha=0.85, label=lab)
            for s in segs:
                ax.axvline(int(s), color="gray", ls="--", lw=0.7, alpha=0.6)
            ax.set_title(dim_labels[d], fontsize=9)
            ax.grid(alpha=0.25)
        for d in range(n_dims, len(axes)):
            axes[d].axis("off")
        handles, leg = axes[0].get_legend_handles_labels()
        fig.legend(handles, leg, loc="upper right", fontsize=9, ncol=len(labels) + 1)
        fig.suptitle(
            f"z₀ first-action tracking vs GT — traj {traj_id} "
            f"(dashed = segment starts; z₀ = MemoryEncoder(c), deterministic)",
            fontsize=12,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(out_dir / f"z0track_traj{traj_id}.png", dpi=150)
        plt.close(fig)

    # ---- figure 2: full-horizon chunk snapshots ----
    dims_show = list(range(min(max_chunk_dims, n_dims)))
    for traj_id in traj_ids:
        sk = f"snap_steps_traj{traj_id}"
        if sk not in first:
            continue
        snap_steps = [int(s) for s in first[sk]]
        gt = gt_of(traj_id)
        nrows2 = len(snap_steps)
        ncols2 = len(dims_show)
        fig, axes = plt.subplots(
            nrows2, ncols2, figsize=(4.2 * ncols2, 2.6 * nrows2), squeeze=False
        )
        for r, s in enumerate(snap_steps):
            H = first[f"snap_chunks_traj{traj_id}"].shape[1]
            gt_chunk = gt[s:][:H]
            if len(gt_chunk) < H:
                pad = np.repeat(gt_chunk[-1:], H - len(gt_chunk), axis=0)
                gt_chunk = np.concatenate([gt_chunk, pad], axis=0)
            for cidx, d in enumerate(dims_show):
                ax = axes[r][cidx]
                ax.plot(gt_chunk[:, d], color="black", lw=1.8, label="GT chunk")
                for lab in labels:
                    c = caches[lab]
                    if sk not in c:
                        continue
                    ss = [int(x) for x in c[sk]]
                    if s not in ss:
                        continue
                    ch = c[f"snap_chunks_traj{traj_id}"][ss.index(s)]
                    ax.plot(ch[:, d], color=colors[lab], lw=1.1, label=lab)
                ax.set_title(f"{dim_labels[d]} @ step {s}", fontsize=9)
                ax.grid(alpha=0.25)
        handles, leg = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, leg, loc="upper right", fontsize=9, ncol=len(labels) + 1)
        fig.suptitle(
            f"z₀ full-horizon chunk vs GT chunk — traj {traj_id}", fontsize=12
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(out_dir / f"z0chunks_traj{traj_id}.png", dpi=150)
        plt.close(fig)

    # ---- figure 3: summary bars ----
    fig, axes = plt.subplots(1, 2, figsize=(6.4 * 2, 4.2))
    for ax, metric, title in (
        (axes[0], first_mse, "z₀ first-action MSE vs GT (mean ± std over trajs)"),
        (axes[1], chunk_mse, "z₀ full-chunk MSE vs GT (mean ± std over snapshots)"),
    ):
        means = [np.mean(metric[lab]) for lab in labels]
        stds = [np.std(metric[lab]) for lab in labels]
        ax.bar(labels, means, yerr=stds, capsize=4,
               color=[colors[lab] for lab in labels], alpha=0.85)
        for i, m in enumerate(means):
            ax.text(i, m, f"{m:.4f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25, axis="y")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_dir / "z0_summary.png", dpi=150)
    plt.close(fig)
    logger.info(f"figures written to {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: ArgsConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _fh = logging.FileHandler(out_dir / "log.txt", mode="a", encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logging.getLogger().addHandler(_fh)

    if args.plot_only:
        caches = _load_caches(out_dir)
        traj_ids = _caches_traj_ids(caches)
        render_figures(out_dir, caches, traj_ids, args.max_chunk_dims)
        return

    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.eval.compare_flow_sources import extract_gt
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    args.embodiment_tag = EmbodimentTag.resolve(args.embodiment_tag)
    if args.state_extra_modality_keys:
        from gr00t.configs.data.state_extra_keys import (
            apply_state_extra_keys,
            parse_extra_keys,
        )

        apply_state_extra_keys(
            args.embodiment_tag.value, parse_extra_keys(args.state_extra_modality_keys)
        )

    assert args.model_paths, "capture mode requires --model-paths"
    labels = args.labels or [
        Path(p).parent.name + "/" + Path(p).name for p in args.model_paths
    ]
    assert len(labels) == len(args.model_paths), "labels must match model_paths length"

    # Modality config from the first checkpoint (arms share the data config).
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
    # Per-key action widths are inferred from the data itself (ModalityConfig
    # carries no dim metadata).
    first_traj = dataset[args.traj_ids[0]]
    dim_labels = np.asarray(
        [
            f"{key}.{i}"
            for key in action_keys
            for i in range(
                np.asarray(first_traj[f"action.{key}"].iloc[0], dtype=np.float32).shape[-1]
            )
        ]
    )

    hl_from_meta: dict[int, str] = {}
    if args.high_level_task_from_meta:
        meta_path = Path(args.dataset_path) / "meta" / "episodes.jsonl"
        with open(meta_path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("high_level_task"):
                    hl_from_meta[int(rec["episode_index"])] = rec["high_level_task"]
        logger.info(f"high_level_task_from_meta: loaded {len(hl_from_meta)} episode tasks")

    # GT + segment starts + snapshot steps (arm-independent).
    gt_payload: dict[str, Any] = {"dim_labels": dim_labels}
    snap_steps_by_traj: dict[int, list[int]] = {}
    segs_by_traj: dict[int, list[int]] = {}
    if args.segment_meta_path:
        with open(args.segment_meta_path) as f:
            for line in f:
                rec = json.loads(line)
                segs_by_traj.setdefault(int(rec["episode_index"]), []).append(
                    int(rec["start"])
                )
    for traj_id in args.traj_ids:
        if traj_id >= len(dataset):
            continue
        traj = dataset[traj_id]
        actual_steps = min(args.steps, len(traj))
        gt_payload[f"gt_traj{traj_id}"] = extract_gt(traj, action_keys, actual_steps)
        starts = sorted(s for s in segs_by_traj.get(traj_id, []) if s < actual_steps)
        gt_payload[f"seg_starts_traj{traj_id}"] = np.asarray(starts, dtype=np.int64)
        # Snapshot steps: episode start + segment starts, capped.
        snaps = sorted({0, *starts})[: args.max_chunk_snapshots]
        if len(snaps) < args.max_chunk_snapshots:
            extra = np.linspace(0, actual_steps - 1, args.max_chunk_snapshots + 2)[1:-1]
            snaps = sorted({*snaps, *(int(e) for e in extra)})[: args.max_chunk_snapshots]
        snap_steps_by_traj[traj_id] = snaps

    for label, model_path in zip(labels, args.model_paths):
        logger.info(f"=== Arm {label}: {model_path} ===")
        policy = Gr00tPolicy(
            embodiment_tag=args.embodiment_tag,
            model_path=model_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        policy.model.action_head.num_inference_timesteps = args.denoising_steps
        payload = capture_arm(
            policy, dataset, args.traj_ids, args, action_keys,
            hl_from_meta, snap_steps_by_traj,
        )
        payload.update(gt_payload)
        cache_path = out_dir / f"z0cache_{_sanitize(label)}.npz"
        np.savez_compressed(cache_path, **payload)
        logger.info(f"saved {cache_path}")
        del policy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    caches = _load_caches(out_dir)
    render_figures(out_dir, caches, _caches_traj_ids(caches), args.max_chunk_dims)


def _load_caches(out_dir: Path) -> dict[str, dict[str, Any]]:
    caches: dict[str, dict[str, Any]] = {}
    for p in sorted(out_dir.glob("z0cache_*.npz")):
        data = np.load(p, allow_pickle=False)
        label = p.stem[len("z0cache_"):]
        caches[label] = {k: data[k] for k in data.files}
    if not caches:
        raise FileNotFoundError(f"no z0cache_*.npz under {out_dir}")
    return caches


def _caches_traj_ids(caches: dict[str, dict[str, Any]]) -> list[int]:
    first = caches[next(iter(caches))]
    ids = []
    for k in first:
        m = re.fullmatch(r"gt_traj(\d+)", k)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


if __name__ == "__main__":
    main(tyro.cli(ArgsConfig))
