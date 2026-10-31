"""visualize_merged_dataset.py — 合并数据集可视化验收（MoS Phase-0 质检门）。

纯 pandas/numpy/matplotlib/cv2 实现，不依赖 gr00t 包，任何环境可跑。

检查内容：
  1. 轨迹图：抽样 episode 的 action / observation.state 17 维分组曲线
     （eef_xyz / eef_rot6d / gripper / joints 四联图；有段表则画段界与 teleport 点）
  2. 视频帧网格：两个相机各抽 K 帧，验证视频完整且与 parquet 对齐
  3. stats 总览：stats.json 的 mean±std 条形图；relative_stats.json 的
     eef_9d q01–q99 带（归一化区间合理性）
  4. 数值质检：维度、NaN/Inf、task 文本、episode 计数 vs info.json

用法：
  uv run python scripts/visualize_merged_dataset.py \
    --dataset-path data/mix_droid_synth_train \
    --boundary 300 \
    --output-dir runs/viz_mix_train

  --boundary N：域边界（源 B 的第一集 index），auto 抽样会取
  [0, N-1, N, 最后一集]；不传则均匀抽 4 集。
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("viz_merged")

# 17D 布局分组（与 modality.json 的 eef_9d/gripper_position/joint_position 对应）
GROUPS = [
    ("eef_xyz (0-2)", [0, 1, 2]),
    ("eef_rot6d (3-8)", [3, 4, 5, 6, 7, 8]),
    ("gripper (9)", [9]),
    ("joints (10-16)", [10, 11, 12, 13, 14, 15, 16]),
]


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def extract_frames(video_path: Path, frame_indices: list[int]) -> list[np.ndarray]:
    """cv2 抽帧；失败返回空列表（调用方记 WARN）。"""
    try:
        import cv2
    except ImportError:
        return []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def plot_trajectory(mat: np.ndarray, title: str, out_path: Path,
                    segments: list[dict] | None = None) -> None:
    """17D 矩阵画四联分组曲线；segments 画段界竖线。"""
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    for ax, (gname, dims) in zip(axes, GROUPS):
        for d in dims:
            ax.plot(mat[:, d], linewidth=0.8, label=f"dim{d}")
        ax.set_ylabel(gname, fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, ncol=min(len(dims), 7), loc="upper right")
        if segments:
            ymin, ymax = ax.get_ylim()
            for seg in segments:
                color = "red" if seg.get("teleported") else "gray"
                ax.axvline(seg["start"], color=color, linestyle="--", alpha=0.5, linewidth=0.8)
                ax.text(seg["start"], ymax, seg.get("kind", "")[:6],
                        fontsize=6, rotation=90, va="top", color=color)
            last_end = segments[-1]["end"]
            ax.axvline(last_end, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    axes[-1].set_xlabel("frame")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_frames_grid(cam_frames: dict[str, list[np.ndarray]], title: str,
                     out_path: Path) -> bool:
    cams = [c for c, fs in cam_frames.items() if fs]
    if not cams:
        return False
    k = max(len(cam_frames[c]) for c in cams)
    fig, axes = plt.subplots(len(cams), k, figsize=(2.2 * k, 2.2 * len(cams)))
    if len(cams) == 1:
        axes = [axes]
    for row, cam in enumerate(cams):
        for col in range(k):
            ax = axes[row][col]
            if col < len(cam_frames[cam]):
                ax.imshow(cam_frames[cam][col])
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(cam.split(".")[-1], fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return True


def plot_stats_overview(stats: dict, out_path: Path) -> None:
    panels = [k for k in ["observation.state", "action"] if k in stats]
    if not panels:
        return
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 4 * len(panels)))
    if len(panels) == 1:
        axes = [axes]
    for ax, key in zip(axes, panels):
        mean = np.asarray(stats[key]["mean"], dtype=float).ravel()
        std = np.asarray(stats[key]["std"], dtype=float).ravel()
        x = np.arange(len(mean))
        ax.bar(x, mean, yerr=std, capsize=2, alpha=0.75)
        ax.set_title(f"{key}: per-dim mean +/- std", fontsize=10)
        ax.set_xlabel("dim"); ax.grid(alpha=0.3, axis="y")
        ax.set_xticks(x)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_rel_stats(rel: dict, out_path: Path) -> None:
    if "eef_9d" not in rel:
        return
    q01 = np.asarray(rel["eef_9d"]["q01"])
    q99 = np.asarray(rel["eef_9d"]["q99"])
    mean = np.asarray(rel["eef_9d"]["mean"])
    if q01.ndim == 1:
        q01, q99, mean = q01[None], q99[None], mean[None]
    n_dim = q01.shape[-1]
    ncol = 3
    nrow = int(np.ceil(n_dim / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 3 * nrow), sharex=True)
    for d in range(n_dim):
        ax = axes[d // ncol][d % ncol]
        ax.fill_between(range(len(mean)), q01[:, d], q99[:, d], alpha=0.3, label="q01-q99")
        ax.plot(mean[:, d], linewidth=1.0, label="mean")
        ax.axhline(0, color="k", linewidth=0.5, alpha=0.5)
        ax.set_title(f"eef_9d dim{d}", fontsize=9)
        ax.grid(alpha=0.3)
        if d == 0:
            ax.legend(fontsize=7)
    for d in range(n_dim, nrow * ncol):
        axes[d // ncol][d % ncol].axis("off")
    fig.suptitle("relative_stats eef_9d: q01-q99 band across action horizon", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--episode-ids", nargs="*", type=int, default=None)
    ap.add_argument("--boundary", type=int, default=None,
                    help="域边界（源 B 首集 index），auto 抽样取 [0, N-1, N, 末集]")
    ap.add_argument("--frames-per-cam", type=int, default=6)
    args = ap.parse_args()

    root = Path(args.dataset_path)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    warns, fails = [], []

    # ── meta 载入 ─────────────────────────────────────────────────────
    info = json.loads((root / "meta/info.json").read_text())
    modality = json.loads((root / "meta/modality.json").read_text())
    episodes = {e["episode_index"]: e for e in load_jsonl(root / "meta/episodes.jsonl")}
    n_ep = info["total_episodes"]
    fps = info["fps"]
    state_dim = sum(v["end"] - v["start"] for v in modality["state"].values())
    action_dim = sum(v["end"] - v["start"] for v in modality["action"].values())
    logger.info("dataset: %s | episodes=%d fps=%d state=%dD action=%dD",
                root, n_ep, fps, state_dim, action_dim)

    if len(episodes) != n_ep:
        fails.append(f"episodes.jsonl 条数 {len(episodes)} != info.total_episodes {n_ep}")

    # ── 抽样 episode ──────────────────────────────────────────────────
    if args.episode_ids:
        sample = sorted(set(i for i in args.episode_ids if 0 <= i < n_ep))
    elif args.boundary is not None:
        sample = sorted({0, args.boundary - 1, args.boundary, n_ep - 1} - {-1})
    else:
        sample = sorted({int(x) for x in np.linspace(0, n_ep - 1, 4)})
    logger.info("sampled episodes: %s", sample)

    segments_by_ep: dict[int, list[dict]] = {}
    for seg in load_jsonl(root / "meta/segment_schedule.jsonl"):
        segments_by_ep.setdefault(seg["episode_index"], []).append(seg)

    summary = [f"dataset={root} episodes={n_ep} fps={fps} state={state_dim}D action={action_dim}D",
               f"sampled={sample}", ""]

    # ── 逐集检查与绘图 ────────────────────────────────────────────────
    for ep in sample:
        pq = root / f"data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet"
        if not pq.is_file():
            fails.append(f"episode {ep}: parquet 缺失 {pq}")
            continue
        df = pd.read_parquet(pq)
        state = np.stack(df["observation.state"].map(np.asarray).values)
        action = np.stack(df["action"].map(np.asarray).values)
        task = "; ".join(episodes.get(ep, {}).get("tasks", [])) or "(empty)"

        if state.shape[1] != state_dim:
            fails.append(f"episode {ep}: state 维 {state.shape[1]} != {state_dim}")
        if action.shape[1] != action_dim:
            fails.append(f"episode {ep}: action 维 {action.shape[1]} != {action_dim}")
        for name, mat in [("state", state), ("action", action)]:
            n_bad = int((~np.isfinite(mat)).sum())
            if n_bad:
                fails.append(f"episode {ep}: {name} 含 {n_bad} 个 NaN/Inf")
        if not task or task == "(empty)":
            warns.append(f"episode {ep}: task 文本为空")
        exp_len = episodes.get(ep, {}).get("length")
        if exp_len is not None and exp_len != len(df):
            warns.append(f"episode {ep}: parquet 帧数 {len(df)} != episodes.jsonl {exp_len}")

        summary.append(f"[ep {ep:06d}] frames={len(df)} state={state.shape[1]}D "
                       f"action={action.shape[1]}D task=\"{task[:80]}\"")
        segs = segments_by_ep.get(ep)
        if segs:
            summary.append(f"    segments={len(segs)} kinds={[s.get('kind') for s in segs]}")
        title = f"ep{ep:06d} | {task[:90]}"
        plot_trajectory(action, f"ACTION | {title}", out / f"ep{ep:06d}_action.png", segs)
        plot_trajectory(state, f"STATE | {title}", out / f"ep{ep:06d}_state.png", segs)

        # 视频帧（注意：modality["video"] 的键是短名 exterior_1_left，
        # 盘上目录名在其 original_key 里 observation.images.exterior_1_left）
        cam_frames = {}
        for cam, vspec in modality.get("video", {}).items():
            dir_name = vspec.get("original_key", cam) if isinstance(vspec, dict) else cam
            vp = root / f"videos/chunk-{ep // 1000:03d}/{dir_name}/episode_{ep:06d}.mp4"
            if not vp.is_file():
                warns.append(f"episode {ep}: 视频缺失 {vp.name} ({cam})")
                continue
            idxs = [int(x) for x in np.linspace(0, len(df) - 1, args.frames_per_cam)]
            fs = extract_frames(vp, idxs)
            if len(fs) < len(idxs):
                warns.append(f"episode {ep}: {cam} 仅解出 {len(fs)}/{len(idxs)} 帧")
            cam_frames[cam.split(".")[-1]] = fs
        if cam_frames:
            plot_frames_grid(cam_frames, f"FRAMES | {title}", out / f"ep{ep:06d}_frames.png")
        summary.append("")

    # ── stats 图 ──────────────────────────────────────────────────────
    stats_path = root / "meta/stats.json"
    if stats_path.is_file():
        plot_stats_overview(json.loads(stats_path.read_text()), out / "stats_overview.png")
        summary.append("stats_overview.png written")
    else:
        fails.append("meta/stats.json 缺失")
    rel_path = root / "meta/relative_stats.json"
    if rel_path.is_file():
        plot_rel_stats(json.loads(rel_path.read_text()), out / "rel_stats_eef9d.png")
        summary.append("rel_stats_eef9d.png written")
    else:
        fails.append("meta/relative_stats.json 缺失")

    # ── 汇总 ──────────────────────────────────────────────────────────
    summary.append("── WARN ──")
    summary.extend(f"  {w}" for w in warns) if warns else summary.append("  (none)")
    summary.append("── FAIL ──")
    summary.extend(f"  {f}" for f in fails) if fails else summary.append("  (none)")
    verdict = "VIS CHECK PASSED" if not fails else "VIS CHECK FAILED"
    summary.append(f"\n== {verdict} (warns={len(warns)}, fails={len(fails)}) ==")
    text = "\n".join(summary)
    (out / "summary.txt").write_text(text)
    logger.info("\n%s", text)
    logger.info("artifacts -> %s", out)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
