#!/usr/bin/env python
"""merge_lerobot_datasets.py — 合并 N 个 LeRobot v2 数据集为一个（MoS 合并冠军数据臂）。

功能（对应"17D 统一 + 索引重编 + stats 重算"三需求）：

1. **状态维度统一**：`--drop-state-keys` 按 modality.json 的 start/end 切片
   observation.state（如从合成 22D 丢弃 target_relative_pose → 17D），
   modality.json / info.json features 同步更新。
2. **索引重编**：episode_index 全局重编号（按输入顺序偏移）；task 表全局重建
   （按任务字符串去重、先到先得），逐帧 parquet 的 task_index 列随之重映射；
   episodes.jsonl、segment_schedule.jsonl（若存在）同步偏移。
3. **stats 重算**：合并完成后直接调用仓库自带的
   `gr00t.data.stats.generate_stats / generate_rel_stats`（与下载器/构建器同一公式），
   不重新发明统计语义。

前置断言：各源 fps 一致、视频键集合一致、action 模态键一致、单 chunk（chunk-000）。
用法：

    uv run python scripts/merge_lerobot_datasets.py \
      --inputs data/droid_subset_train data/synth_mixture_train_db7 \
      --drop-state-keys target_relative_pose \
      --output-dir data/mix_droid_synth_train

    # 快速自检（不写视频）：
    uv run python scripts/merge_lerobot_datasets.py \
      --inputs data/droid_subset_eval data/synth_mixture_eval_db7 \
      --drop-state-keys target_relative_pose \
      --output-dir data/mix_droid_synth_eval --skip-videos
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

logger = logging.getLogger("merge_lerobot")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_json(p: Path):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(p: Path):
    if not p.is_file():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(p: Path, records):
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def state_keep_dims(modality: dict, drop_keys: list[str]) -> tuple[np.ndarray, list[str]]:
    """按 modality.json state 段计算保留维索引与保留键名。"""
    keep_idx, keep_keys = [], []
    for key, spec in modality["state"].items():
        s, e = spec["start"], spec["end"]
        if key in drop_keys:
            continue
        keep_idx.extend(range(s, e))
        keep_keys.append(key)
    return np.array(keep_idx, dtype=int), keep_keys


def compact_state_modality(modality: dict, drop_keys: list[str]) -> dict:
    """drop 后重排 state 段 start/end 使其连续（eef_9d 0-9, grip 9-10, joint 10-17 ...）。

    仅当源条目显式携带 ``original_key`` 时才透传——缺省语义是"从默认矩阵列
    （observation.state/action）切片"；擅自补 original_key=<组名> 会让
    LeRobotEpisodeLoader._extract_joint_groups 去找不存在的按组名列，报 KeyError。
    """
    out, cursor = {}, 0
    for key, spec in modality["state"].items():
        if key in drop_keys:
            continue
        n = spec["end"] - spec["start"]
        entry = {"start": cursor, "end": cursor + n}
        if "original_key" in spec:
            entry["original_key"] = spec["original_key"]
        out[key] = entry
        cursor += n
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True, help="源数据集目录（按此顺序偏移 episode_index）")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--drop-state-keys", nargs="*", default=[], help="从 observation.state 丢弃的模态键")
    ap.add_argument("--embodiment-tag", default="OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT",
                    help="传给 generate_rel_stats（须与训练用 tag 一致）")
    ap.add_argument("--skip-videos", action="store_true", help="只合并 meta+parquet（快速自检）")
    ap.add_argument("--skip-stats", action="store_true",
                    help="跳过 stats 重算（本地无项目 venv/torch 时自检用；服务器正式跑勿加）")
    ap.add_argument("--move", action="store_true", help="视频用硬链接替代复制（同盘省空间）")
    args = ap.parse_args()

    srcs = [Path(p) for p in args.inputs]
    out = Path(args.output_dir)
    if out.exists():
        logger.error("输出目录已存在（防覆盖）：%s", out)
        return 1
    for s in srcs:
        for rel in ["meta/info.json", "meta/modality.json", "meta/tasks.jsonl", "meta/episodes.jsonl"]:
            if not (s / rel).is_file():
                logger.error("源缺文件：%s/%s", s, rel)
                return 1

    # ── 前置一致性断言 ────────────────────────────────────────────────
    infos = [load_json(s / "meta/info.json") for s in srcs]
    mods = [load_json(s / "meta/modality.json") for s in srcs]
    fps_set = {i["fps"] for i in infos}
    assert len(fps_set) == 1, f"fps 不一致: {fps_set}"
    video_keys = [sorted(m["video"].keys()) for m in mods]
    assert all(v == video_keys[0] for v in video_keys), f"视频键集合不一致: {video_keys}"
    action_keys = [sorted(m["action"].keys()) for m in mods]
    assert all(a == action_keys[0] for a in action_keys), f"action 键不一致: {action_keys}"
    for m in mods:
        missing = set(args.drop_state_keys) - set(m["state"].keys())
        if missing and m is not mods[0]:
            pass  # 某些源本来就没有这些键（如真机 17D），允许
    logger.info("断言通过：fps=%d, video=%s, action keys 一致", infos[0]["fps"], video_keys[0])

    # ── 全局 task 表（字符串去重，先到先得） ─────────────────────────
    global_tasks: dict[str, int] = {}
    src_task_maps: list[dict[int, int]] = []
    for s in srcs:
        old_tasks = {r["task_index"]: r["task"] for r in load_jsonl(s / "meta/tasks.jsonl")}
        old2new = {}
        for old_idx, text in sorted(old_tasks.items()):
            if text not in global_tasks:
                global_tasks[text] = len(global_tasks)
            old2new[old_idx] = global_tasks[text]
        src_task_maps.append(old2new)
    logger.info("全局 task 表：%d 项（各源 %s）", len(global_tasks), [len(m) for m in src_task_maps])

    # ── 逐源逐集复制 + 重映射 ────────────────────────────────────────
    (out / "data/chunk-000").mkdir(parents=True)
    merged_episodes, merged_schedule = [], []
    ep_offset, frame_total = 0, 0
    for si, s in enumerate(srcs):
        episodes = sorted(load_jsonl(s / "meta/episodes.jsonl"), key=lambda r: r["episode_index"])
        old2new_task = src_task_maps[si]
        keep_idx, _ = state_keep_dims(mods[si], args.drop_state_keys)
        for rec in episodes:
            old_idx, new_idx = rec["episode_index"], ep_offset + rec["episode_index"]
            pq_src = s / f"data/chunk-000/episode_{old_idx:06d}.parquet"
            df = pd.read_parquet(pq_src)
            # task_index 重映射
            if "task_index" in df.columns:
                df["task_index"] = df["task_index"].map(old2new_task).astype(np.int64)
            # state 切片
            state_arr = np.stack(df["observation.state"].map(np.asarray).values)
            if state_arr.shape[1] != len(keep_idx):
                df["observation.state"] = list(state_arr[:, keep_idx].astype(np.float32))
            df.to_parquet(out / f"data/chunk-000/episode_{new_idx:06d}.parquet")
            # 视频
            if not args.skip_videos:
                for vkey in video_keys[0]:
                    full_key = vkey if vkey.startswith("observation.images") else f"observation.images.{vkey}"
                    vdir_src = s / f"videos/chunk-000/{full_key}"
                    vdir_dst = out / f"videos/chunk-000/{full_key}"
                    vdir_dst.mkdir(parents=True, exist_ok=True)
                    vsrc = vdir_src / f"episode_{old_idx:06d}.mp4"
                    if vsrc.is_file():
                        if args.move:
                            try:
                                (vdir_dst / f"episode_{new_idx:06d}.mp4").hardlink_to(vsrc)
                            except OSError:
                                shutil.copy(vsrc, vdir_dst / f"episode_{new_idx:06d}.mp4")
                        else:
                            shutil.copy(vsrc, vdir_dst / f"episode_{new_idx:06d}.mp4")
                    else:
                        logger.warning("缺视频：%s", vsrc)
            new_rec = dict(rec)
            new_rec["episode_index"] = new_idx
            merged_episodes.append(new_rec)
            frame_total += rec["length"]
        # segment_schedule.jsonl（合成集分层评估用）
        sched_path = s / "meta/segment_schedule.jsonl"
        if sched_path.is_file():
            for r in load_jsonl(sched_path):
                r = dict(r)
                r["episode_index"] = ep_offset + r["episode_index"]
                merged_schedule.append(r)
        logger.info("源 %s：%d 集并入（偏移 %d）", s, len(episodes), ep_offset)
        ep_offset += len(episodes)

    # ── meta 写盘 ────────────────────────────────────────────────────
    meta = out / "meta"
    meta.mkdir(exist_ok=True)
    write_jsonl(meta / "episodes.jsonl", merged_episodes)
    write_jsonl(meta / "tasks.jsonl",
                [{"task_index": i, "task": t} for t, i in sorted(global_tasks.items(), key=lambda kv: kv[1])])
    if merged_schedule:
        write_jsonl(meta / "segment_schedule.jsonl", merged_schedule)

    new_mod = dict(mods[0])
    new_mod["state"] = compact_state_modality(mods[0], args.drop_state_keys)
    with open(meta / "modality.json", "w") as f:
        json.dump(new_mod, f, indent=2)

    new_info = dict(infos[0])
    new_state_dim = sum(spec["end"] - spec["start"] for spec in new_mod["state"].values())
    if "observation.state" in new_info.get("features", {}):
        new_info["features"]["observation.state"]["shape"] = [new_state_dim]
        if "names" in new_info["features"]["observation.state"]:
            names = new_info["features"]["observation.state"]["names"]
            _, keep_keys = state_keep_dims(mods[0], args.drop_state_keys)
            keep_names = [n for k in keep_keys for n in
                          (names[mods[0]["state"][k]["start"]:mods[0]["state"][k]["end"]]
                           if len(names) >= mods[0]["state"][k]["end"] else [k])]
            new_info["features"]["observation.state"]["names"] = keep_names
    new_info.update({
        "total_episodes": len(merged_episodes),
        "total_frames": frame_total,
        "total_videos": len(merged_episodes) * len(video_keys[0]),
        "total_tasks": len(global_tasks),
        "total_chunks": 1,
        "splits": {"train": f"0:{len(merged_episodes)}"},
    })
    with open(meta / "info.json", "w") as f:
        json.dump(new_info, f, indent=2)
    logger.info("meta 写盘完成：%d 集 / %d 帧 / state %dD", len(merged_episodes), frame_total, new_state_dim)

    # ── stats 重算（仓库自带公式） ───────────────────────────────────
    if not args.skip_stats:
        from gr00t.data.stats import generate_rel_stats, generate_stats
        from gr00t.data.embodiment_tags import EmbodimentTag

        logger.info("重算 stats.json / relative_stats.json ...")
        generate_stats(out)
        generate_rel_stats(out, EmbodimentTag.resolve(args.embodiment_tag))
    else:
        logger.info("--skip-stats：跳过 stats 重算（服务器正式跑必须重算）")

    # ── 验证门 ───────────────────────────────────────────────────────
    ok = True

    def check(name, cond):
        nonlocal ok
        logger.info("  [%s] %s", "PASS" if cond else "FAIL", name)
        ok = ok and bool(cond)

    logger.info("── validation ──")
    for rec in merged_episodes[:: max(1, len(merged_episodes) // 20)]:
        df = pd.read_parquet(out / f"data/chunk-000/episode_{rec['episode_index']:06d}.parquet")
        check(f"ep {rec['episode_index']} 帧数={len(df)}=={rec['length']}", len(df) == rec["length"])
        check(f"ep {rec['episode_index']} state={np.asarray(df['observation.state'].iloc[0]).shape[0]}D=={new_state_dim}D",
              np.asarray(df["observation.state"].iloc[0]).shape[0] == new_state_dim)
        if "task_index" in df.columns:
            check(f"ep {rec['episode_index']} task_index 在界内", df["task_index"].max() < len(global_tasks))
    # task 文本抽查：第二源首集的任务字符串应与原数据集一致
    if len(srcs) > 1:
        src1_eps = load_jsonl(srcs[1] / "meta/episodes.jsonl")
        first_second_old = src1_eps[0]["episode_index"]
        first_second_new = len(load_jsonl(srcs[0] / "meta/episodes.jsonl")) + first_second_old
        old_texts = set(src1_eps[0].get("tasks", []))
        df = pd.read_parquet(out / f"data/chunk-000/episode_{first_second_new:06d}.parquet")
        inv = {v: k for k, v in global_tasks.items()}
        new_texts = {inv[int(t)] for t in df["task_index"].unique()}
        check(f"第二源首集任务文本保持（{sorted(new_texts)[:2]} vs 原文 {sorted(old_texts)[:2]}）",
              (not old_texts) or bool(old_texts & new_texts))
    if not args.skip_videos:
        n_vid = len(list(out.glob("videos/chunk-000/*/episode_*.mp4")))
        check(f"视频文件数 {n_vid} == {len(merged_episodes) * len(video_keys[0])}",
              n_vid == len(merged_episodes) * len(video_keys[0]))
    check("meta/stats.json 存在", args.skip_stats or (meta / "stats.json").is_file())
    check("relative_stats 存在", args.skip_stats or any("relative" in p.name for p in meta.glob("*.json")))

    logger.info("── validation %s ──", "PASSED" if ok else "FAILED")
    logger.info("合并完成：%s（%d 集 / %d 帧 / state %dD）。训练时不要再传 --state-extra-modality-keys；"
                "可加 --expected-state-dim %d 作断言。", out, len(merged_episodes), frame_total,
                new_state_dim, new_state_dim)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
