---
license: Apache-2.0
task_categories:
  - robotics
multi_modal:
  robotics:
    language:
      - en
language:
  - en
tags:
  - robotics
  - vision-language-action
  - DROID
  - synthetic
  - flow-matching
  - GR00T
configs:
  - config_name: preview
    default: true
    data_files:
      - split: train
        path: "preview/mos_preview.parquet"
---

# MoS Mixed-Start Datasets — 数据卡

本数据仓支撑 MoS（Mixed-Start Flow Matching）在 GR00T N1.7-3B 上的发布权重训练、开环评估和论文复现。真实 DROID 子集与合成多段数据统一为 **LeRobot v2.1、17D 状态／动作、双视角**格式。

- 模型：[GR00T-N1.7-MixedStart](https://www.modelscope.cn/models/joyogzw/GR00T-N1.7-MixedStart)
- 代码与论文：[GUOZHIWEN/GR00T-N1.7-MixedStart](https://github.com/GUOZHIWEN/GR00T-N1.7-MixedStart)
- 完整复现：[REPRODUCE.md](https://github.com/GUOZHIWEN/GR00T-N1.7-MixedStart/blob/main/docs/REPRODUCE.md)

## 📌 固定版本与完整性

数据仓为 `joyogzw/GR00T-N1.7-MixedStart-Datasets`。四个数据归档使用同一不可变 revision，以代码仓 [`DATA_MANIFEST.json`](https://github.com/GUOZHIWEN/GR00T-N1.7-MixedStart/blob/main/docs/DATA_MANIFEST.json) 中的 `base_revision` 为准；该清单同时记录每个归档的字节数、SHA256、episode 数和帧数。

下载脚本只从该固定 revision 获取数据，校验 SHA256 和文件大小，拒绝路径穿越及归档链接，并在解包后检查集数、帧数、17D 状态／动作、双视角视频和必需段表。不会用本地暂存文件或其他数据目录替代缺失归档。

## 🔎 网页预览

`preview/mos_preview.parquet` 是仅供网页表格预览的小型抽样文件：从合成、DROID 两个 held-out 归档各取前 2 集，每集均匀抽取 10 帧，共 40 行。状态与动作保留原始 17D 数值，展开为 `state_00…state_16` 和 `action_00…action_16`，另附域、来源归档、集次、帧号、时间戳及主指令。该文件不包含视频，**不是训练集或独立评估集**，不能用于复现发布指标；完整输入仍以四个归档为准。预览可由 `python scripts/build_mos_dataset_preview.py` 重建，并需由数据维护者通过发布脚本上传；截至本次整理，远端尚未提供该文件，不影响四个归档的固定版本下载与复现。

## 📦 数据集清单

| 归档文件 | 解包目录 | 集数／帧数 | 构成与用途 |
|---|---|---:|---|
| `mix_droid_synth_train.tar.gz` | `mix_droid_synth_train` | 400／108,292 | DROID 300 集＋合成 100 集；发布权重训练 |
| `mix_droid_synth_eval.tar.gz` | `mix_droid_synth_eval` | 40／10,735 | DROID 30 集＋合成 10 集；合并域数据分析 |
| `droid_subset_eval.tar.gz` | `droid_subset_eval` | 30／8,735 | 独立 DROID held-out；发布结果的 DROID 评估协议 |
| `synth_mixture_eval_db7_17d.tar.gz` | `synth_mixture_eval_db7_17d` | 10／2,000 | 17D 合成 held-out，含段表；发布结果的合成评估协议 |

训练与评估使用不同 episode。发布评估使用每个评估集的前 10 条轨迹；合成域聚合评估随机种子 0、1、2、3，DROID 使用评估随机种子 0。合并评估集不能冒充独立 DROID 评估输入。

## 🚀 下载与校验

```bash
git clone https://github.com/GUOZHIWEN/GR00T-N1.7-MixedStart.git
cd GR00T-N1.7-MixedStart
pip install uv && uv sync
uv pip install modelscope

# 固定版本下载、SHA256 校验、安全解包与结构验证
uv run python scripts/prepare_mos_data.py --output-dir data

# 对已有解包目录重新检查
uv run python scripts/verify_mos_data.py --data-dir data
```

训练、评估、模型下载和最小推理命令见[复现指南](https://github.com/GUOZHIWEN/GR00T-N1.7-MixedStart/blob/main/docs/REPRODUCE.md)。

## 🗂️ 格式与输入契约

- **目录结构**：`data/chunk-000/episode_*.parquet`、`videos/chunk-000/<view>/episode_*.mp4` 和 `meta/`。
- **状态与动作均为 17D**：EEF `eef_9d` 9D＋`gripper_position` 1D＋`joint_position` 7D，不是“9D＋8D”。
- **视频**：`observation.images.exterior_1_left` 与 `observation.images.wrist_left`，15 fps，180×320 RGB。
- **文本**：`meta/tasks.jsonl` 和逐集 `meta/episodes.jsonl`。
- **归一化**：`meta/stats.json`；相对量统计见 `meta/relative_stats.json`；键映射以 `meta/modality.json` 为准。
- **embodiment tag**：`OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT`。
- 17D 协议不要传 `--state-extra-modality-keys` 或 `--expected-data-version`。

## 🔗 来源、转换与许可

- **合成部分**：由 `scripts/build_synthetic_mixture_dataset.py` 生成，seed 0；训练与评估采用不同任务参数网格。合成代码和新生成数据随项目以 Apache-2.0 发布。
- **DROID 部分**：源自 ModelScope `lerobot/droid_1.0.1`，300 训练＋30 评估集次。许可、署名和使用条件沿用原始 DROID 数据集，不由本数据仓的 Apache-2.0 元数据替代。
- **子集固定**：选中集次见本仓 `droid_subset_train_episodes.jsonl` 与 `droid_subset_eval_episodes.jsonl`。上游固定 revision 为 `b697b419ca258bf377cbc0c3279e5348fc5c7c6f`，可用 `scripts/download_droid_subset.py` 从源仓重建。
- **来源元数据**：`droid_source_tree.json` 记录所选上游版本中的 `meta/` 文件信息；原始逐集数据通过上述重建脚本获取。
- **合并与质检**：`scripts/merge_lerobot_datasets.py` 统一 17D、重编索引并重算统计；`scripts/visualize_merged_dataset.py` 用于质检。

使用 DROID 数据时请同时引用原始 DROID 论文并遵守其原始许可。

## ⚠️ 不包含与使用边界

- 不包含 DROID 原始全量数据；完整数据请从 `lerobot/droid_1.0.1` 获取并遵守其许可。
- 不包含当前发布协议之外的 22D 强条件历史数据；旧实验材料不属于当前 17D 复现入口。
- 数据用于开环 replay，不应描述为闭环 rollout、真机采集或任务成功率证据。

## 📬 联系方式

郭志文 · [zhiwenguo99@gmail.com](mailto:zhiwenguo99@gmail.com)

