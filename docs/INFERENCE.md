# MoS 单样本推理契约

`examples/mos_inference.py` 从真实机器人或评估流程保存的 NPZ 读取一条观测，加载本地权重并输出一个动作块。**示例不会连接或驱动机器人，也不要把随机模板输入生成的动作用于执行。**

## 输入

单个 NPZ 必须包含下列数组，批量大小固定为 1：

| NPZ 键 | dtype | shape | 含义 |
|---|---|---|---|
| `video.exterior_image_1_left` | `uint8` | `(1, 2, H, W, 3)` | 外部视角连续两帧 RGB |
| `video.wrist_image_left` | `uint8` | `(1, 2, H, W, 3)` | 腕部视角连续两帧 RGB |
| `state.eef_9d` | `float32` | `(1, 1, 9)` | XYZ＋旋转 6D |
| `state.gripper_position` | `float32` | `(1, 1, 1)` | 夹爪位置 |
| `state.joint_position` | `float32` | `(1, 1, 7)` | 关节位置 |

输入必须使用与训练数据相同的坐标系、物理单位和两帧时间顺序；本脚本不负责重新标定。语言指令通过 `--task` 传入，内部键为 `annotation.language.language_instruction`，形状为 `[[str]]`。

代码仓已附带 `examples/real_observation.npz`：它来自合成评估集第 0 条 episode 的第 15、16 帧及第 16 帧状态，**不是随机模板，也不是实机采集**。配套 `examples/real_observation.json` 记录图像、状态和指令来源。发布配方使用单文本口，因此示例指令采用该 episode 的高层任务。

`runs/trajectory_*/traj_*_trajectory.npz` 保存的是**动作真值与预测输出**，不包含图像和状态，不能用作这里的输入。使用自有文件时，绝对路径以 `/` 开头，例如 `/root/autodl-tmp/...`；`root/autodl-tmp/...` 会被解释为当前目录下的相对路径。

## 运行

```bash
uv run python examples/mos_inference.py \
  --model-path release/mos_best \
  --sample-npz examples/real_observation.npz \
  --task "indicate all the blocks one after another" \
  --device cuda --denoising-steps 4 \
  --fixed-beta 0 --fixed-noise-std 0 \
  --history-replan-stride 16 \
  --output runs/mos_inference_action.npz
```

使用已准备的评估数据重新导出同一份观测：

```bash
uv run python scripts/export_real_observation.py \
  --dataset-path data/synth_mixture_eval_db7_17d \
  --episode-index 0 --frame-index 16 \
  --output examples/real_observation.npz
```

CPU 可用 `--device cpu` 做接口排障，但完整模型推理资源开销较大。脚本验证输入、输出 shape、dtype 和有限性；这些检查不能替代真实权重加载、数值验收和机器人安全审查。

## 输出

输出 NPZ 保存由模型处理器**反归一化后**的 `float32` 动作：

| 键 | shape | 说明 |
|---|---|---|
| `action.eef_9d` | `(40, 9)` | 40 步 EEF 动作块 |
| `action.gripper_position` | `(40, 1)` | 夹爪动作 |
| `action.joint_position` | `(40, 7)` | 关节动作 |

`Gr00tPolicy.get_action()` 的 Python 返回值带有批量维，即 `(1, 40, D)`。相对 EEF／关节量由处理器按 checkpoint 的统计与当前状态解码，不应再手工重复反归一化。

## 历史、执行窗口与 reset

- 一次 `get_action()` 生成 40 步动作块；示例默认重规划步幅为 16。真实系统应在执行窗口后采集**新的图像和状态**再推理，不能把同一观测重复输入当作闭环。
- Policy 会缓存上一次的反归一化动作块和对应原始状态，供历史条件路径使用。`history_replan_stride=16` 表示缓存窗口在重规划时已过去 16 步。
- **每个新 episode 必须调用 `policy.reset()`**。reset 清空动作历史，不会重建模型；后续起点按 checkpoint 与部署覆盖配置生成。官方语义配方 `β=0` 时不混入历史高斯回退项。

## β、η 与去噪步数不是同一参数

- `β` 控制训练源中历史 `h` 与语义起点 `z₀` 的混合；`--fixed-beta 0` 关闭部署期历史混合。
- `η` 控制语义源上的噪声强度；`--fixed-noise-std 0` 才令语义源本身为 `z₀`。它**不是**高斯对照臂的标准差覆盖参数。
- `--denoising-steps` 是流场 Euler 积分步数。改变它仍需重新评估精度与运行间分歧。

示例默认 `β=0, η=0`，便于确定性接口验收；发布模型卡中的主结果使用 `η=0.04`，两种协议不能混报。随机高斯臂的固定噪声部署属于另一项控制实验。

## 发布验收门槛

```bash
uv run python scripts/validate_mos_release.py --release-dir release --write-sums
uv run python scripts/validate_mos_release.py --release-dir release
# 真实权重加载必须显式请求，离线伪权重测试不代表此步通过：
uv run python scripts/validate_mos_release.py --release-dir release --load --device cuda
```

完整门槛为：分片和 SHA256 校验通过、处理器与统计文件齐全、真实 `Gr00tPolicy` 加载成功、真实 NPZ 单样本推理产生有限动作，并完成开环或实机协议验收。伪权重单元测试仅覆盖目录和哈希逻辑。
