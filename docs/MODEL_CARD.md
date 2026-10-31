# MoS 模型卡入口

**唯一权威模型卡已迁移至 [`release/README.md`](../release/README.md)。** 该文件同时用于 ModelScope 模型仓，包含：

- `mos_best` 与 `mos_best_no_mem` 的架构、训练配置和下载方式；
- 当前发布权重的合成域与 DROID 开环结果及配对置信区间；
- 最小推理、输入输出契约和完整复现入口；
- 适用范围、许可和局限。

本文件不再复制实验数字，以避免不同文档之间发生版本漂移。

## 当前结论边界

- 发布模型从**预训练 Action DiT**继续微调；论文多种子主结果使用**从零初始化 Action DiT**，两条证据线必须分开解读。
- 合成 held-out 上观察到精度和 repeat divergence 显著改善。
- DROID replay 上 repeat divergence 显著改善，但精度配对区间包含 1，**未观察到显著精度提升**。
- 当前证据均为开环 replay，不代表闭环成功率或真机部署收益。

早期模型卡中的实验数字和旧配方不再作为当前发布结论。历史过程记录仅见 `docs/history/`，不属于公开复现入口。
