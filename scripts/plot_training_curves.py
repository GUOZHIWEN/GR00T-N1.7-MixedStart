"""plot_training_curves.py — 从 launch_finetune 训练日志提取全部损失曲线并出图。

解析两类日志行：
  - HF trainer 行：{'loss': ..., 'grad_norm': ..., 'learning_rate': ...}（每 10 步）
  - MoS dbg 行：{'dbg_flow_loss': ..., 'dbg_anchor_loss': ..., 'dbg_conf_loss': ...}（每 5 步）

用法：
  python scripts/plot_training_curves.py \
    --logs mos_best=runs/mos_best/mos_best_log.txt \
           mos_best_no_mem=runs/mos_best_no_mem/mos_best_no_mem_log.txt \
    --out runs/mos_loss_curves.png
"""

import argparse
import ast
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DBG_KEYS = ["dbg_flow_loss", "dbg_anchor_loss", "dbg_conf_loss"]


def parse_log(path: str) -> dict:
    trainer, dbg = [], []
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            d = ast.literal_eval(line)
        except Exception:
            continue
        if "loss" in d and "grad_norm" in d:
            trainer.append(d)
        elif "dbg_flow_loss" in d:
            dbg.append(d)
    out = {"trainer": trainer, "dbg": dbg}
    # x 轴重建：trainer 每 10 步一行，dbg 每 5 步一行
    out["trainer_x"] = (np.arange(len(trainer)) + 1) * 10
    out["dbg_x"] = (np.arange(len(dbg)) + 1) * 5
    return out


def ema(y: np.ndarray, alpha: float = 0.06) -> np.ndarray:
    out = np.empty_like(y, dtype=float)
    acc = y[0]
    for i, v in enumerate(y):
        acc = alpha * v + (1 - alpha) * acc
        out[i] = acc
    return out


def series(rows: list[dict], key: str) -> np.ndarray:
    return np.array([r[key] for r in rows if key in r and np.isfinite(r[key])], dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", nargs="+", required=True, help="label=log_path 形式，可多个")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    arms = {}
    for item in args.logs:
        label, _, path = item.partition("=")
        arms[label] = parse_log(path)
        print(f"{label}: trainer {len(arms[label]['trainer'])} 行, dbg {len(arms[label]['dbg'])} 行")

    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.5))

    # ── (0,0) 总训练 loss ─────────────────────────────────────────────
    ax = axes[0][0]
    for i, (lb, d) in enumerate(arms.items()):
        y = series(d["trainer"], "loss")
        ax.plot(d["trainer_x"][: len(y)], y, color=colors[i], alpha=0.15, linewidth=0.6)
        ax.plot(d["trainer_x"][: len(y)], ema(y), color=colors[i], linewidth=1.6,
                label=f"{lb} (final {y[-100:].mean():.4f})")
    ax.set_title("total training loss (EMA)")
    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # ── (0,1) flow matching loss ──────────────────────────────────────
    ax = axes[0][1]
    for i, (lb, d) in enumerate(arms.items()):
        y = series(d["dbg"], "dbg_flow_loss")
        ax.plot(d["dbg_x"][: len(y)], y, color=colors[i], alpha=0.15, linewidth=0.6)
        ax.plot(d["dbg_x"][: len(y)], ema(y), color=colors[i], linewidth=1.6,
                label=f"{lb} (last-500 {y[-100:].mean():.4f})")
    ax.set_title("flow matching loss (dbg_flow_loss)")
    ax.set_xlabel("step"); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # ── (0,2) anchor loss + λ 调度（仅记忆臂） ────────────────────────
    ax = axes[0][2]
    plotted = False
    for i, (lb, d) in enumerate(arms.items()):
        y = series(d["dbg"], "dbg_anchor_loss")
        if len(y) == 0:
            continue
        plotted = True
        ax.plot(d["dbg_x"][: len(y)], y, color=colors[i], alpha=0.15, linewidth=0.6)
        ax.plot(d["dbg_x"][: len(y)], ema(y), color=colors[i], linewidth=1.6, label=f"{lb} anchor")
        lam = series(d["dbg"], "dbg_anchor_lambda_eff")
        if len(lam):
            ax2 = ax.twinx()
            ax2.plot(d["dbg_x"][: len(lam)], lam, color="gray", linestyle="--", linewidth=1.2,
                     label="lambda_eff")
            ax2.set_ylabel("lambda_eff", color="gray")
            ax2.tick_params(axis="y", labelcolor="gray")
            ax2.legend(fontsize=8, loc="lower left")
    ax.set_title("role anchor loss + lambda schedule")
    ax.set_xlabel("step"); ax.grid(alpha=0.3)
    if plotted:
        ax.legend(fontsize=8, loc="upper right")
    else:
        ax.text(0.5, 0.5, "(no memory arm)", ha="center", va="center", transform=ax.transAxes)

    # ── (1,0) confidence loss（仅记忆臂） ─────────────────────────────
    ax = axes[1][0]
    plotted = False
    for i, (lb, d) in enumerate(arms.items()):
        y = series(d["dbg"], "dbg_conf_loss")
        if len(y) == 0:
            continue
        plotted = True
        ax.plot(d["dbg_x"][: len(y)], y, color=colors[i], alpha=0.15, linewidth=0.6)
        ax.plot(d["dbg_x"][: len(y)], ema(y), color=colors[i], linewidth=1.6, label=lb)
    ax.set_title("confidence (beta routing) loss (dbg_conf_loss)")
    ax.set_xlabel("step"); ax.grid(alpha=0.3)
    if plotted:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "(no memory arm)", ha="center", va="center", transform=ax.transAxes)

    # ── (1,1) β* vs β̂（诊断，仅记忆臂） ──────────────────────────────
    ax = axes[1][1]
    plotted = False
    for i, (lb, d) in enumerate(arms.items()):
        bs = series(d["dbg"], "dbg_beta_star_mean")
        bp = series(d["dbg"], "dbg_beta_pred_mean")
        if len(bs) == 0:
            continue
        plotted = True
        x = d["dbg_x"][: len(bs)]
        ax.plot(x, ema(bs), color=colors[i], linewidth=1.6, label=f"{lb} beta* (target)")
        ax.plot(x, ema(bp), color=colors[i], linewidth=1.2, linestyle="--", label=f"{lb} beta_pred")
    ax.set_title("routing diagnostics: beta_star vs beta_pred (EMA)")
    ax.set_xlabel("step"); ax.set_ylabel("beta"); ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    if plotted:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "(no memory arm)", ha="center", va="center", transform=ax.transAxes)

    # ── (1,2) grad_norm ───────────────────────────────────────────────
    ax = axes[1][2]
    for i, (lb, d) in enumerate(arms.items()):
        y = series(d["trainer"], "grad_norm")
        ax.plot(d["trainer_x"][: len(y)], y, color=colors[i], alpha=0.15, linewidth=0.6)
        ax.plot(d["trainer_x"][: len(y)], ema(y), color=colors[i], linewidth=1.6, label=lb)
    ax.set_title("grad norm (EMA)")
    ax.set_xlabel("step"); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle("MoS merged-data arms: training loss curves (5000 steps)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
