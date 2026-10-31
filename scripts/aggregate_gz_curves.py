"""aggregate_gz_curves.py — db28 G/Z 从头训练双臂 × 多 seed 收敛曲线聚合。

对每个臂的多个 seed 日志解析 dbg_flow_loss / dbg_anchor_loss（MoS dbg 行，
每 5 步一条）与 trainer loss（每 10 步一条），按最短长度截断后计算
mean±std，出三联图（step 轴），并汇总各 run 的 train_runtime（wall-clock）
与末段均损到 JSON。

用法：
  python scripts/aggregate_gz_curves.py \
    --arm G=runs/db28g_s0/log.txt,runs/db28g_s1/log.txt,runs/db28g_s2/log.txt \
    --arm Z=runs/db28z_s0/log.txt,runs/db28z_s1/log.txt,runs/db28z_s2/log.txt \
    --out runs/db28_gz_curves.png --json runs/db28_gz_curves.json
"""

import argparse
import ast
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DBG_X_STEP = 5
TRAINER_X_STEP = 10


def parse_log(path: str) -> dict:
    dbg_flow, dbg_anchor, trainer_loss = [], [], []
    runtime = None
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line.startswith("{"):
            m = re.search(r"'train_runtime': ([0-9.]+)", line)
            if m:
                runtime = float(m.group(1))
            continue
        try:
            d = ast.literal_eval(line)
        except Exception:
            continue
        if "train_runtime" in d:
            runtime = float(d["train_runtime"])
        if "dbg_flow_loss" in d and np.isfinite(d["dbg_flow_loss"]):
            dbg_flow.append(float(d["dbg_flow_loss"]))
            if "dbg_anchor_loss" in d and np.isfinite(d["dbg_anchor_loss"]):
                dbg_anchor.append(float(d["dbg_anchor_loss"]))
        elif "loss" in d and "grad_norm" in d and np.isfinite(d["loss"]):
            trainer_loss.append(float(d["loss"]))
    return {
        "flow": np.array(dbg_flow),
        "anchor": np.array(dbg_anchor),
        "trainer": np.array(trainer_loss),
        "runtime": runtime,
    }


def stack_runs(series_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = min(len(s) for s in series_list)
    if n == 0:
        return np.array([]), np.array([]), np.array([])
    mat = np.stack([s[:n] for s in series_list])
    return mat.mean(axis=0), mat.std(axis=0), mat


def ema(y: np.ndarray, alpha: float = 0.10) -> np.ndarray:
    out = np.empty_like(y, dtype=float)
    acc = y[0]
    for i, v in enumerate(y):
        acc = alpha * v + (1 - alpha) * acc
        out[i] = acc
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--arm", nargs="+", required=True, action="append",
                    help="LABEL=path1,path2,path3（每臂 3 个 seed 日志）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    # argparse append+nargs 会把多个 --arm 收成 list[list]，拍平
    arm_specs = [item for group in args.arm for item in group]

    arms: dict[str, list[dict]] = {}
    for spec in arm_specs:
        label, _, paths = spec.partition("=")
        arms[label] = [parse_log(p) for p in paths.split(",")]
        for p, d in zip(paths.split(","), arms[label]):
            print(f"[{label}] {p}: flow {len(d['flow'])} 行, trainer {len(d['trainer'])} 行, "
                  f"runtime={d['runtime']}")

    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, (label, runs) in enumerate(arms.items()):
        c = colors[i % 10]
        # ── flow loss mean±std ──
        mean, std, mat = stack_runs([r["flow"] for r in runs])
        if len(mean):
            x = (np.arange(len(mean)) + 1) * DBG_X_STEP
            axes[0].plot(x, ema(mean), color=c, label=f"{label} (n={mat.shape[0]})")
            axes[0].fill_between(x, ema(mean - std), ema(mean + std), color=c, alpha=0.18)
            for row in mat:
                axes[0].plot(x, ema(row), color=c, alpha=0.15, linewidth=0.6)
        # ── anchor loss mean±std ──
        mean_a, std_a, mat_a = stack_runs([r["anchor"] for r in runs])
        if len(mean_a):
            x = (np.arange(len(mean_a)) + 1) * DBG_X_STEP
            axes[1].plot(x, ema(mean_a), color=c, label=label)
            axes[1].fill_between(x, ema(mean_a - std_a), ema(mean_a + std_a), color=c, alpha=0.18)
        # ── wall-clock bar ──
        rt = [r["runtime"] for r in runs if r["runtime"] is not None]
        if rt:
            axes[2].bar(i, np.mean(rt), yerr=np.std(rt), color=c, alpha=0.8, capsize=4)
            axes[2].text(i, np.mean(rt), f"{np.mean(rt)/3600:.2f}h", ha="center", va="bottom")

    axes[0].set_title("dbg_flow_loss (mean ± seed-std, EMA)")
    axes[0].set_xlabel("step")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].set_title("dbg_anchor_loss (mean ± seed-std, EMA)")
    axes[1].set_xlabel("step")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[2].set_title("train_runtime (wall-clock, mean ± std)")
    axes[2].set_xticks(range(len(arms)))
    axes[2].set_xticklabels(list(arms.keys()))
    axes[2].grid(alpha=0.3, axis="y")

    fig.suptitle("db28 G/Z from-scratch convergence (3 seeds per arm)")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"saved -> {args.out}")

    if args.json:
        summary = {}
        for label, runs in arms.items():
            mean, std, mat = stack_runs([r["flow"] for r in runs])
            tail = mat[:, -40:].mean(axis=1) if mat.size else []
            rt = [r["runtime"] for r in runs if r["runtime"] is not None]
            summary[label] = {
                "n_seeds": len(runs),
                "flow_final_mean": float(mean[-1]) if len(mean) else None,
                "flow_tail40_per_seed": [float(v) for v in tail],
                "runtime_mean_s": float(np.mean(rt)) if rt else None,
                "runtime_std_s": float(np.std(rt)) if rt else None,
            }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"saved -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
