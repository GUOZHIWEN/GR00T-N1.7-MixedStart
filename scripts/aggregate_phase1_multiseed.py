#!/usr/bin/env python
"""Aggregate multi-seed Phase-1 emulator runs into cluster-bootstrap CIs.

Paper §5.3 / §8 item 9 (CI-ization of the strong-conditioning suite).

Reads N ``phase1_metrics.json`` files (one per seed; same arms, strides and
trajectory set), pools (traj × seed) records with the TRAJECTORY as the
cluster unit — seeds within one trajectory are correlated — and reports,
per stride and metric:

  * per-arm pooled mean + 95% cluster-bootstrap CI
    (``gr00t.eval.compare_flow_sources._cluster_bootstrap_ci95``, the exact
    implementation used for the §5.2 multiseed table, imported so the
    statistical contract cannot drift);
  * strong/baseline paired ratio + 95% paired cluster-bootstrap CI (same
    resampled trajectory index set applied to both arms per replicate).

Usage:

  uv run python scripts/aggregate_phase1_multiseed.py \
    --runs runs/phase1_sc_s0 runs/phase1_sc_s1 runs/phase1_sc_s2 \
    --strong-arm strongcond_z0 --baseline-arm no_mem \
    --strides 8 16 24 32 --resamples 10000 \
    --out-json runs/phase1_sc_multiseed_ci.json
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tyro

logger = logging.getLogger(__name__)

# (report name, extractor from a per-traj record)
METRICS = (
    ("mse_eff", lambda t: t.get("mse_eff")),
    ("repeat‖Δ‖", lambda t: t.get("repeat_divergence")),
    ("M4_boundary_jump", lambda t: (t.get("M4") or {}).get("boundary_jump_mean")),
)


@dataclass
class Args:
    runs: list[str]
    """phase1 emulator out-dirs, ONE PER SEED (each holds phase1_metrics.json)."""
    strong_arm: str
    """label of the strong-conditioning (memory, β=0) arm inside the jsons."""
    baseline_arm: str
    """label of the gaussian baseline arm inside the jsons."""
    strides: list[int] = field(default_factory=lambda: [8, 16, 24, 32])
    resamples: int = 10000
    seed: int = 0
    out_json: str | None = None


def _pooled_clusters(metrics_by_traj: dict[str, float]) -> list[list[float]]:
    """dict traj_key -> list of per-seed values -> cluster list (finite only)."""
    clusters = []
    for key in sorted(metrics_by_traj, key=lambda k: int(k) if str(k).isdigit() else str(k)):
        vals = [v for v in metrics_by_traj[key] if v is not None and np.isfinite(v)]
        if vals:
            clusters.append(vals)
    return clusters


def _flat(clusters: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
    sizes = np.array([len(c) for c in clusters])
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    flat = np.concatenate([np.asarray(c, dtype=np.float64) for c in clusters])
    return flat, offsets


def main(args: Args) -> int:
    from gr00t.eval.compare_flow_sources import _cluster_bootstrap_ci95

    rng = np.random.default_rng(args.seed)
    per_seed: list[dict] = []
    for run_dir in args.runs:
        path = Path(run_dir) / "phase1_metrics.json"
        if not path.exists():
            raise FileNotFoundError(f"missing {path} — run the emulator with the matching seed first")
        per_seed.append(json.load(open(path)))
        logger.info(f"loaded {path}")

    for label in (args.strong_arm, args.baseline_arm):
        for j, blob in enumerate(per_seed):
            if label not in blob.get("arms", {}):
                raise KeyError(f"arm {label!r} not in {args.runs[j]}/phase1_metrics.json (arms: {list(blob.get('arms', {}))})")

    report: dict[str, dict] = {}
    print(f"\n{'stride':>6} {'metric':>18} {'strong mean [CI95]':>30} {'base mean [CI95]':>30} {'ratio [CI95]':>26}")
    for stride in args.strides:
        skey = str(stride)
        report[skey] = {}
        for metric_name, extract in METRICS:
            arm_clusters: dict[str, dict[str, list[float]]] = {}
            for label in (args.strong_arm, args.baseline_arm):
                by_traj: dict[str, list[float]] = {}
                for blob in per_seed:
                    strides_blob = blob["arms"][label].get("strides", {})
                    if skey not in strides_blob:
                        raise KeyError(f"stride {skey} missing for arm {label!r} in one run")
                    for tkey, trec in strides_blob[skey].get("trajs", {}).items():
                        by_traj.setdefault(str(tkey), []).append(extract(trec))
                arm_clusters[label] = by_traj

            row: dict[str, dict] = {}
            means: dict[str, float] = {}
            for label, tag in ((args.strong_arm, "strong"), (args.baseline_arm, "baseline")):
                clusters = _pooled_clusters(arm_clusters[label])
                flat_vals = np.concatenate([np.asarray(c) for c in clusters])
                mean = float(flat_vals.mean())
                lo, hi = _cluster_bootstrap_ci95(clusters, rng, args.resamples)
                row[tag] = {"mean": mean, "ci95": [lo, hi], "n_traj": len(clusters)}
                means[tag] = mean

            # paired ratio bootstrap: shared resampled trajectory indices
            cs = _pooled_clusters(arm_clusters[args.strong_arm])
            cb = _pooled_clusters(arm_clusters[args.baseline_arm])
            n_cl = min(len(cs), len(cb))
            cs, cb = cs[:n_cl], cb[:n_cl]
            fs, of_ = _flat(cs)
            fb, ob = _flat(cb)
            idx = rng.integers(0, n_cl, size=(args.resamples, n_cl))
            stats = np.empty(args.resamples)
            for b in range(args.resamples):
                ms = np.concatenate([fs[of_[i]:of_[i + 1]] for i in idx[b]]).mean()
                mb = np.concatenate([fb[ob[i]:ob[i + 1]] for i in idx[b]]).mean()
                stats[b] = ms / mb if mb != 0 else np.nan
            stats = stats[np.isfinite(stats)]
            r_lo, r_hi = (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))) if len(stats) else (float("nan"),) * 2
            ratio = means["strong"] / means["baseline"] if means["baseline"] else float("nan")
            row["ratio"] = {"mean": ratio, "ci95": [r_lo, r_hi]}
            report[skey][metric_name] = row

            s_ = row["strong"]
            b_ = row["baseline"]
            print(
                f"{stride:>6} {metric_name:>18} "
                f"{s_['mean']:>13.4f} [{s_['ci95'][0]:.4f},{s_['ci95'][1]:.4f}] "
                f"{b_['mean']:>13.4f} [{b_['ci95'][0]:.4f},{b_['ci95'][1]:.4f}] "
                f"{ratio:>10.3f} [{r_lo:.3f},{r_hi:.3f}]"
            )

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        logger.info(f"written: {out}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="INFO:%(name)s:%(message)s")
    sys.exit(main(tyro.cli(Args)))
