#!/usr/bin/env python3
"""Aggregate semantic-start tracking NMSE for direct-y vs mixed-source loss.

Inputs are produced by visualize_memory_start.py. Both runs must use the same
recorded deployment protocol, ten trajectories, and beta=0.3. Their checkpoint
configs must identify the intended supervision targets.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load(path: Path, expected_target: str) -> dict:
    blob = json.loads(path.read_text(encoding="utf-8"))
    if set(blob.get("trajs", {})) != {str(i) for i in range(10)}:
        raise ValueError(f"{path}: expected trajectories 0..9")
    protocol = blob.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError(f"{path}: missing deployment protocol; regenerate the diagnostic")
    if protocol.get("fixed_beta") != 0.3:
        raise ValueError(f"{path}: fixed_beta must be 0.3, got {protocol.get('fixed_beta')!r}")
    model_path = blob.get("model_path")
    if not model_path:
        raise ValueError(f"{path}: missing model_path")
    config_path = Path(model_path) / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for key, expected in (
        ("memory_start_anchor_target", expected_target),
        ("memory_start_anchor_lambda", 0.1),
        ("memory_anchor_lambda_schedule", "cosine"),
        ("zero_init_dit", False),
    ):
        if config.get(key) != expected:
            raise ValueError(f"{config_path}: {key}={config.get(key)!r}, expected {expected!r}")
    return blob


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct", type=Path, required=True,
                        help="Direct-target memory_start_metrics.json")
    parser.add_argument("--mixed", type=Path, required=True,
                        help="Mixed-source-consistency memory_start_metrics.json")
    parser.add_argument("--out-json", type=Path, default=Path("runs/mos_anchor_supervision.json"))
    args = parser.parse_args()
    arms = {
        "direct_target": load(args.direct, "y"),
        "mixed_source_consistency": load(args.mixed, "role"),
    }
    protocols = [blob["protocol"] for blob in arms.values()]
    if protocols[0] != protocols[1]:
        raise ValueError(f"deployment protocols differ: {protocols}")
    report = {
        "scope": "pretrained-DiT semantic-start tracking; open-loop replay",
        "sources": {}, "protocol": protocols[0], "n_trajectories": 10,
    }
    for label, blob in arms.items():
        report["sources"][label] = {
            "model_path": blob["model_path"],
            "metrics_path": str(args.direct if label == "direct_target" else args.mixed),
            "trajectories": sorted(blob["trajs"], key=int),
        }
        for split, key in (("segment_head", "z0role_head"), ("segment_mid", "z0role_mid")):
            values = [float(record["mean_nmse_live_dims"][key])
                      for record in blob["trajs"].values()]
            if not np.isfinite(values).all():
                raise ValueError(f"{label}/{split}: non-finite NMSE")
            report.setdefault(label, {})[split] = float(np.mean(values))
            report[label][f"{split}_per_trajectory"] = values
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
