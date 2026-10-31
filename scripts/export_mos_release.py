#!/usr/bin/env python
"""Export loadable checkpoint-5000 directories as the two official release models.

uv run python scripts/export_mos_release.py --run-suffix _v2 --output-dir release
Creates release/mos_best and release/mos_best_no_mem without optimizer/log files.
"""
import argparse
import json
from pathlib import Path
import shutil

RECIPE = {
    "mos_best": {
        "flow_source_mode": "memory_history_mix", "memory_start_enabled": True,
        "memory_encoder_arch": "mlp", "memory_start_mix_beta_max": 0.3,
        "memory_start_mix_beta_min": 0.0, "memory_start_mix_beta_sampling": "uniform",
        "memory_start_anchor_target": "role", "memory_start_anchor_lambda": 0.1,
        "memory_anchor_lambda_schedule": "cosine", "memory_start_noise_std": 0.04,
    },
    "mos_best_no_mem": {
        "flow_source_mode": "gaussian", "memory_start_enabled": False,
        "gaussian_source_noise_std": 1.0,
    },
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default="runs")
    ap.add_argument("--run-suffix", default="_v2")
    ap.add_argument("--output-dir", default="release")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output model directories after checkpoint recipe validation.",
    )
    args = ap.parse_args()
    run_dir, out_root = Path(args.run_dir), Path(args.output_dir)
    for name, expected in RECIPE.items():
        src = run_dir / f"{name}{args.run_suffix}" / "checkpoint-5000"
        if not src.is_dir():
            raise FileNotFoundError(f"missing completed checkpoint: {src}")
        cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
        common = {"history_action_align": "shift", "history_action_offset": 1,
                  "zero_init_dit": False, "confidence_tokens_enabled": False,
                  "confidence_routing_enabled": False, "sub_instruction_mode": "off",
                  "tune_projector": True, "tune_diffusion_model": True, "tune_vlln": True}
        for key, want in (common | expected).items():
            if cfg.get(key) != want:
                raise ValueError(f"{name}: {key}={cfg.get(key)!r}, expected {want!r}")
        dst = out_root / name
        if dst.exists():
            if not args.overwrite:
                raise FileExistsError(
                    f"refusing to overwrite {dst}; use --overwrite to replace it "
                    "or choose a new --output-dir"
                )
            if not dst.is_dir() or dst.is_symlink():
                raise FileExistsError(f"refusing to replace non-directory or symlink: {dst}")
            shutil.rmtree(dst)
        ignore = shutil.ignore_patterns("optimizer.pt", "scheduler.pt", "rng_state*",
                                        "trainer_state.json", "training_args.bin", "*.log")
        shutil.copytree(src, dst, ignore=ignore)
        # Checkpoints may contain processor files at their root (the training
        # callback's layout) or under processor/. Gr00tPolicy accepts both.
        has_processor = (dst / "processor_config.json").is_file() or (
            dst / "processor" / "processor_config.json"
        ).is_file()
        if not has_processor:
            processor_src = src.parent / "processor"
            if (processor_src / "processor_config.json").is_file():
                shutil.copytree(processor_src, dst / "processor")
                print(f"Copied processor fallback {processor_src} -> {dst / 'processor'}")
        required = ["config.json", "statistics.json"]
        missing = [x for x in required if not (dst / x).exists()]
        if not (
            (dst / "processor_config.json").is_file()
            or (dst / "processor" / "processor_config.json").is_file()
        ):
            missing.append("processor_config.json (root or processor/)")
        if not ((dst / "model.safetensors").exists() or (dst / "model.safetensors.index.json").exists()):
            missing.append("model weights")
        if missing:
            raise RuntimeError(f"incomplete export {dst}: {missing}")
        print(f"Exported {src} -> {dst}")


if __name__ == "__main__":
    main()
