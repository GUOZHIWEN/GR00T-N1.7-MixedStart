#!/usr/bin/env python
"""CLI regression tests for the official MoS checkpoint exporter."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("export_mos_release.py")
COMMON = {
    "history_action_align": "shift",
    "history_action_offset": 1,
    "zero_init_dit": False,
    "confidence_tokens_enabled": False,
    "confidence_routing_enabled": False,
    "sub_instruction_mode": "off",
    "tune_projector": True,
    "tune_diffusion_model": True,
    "tune_vlln": True,
}
RECIPES = {
    "mos_best": {
        "flow_source_mode": "memory_history_mix",
        "memory_start_enabled": True,
        "memory_encoder_arch": "mlp",
        "memory_start_mix_beta_max": 0.3,
        "memory_start_mix_beta_min": 0.0,
        "memory_start_mix_beta_sampling": "uniform",
        "memory_start_anchor_target": "role",
        "memory_start_anchor_lambda": 0.1,
        "memory_anchor_lambda_schedule": "cosine",
        "memory_start_noise_std": 0.04,
    },
    "mos_best_no_mem": {
        "flow_source_mode": "gaussian",
        "memory_start_enabled": False,
        "gaussian_source_noise_std": 1.0,
    },
}


class ExportMosReleaseTest(unittest.TestCase):
    def test_default_refuses_overwrite_and_explicit_flag_replaces(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            for name, recipe in RECIPES.items():
                checkpoint = root / "runs" / f"{name}_v2" / "checkpoint-5000"
                checkpoint.mkdir(parents=True)
                (checkpoint / "processor").mkdir()
                (checkpoint / "processor" / "processor_config.json").write_text(
                    '{"processor_class": "test"}', encoding="utf-8"
                )
                (checkpoint / "config.json").write_text(
                    json.dumps(COMMON | recipe), encoding="utf-8"
                )
                (checkpoint / "statistics.json").write_text("{}", encoding="utf-8")
                (checkpoint / "model.safetensors").write_bytes(b"test weights")
                (checkpoint / "optimizer.pt").write_bytes(b"not exported")

            command = [
                sys.executable,
                str(SCRIPT),
                "--run-dir",
                "runs",
                "--run-suffix",
                "_v2",
                "--output-dir",
                "release",
            ]
            first = subprocess.run(command, cwd=root, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)

            stale = root / "release" / "mos_best" / "stale.txt"
            stale.write_text("old export", encoding="utf-8")
            refused = subprocess.run(command, cwd=root, capture_output=True, text=True)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("use --overwrite", refused.stderr)
            self.assertTrue(stale.exists())

            replaced = subprocess.run(
                [*command, "--overwrite"], cwd=root, capture_output=True, text=True
            )
            self.assertEqual(replaced.returncode, 0, replaced.stderr)
            self.assertFalse(stale.exists())
            for name in RECIPES:
                exported = root / "release" / name
                self.assertTrue((exported / "model.safetensors").is_file())
                self.assertTrue((exported / "processor").is_dir())
                self.assertFalse((exported / "optimizer.pt").exists())

    def test_accepts_root_processor_and_uses_run_directory_fallback(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            for name, recipe in RECIPES.items():
                run = root / "runs" / f"{name}_v2"
                checkpoint = run / "checkpoint-5000"
                checkpoint.mkdir(parents=True)
                (checkpoint / "config.json").write_text(
                    json.dumps(COMMON | recipe), encoding="utf-8"
                )
                (checkpoint / "statistics.json").write_text("{}", encoding="utf-8")
                (checkpoint / "model.safetensors").write_bytes(b"test weights")
                if name == "mos_best":
                    (checkpoint / "processor_config.json").write_text(
                        '{"processor_class": "test"}', encoding="utf-8"
                    )
                else:
                    (run / "processor").mkdir()
                    (run / "processor" / "processor_config.json").write_text(
                        '{"processor_class": "test"}', encoding="utf-8"
                    )

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--run-dir",
                    "runs",
                    "--run-suffix",
                    "_v2",
                    "--output-dir",
                    "release",
                ],
                cwd=root,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(
                (root / "release" / "mos_best" / "processor_config.json").is_file()
            )
            self.assertTrue(
                (
                    root
                    / "release"
                    / "mos_best_no_mem"
                    / "processor"
                    / "processor_config.json"
                ).is_file()
            )


if __name__ == "__main__":
    unittest.main()
