"""Offline regression tests for public aggregation and legacy-audit gates."""
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from aggregate_anchor_supervision import load as load_anchor
from aggregate_mos_controls import load_control
from eval_result_utils import legacy_synth_overrides


PROOF = {
    "status": "PASS",
    "legacy_dataset": "data/synth_mixture_eval_db7",
    "canonical_dataset": "data/synth_mixture_eval_db7_17d",
    "episodes": 10,
    "state_prefix_17d_equal": True,
    "actions_equal": True,
    "task_indices_equal": True,
    "video_sha256_equal": True,
    "metadata_equal": True,
    "evaluation_statistics_source": "checkpoint",
    "dataset_statistics_compared": False,
}


class ReproductionGateTest(unittest.TestCase):
    def test_legacy_alias_requires_audited_projection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "proof.json"
            path.write_text(json.dumps(PROOF), encoding="utf-8")
            overrides = legacy_synth_overrides(path)
            self.assertEqual(overrides["dataset_path"], PROOF["legacy_dataset"])
            invalid = PROOF | {"video_sha256_equal": False}
            path.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaises(ValueError):
                legacy_synth_overrides(path)
        self.assertIsNone(legacy_synth_overrides(None))

    def test_anchor_checkpoint_and_protocol_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            model.mkdir()
            config = {
                "memory_start_anchor_target": "role",
                "memory_start_anchor_lambda": 0.1,
                "memory_anchor_lambda_schedule": "cosine",
                "zero_init_dit": False,
            }
            (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
            metrics = root / "metrics.json"
            metrics.write_text(json.dumps({
                "model_path": str(model),
                "protocol": {"fixed_beta": 0.3},
                "trajs": {str(i): {} for i in range(10)},
            }), encoding="utf-8")
            self.assertEqual(load_anchor(metrics, "role")["protocol"]["fixed_beta"], 0.3)
            with self.assertRaises(ValueError):
                load_anchor(metrics, "y")

    def test_long_eval_custom_path_and_full_horizon_are_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = root / "eval_mos_scratch_mos_best_no_mem_ckpt5000_long_es0_s2"
            result.mkdir()
            args = {
                "dataset_path": "data/synth_mixture_eval_long_17d",
                "model_paths": [f"{root}/db28g_s2/checkpoint-5000"],
                "seed": "0", "steps": "400", "execution_horizon": "16",
                "denoising_steps": "4", "history_source": "pred",
                "fixed_beta_value": "0.0", "force_fixed_beta": "True",
                "fixed_noise_draw": "False", "fixed_noise_std": None,
                "obs_image_noise_std": "0.0", "obs_state_noise_std": "0.0",
            }
            trajs = {str(i): {"mse_effective": 0.1, "repeat_divergence": 0.01}
                     for i in range(10)}
            (result / "compare_metrics.json").write_text(json.dumps({
                "args": args, "arms": {"G": {"trajs": trajs}},
            }), encoding="utf-8")
            record = load_control(
                root, "mos_best_no_mem", "long", 2, steps=400,
                long_dataset="data/synth_mixture_eval_long_17d",
            )
            self.assertEqual(len(record["trajs"]), 10)
            with self.assertRaises(ValueError):
                load_control(root, "mos_best_no_mem", "long", 2, steps=200,
                             long_dataset="data/synth_mixture_eval_long_17d")


if __name__ == "__main__":
    unittest.main()
