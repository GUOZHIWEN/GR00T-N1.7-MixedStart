"""Validate the real-NPZ input contract without loading torch or model weights."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "examples" / "mos_inference.py"
SPEC = importlib.util.spec_from_file_location("mos_inference_input", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class InferenceInputTest(unittest.TestCase):
    def observation(self, root: Path, **overrides) -> Path:
        values = {
            "video.exterior_image_1_left": np.zeros((1, 2, 180, 320, 3), dtype=np.uint8),
            "video.wrist_image_left": np.zeros((1, 2, 180, 320, 3), dtype=np.uint8),
            "state.eef_9d": np.zeros((1, 1, 9), dtype=np.float32),
            "state.gripper_position": np.zeros((1, 1, 1), dtype=np.float32),
            "state.joint_position": np.zeros((1, 1, 7), dtype=np.float32),
        }
        values.update(overrides)
        path = root / "observation.npz"
        np.savez(path, **values)
        return path

    def test_valid_observation_preserves_units_and_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            observation = MODULE.load_observation(self.observation(Path(tmp)), "Pick up the block")
        self.assertEqual(observation["video"]["exterior_image_1_left"].shape, (1, 2, 180, 320, 3))
        self.assertEqual(observation["state"]["eef_9d"].dtype, np.float32)
        self.assertEqual(observation["state"]["joint_position"].shape, (1, 1, 7))
        self.assertEqual(
            observation["language"]["annotation.language.language_instruction"],
            [["Pick up the block"]],
        )

    def test_wrong_dtype_shape_nonfinite_and_empty_task_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = (
                {"state.eef_9d": np.zeros((1, 1, 9), dtype=np.float64)},
                {"state.joint_position": np.zeros((1, 1, 8), dtype=np.float32)},
                {"state.gripper_position": np.full((1, 1, 1), np.nan, dtype=np.float32)},
                {"video.wrist_image_left": np.zeros((1, 1, 180, 320, 3), dtype=np.uint8)},
            )
            for override in cases:
                with self.subTest(key=next(iter(override))), self.assertRaises(ValueError):
                    MODULE.load_observation(self.observation(root, **override), "task")
            with self.assertRaises(ValueError):
                MODULE.load_observation(self.observation(root), "  ")
            path = root / "missing.npz"
            np.savez(path, **{"state.eef_9d": np.zeros((1, 1, 9), dtype=np.float32)})
            with self.assertRaises(KeyError):
                MODULE.load_observation(path, "task")

    def test_packaged_real_observation_contract(self):
        import json

        sample = SOURCE.parent / "real_observation.npz"
        metadata = json.loads(sample.with_suffix(".json").read_text(encoding="utf-8"))
        observation = MODULE.load_observation(sample, metadata["task"])
        with np.load(sample, allow_pickle=False) as values:
            self.assertEqual(set(values.files), set(MODULE.VIDEO_SPECS) | set(MODULE.STATE_SPECS))
            for key in MODULE.VIDEO_SPECS:
                self.assertEqual(values[key].shape, (1, 2, 180, 320, 3))
                self.assertGreater(float(values[key].std()), 1.0)
                self.assertFalse(np.array_equal(values[key][0, 0], values[key][0, 1]))
        self.assertEqual(
            observation["language"]["annotation.language.language_instruction"],
            [[metadata["task"]]],
        )

    def test_missing_path_reports_example_and_resolved_path(self):
        with self.assertRaisesRegex(FileNotFoundError, "examples/real_observation.npz"):
            MODULE.load_observation(Path("definitely_missing_observation.npz"), "task")


if __name__ == "__main__":
    unittest.main()
