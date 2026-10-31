"""Fake-weight tests for release structure only; real loading is not covered."""
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from export_mos_release import RECIPE  # noqa: E402
from validate_mos_release import COMMON, validate_model, verify_checksums  # noqa: E402


class ReleaseValidationTest(unittest.TestCase):
    def make_model(self, root, name, sharded=True):
        model = root / name
        (model / "processor").mkdir(parents=True)
        (model / "config.json").write_text(json.dumps(COMMON | RECIPE[name]), encoding="utf-8")
        (model / "statistics.json").write_text('{"state": {}}', encoding="utf-8")
        (model / "processor" / "processor_config.json").write_text('{"processor_class": "fake"}', encoding="utf-8")
        if sharded:
            shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
            for shard in shards:
                (model / shard).write_bytes(b"fake tensor bytes")
            (model / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {"a": shards[0], "b": shards[1]},
            }), encoding="utf-8")
        else:
            (model / "model.safetensors").write_bytes(b"fake tensor bytes")
        return model

    def test_sharded_and_single_file_checksums(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, sharded in (("mos_best", True), ("mos_best_no_mem", False)):
                model = self.make_model(Path(tmp), name, sharded)
                report = validate_model(model, name, write_sums=True)
                self.assertEqual(report["status"], "PASS")
                self.assertFalse(report["real_model_loaded"])
                self.assertGreaterEqual(report["checksum_entries"], 4)
                self.assertEqual(verify_checksums(model), report["checksum_entries"])

    def test_missing_shard_and_modified_file_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = self.make_model(Path(tmp), "mos_best")
            (model / "model-00002-of-00002.safetensors").unlink()
            with self.assertRaises(FileNotFoundError):
                validate_model(model, "mos_best", write_sums=True)
        with tempfile.TemporaryDirectory() as tmp:
            model = self.make_model(Path(tmp), "mos_best_no_mem", sharded=False)
            validate_model(model, "mos_best_no_mem", write_sums=True)
            (model / "model.safetensors").write_bytes(b"modified")
            with self.assertRaises(ValueError):
                verify_checksums(model)

    def test_wrong_recipe_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = self.make_model(Path(tmp), "mos_best", sharded=False)
            config = json.loads((model / "config.json").read_text(encoding="utf-8"))
            config["memory_encoder_arch"] = "transformer"
            (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_model(model, "mos_best", write_sums=True)


if __name__ == "__main__":
    unittest.main()
