"""Offline checks for ModelScope classification and real tabular previews."""
from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))
from build_mos_dataset_preview import (
    ACTION_COLUMNS, SOURCES, STATE_COLUMNS, build_preview, validate_preview,
)
import publish_mos_datasets as publisher


def make_archive(path: Path, domain_offset: int) -> None:
    dataset_dir = path.name.removesuffix(".tar.gz")
    episodes = []
    with tarfile.open(path, "w:gz") as archive:
        for episode_index, length in enumerate((12, 10, 8)):
            episodes.append({
                "episode_index": episode_index,
                "tasks": [f"task {domain_offset + episode_index}"],
                "length": length,
            })
            states = (domain_offset + episode_index + np.linspace(0, 1, length))[:, None]
            states = np.repeat(states.astype(np.float32), 17, axis=1)
            frame = pd.DataFrame({
                "episode_index": np.full(length, episode_index, dtype=np.int64),
                "frame_index": np.arange(length, dtype=np.int64),
                "timestamp": np.arange(length, dtype=np.float64) / 15,
                "observation.state": list(states),
                "action": list(-states),
            })
            payload = io.BytesIO()
            frame.to_parquet(payload, index=False)
            for name, data in (
                (f"{dataset_dir}/data/chunk-000/episode_{episode_index:06d}.parquet",
                 payload.getvalue()),
            ):
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
        payload = "".join(json.dumps(record) + "\n" for record in episodes).encode()
        member = tarfile.TarInfo(f"{dataset_dir}/meta/episodes.jsonl")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))


class DatasetPreviewTest(unittest.TestCase):
    def test_card_declares_domain_task_and_default_preview(self):
        card = (ROOT / "docs/DATASET_CARD.md").read_text(encoding="utf-8")
        metadata = yaml.safe_load(card.split("---", 2)[1])
        self.assertEqual(metadata["task_categories"], ["robotics"])
        self.assertEqual(metadata["multi_modal"]["robotics"]["language"], ["en"])
        config = metadata["configs"][0]
        self.assertEqual(config["config_name"], "preview")
        self.assertIs(config["default"], True)
        self.assertEqual(config["data_files"], [{
            "split": "train", "path": publisher.PREVIEW_FILE,
        }])
        self.assertIn("不是训练集或独立评估集", card)

    def test_sampling_preserves_real_values_and_excludes_later_episodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp)
            for domain_offset, (_, name) in enumerate(SOURCES):
                make_archive(staging / name, domain_offset * 10)
            output = staging / publisher.PREVIEW_FILE
            preview = build_preview(staging, output, episodes_per_domain=2,
                                    frames_per_episode=5)
            self.assertEqual(len(preview), 20)
            self.assertEqual(preview.groupby("domain").size().to_dict(),
                             {"DROID": 10, "synthetic": 10})
            first_synth_ep1 = preview[(preview["domain"] == "synthetic") &
                                      (preview["episode_index"] == 1)].iloc[0]
            self.assertEqual(first_synth_ep1["frame_index"], 0)
            np.testing.assert_array_equal(
                first_synth_ep1[STATE_COLUMNS].to_numpy(dtype=np.float32),
                np.ones(17, dtype=np.float32),
            )
            first_droid_ep1 = preview[(preview["domain"] == "DROID") &
                                      (preview["episode_index"] == 1)].iloc[0]
            np.testing.assert_array_equal(
                first_droid_ep1[STATE_COLUMNS].to_numpy(dtype=np.float32),
                np.full(17, 11, dtype=np.float32),
            )
            np.testing.assert_array_equal(
                first_droid_ep1[ACTION_COLUMNS].to_numpy(dtype=np.float32),
                np.full(17, -11, dtype=np.float32),
            )
            self.assertNotIn(2, preview["episode_index"].unique())
            validate_preview(output, expected=output)

    def test_remote_verification_rejects_modified_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp)
            for _, name in SOURCES:
                make_archive(staging / name, 0)
            output = staging / publisher.PREVIEW_FILE
            build_preview(staging, output)
            preview = pd.read_parquet(output)
            preview.loc[0, "state_00"] += 1
            changed = staging / "changed.parquet"
            preview.to_parquet(changed, index=False)
            with self.assertRaises(AssertionError):
                validate_preview(changed, expected=output)

    def test_real_staged_preview_rebuilds_exactly_when_available(self):
        staging = WORKSPACE / "release/dataset_updates"
        if not (staging / publisher.PREVIEW_FILE).is_file():
            self.skipTest("staging was removed after verified publication")
        with tempfile.TemporaryDirectory() as tmp:
            regenerated = build_preview(staging, Path(tmp) / "preview.parquet")
            actual = validate_preview(staging / publisher.PREVIEW_FILE)
            pd.testing.assert_frame_equal(actual, regenerated, check_exact=True)
            self.assertEqual(len(actual), 40)

    def test_upload_and_revision_checks_cover_preview(self):
        self.assertIn(publisher.PREVIEW_FILE, publisher.UPLOAD_FILES)
        source = (ROOT / "scripts/publish_mos_datasets.py").read_text(encoding="utf-8")
        self.assertIn('records[PREVIEW_FILE]["Revision"]', source)
        self.assertIn("verify_archive(cache / PREVIEW_FILE", source)
        self.assertIn("validate_preview(cache / PREVIEW_FILE, expected=preview)", source)
        for name in ("build_mos_dataset_preview.py", "publish_mos_datasets.py"):
            self.assertEqual((ROOT / "scripts" / name).read_bytes(),
                             (WORKSPACE / "scripts" / name).read_bytes())


if __name__ == "__main__":
    unittest.main()
