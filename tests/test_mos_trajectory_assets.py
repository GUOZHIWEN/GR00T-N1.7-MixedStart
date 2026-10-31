"""Offline checks for exact paired trajectory export and shared-axis figures."""
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import unittest

ROOT = Path(__file__).resolve().parents[1]
# Prefer this checkout when tests run from the parent research workspace.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_result_utils import COMMON_PROTOCOL, DOMAIN_PROTOCOL, load_compare_result
from generate_mos_release_trajectory_assets import (
    load_domain,
    load_trajectory,
    plot_domain,
    select_representative,
)
from gr00t.eval.compare_flow_sources import save_trajectory_arrays


LABELS = ["mos_best", "mos_best_no_mem"]
MODELS = ["release/mos_best", "release/mos_best_no_mem"]


def make_evaluation(root, domain="synth", npz_models=None):
    root.mkdir(parents=True, exist_ok=True)
    args = {key: "None" for key in COMMON_PROTOCOL | DOMAIN_PROTOCOL[domain]}
    args.update(COMMON_PROTOCOL)
    args.update(DOMAIN_PROTOCOL[domain])
    args.update({
        "model_paths": str(MODELS), "labels": str(LABELS),
        "traj_ids": str(list(range(10))), "seed": "0", "seeds": "None",
    })
    arms = {}
    for label, model, base in zip(LABELS, MODELS, (0.04, 0.05)):
        trajs = {
            str(tid): {"mse_effective": base + tid * 0.001, "repeat_divergence": base / 4}
            for tid in range(10)
        }
        arms[label] = {"model_path": model, "trajs": trajs}
    (root / "compare_metrics.json").write_text(
        json.dumps({"args": args, "arms": arms}), encoding="utf-8"
    )
    t = np.arange(200, dtype=np.float32)[:, None]
    dims = np.arange(17, dtype=np.float32)[None, :]
    gt = np.sin(t / 8 + dims / 4).astype(np.float32)
    preds = {
        "mos_best": gt + 0.025 * np.cos(t / 6 + dims),
        "mos_best_no_mem": gt + 0.08 * np.sin(t / 5 + dims),
    }
    save_args = SimpleNamespace(
        dataset_path=DOMAIN_PROTOCOL[domain]["dataset_path"], seed=0,
        execution_horizon=16, denoising_steps=4,
    )
    for tid in range(10):
        save_trajectory_arrays(
            gt, preds, arms, save_args, tid, root / f"traj_{tid}_trajectory.npz"
        )
    if npz_models is not None:
        path = root / "traj_5_trajectory.npz"
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        arrays["model_paths"] = np.asarray(npz_models)
        np.savez_compressed(path, **arrays)


class TrajectoryAssetTest(unittest.TestCase):
    def setUp(self):
        self.validation = ROOT.parent / ".validation"
        self.validation.mkdir(exist_ok=True)
        self.root = self.validation / "trajectory_fixture"
        if self.root.exists():
            shutil.rmtree(self.root)
        make_evaluation(self.root)

    def test_arrays_are_pickle_free_and_aligned(self):
        trajectory = load_trajectory(
            self.root, 5, *LABELS, *MODELS, DOMAIN_PROTOCOL["synth"]["dataset_path"]
        )
        self.assertEqual(trajectory["gt"].shape, (200, 17))
        self.assertEqual(trajectory["semantic"].shape, (200, 17))
        self.assertEqual(trajectory["horizon"], 16)
        with np.load(self.root / "traj_5_trajectory.npz", allow_pickle=False) as data:
            self.assertEqual(data["predictions"].shape, (2, 200, 17))
            self.assertEqual(data["labels"].tolist(), LABELS)

    def test_short_episode_is_valid_at_maximum_step_budget(self):
        path = self.root / "traj_5_trajectory.npz"
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        arrays["gt"] = arrays["gt"][:137]
        arrays["predictions"] = arrays["predictions"][:, :137]
        np.savez_compressed(path, **arrays)
        trajectory = load_trajectory(
            self.root, 5, *LABELS, *MODELS, DOMAIN_PROTOCOL["synth"]["dataset_path"]
        )
        self.assertEqual(trajectory["gt"].shape, (137, 17))

    def test_selection_is_upper_median_not_best_case(self):
        records = load_domain(self.plot_args("synth"))[1]
        tid, ratio = select_representative(records["semantic"]["trajs"], records["gaussian"]["trajs"])
        self.assertEqual(tid, 4)
        self.assertAlmostEqual(ratio, 0.054 / 0.044)

    def test_clear_gap_selection_is_explicit_and_deterministic(self):
        records = load_domain(self.plot_args("synth"))[1]
        tid, ratio = select_representative(
            records["semantic"]["trajs"], records["gaussian"]["trajs"], selection="max-ratio",
        )
        self.assertEqual(tid, 0)
        self.assertAlmostEqual(ratio, 0.05 / 0.04)
        with self.assertRaises(ValueError):
            select_representative(records["semantic"]["trajs"], records["gaussian"]["trajs"],
                                  selection="visual-best")

    def test_shared_axis_plot_and_provenance(self):
        for domain in ("synth", "droid"):
            make_evaluation(self.root, domain=domain)
            args = self.plot_args(domain)
            trajectory, records, tid, ratio = load_domain(args)
            output = self.validation / f"trajectory_panel_{domain}.png"
            plot_domain(domain, trajectory, records, tid, ratio, [0, 1, 3, 4], output,
                        args.semantic_label, args.gaussian_label)
            self.assertGreater(output.stat().st_size, 50_000)

    def test_mislabeled_npz_and_ambiguous_arm_are_rejected(self):
        make_evaluation(self.root, npz_models=[MODELS[1], MODELS[0]])
        with self.assertRaisesRegex(ValueError, "not release/mos_best"):
            load_trajectory(
                self.root, 5, *LABELS, *MODELS, DOMAIN_PROTOCOL["synth"]["dataset_path"]
            )
        with self.assertRaises(ValueError):
            load_compare_result(self.root, MODELS[0], "synth")

    @staticmethod
    def plot_args(domain):
        return SimpleNamespace(
            evaluation_dir=ROOT.parent / ".validation" / "trajectory_fixture",
            semantic_label=LABELS[0], gaussian_label=LABELS[1],
            semantic_model=MODELS[0], gaussian_model=MODELS[1],
            domain=domain, legacy_synth_equivalence=None,
        )


if __name__ == "__main__":
    unittest.main()
