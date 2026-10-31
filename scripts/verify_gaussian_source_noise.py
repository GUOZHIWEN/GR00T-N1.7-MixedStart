#!/usr/bin/env python
"""G-small regression checks; run before training, without loading model weights.

    uv run python scripts/verify_gaussian_source_noise.py

Checks the real action-head source builder, legacy fallback, config round-trip,
CLI validation, and the two model-loading propagation chokepoints. It does not
replace the 200-step GPU smoke or the checkpoint-level deployment acceptance.
"""

from dataclasses import replace
import inspect
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from transformers.feature_extraction_utils import BatchFeature

from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead


def bare_head(config):
    """Construct only the config-bearing shell needed by _build_flow_source."""
    head = Gr00tN1d7ActionHead.__new__(Gr00tN1d7ActionHead)
    torch.nn.Module.__init__(head)
    head.config = config
    return head


def build(config, noise):
    return bare_head(config)._build_flow_source(
        torch.zeros_like(noise), BatchFeature(data={}), noise
    )


def main() -> None:
    noise = torch.linspace(-2.0, 2.0, 2 * 40 * 17, dtype=torch.float32).reshape(2, 40, 17)

    # Old checkpoints have no new field: return the original tensor, not merely
    # a numerically close multiplication by one.
    legacy = build(SimpleNamespace(flow_source_mode="gaussian"), noise)
    assert legacy is noise, "legacy gaussian source must preserve tensor identity"
    explicit_default = build(
        SimpleNamespace(flow_source_mode="gaussian", gaussian_source_noise_std=1.0),
        noise,
    )
    assert explicit_default is noise
    print("PASS legacy/default: exact source tensor preserved")

    small = build(
        SimpleNamespace(flow_source_mode="gaussian", gaussian_source_noise_std=0.04),
        noise,
    )
    torch.testing.assert_close(small, 0.04 * noise, rtol=0.0, atol=0.0)
    assert not torch.equal(small, noise)
    print("PASS G-small: source == 0.04 * epsilon (exact float comparison)")

    cfg = Gr00tN1d7Config(gaussian_source_noise_std=0.04)
    saved = json.loads(cfg.to_json_string())
    assert saved["gaussian_source_noise_std"] == 0.04
    assert Gr00tN1d7Config().gaussian_source_noise_std == 1.0
    ft = FinetuneConfig(
        base_model_path="unused-in-unit-test",
        dataset_path="unused-in-unit-test",
        embodiment_tag="OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT",
        gaussian_source_noise_std=0.04,
    )
    assert ft.gaussian_source_noise_std == 0.04
    for invalid in (
        {"gaussian_source_noise_std": 0.0},
        {
            "flow_source_mode": "memory_history_mix",
            "memory_start_enabled": True,
            "gaussian_source_noise_std": 0.04,
        },
    ):
        try:
            replace(ft, **invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid G-small config was accepted: {invalid}")
    print("PASS config: checkpoint serialization + invalid-combination guards")

    for method_name in ("forward", "get_action_with_features"):
        source = inspect.getsource(getattr(Gr00tN1d7ActionHead, method_name))
        assert "self._build_flow_source(" in source, method_name
    print("PASS deployment/training: both call the tested shared source builder")

    required_snippets = {
        "gr00t/experiment/launch_finetune.py": (
            "config.model.gaussian_source_noise_std = ft_config.gaussian_source_noise_std",
            '"gaussian_source_noise_std",',
        ),
        "gr00t/model/gr00t_n1d7/setup.py": (
            "gaussian_source_noise_std=getattr(",
            '"gaussian_source_noise_std",',
        ),
    }
    for rel, snippets in required_snippets.items():
        text = (ROOT / rel).read_text(encoding="utf-8")
        for snippet in snippets:
            assert snippet in text, f"{rel}: missing {snippet}"
    print("PASS propagation: launch + from_pretrained kwargs + landing assertions")
    print("ALL G-SMALL CPU CHECKS PASSED")


if __name__ == "__main__":
    main()
