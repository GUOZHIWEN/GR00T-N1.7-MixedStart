# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for the P0 transfer features in the action head:

- P0-1: ``has_real_action`` per-sample loss weighting (DreamZero-style mask).
- P0-2: history-conditioned flow matching source (UniWAM-style), shared by the
  training forward and the inference denoising init.

These tests instantiate Gr00tN1d7ActionHead directly with a tiny config, in the
same style as tests/gr00t/model/test_action_head.py.
"""

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
import torch
from transformers.feature_extraction_utils import BatchFeature


def _small_config(**overrides) -> Gr00tN1d7Config:
    defaults = dict(
        backbone_embedding_dim=64,
        hidden_size=64,
        input_embedding_dim=64,
        max_state_dim=7,
        max_action_dim=7,
        action_horizon=4,
        state_history_length=1,
        num_inference_timesteps=2,
        max_num_embodiments=4,
        add_pos_embed=True,
        use_vlln=True,
        max_seq_len=32,
        use_alternate_vl_dit=False,
        attend_text_every_n_blocks=2,
        tune_projector=True,
        tune_diffusion_model=True,
        tune_vlln=True,
        state_dropout_prob=0.0,
        noise_beta_alpha=1.5,
        noise_beta_beta=1.0,
        noise_s=0.999,
        num_timestep_buckets=1000,
        attn_dropout=0.0,
        diffusion_model_cfg={
            "positional_embeddings": None,
            "num_layers": 2,
            "num_attention_heads": 2,
            "attention_head_dim": 32,
            "norm_type": "ada_norm",
            "dropout": 0.0,
            "final_dropout": False,
            "output_dim": 64,
            "interleave_self_attention": True,
        },
    )
    defaults.update(overrides)
    return Gr00tN1d7Config(**defaults)


def _make_backbone_output(config, batch_size=2, seq_len=8):
    return BatchFeature(
        data={
            "backbone_features": torch.randn(batch_size, seq_len, config.backbone_embedding_dim),
            "backbone_attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            "image_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
        }
    )


def _make_action_input(config, batch_size=2, **extras):
    data = {
        "state": torch.randn(batch_size, config.state_history_length, config.max_state_dim),
        "action": torch.randn(batch_size, config.action_horizon, config.max_action_dim),
        "embodiment_id": torch.zeros(batch_size, dtype=torch.long),
        "action_mask": torch.ones(batch_size, config.action_horizon, config.max_action_dim),
    }
    data.update(extras)
    return BatchFeature(data=data)


# ---------------------------------------------------------------------------
# P0-2: _build_flow_source
# ---------------------------------------------------------------------------


class TestBuildFlowSource:
    def test_gaussian_mode_returns_noise(self):
        config = _small_config(flow_source_mode="gaussian")
        head = Gr00tN1d7ActionHead(config)
        actions = torch.randn(2, config.action_horizon, config.max_action_dim)
        noise = torch.randn_like(actions)
        action_input = BatchFeature(data={})
        source = head._build_flow_source(actions, action_input, noise)
        assert source is noise

    def test_history_mode_uses_history_chunk(self):
        config = _small_config(flow_source_mode="history", flow_source_action_noise_std=0.0)
        head = Gr00tN1d7ActionHead(config)
        actions = torch.randn(2, config.action_horizon, config.max_action_dim)
        noise = torch.randn_like(actions)
        history = torch.randn_like(actions)
        action_input = BatchFeature(
            data={
                "history_action": history,
                "has_action_history": torch.ones(2, dtype=torch.float32),
            }
        )
        source = head._build_flow_source(actions, action_input, noise)
        torch.testing.assert_close(source, history)

    def test_history_mode_falls_back_without_flag(self):
        """Samples with has_action_history=0 (episode start) fall back to noise."""
        config = _small_config(flow_source_mode="history", flow_source_action_noise_std=0.0)
        head = Gr00tN1d7ActionHead(config)
        actions = torch.randn(2, config.action_horizon, config.max_action_dim)
        noise = torch.randn_like(actions)
        history = torch.randn_like(actions)
        action_input = BatchFeature(
            data={
                "history_action": history,
                # sample 0 has history, sample 1 does not
                "has_action_history": torch.tensor([1.0, 0.0]),
            }
        )
        source = head._build_flow_source(actions, action_input, noise)
        torch.testing.assert_close(source[0], history[0])
        torch.testing.assert_close(source[1], noise[1])

    def test_history_mode_missing_key_falls_back(self):
        """Old checkpoints / datasets without the history key fall back to noise."""
        config = _small_config(flow_source_mode="history")
        head = Gr00tN1d7ActionHead(config)
        actions = torch.randn(2, config.action_horizon, config.max_action_dim)
        noise = torch.randn_like(actions)
        source = head._build_flow_source(actions, BatchFeature(data={}), noise)
        assert source is noise

    def test_rtc_alignment_slices_prefix_and_holds_tail(self):
        """rtc: source[j] = prev[offset + j], tail holds prev[-1]."""
        offset = 1
        config = _small_config(
            flow_source_mode="history",
            flow_source_action_noise_std=0.0,
            history_action_offset=offset,
            history_action_align="rtc",
        )
        head = Gr00tN1d7ActionHead(config)
        H, D = config.action_horizon, config.max_action_dim
        actions = torch.randn(1, H, D)
        noise = torch.randn_like(actions)
        history = torch.randn_like(actions)
        action_input = BatchFeature(
            data={
                "history_action": history,
                "has_action_history": torch.ones(1, dtype=torch.float32),
            }
        )
        source = head._build_flow_source(actions, action_input, noise)
        expected = torch.cat([history[:, offset:], history[:, -1:].expand(-1, offset, -1)], dim=1)
        torch.testing.assert_close(source, expected)
        # Prefix matches the shifted window; tail holds the last action.
        torch.testing.assert_close(source[0, : H - offset], history[0, offset:])
        torch.testing.assert_close(source[0, H - offset:], history[0, -1:].expand(offset, D))

    def test_eta_noise_perturbs_source(self):
        config = _small_config(flow_source_mode="history", flow_source_action_noise_std=0.04)
        head = Gr00tN1d7ActionHead(config)
        actions = torch.randn(2, config.action_horizon, config.max_action_dim)
        noise = torch.randn_like(actions)
        history = torch.randn_like(actions)
        action_input = BatchFeature(
            data={
                "history_action": history,
                "has_action_history": torch.ones(2, dtype=torch.float32),
            }
        )
        source = head._build_flow_source(actions, action_input, noise)
        assert not torch.allclose(source, history)
        # Perturbation should be small relative to the signal.
        assert (source - history).std() < 0.2


# ---------------------------------------------------------------------------
# P0-1: has_real_action loss masking
# ---------------------------------------------------------------------------


class TestHasRealActionLoss:
    def _forward(self, config, action_input, seed=0):
        torch.manual_seed(seed)
        head = Gr00tN1d7ActionHead(config)
        head.train()
        return head.forward(_make_backbone_output(config), action_input)

    def test_disabled_flag_ignores_field(self):
        """With the feature off, has_real_action in the batch is ignored."""
        config = _small_config(use_has_real_action=False)
        action_input = _make_action_input(
            config, has_real_action=torch.tensor([1.0, 0.0], dtype=torch.float32)
        )
        out = self._forward(config, action_input)
        assert torch.isfinite(out["loss"])
        assert out["loss"].abs() > 0

    def test_drop_unlabeled_zeroes_pseudo_sample_loss(self):
        config = _small_config(use_has_real_action=True, drop_unlabeled_action=True)
        action_input = _make_action_input(
            config, has_real_action=torch.tensor([1.0, 0.0], dtype=torch.float32)
        )
        out = self._forward(config, action_input)
        # Pseudo sample contributes exactly zero to the per-element loss.
        assert torch.count_nonzero(out["action_loss"][1]) == 0
        # And the scalar loss is normalized over real samples only.
        action_mask = action_input.action_mask
        expected = out["action_loss"][0].sum() / (action_mask[0].sum() + 1e-6)
        torch.testing.assert_close(out["loss"], expected)

    def test_all_pseudo_drop_gives_zero_loss(self):
        config = _small_config(use_has_real_action=True, drop_unlabeled_action=True)
        action_input = _make_action_input(
            config, has_real_action=torch.zeros(2, dtype=torch.float32)
        )
        out = self._forward(config, action_input)
        torch.testing.assert_close(out["loss"], torch.zeros_like(out["loss"]))

    def test_pseudo_weight_scales_contribution(self):
        """weight=0.5: pseudo sample contributes half its MSE to the numerator."""
        config = _small_config(
            use_has_real_action=True, pseudo_action_loss_weight=0.5, drop_unlabeled_action=False
        )
        action_input = _make_action_input(
            config, has_real_action=torch.tensor([1.0, 0.0], dtype=torch.float32)
        )
        out = self._forward(config, action_input)
        action_mask = action_input.action_mask
        # action_loss already carries the per-sample weight; the denominator
        # must use the same weighting.
        weighted_mask = action_mask.clone()
        weighted_mask[1] = weighted_mask[1] * 0.5
        expected = out["action_loss"].sum() / (weighted_mask.sum() + 1e-6)
        torch.testing.assert_close(out["loss"], expected)

    def test_all_real_matches_unmasked_loss(self):
        """flag=1 everywhere must reproduce the unmasked loss bit-for-bit."""
        config_on = _small_config(use_has_real_action=True)
        config_off = _small_config(use_has_real_action=False)
        # Share the exact same inputs between the two forwards.
        base_input = _make_action_input(config_off)
        flagged_input = BatchFeature(
            data={**base_input, "has_real_action": torch.ones(2, dtype=torch.float32)}
        )
        out_on = self._forward(config_on, flagged_input, seed=42)
        out_off = self._forward(config_off, base_input, seed=42)
        torch.testing.assert_close(out_on["loss"], out_off["loss"])


# ---------------------------------------------------------------------------
# P0-2: training forward + inference with history inputs (smoke)
# ---------------------------------------------------------------------------


class TestHistoryModeForwardInference:
    def test_forward_with_history_runs(self):
        config = _small_config(flow_source_mode="history", flow_source_action_noise_std=0.04)
        torch.manual_seed(0)
        head = Gr00tN1d7ActionHead(config)
        head.train()
        action_input = _make_action_input(
            config,
            history_action=torch.randn(2, config.action_horizon, config.max_action_dim),
            has_action_history=torch.tensor([1.0, 0.0]),
        )
        out = head.forward(_make_backbone_output(config), action_input)
        assert torch.isfinite(out["loss"])

    def test_get_action_with_history_source(self):
        config = _small_config(flow_source_mode="history", flow_source_action_noise_std=0.0)
        torch.manual_seed(0)
        head = Gr00tN1d7ActionHead(config)
        head.eval()
        action_input = _make_action_input(
            config,
            history_action=torch.randn(2, config.action_horizon, config.max_action_dim),
            has_action_history=torch.ones(2, dtype=torch.float32),
        )
        del action_input["action"]  # no ground truth at inference
        out = head.get_action(_make_backbone_output(config), action_input)
        assert out["action_pred"].shape == (2, config.action_horizon, config.max_action_dim)
        assert torch.isfinite(out["action_pred"]).all()

    def test_get_action_without_history_still_works(self):
        """A history-trained checkpoint must still run from a cold start (noise)."""
        config = _small_config(flow_source_mode="history")
        torch.manual_seed(0)
        head = Gr00tN1d7ActionHead(config)
        head.eval()
        action_input = _make_action_input(config)
        del action_input["action"]
        out = head.get_action(_make_backbone_output(config), action_input)
        assert out["action_pred"].shape == (2, config.action_horizon, config.max_action_dim)

    def test_num_inference_timesteps_one_step(self):
        """History conditioning targets 1-2 denoising steps; N=1 must run."""
        config = _small_config(
            flow_source_mode="history",
            flow_source_action_noise_std=0.0,
            num_inference_timesteps=1,
        )
        torch.manual_seed(0)
        head = Gr00tN1d7ActionHead(config)
        head.eval()
        action_input = _make_action_input(
            config,
            history_action=torch.randn(2, config.action_horizon, config.max_action_dim),
            has_action_history=torch.ones(2, dtype=torch.float32),
        )
        del action_input["action"]
        out = head.get_action(_make_backbone_output(config), action_input)
        assert torch.isfinite(out["action_pred"]).all()
