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
Tests for T1 (mixed-source paper §4.4): confidence routing and the
inference-time history rtc-offset override.

- ConfidenceGapHead num_outputs=4: construction, zero-init, reset recovery.
- Action head builds the 4-output head iff confidence_routing_enabled.
- _build_flow_source: per-sample beta_override and rtc_offset threading.
- Training forward: dual per-source gap targets (ĝ_hist, ĝ_mem) and debug
  stats; finite loss; gradient flows.
- Inference: β̂ = ĝ_mem²/(ĝ_hist² + ĝ_mem² + ε) routing (and the L1 ablation),
  beta_hat exposed in the output; options["history_rtc_offset"] changes the
  source alignment.
- FinetuneConfig validation for the routing flags.
"""

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
from gr00t.model.modules.action_memory_encoder import ConfidenceGapHead
import torch
from transformers.feature_extraction_utils import BatchFeature

from tests.gr00t.model.test_p0_flow_source_and_mask import (
    _make_action_input,
    _make_backbone_output,
    _small_config,
)


K, C = 4, 2  # memory / confidence query counts used throughout


def _routing_config(**overrides) -> Gr00tN1d7Config:
    defaults = dict(
        flow_source_mode="memory_history_mix",
        memory_start_enabled=True,
        memory_start_hidden_dim=32,
        memory_num_queries=K,
        memory_query_dim=8,
        memory_query_path="backbone",
        memory_start_mix_beta=0.5,
        memory_start_noise_std=0.0,
        memory_start_anchor_lambda=0.0,
        history_action_align="rtc",
        history_action_offset=1,
        confidence_tokens_enabled=True,
        confidence_num_queries=C,
        confidence_loss_weight=0.1,
        confidence_every_k=1,
        confidence_routing_enabled=True,
    )
    defaults.update(overrides)
    return _small_config(**defaults)


def _routing_inputs(config, batch_size=2):
    """Backbone output with the K+C tail + action input with history."""
    backbone_output = _make_backbone_output(config, batch_size)
    backbone_output["memory_query_features"] = torch.randn(
        batch_size, K + C, config.backbone_embedding_dim
    )
    action_input = _make_action_input(
        config,
        batch_size,
        history_action=torch.randn(batch_size, config.action_horizon, config.max_action_dim),
        has_action_history=torch.ones(batch_size, dtype=torch.float32),
    )
    return backbone_output, action_input


# ---------------------------------------------------------------------------
# ConfidenceGapHead with num_outputs=4
# ---------------------------------------------------------------------------


class TestConfidenceGapHead4Out:
    def test_construction_and_zero_init(self):
        head = ConfidenceGapHead(cond_dim=16, num_queries=3, hidden_dim=8, num_outputs=4)
        assert head.readout[2].out_features == 4
        out = head(torch.randn(2, 3, 16))
        assert out.shape == (2, 4)
        assert torch.all(out == 0)  # zero-init readout

    def test_num_outputs_validation(self):
        try:
            ConfidenceGapHead(cond_dim=8, num_outputs=3)
        except AssertionError:
            return
        raise AssertionError("expected AssertionError for num_outputs=3")

    def test_reset_recovers_from_garbage(self):
        head = ConfidenceGapHead(cond_dim=8, num_queries=2, hidden_dim=8, num_outputs=4)
        with torch.no_grad():
            head.queries.fill_(1.0e31)
            head.readout[2].weight.fill_(2.3e31)
        head.reset_parameters()
        out = head(torch.randn(1, 2, 8))
        assert torch.isfinite(out).all() and torch.all(out == 0)


# ---------------------------------------------------------------------------
# Action-head wiring
# ---------------------------------------------------------------------------


class TestRoutingHeadWiring:
    def test_routing_builds_4out_head(self):
        head = Gr00tN1d7ActionHead(_routing_config())
        assert head.confidence_head is not None
        assert head.confidence_head.num_outputs == 4

    def test_default_keeps_2out_head(self):
        config = _routing_config(confidence_routing_enabled=False)
        head = Gr00tN1d7ActionHead(config)
        assert head.confidence_head.num_outputs == 2


# ---------------------------------------------------------------------------
# _build_flow_source: beta_override + rtc_offset
# ---------------------------------------------------------------------------


class TestBetaOverrideAndRtcOffset:
    def test_beta_override_per_sample(self):
        config = _routing_config()
        head = Gr00tN1d7ActionHead(config)
        B = 2
        actions = torch.randn(B, config.action_horizon, config.max_action_dim)
        noise = torch.randn_like(actions)
        history = torch.randn_like(actions)
        z0 = torch.randn_like(actions)
        action_input = BatchFeature(
            data={
                "history_action": history,
                "has_action_history": torch.ones(B, dtype=torch.float32),
            }
        )
        beta = torch.tensor([1.0, 0.0])  # sample 0 -> history, sample 1 -> memory
        source = head._build_flow_source(
            actions, action_input, noise, memory_start=z0, beta_override=beta
        )
        # rtc offset=1: aligned history = history[1:] + hold last.
        aligned = torch.cat([history[:, 1:], history[:, -1:]], dim=1)
        torch.testing.assert_close(source[0], aligned[0])
        torch.testing.assert_close(source[1], z0[1])

    def test_beta_override_none_uses_config(self):
        config = _routing_config(memory_start_mix_beta=0.25)
        head = Gr00tN1d7ActionHead(config)
        B = 2
        actions = torch.randn(B, config.action_horizon, config.max_action_dim)
        noise = torch.randn_like(actions)
        history = torch.randn_like(actions)
        z0 = torch.randn_like(actions)
        action_input = BatchFeature(
            data={
                "history_action": history,
                "has_action_history": torch.ones(B, dtype=torch.float32),
            }
        )
        source = head._build_flow_source(actions, action_input, noise, memory_start=z0)
        aligned = torch.cat([history[:, 1:], history[:, -1:]], dim=1)
        torch.testing.assert_close(source, 0.25 * aligned + 0.75 * z0)

    def test_rtc_offset_override_changes_slice(self):
        config = _routing_config(history_action_offset=1)
        head = Gr00tN1d7ActionHead(config)
        B = 2
        actions = torch.randn(B, config.action_horizon, config.max_action_dim)
        history = torch.randn_like(actions)
        action_input = BatchFeature(data={"history_action": history})
        src_default = head._aligned_history_source(actions, action_input, eta=0.0)
        src_off3 = head._aligned_history_source(actions, action_input, eta=0.0, rtc_offset=3)
        # offset=3: source[j] = history[3 + j], tail holds history[-1].
        expected = torch.cat([history[:, 3:], history[:, -1:].expand(-1, 3, -1)], dim=1)
        torch.testing.assert_close(src_off3, expected)
        assert not torch.allclose(src_default, src_off3)

    def test_rtc_offset_threads_through_mix(self):
        config = _routing_config(history_action_offset=1, memory_start_mix_beta=1.0)
        head = Gr00tN1d7ActionHead(config)
        B = 2
        actions = torch.randn(B, config.action_horizon, config.max_action_dim)
        noise = torch.randn_like(actions)
        history = torch.randn_like(actions)
        z0 = torch.randn_like(actions)
        action_input = BatchFeature(
            data={
                "history_action": history,
                "has_action_history": torch.ones(B, dtype=torch.float32),
            }
        )
        source = head._build_flow_source(
            actions, action_input, noise, memory_start=z0, rtc_offset=2
        )
        expected = torch.cat([history[:, 2:], history[:, -1:].expand(-1, 2, -1)], dim=1)
        torch.testing.assert_close(source, expected)  # beta=1 -> pure history term


# ---------------------------------------------------------------------------
# Training forward with dual per-source gap targets
# ---------------------------------------------------------------------------


class TestRoutingForward:
    def test_routing_loss_and_debug_stats(self):
        config = _routing_config()
        torch.manual_seed(0)
        head = Gr00tN1d7ActionHead(config)
        head.train()
        backbone_output, action_input = _routing_inputs(config)
        out = head.forward(backbone_output, action_input)
        assert out["confidence_loss"] is not None
        assert torch.isfinite(out["confidence_loss"])
        assert torch.isfinite(out["loss"])
        for key in (
            "conf_loss",
            "gap_target_mean",
            "gap_hist_target_mean",
            "gap_mem_target_mean",
            "gap_hist_pred_mean",
            "gap_mem_pred_mean",
        ):
            assert key in out["debug_stats"], key
        # Zero-init readout predicts zero gaps.
        assert out["debug_stats"]["gap_hist_pred_mean"] == 0.0
        assert out["debug_stats"]["gap_mem_pred_mean"] == 0.0

    def test_routing_loss_backward_flows(self):
        config = _routing_config()
        torch.manual_seed(0)
        head = Gr00tN1d7ActionHead(config)
        head.train()
        backbone_output, action_input = _routing_inputs(config)
        out = head.forward(backbone_output, action_input)
        out["loss"].backward()
        grad = head.confidence_head.readout[2].weight.grad
        assert grad is not None and grad.abs().sum() > 0

    def test_2out_head_skips_routing_targets(self):
        config = _routing_config(confidence_routing_enabled=False)
        torch.manual_seed(0)
        head = Gr00tN1d7ActionHead(config)
        head.train()
        backbone_output, action_input = _routing_inputs(config)
        out = head.forward(backbone_output, action_input)
        assert out["confidence_loss"] is not None
        assert "gap_hist_target_mean" not in out["debug_stats"]


# ---------------------------------------------------------------------------
# Inference: β̂ routing + rtc offset via options
# ---------------------------------------------------------------------------


class TestRoutingInference:
    def _get_action(self, config, seed=0, options=None):
        torch.manual_seed(seed)
        head = Gr00tN1d7ActionHead(config)
        head.eval()
        backbone_output, action_input = _routing_inputs(config)
        del action_input["action"]
        return head, head.get_action(backbone_output, action_input, options=options)

    def test_zero_init_routes_to_pure_memory(self):
        """Zero-init readout: ĝ_hist = ĝ_mem = 0 -> β̂ = 0 (pure memory arm)."""
        config = _routing_config()
        _, out = self._get_action(config)
        assert "beta_hat" in out
        assert out["beta_hat"].shape == (2,)
        assert torch.all(out["beta_hat"] == 0)
        assert out["confidence"].shape == (2, 4)

    def test_bias_manipulation_routes_to_history(self):
        """g_hist=0, g_mem=2 -> β̂ = 4/(0+4) = 1 (pure history arm)."""
        config = _routing_config()
        head, _ = self._get_action(config)
        with torch.no_grad():
            head.confidence_head.readout[2].bias.copy_(torch.tensor([0.0, 0.0, 0.0, 2.0]))
        backbone_output, action_input = _routing_inputs(config)
        del action_input["action"]
        out = head.get_action(backbone_output, action_input)
        torch.testing.assert_close(out["beta_hat"], torch.ones(2))

    def test_l1_ablation_differs_from_squared(self):
        """g_hist=2, g_mem=1: L1 -> 1/3; squared -> 1/5."""
        config = _routing_config(confidence_routing_l1=True)
        head, _ = self._get_action(config)
        with torch.no_grad():
            head.confidence_head.readout[2].bias.copy_(torch.tensor([0.0, 0.0, 2.0, 1.0]))
        backbone_output, action_input = _routing_inputs(config)
        del action_input["action"]
        out = head.get_action(backbone_output, action_input)
        torch.testing.assert_close(out["beta_hat"], torch.full((2,), 1.0 / 3.0))

    def test_routing_disabled_no_beta_hat(self):
        config = _routing_config(confidence_routing_enabled=False)
        _, out = self._get_action(config)
        assert "beta_hat" not in out
        assert out["confidence"].shape == (2, 2)

    def test_options_rtc_offset_changes_output(self):
        """options['history_rtc_offset'] realigns the source -> new actions."""
        config = _routing_config(confidence_routing_enabled=False, memory_start_mix_beta=1.0)
        _, out_default = self._get_action(config, seed=7)
        _, out_offset = self._get_action(config, seed=7, options={"history_rtc_offset": 3})
        assert not torch.allclose(out_default["action_pred"], out_offset["action_pred"])


# ---------------------------------------------------------------------------
# FinetuneConfig validation (CLI surface)
# ---------------------------------------------------------------------------


class TestRoutingConfigValidation:
    _REQUIRED = dict(base_model_path="m", dataset_path="d", embodiment_tag="new_embodiment")

    def test_defaults_off(self):
        from gr00t.configs.finetune_config import FinetuneConfig

        cfg = FinetuneConfig(**self._REQUIRED)
        assert cfg.confidence_routing_enabled is False
        assert cfg.confidence_routing_l1 is False

    def test_routing_requires_confidence_tokens(self):
        from gr00t.configs.finetune_config import FinetuneConfig

        try:
            FinetuneConfig(
                **self._REQUIRED,
                flow_source_mode="memory_history_mix",
                memory_start_enabled=True,
                confidence_routing_enabled=True,
            )
        except ValueError as exc:
            assert "confidence_tokens_enabled" in str(exc)
            return
        raise AssertionError("expected ValueError (confidence tokens off)")

    def test_routing_requires_mix_mode(self):
        from gr00t.configs.finetune_config import FinetuneConfig

        try:
            FinetuneConfig(
                **self._REQUIRED,
                flow_source_mode="memory",
                memory_start_enabled=True,
                confidence_tokens_enabled=True,
                confidence_routing_enabled=True,
            )
        except ValueError as exc:
            assert "memory_history_mix" in str(exc)
            return
        raise AssertionError("expected ValueError (not mix mode)")

    def test_routing_valid_combo_accepted(self):
        from gr00t.configs.finetune_config import FinetuneConfig

        cfg = FinetuneConfig(
            **self._REQUIRED,
            flow_source_mode="memory_history_mix",
            memory_start_enabled=True,
            confidence_tokens_enabled=True,
            confidence_routing_enabled=True,
        )
        assert cfg.confidence_routing_enabled is True


# ---------------------------------------------------------------------------
# Direct-β routing (survey §13.18): 5th output regresses closed-form β*
# ---------------------------------------------------------------------------


class TestDirectBetaRouting:
    def test_direct_beta_builds_5out_head(self):
        config = _routing_config(confidence_routing_mode="direct_beta")
        head = Gr00tN1d7ActionHead(config)
        assert head.confidence_head.num_outputs == 5

    def test_gap_ratio_default_keeps_4out_head(self):
        config = _routing_config(confidence_routing_mode="gap_ratio")
        head = Gr00tN1d7ActionHead(config)
        assert head.confidence_head.num_outputs == 4

    def test_closed_form_beta_star_exact(self):
        """y = 0.7·h + 0.3·z0 → β* = 0.7 (masked least squares)."""
        B, T, D = 3, 5, 4
        h = torch.randn(B, T, D)
        z0 = torch.randn(B, T, D)
        y = 0.7 * h + 0.3 * z0
        mask = torch.ones(B, T, D)
        beta = Gr00tN1d7ActionHead._closed_form_beta_star(y, h, z0, mask)
        torch.testing.assert_close(beta, torch.full((B,), 0.7))

    def test_closed_form_beta_star_extremes_and_degenerate(self):
        B, T, D = 4, 5, 4
        h = torch.randn(B, T, D)
        z0 = torch.randn(B, T, D)
        mask = torch.ones(B, T, D)
        # y = z0 -> 0 ; y = h -> 1
        beta = Gr00tN1d7ActionHead._closed_form_beta_star(
            torch.stack([z0[0], h[1], z0[2], z0[3]]), h, z0, mask
        )
        assert beta[0].item() == 0.0 and beta[1].item() == 1.0
        # h == z0 (degenerate) -> 0.5 default
        beta_deg = Gr00tN1d7ActionHead._closed_form_beta_star(
            z0[2:], z0[2:], z0[2:], mask[2:]
        )
        assert torch.all(beta_deg == 0.5)

    def test_closed_form_beta_star_respects_mask(self):
        B, T, D = 1, 2, 2
        h = torch.ones(B, T, D)
        z0 = torch.zeros(B, T, D)
        # masked-out half says β=1, live half says β=0 -> mask must win
        y = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])
        mask = torch.tensor([[[1.0, 1.0], [0.0, 0.0]]])
        beta = Gr00tN1d7ActionHead._closed_form_beta_star(y, h, z0, mask)
        torch.testing.assert_close(beta, torch.zeros(B))

    def test_direct_beta_inference_uses_fifth_output(self):
        """bias[4]=0.8 (others 0) → β̂ = 0.8 directly, no gap ratio."""
        config = _routing_config(confidence_routing_mode="direct_beta")
        torch.manual_seed(0)
        head = Gr00tN1d7ActionHead(config)
        head.eval()
        with torch.no_grad():
            head.confidence_head.readout[2].bias.copy_(
                torch.tensor([0.0, 0.0, 0.0, 0.0, 0.8])
            )
        backbone_output, action_input = _routing_inputs(config)
        del action_input["action"]
        out = head.get_action(backbone_output, action_input)
        torch.testing.assert_close(out["beta_hat"], torch.full((2,), 0.8))

    def test_direct_beta_inference_clamps_to_unit_interval(self):
        config = _routing_config(confidence_routing_mode="direct_beta")
        torch.manual_seed(0)
        head = Gr00tN1d7ActionHead(config)
        head.eval()
        with torch.no_grad():
            head.confidence_head.readout[2].bias.copy_(
                torch.tensor([0.0, 0.0, 0.0, 0.0, 2.0])
            )
        backbone_output, action_input = _routing_inputs(config)
        del action_input["action"]
        out = head.get_action(backbone_output, action_input)
        assert torch.all(out["beta_hat"] == 1.0)

    def test_routing_mode_validation(self):
        from gr00t.configs.finetune_config import FinetuneConfig

        try:
            FinetuneConfig(
                base_model_path="m",
                dataset_path="d",
                embodiment_tag="new_embodiment",
                confidence_routing_mode="bogus",
            )
        except ValueError as exc:
            assert "confidence_routing_mode" in str(exc)
            return
        raise AssertionError("expected ValueError for bogus routing mode")
