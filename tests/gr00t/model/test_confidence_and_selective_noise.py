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
Tests for T0-2 (ΔFM selective source-noise mask) and v4.1 (confidence query
tokens + detached one-step endpoint gap target):

- build_action_noise_mask: gripper/padding exemption layout.
- _source_noise / _build_flow_source: masked eta on history & memory arms.
- ConfidenceGapHead: zero-init, reset-from-garbage, readout shape.
- process_backbone_output: K/C tail split for memory/confidence queries.
- Training forward: confidence loss present, finite, amortised by every_k.
- Inference: confidence readout exposed under "confidence".
- Gr00tN1d7._backbone_query_embeds: memory+confidence concatenation order.
"""

from types import SimpleNamespace

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7, Gr00tN1d7ActionHead
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import build_action_noise_mask
from gr00t.model.modules.action_memory_encoder import ConfidenceGapHead
import torch
from transformers.feature_extraction_utils import BatchFeature

from tests.gr00t.model.test_p0_flow_source_and_mask import (
    _make_action_input,
    _make_backbone_output,
    _small_config,
)


# ---------------------------------------------------------------------------
# T0-2: build_action_noise_mask (pure function, processor-side)
# ---------------------------------------------------------------------------


class TestBuildActionNoiseMask:
    def test_gripper_column_exempt(self):
        # eef_9d (9) + gripper_position (1) + joint_position (7) = 17 cols.
        mask = build_action_noise_mask(
            [("eef_9d", 9), ("gripper_position", 1), ("joint_position", 7)],
            max_action_horizon=4,
            max_action_dim=20,
            valid_horizon=3,
        )
        assert mask.shape == (4, 20)
        assert mask[:3, :9].all()  # eef: noisy
        assert not mask[:3, 9].any()  # gripper: exempt
        assert mask[:3, 10:17].all()  # joint: noisy
        assert not mask[:, 17:].any()  # padding cols: exempt
        assert not mask[3:, :].any()  # padding rows: exempt

    def test_full_mask_when_no_exempt_key(self):
        mask = build_action_noise_mask(
            [("joint_position", 7)], max_action_horizon=4, max_action_dim=10, valid_horizon=4
        )
        assert mask[:, :7].all()
        assert not mask[:, 7:].any()

    def test_none_valid_horizon_keeps_all_rows(self):
        mask = build_action_noise_mask(
            [("eef", 3)], max_action_horizon=5, max_action_dim=6, valid_horizon=None
        )
        assert mask[:, :3].all()  # all 5 horizon rows active
        assert not mask[:, 3:].any()

    def test_overflow_width_clipped(self):
        mask = build_action_noise_mask(
            [("a", 5), ("gripper_x", 100)], max_action_horizon=2, max_action_dim=8
        )
        assert mask[:, :5].all()
        assert not mask[:, 5:].any()


# ---------------------------------------------------------------------------
# T0-2: masked eta in the action head (history & memory arms)
# ---------------------------------------------------------------------------


def _noise_mask(config, batch_size):
    """All-ones mask with the LAST action column exempted (fake gripper)."""
    mask = torch.ones(batch_size, config.action_horizon, config.max_action_dim)
    mask[:, :, -1] = 0.0
    return mask


class TestSelectiveSourceNoise:
    def test_history_eta_masked_dims_stay_clean(self):
        config = _small_config(
            flow_source_mode="history",
            flow_source_action_noise_std=0.5,
            flow_source_noise_selective=True,
        )
        torch.manual_seed(0)
        head = Gr00tN1d7ActionHead(config)
        actions = torch.randn(2, config.action_horizon, config.max_action_dim)
        history = torch.randn_like(actions)
        noise = torch.randn_like(actions)
        action_input = BatchFeature(
            data={
                "history_action": history,
                "has_action_history": torch.ones(2),
                "action_noise_mask": _noise_mask(config, 2),
            }
        )
        source = head._build_flow_source(actions, action_input, noise)
        delta = source - history
        assert torch.all(delta[:, :, -1] == 0)  # exempt dim: no noise
        assert delta[:, :, :-1].abs().sum() > 0  # continuous dims: noisy

    def test_selective_off_ignores_mask(self):
        config = _small_config(
            flow_source_mode="history",
            flow_source_action_noise_std=0.5,
            flow_source_noise_selective=False,
        )
        torch.manual_seed(0)
        head = Gr00tN1d7ActionHead(config)
        actions = torch.randn(2, config.action_horizon, config.max_action_dim)
        history = torch.randn_like(actions)
        noise = torch.randn_like(actions)
        action_input = BatchFeature(
            data={
                "history_action": history,
                "has_action_history": torch.ones(2),
                "action_noise_mask": _noise_mask(config, 2),
            }
        )
        source = head._build_flow_source(actions, action_input, noise)
        assert (source - history)[:, :, -1].abs().sum() > 0  # gripper noisy too

    def test_missing_mask_falls_back_to_plain_noise(self):
        config = _small_config(
            flow_source_mode="history",
            flow_source_action_noise_std=0.5,
            flow_source_noise_selective=True,
        )
        torch.manual_seed(0)
        head = Gr00tN1d7ActionHead(config)
        actions = torch.randn(2, config.action_horizon, config.max_action_dim)
        history = torch.randn_like(actions)
        noise = torch.randn_like(actions)
        action_input = BatchFeature(
            data={"history_action": history, "has_action_history": torch.ones(2)}
        )
        source = head._build_flow_source(actions, action_input, noise)
        assert (source - history)[:, :, -1].abs().sum() > 0

    def test_memory_arm_eta_masked(self):
        config = _small_config(
            flow_source_mode="memory",
            memory_start_enabled=True,
            memory_start_noise_std=0.5,
            flow_source_noise_selective=True,
            memory_start_hidden_dim=32,
            memory_num_queries=4,
            memory_query_dim=8,
            memory_query_path="post",
        )
        torch.manual_seed(0)
        head = Gr00tN1d7ActionHead(config)
        actions = torch.randn(2, config.action_horizon, config.max_action_dim)
        noise = torch.randn_like(actions)
        mem_start = torch.randn_like(actions)
        action_input = BatchFeature(data={"action_noise_mask": _noise_mask(config, 2)})
        source = head._build_flow_source(actions, action_input, noise, memory_start=mem_start)
        delta = source - mem_start
        assert torch.all(delta[:, :, -1] == 0)
        assert delta[:, :, :-1].abs().sum() > 0


# ---------------------------------------------------------------------------
# v4.1: ConfidenceGapHead module
# ---------------------------------------------------------------------------


class TestConfidenceGapHead:
    def test_zero_init_gives_zero_readout(self):
        head = ConfidenceGapHead(cond_dim=16, num_queries=3, hidden_dim=8)
        out = head(torch.randn(2, 3, 16))
        assert out.shape == (2, 2)
        assert torch.all(out == 0)

    def test_queries_shape_and_grad(self):
        head = ConfidenceGapHead(cond_dim=16, num_queries=4)
        assert head.queries.shape == (4, 16)
        assert head.queries.requires_grad

    def test_reset_recovers_from_garbage(self):
        head = ConfidenceGapHead(cond_dim=16, num_queries=3, hidden_dim=8)
        with torch.no_grad():
            head.queries.fill_(1.0e31)
            head.readout[2].weight.fill_(2.3e31)
        garbage_out = head(torch.randn(1, 3, 16))
        assert (not torch.isfinite(garbage_out).all()) or garbage_out.abs().max() > 1e10
        head.reset_parameters()
        out = head(torch.randn(1, 3, 16))
        assert torch.isfinite(out).all() and torch.all(out == 0)

    def test_readout_breaks_zero_init_after_update(self):
        head = ConfidenceGapHead(cond_dim=8, num_queries=2, hidden_dim=8)
        opt = torch.optim.SGD(head.parameters(), lr=0.1)
        target = torch.tensor([[0.5, 0.9]])
        loss = ((head(torch.randn(1, 2, 8)) - target) ** 2).mean()
        loss.backward()
        opt.step()
        assert head.readout[2].weight.abs().sum() > 0


# ---------------------------------------------------------------------------
# v4.1: tail split in process_backbone_output
# ---------------------------------------------------------------------------


def _conf_config(**overrides):
    defaults = dict(
        confidence_tokens_enabled=True,
        confidence_num_queries=2,
        confidence_loss_weight=0.1,
        confidence_every_k=1,
    )
    defaults.update(overrides)
    return _small_config(**defaults)


def _tail_backbone_output(config, batch_size=2, seq_len=8, num_tail=0, tail_fill=None):
    out = _make_backbone_output(config, batch_size, seq_len)
    if num_tail > 0:
        tail = torch.randn(batch_size, num_tail, config.backbone_embedding_dim)
        if tail_fill is not None:
            tail = torch.full(
                (batch_size, num_tail, config.backbone_embedding_dim), float(tail_fill)
            )
        out["memory_query_features"] = tail
    return out


class TestTailQuerySplit:
    def test_confidence_only_gets_full_tail(self):
        config = _conf_config()
        head = Gr00tN1d7ActionHead(config)
        out = head.process_backbone_output(
            _tail_backbone_output(config, num_tail=config.confidence_num_queries)
        )
        assert "confidence_query_features" in out
        assert out["confidence_query_features"].shape == (
            2,
            config.confidence_num_queries,
            config.backbone_embedding_dim,
        )
        # No memory arm: the shared key must be consumed (not leak through).
        assert "memory_query_features" not in out

    def test_memory_backbone_plus_confidence_split_order(self):
        K, C = 4, 2
        config = _conf_config(
            memory_start_enabled=True,
            flow_source_mode="memory",
            memory_start_hidden_dim=32,
            memory_num_queries=K,
            memory_query_dim=8,
            memory_query_path="backbone",
            confidence_num_queries=C,
        )
        head = Gr00tN1d7ActionHead(config)
        B = 2
        out0 = _make_backbone_output(config, B)
        tail = torch.zeros(B, K + C, config.backbone_embedding_dim)
        tail[:, :K] = 1.0  # memory rows
        tail[:, K:] = 2.0  # confidence rows
        out0["memory_query_features"] = tail
        out = head.process_backbone_output(out0)
        assert out["memory_query_features"].shape == (B, K, config.backbone_embedding_dim)
        assert torch.all(out["memory_query_features"] == 1.0)
        assert out["confidence_query_features"].shape == (B, C, config.backbone_embedding_dim)
        assert torch.all(out["confidence_query_features"] == 2.0)

    def test_memory_post_plus_confidence(self):
        """post path: tail = C rows only; memory features rebuilt by v2 append."""
        K, C = 4, 2
        config = _conf_config(
            memory_start_enabled=True,
            flow_source_mode="memory",
            memory_start_hidden_dim=32,
            memory_num_queries=K,
            memory_query_dim=8,
            memory_query_path="post",
            confidence_num_queries=C,
        )
        head = Gr00tN1d7ActionHead(config)
        out = head.process_backbone_output(_tail_backbone_output(config, num_tail=C))
        assert out["confidence_query_features"].shape == (2, C, config.backbone_embedding_dim)
        assert out["memory_query_features"].shape == (2, K, config.backbone_embedding_dim)

    def test_no_confidence_tail_passes_through(self):
        """Without a confidence head the memory tail is unchanged (legacy)."""
        K = 4
        config = _small_config(
            memory_start_enabled=True,
            flow_source_mode="memory",
            memory_start_hidden_dim=32,
            memory_num_queries=K,
            memory_query_dim=8,
            memory_query_path="backbone",
        )
        head = Gr00tN1d7ActionHead(config)
        out = head.process_backbone_output(_tail_backbone_output(config, num_tail=K))
        assert "confidence_query_features" not in out
        assert out["memory_query_features"].shape == (2, K, config.backbone_embedding_dim)


# ---------------------------------------------------------------------------
# v4.1: training forward + inference readout
# ---------------------------------------------------------------------------


class TestConfidenceForward:
    def _forward(self, config, seed=0):
        torch.manual_seed(seed)
        head = Gr00tN1d7ActionHead(config)
        head.train()
        backbone_output = _tail_backbone_output(config, num_tail=config.confidence_num_queries)
        return head, head.forward(backbone_output, _make_action_input(config))

    def test_confidence_loss_present_and_finite(self):
        config = _conf_config()
        _, out = self._forward(config)
        assert out["confidence_loss"] is not None
        assert torch.isfinite(out["confidence_loss"])
        assert torch.isfinite(out["loss"])
        for key in ("conf_loss", "gap_target_mean", "gap_pred_mean"):
            assert key in out["debug_stats"]

    def test_zero_init_predicts_zero_gap(self):
        config = _conf_config()
        _, out = self._forward(config)
        # Zero-init readout: ĝ = 0, so conf_loss == MSE(0, gap) + pinball(0, gap).
        assert out["debug_stats"]["gap_pred_mean"] == 0.0

    def test_every_k_skips_computation(self):
        config = _conf_config(confidence_every_k=2)
        torch.manual_seed(0)
        head = Gr00tN1d7ActionHead(config)
        head.train()
        head.set_anchor_lambda_progress(step=1, max_steps=10)  # 1 % 2 != 0
        out = head.forward(
            _tail_backbone_output(config, num_tail=config.confidence_num_queries),
            _make_action_input(config),
        )
        assert out["confidence_loss"] is None
        assert "conf_loss" not in out["debug_stats"]
        head.set_anchor_lambda_progress(step=2, max_steps=10)  # 2 % 2 == 0
        out = head.forward(
            _tail_backbone_output(config, num_tail=config.confidence_num_queries),
            _make_action_input(config),
        )
        assert out["confidence_loss"] is not None

    def test_missing_features_raise(self):
        config = _conf_config()
        torch.manual_seed(0)
        head = Gr00tN1d7ActionHead(config)
        head.train()
        try:
            head.forward(_make_backbone_output(config), _make_action_input(config))
        except RuntimeError as exc:
            assert "confidence_query_features" in str(exc)
            return
        raise AssertionError("expected RuntimeError for missing confidence_query_features")

    def test_get_action_exposes_confidence(self):
        config = _conf_config()
        torch.manual_seed(0)
        head = Gr00tN1d7ActionHead(config)
        head.eval()
        action_input = _make_action_input(config)
        del action_input["action"]
        backbone_output = _tail_backbone_output(config, num_tail=config.confidence_num_queries)
        out = head.get_action(backbone_output, action_input)
        assert "confidence" in out
        assert out["confidence"].shape == (2, 2)
        # Zero-init readout predicts zero gap.
        assert torch.all(out["confidence"] == 0)

    def test_gaussian_arm_also_trains_head(self):
        """Arm independence: flow_source_mode stays gaussian, conf loss on."""
        config = _conf_config(flow_source_mode="gaussian")
        _, out = self._forward(config)
        assert out["confidence_loss"] is not None


# ---------------------------------------------------------------------------
# v4.1: Gr00tN1d7._backbone_query_embeds concatenation (memory first)
# ---------------------------------------------------------------------------


class TestBackboneQueryEmbedsConcat:
    def _stub(self, config):
        head = Gr00tN1d7ActionHead(config)
        return SimpleNamespace(action_head=head, config=config)

    def test_memory_plus_confidence_order(self):
        K, C = 4, 2
        config = _conf_config(
            memory_start_enabled=True,
            flow_source_mode="memory",
            memory_start_hidden_dim=32,
            memory_num_queries=K,
            memory_query_dim=8,
            memory_query_path="backbone",
            confidence_num_queries=C,
        )
        stub = self._stub(config)
        # Mark the two parameter blocks so the concat order is observable.
        with torch.no_grad():
            stub.action_head.memory_encoder.queries.fill_(1.0)
            stub.action_head.confidence_head.queries.fill_(2.0)
        embeds = Gr00tN1d7._backbone_query_embeds(stub, batch_size=3)
        assert embeds.shape == (3, K + C, config.backbone_embedding_dim)
        assert torch.all(embeds[:, :K] == 1.0)
        assert torch.all(embeds[:, K:] == 2.0)

    def test_confidence_only(self):
        config = _conf_config()
        embeds = Gr00tN1d7._backbone_query_embeds(self._stub(config), batch_size=2)
        assert embeds.shape == (2, config.confidence_num_queries, config.backbone_embedding_dim)

    def test_post_path_memory_still_threads_confidence(self):
        config = _conf_config(
            memory_start_enabled=True,
            flow_source_mode="memory",
            memory_start_hidden_dim=32,
            memory_num_queries=4,
            memory_query_dim=8,
            memory_query_path="post",
            confidence_num_queries=2,
        )
        embeds = Gr00tN1d7._backbone_query_embeds(self._stub(config), batch_size=2)
        # post memory queries do NOT ride the backbone; confidence does.
        assert embeds.shape == (2, 2, config.backbone_embedding_dim)

    def test_both_disabled_returns_none(self):
        config = _small_config()
        assert Gr00tN1d7._backbone_query_embeds(self._stub(config), 2) is None


# ---------------------------------------------------------------------------
# FinetuneConfig validation (CLI surface)
# ---------------------------------------------------------------------------


class TestConfidenceConfigValidation:
    _REQUIRED = dict(base_model_path="m", dataset_path="d", embodiment_tag="new_embodiment")

    def test_defaults_off(self):
        from gr00t.configs.finetune_config import FinetuneConfig

        cfg = FinetuneConfig(**self._REQUIRED)
        assert cfg.flow_source_noise_selective is False
        assert cfg.confidence_tokens_enabled is False
        assert cfg.confidence_num_queries == 4
        assert cfg.confidence_loss_weight == 0.1
        assert cfg.confidence_every_k == 1

    def test_invalid_values_raise(self):
        from gr00t.configs.finetune_config import FinetuneConfig

        for bad in (
            dict(confidence_num_queries=0),
            dict(confidence_loss_weight=-0.1),
            dict(confidence_every_k=0),
        ):
            try:
                FinetuneConfig(**self._REQUIRED, confidence_tokens_enabled=True, **bad)
            except ValueError:
                continue
            raise AssertionError(f"expected ValueError for {bad}")
