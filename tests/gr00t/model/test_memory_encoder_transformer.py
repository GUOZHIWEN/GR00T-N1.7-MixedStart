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

"""Unit tests for ActionMemoryEncoderTransformer (P2 transformer trunk).

Verifies the contract shared with the MLP ActionMemoryEncoder (K learnable
queries, zero-initialised output head, ``forward(query_features,
embodiment_id) -> (B, H, D)``), the gradient flow through every submodule
(after the zero-init head has left its exact-zero initial state), and the
parameter-efficiency claim (~58x fewer params than the MLP trunk).
"""

import torch

from gr00t.model.modules.action_memory_encoder import (
    ActionMemoryEncoder,
    ActionMemoryEncoderTransformer,
)


def _build(**overrides):
    kwargs = dict(
        cond_dim=2048,
        action_dim=132,
        action_horizon=40,
        num_embodiments=32,
        d_model=256,
        num_layers=2,
        num_queries=64,
    )
    kwargs.update(overrides)
    return ActionMemoryEncoderTransformer(**kwargs)


class TestInterface:
    def test_output_shape(self):
        enc = _build()
        z0 = enc(torch.randn(3, 64, 2048), torch.tensor([0, 5, 31]))
        assert z0.shape == (3, 40, 132)

    def test_owns_learnable_queries_like_mlp_variant(self):
        """The action head threads ``queries`` through the frozen backbone;
        both trunks must expose the same (K, cond_dim) Parameter."""
        enc = _build(num_queries=16)
        assert enc.queries.shape == (16, 2048)
        assert enc.num_queries == 16
        assert enc.queries.requires_grad

    def test_param_efficiency_vs_mlp(self):
        mlp = ActionMemoryEncoder(
            cond_dim=2048, action_dim=132, action_horizon=40, num_embodiments=32
        )
        tr = _build()
        n_mlp = sum(p.numel() for p in mlp.parameters())
        n_tr = sum(p.numel() for p in tr.parameters())
        assert n_tr < n_mlp / 10  # spec: ~179M vs ~3.1M (~58x)

    def test_zero_init_head_gives_zero_z0(self):
        """UVM bitwise-identity preconditioning: at init z0 == 0 so the flow
        source is not yanked away from the pretrained distribution."""
        enc = _build()
        z0 = enc(torch.randn(2, 64, 2048), torch.tensor([0, 1]))
        assert torch.count_nonzero(z0) == 0

    def test_bf16_module_forward(self):
        """Deployment condition under --load-bf16: module AND input bf16."""
        enc = _build().to(torch.bfloat16)
        z0 = enc(torch.randn(2, 64, 2048, dtype=torch.bfloat16), torch.tensor([1, 2]))
        assert z0.dtype == torch.bfloat16
        assert z0.shape == (2, 40, 132)


class TestGradientFlow:
    def test_full_chain_receives_gradient(self):
        """With a non-zero head (post step-1 state), every submodule — input
        projection, query self-attention, step-queries, embodiment embedding,
        cross-attention — must receive gradient."""
        enc = _build()
        with torch.no_grad():
            enc.head.weight.normal_(0, 0.01)
        x = torch.randn(3, 64, 2048, requires_grad=True)
        loss = (enc(x, torch.tensor([0, 5, 31])) - torch.randn(3, 40, 132)).pow(2).mean()
        loss.backward()
        assert x.grad.abs().sum() > 0
        assert enc.step_queries.grad.abs().sum() > 0
        assert enc.embodiment_embed.weight.grad.abs().sum() > 0
        assert enc.input_proj.weight.grad.abs().sum() > 0
        for p in enc.self_attn.parameters():
            assert p.grad is not None and p.grad.abs().sum() > 0
        for p in enc.cross_attn.parameters():
            assert p.grad is not None and p.grad.abs().sum() > 0

    def test_zero_head_blocks_upstream_grad_at_init(self):
        """Documented zero-init property: at EXACT init the upstream gradient
        is zero through the (zero) head — the head itself still trains, so
        step 1 unblocks the trunk. Mirrors the MLP variant's contract."""
        enc = _build()
        x = torch.randn(2, 64, 2048, requires_grad=True)
        loss = (enc(x, torch.tensor([0, 1])) - torch.randn(2, 40, 132)).pow(2).mean()
        loss.backward()
        assert enc.head.weight.grad.abs().sum() > 0
        assert torch.count_nonzero(x.grad) == 0

    def test_embodiment_id_differentiates_output(self):
        enc = _build()
        with torch.no_grad():
            enc.head.weight.normal_(0, 0.01)
        x = torch.randn(1, 64, 2048)
        assert (enc(x, torch.tensor([0])) - enc(x, torch.tensor([7]))).abs().sum() > 0


class TestResetParameters:
    def test_reset_rezeros_head_and_keeps_finite(self):
        """The meta-device safeguard path (setup.py / gr00t_policy.py): reset
        must restore the zero head and finite real tensors throughout."""
        enc = _build()
        with torch.no_grad():
            enc.head.weight.normal_(0, 0.01)
            enc.queries.mul_(1000)
        enc.reset_parameters()
        assert torch.count_nonzero(enc.head.weight) == 0
        assert torch.count_nonzero(enc.head.bias) == 0
        for name, p in enc.named_parameters():
            assert torch.isfinite(p).all(), f"{name} not finite after reset"

    def test_reset_reinitializes_queries(self):
        enc = _build()
        before = enc.queries.detach().clone()
        with torch.no_grad():
            enc.queries.zero_()
        enc.reset_parameters()
        assert not torch.equal(before, enc.queries.detach())
        assert enc.queries.abs().sum() > 0
