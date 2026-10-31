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

"""Learnable action memory-start encoder (P2, UVM-style, learnable-query form).

Transplant of Unified-Vision-Modeling's memory-start idea
(``z0 = StartEncoderVRES(und_branch, c_text, queries)``, bridge flow matching
``z_t = t*y0 + (1-t)*s`` with ``s = z0``) onto the GR00T N1.7 action DiT.

UVM-faithful input path: this module owns K LEARNABLE QUERY TOKENS
(``self.queries``, cond_dim each). The action head appends them to the frozen
backbone's VL feature sequence and runs the combined sequence through the
*trainable* ``vl_self_attention`` (the frozen VLM itself is never touched).
The K query outputs — content-adaptive semantic views of the observation +
instruction — are then LayerNorm'ed, projected per token, flattened, and
mapped by an MLP trunk + per-embodiment head to the memory start ``z0`` of
shape (B, action_horizon, max_action_dim), which serves as (part of) the
flow-matching source endpoint instead of pure Gaussian noise.

Design notes:
  * Per-embodiment output head (CategorySpecificLinear), consistent with the
    state/action encoders in this codebase.
  * The output head is ZERO-initialised (UVM "bitwise-identity
    preconditioning"): at init z0 == 0, so enabling the module does not yank
    the source distribution away from the pretrained one on step 0.
  * The anchor loss ``lambda * ||z0 - a0||^2`` lives in the action head
    (gr00t_n1d7.py), not here.
  * ``reset_parameters`` MUST be re-run after ``AutoModel.from_pretrained``
    when the checkpoint lacks ``memory_encoder.*`` keys (meta-device
    instantiation no-ops __init__ inits and HF ``_init_weights`` does not
    cover bare Parameters); setup.py / gr00t_policy.py do this.
"""

import torch
from torch import nn
import torch.nn.functional as F

from gr00t.model.modules.embodiment_conditioned_mlp import CategorySpecificLinear


class ActionMemoryEncoder(nn.Module):
    """K learnable queries (via vl_self_attention) -> action-chunk start z0.

    Args:
        cond_dim: dimension of the backbone/VL features (= query token width).
        action_dim: max action dimension (padded action space).
        action_horizon: chunk length H.
        num_embodiments: number of embodiment categories for the output head.
        hidden_dim: trunk hidden width.
        num_queries: K, number of learnable query tokens (UVM uses 64).
        query_dim: per-query projection width before the flatten+trunk.
    """

    def __init__(
        self,
        cond_dim: int,
        action_dim: int,
        action_horizon: int,
        num_embodiments: int,
        hidden_dim: int = 1024,
        num_queries: int = 64,
        query_dim: int = 64,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.num_queries = num_queries

        # K learnable query tokens; the action head appends them to the VL
        # feature sequence before vl_self_attention (UVM: K=64 learned queries
        # through the understanding branch).
        self.queries = nn.Parameter(0.02 * torch.randn(num_queries, cond_dim))

        self.norm = nn.LayerNorm(cond_dim)
        self.query_proj = nn.Linear(cond_dim, query_dim)
        self.trunk = nn.Sequential(
            nn.Linear(num_queries * query_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        # Per-embodiment linear head: (B, 1, hidden) -> (B, 1, H * action_dim)
        self.head = CategorySpecificLinear(
            num_embodiments, hidden_dim, action_horizon * action_dim
        )
        self.reset_parameters()

    def reset_parameters(self):
        """(Re-)initialise all encoder weights with REAL tensors.

        The output head is ZERO-initialised (UVM "bitwise-identity
        preconditioning"): at init z0 == 0, so enabling the module does not
        yank the source distribution away from the pretrained one on step 0.

        NOTE: this MUST be called explicitly after ``AutoModel.from_pretrained``
        when the checkpoint lacks ``memory_encoder.*`` keys. from_pretrained
        instantiates the model on the meta device (where __init__'s inits are
        no-ops) and its ``_init_weights`` only covers nn.Linear/Embedding —
        CategorySpecificLinear's bare Parameters (head.W/head.b) and
        ``self.queries`` would otherwise stay UNINITIALISED GARBAGE (~1e31),
        poisoning the flow source with inf/NaN from step 0 (the smoke_mem_hard
        bug). See setup.py / gr00t_policy.py.
        """
        nn.init.normal_(self.queries, std=0.02)
        self.norm.reset_parameters()
        self.query_proj.reset_parameters()
        for lin in (self.trunk[0], self.trunk[2]):
            lin.reset_parameters()
        nn.init.zeros_(self.head.W)
        nn.init.zeros_(self.head.b)

    def forward(
        self,
        query_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            query_features: (B, K, cond_dim) hidden states at the K appended
                query positions, AFTER vl_self_attention (see
                Gr00tN1d7ActionHead.process_backbone_output).
            embodiment_id: (B,) long.
            state: accepted for call-site uniformity (db18); always None in
                practice — read_state is only implemented on the transformer
                trunk (the CLI rejects the MLP + read_state combination).

        Returns:
            z0: (B, action_horizon, action_dim) memory start.
        """
        B = query_features.shape[0]
        # fp32 LayerNorm: real VL features can be large-magnitude, and bf16
        # normalisation is needlessly imprecise here.
        n32 = F.layer_norm(
            query_features.float(),
            self.norm.normalized_shape,
            self.norm.weight.float(),
            self.norm.bias.float(),
            self.norm.eps,
        )
        q = self.query_proj(n32.to(dtype=query_features.dtype))  # (B, K, query_dim)
        h = self.trunk(q.reshape(B, self.num_queries * q.shape[-1]))  # (B, hidden)
        z0 = self.head(h.unsqueeze(1), embodiment_id)  # (B, 1, H * D)
        return z0.view(-1, self.action_horizon, self.action_dim)


class ActionMemoryEncoderTransformer(nn.Module):
    """Transformer trunk variant of the action memory-start encoder.

    Same contract as :class:`ActionMemoryEncoder` (owns the K learnable query
    tokens ``self.queries``; ``forward(query_features, embodiment_id)`` ->
    z0 of shape (B, action_horizon, action_dim); zero-initialised output
    head), but the flatten + 2-layer MLP + per-embodiment
    CategorySpecificLinear head (~179M params, 173M of them in the 32
    embodiment heads) is replaced by a horizon-structured Transformer:

      1. fp32 LayerNorm + Linear(cond_dim -> d_model) on the K query features;
      2. ``num_layers`` pre-LN self-attention layers over the K queries
         (content-adaptive interaction between semantic views);
      3. H learnable step-queries (one per action-chunk step) plus a
         per-embodiment embedding (nn.Embedding), cross-attending once to
         the K query outputs;
      4. a SHARED per-step FFN and a shared zero-initialised
         Linear(d_model -> action_dim) output head.

    Why this generalises better than the MLP trunk: the MLP flattens
    (K * query_dim) and regresses all H * D outputs in one shot, so each
    embodiment head must memorise a flat lookup (173M params for 32
    embodiments) with no weight sharing across chunk steps or embodiments.
    Here the K queries interact via attention (permutation-structured,
    content-adaptive), every chunk step shares the same decoder weights
    (H only changes the number of step-queries), and embodiments enter as a
    compact embedding instead of a full per-embodiment matrix — ~3.1M params
    total (~58x fewer), with cross-horizon weight sharing by construction.
    """

    def __init__(
        self,
        cond_dim: int,
        action_dim: int,
        action_horizon: int,
        num_embodiments: int,
        d_model: int = 256,
        num_layers: int = 2,
        num_queries: int = 64,
        num_heads: int = 8,
        dropout: float = 0.0,
        bottleneck_dim: int = 0,
        var_head: bool = False,
        read_state: bool = False,
        state_dim: int = 0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.num_queries = num_queries
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim
        # db17 (paper §8 item 11-①, survey §13.81): explicit source-
        # distribution supervision (LeaP-style). When enabled, a second output
        # head emits per-dim log-variance so the anchor can be a heteroscedastic
        # Gaussian NLL instead of a fixed-variance MSE — MSE is the σ²=const
        # special case whose optimum is E[y|c] (mode averaging); the learned
        # σ(c) absorbs irreducible conditional variance and lets μ concentrate
        # on ONE mode. Deployment still uses z0 = μ(c) only (deterministic
        # start asset preserved); σ is a training-time supervision device.
        self.var_head_enabled = bool(var_head)
        # db18 T2 (paper §8 item 11-⑤): read the flattened proprioceptive
        # state as one extra cross-attention KV token — z₀ = f(H, q). state_dim
        # is max_state_dim × state_history_length (132 × 1 = 132 for DROID).
        # state_proj is a plain (non-zero-init) input projection.
        self.read_state = bool(read_state)
        if self.read_state:
            if state_dim <= 0:
                raise ValueError("read_state=True requires state_dim > 0")
            self.state_proj = nn.Linear(state_dim, d_model)

        # K learnable query tokens, identical role/plumbing to the MLP
        # variant (threaded through the frozen backbone / vl_self_attention).
        self.queries = nn.Parameter(0.02 * torch.randn(num_queries, cond_dim))

        self.norm = nn.LayerNorm(cond_dim)
        self.input_proj = nn.Linear(cond_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.self_attn = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # H learnable step-queries + per-embodiment conditioning embedding.
        self.step_queries = nn.Parameter(0.02 * torch.randn(action_horizon, d_model))
        self.embodiment_embed = nn.Embedding(num_embodiments, d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.out_norm = nn.LayerNorm(d_model)
        # db13 (§6.4 direction 2): optional hard low-dimensional bottleneck on
        # z₀'s content. When > 0, ALL information reaching the output head must
        # pass through an L-dim task-parameter latent: the (B, H, d) trunk
        # output is pooled, projected to L dims, and re-expanded — z₀ can then
        # only carry a rank-L parameterization. NOTE (review P0-1): with the
        # anchor target still being y, the bottleneck does NOT lower the
        # conditional-mean floor — z₀* = proj_M(E[y|c]) has floor
        # E[Var(y|c)] + ‖proj_M(E[y|c]) − E[y|c]‖² ≥ E[Var(y|c)]. What the arm
        # tests is whether the LEARNABLE segment-head content is low-rank,
        # i.e. whether an L-dim manifold suffices to approximate E[y|c] at
        # segment heads. Floor reduction would require re-targeting the anchor
        # at explicit low-dim task parameters (a further variant, not db13).
        # 0 = disabled (legacy behaviour).
        if bottleneck_dim > 0:
            self.to_latent = nn.Linear(d_model, bottleneck_dim)
            self.from_latent = nn.Linear(bottleneck_dim, action_horizon * d_model)
        else:
            self.to_latent = None
            self.from_latent = None
        # Shared per-step output head, ZERO-initialised (same UVM
        # bitwise-identity preconditioning as the MLP variant).
        self.head = nn.Linear(d_model, action_dim)
        if self.var_head_enabled:
            # db17: log-variance head for the NLL anchor. Weight zero-init;
            # bias init log(σ0²) with σ0 = 0.5 lives in reset_parameters (a
            # plain Linear reset would overwrite it).
            self.head_logvar = nn.Linear(d_model, action_dim)
        self.reset_parameters()

    def reset_parameters(self):
        """(Re-)initialise all encoder weights with REAL tensors; the output
        head is ZERO-initialised (at init z0 == 0, so enabling the module does
        not yank the source distribution away from the pretrained one).

        MUST be called after ``AutoModel.from_pretrained`` when the checkpoint
        lacks ``memory_encoder.*`` keys (meta-device instantiation no-ops
        __init__ inits; bare Parameters such as ``queries``/``step_queries``/
        ``cross_attn.in_proj_*`` would otherwise stay uninitialised garbage —
        the smoke_mem_hard bug class). See setup.py / gr00t_policy.py.
        """
        nn.init.normal_(self.queries, std=0.02)
        nn.init.normal_(self.step_queries, std=0.02)
        self.embodiment_embed.reset_parameters()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            elif isinstance(m, nn.LayerNorm):
                m.reset_parameters()
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.xavier_uniform_(m.in_proj_weight)
                nn.init.zeros_(m.in_proj_bias)
                m.out_proj.reset_parameters()
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        if getattr(self, "var_head_enabled", False):
            # db17: zero weight keeps log σ² uniform across dims at init;
            # bias = log(σ0²) with σ0 = 0.5 → NLL starts as a mildly scaled
            # MSE (½·(μ−y)²/0.25 + log 0.25), then learns per-dim spread.
            nn.init.zeros_(self.head_logvar.weight)
            nn.init.constant_(self.head_logvar.bias, -1.3862943611198906)  # log(0.25)

    def forward(
        self,
        query_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        state: torch.Tensor | None = None,
        return_logvar: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            query_features: (B, K, cond_dim) hidden states at the K appended
                query positions (same input as the MLP variant).
            embodiment_id: (B,) long.
            state: db18 T2 — (B, state_dim) flattened proprioceptive state,
                appended as one extra cross-attention KV token when the module
                was built with ``read_state=True`` (z₀ = f(H, q)). REQUIRED
                then; ignored otherwise.
            return_logvar: db17 — when True (requires ``var_head=True`` at
                construction), also return the per-dim log σ² for the NLL
                anchor. Training-only; inference never requests it (deployment
                uses the deterministic z₀ = μ(c) alone).

        Returns:
            z0: (B, action_horizon, action_dim) memory start; or the tuple
            ``(z0, logvar)`` with logvar (B, H, D) clamped to [-8, 4]
            (σ ∈ [0.018, 7.4]) when ``return_logvar=True``.
        """
        B = query_features.shape[0]
        # fp32 LayerNorm, same rationale as the MLP variant.
        n32 = F.layer_norm(
            query_features.float(),
            self.norm.normalized_shape,
            self.norm.weight.float(),
            self.norm.bias.float(),
            self.norm.eps,
        )
        h = self.input_proj(n32.to(dtype=query_features.dtype))  # (B, K, d)
        h = self.self_attn(h)  # (B, K, d) query-query interaction
        kv = h
        if self.read_state:
            # db18 T2: proprioceptive state as one extra cross-attn KV token.
            if state is None:
                raise RuntimeError(
                    "read_state=True but forward received state=None — the "
                    "caller must pass the flattened state (db18 wiring)."
                )
            # dtype follows the weight (bf16 checkpoint at inference, fp32 in
            # training autocast) — never force .float() into a bf16 Linear.
            s = self.state_proj(state.to(dtype=self.state_proj.weight.dtype)).to(dtype=kv.dtype)  # (B, d)
            kv = torch.cat([kv, s.unsqueeze(1)], dim=1)  # (B, K+1, d)
        q = self.step_queries.unsqueeze(0).expand(B, -1, -1)  # (B, H, d)
        q = q + self.embodiment_embed(embodiment_id).unsqueeze(1)  # (B, H, d)
        x = self.cross_norm(q + self.cross_attn(q, kv, kv, need_weights=False)[0])
        x = self.ffn_norm(x + self.ffn(x))  # (B, H, d)
        if self.to_latent is not None:
            # db13: hard rank-L content bottleneck — every output channel must
            # be reconstructed from the L-dim task-parameter latent.
            latent = self.to_latent(x.mean(dim=1))  # (B, L)
            x = self.from_latent(latent).view(B, self.action_horizon, self.d_model)
        out = self.out_norm(x)
        z0 = self.head(out)  # (B, H, action_dim)
        if not return_logvar:
            return z0
        if not self.var_head_enabled:
            raise RuntimeError(
                "return_logvar=True requires var_head=True at construction "
                "(config memory_encoder_var_head)."
            )
        # db17: per-dim log σ², clamped to [-8, 4] (σ ∈ [0.018, 7.4]) so the
        # NLL anchor can neither explode (σ→0) nor switch off (σ→∞).
        logvar = self.head_logvar(out).clamp(-8.0, 4.0)
        return z0, logvar


class ConfidenceGapHead(nn.Module):
    """v4.1: C confidence query tokens -> per-sample gap prediction (ĝ_mean, ĝ_q90).

    Arm-independent companion of ActionMemoryEncoder: its C learnable query
    tokens are threaded through the FROZEN backbone together with the K memory
    queries (same collator placeholder + inputs_embeds pre-hook mechanism,
    appended after the memory queries), so the readout conditions on the same
    observation + instruction context the DiT sees — without touching frozen
    weights. A small MLP maps the C last-layer hidden states to two scalars
    per sample:

      * ĝ_mean ≈ masked L1 gap ‖ŷ1 − y‖ between the one-step endpoint
        ŷ1 = s + v_θ(s, t=0, c) and the ground-truth chunk (MSE target);
      * ĝ_q90  ≈ the 0.9-quantile of that gap (pinball/quantile loss).

    The training target is detached and lives in the action head
    (gr00t_n1d7.py); computing it costs one extra no-grad DiT forward per
    computed step (``confidence_every_k`` amortises). At inference the readout
    is returned under ``confidence`` for downstream adaptive denoising-step
    policies.

    Same construction discipline as ActionMemoryEncoder: the final readout
    layer is zero-initialised (ĝ starts at 0 = "no gap" prior), and
    ``reset_parameters`` MUST be re-run after ``AutoModel.from_pretrained``
    when the checkpoint lacks ``confidence_head.*`` keys (meta-device trap —
    see ActionMemoryEncoder.reset_parameters).
    """

    def __init__(
        self, cond_dim: int, num_queries: int = 4, hidden_dim: int = 256, num_outputs: int = 2
    ):
        super().__init__()
        assert num_outputs in (2, 4, 5), f"num_outputs must be 2, 4 or 5, got {num_outputs}"
        self.num_queries = num_queries
        self.num_outputs = num_outputs
        # C learnable confidence queries; threaded through the frozen backbone
        # (after the memory queries) via the collator/pre-hook mechanism.
        self.queries = nn.Parameter(0.02 * torch.randn(num_queries, cond_dim))
        self.norm = nn.LayerNorm(cond_dim)
        self.readout = nn.Sequential(
            nn.Linear(num_queries * cond_dim, hidden_dim),
            nn.GELU(),
            # (gap_mean, gap_q90); T1 routing adds (gap_hist, gap_mem).
            nn.Linear(hidden_dim, num_outputs),
        )
        self.reset_parameters()

    def reset_parameters(self):
        """(Re-)initialise with REAL tensors; zero-init the final readout layer.

        MUST be called after ``AutoModel.from_pretrained`` when the checkpoint
        lacks ``confidence_head.*`` keys — bare Parameters materialise as
        uninitialised garbage under meta-device instantiation (the
        smoke_mem_hard bug class). See setup.py / gr00t_policy.py.
        """
        nn.init.normal_(self.queries, std=0.02)
        self.norm.reset_parameters()
        self.readout[0].reset_parameters()
        nn.init.zeros_(self.readout[2].weight)
        nn.init.zeros_(self.readout[2].bias)

    def forward(self, query_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_features: (B, C, cond_dim) hidden states at the C confidence
                query positions (last rows of the backbone's threaded queries).

        Returns:
            (B, num_outputs): (ĝ_mean, ĝ_q90) gap predictions; with T1
                confidence routing (num_outputs=4), additionally
                (ĝ_hist, ĝ_mem) — the per-source one-step endpoint gaps.
        """
        B = query_features.shape[0]
        # fp32 LayerNorm, same rationale as ActionMemoryEncoder.
        n32 = F.layer_norm(
            query_features.float(),
            self.norm.normalized_shape,
            self.norm.weight.float(),
            self.norm.bias.float(),
            self.norm.eps,
        )
        return self.readout(n32.to(dtype=query_features.dtype).reshape(B, -1))


class ConstActionStart(nn.Module):
    """db15 "last cut" ablation (paper §6.5): content-free deterministic start.

    Replaces the conditional memory encoder (mlp/transformer) with a single
    learnable table of shape (1, action_horizon, action_dim), broadcast over
    the batch. z0 is therefore CONSTANT — it cannot read c at all. Under
    beta==0 the role anchor degenerates to ||z0 - y||^2 and the table
    converges to the UNCONDITIONAL mean E[y]. Comparison against the
    conditional encoder arm separates:

      * the determinism asset (architecture property: no sampling at the
        source) — expected to survive (repeat||Delta|| ~ 0.014 at eta=0.04),
        from
      * z0's conditional content (~E[y|c], small source gap) — residual value
        exists iff this arm is significantly worse on accuracy.

    The query plumbing contract is kept intact (``num_queries``/``queries``
    are threaded through the backbone exactly like the other encoders; their
    hidden states are simply ignored, so the queries receive no gradient), so
    NO call site, guard, or collator code changes. ``reset_parameters`` is
    required by setup.py (meta-device instantiation contract; see the
    ActionMemoryEncoder module docstring).
    """

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        num_queries: int = 64,
        cond_dim: int = 1,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.num_queries = num_queries
        # Threaded-but-ignored query tokens (contract placeholders only).
        self.queries = nn.Parameter(torch.zeros(num_queries, cond_dim))
        # The start table: THE ONLY learned content of this module.
        self.table = nn.Parameter(torch.zeros(1, action_horizon, action_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.queries, std=0.02)
        nn.init.zeros_(self.table)

    def forward(
        self,
        query_features: torch.Tensor,
        embodiment_id: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Ignores conditioning; returns (B, action_horizon, action_dim).
        ``state`` is accepted for call-site uniformity (db18) and ignored —
        the constant start reads nothing by design."""
        B = query_features.shape[0]
        return self.table.to(dtype=query_features.dtype).expand(B, -1, -1)
