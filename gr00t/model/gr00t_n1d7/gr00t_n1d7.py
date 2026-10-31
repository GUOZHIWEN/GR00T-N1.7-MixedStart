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

import logging
from typing import Any, Tuple

import torch
from torch import nn
from torch.distributions import Beta
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
import tree

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.modules.action_memory_encoder import (
    ActionMemoryEncoder,
    ActionMemoryEncoderTransformer,
    ConfidenceGapHead,
    ConstActionStart,
)
from gr00t.model.modules.dit import AlternateVLDiT, DiT, SelfAttentionTransformer
from gr00t.model.modules.embodiment_conditioned_mlp import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)


logger = logging.getLogger(__name__)


class Gr00tN1d7ActionHead(nn.Module):
    """Action head component for flow matching diffusion policy."""

    supports_gradient_checkpointing = True

    def __init__(self, config: Gr00tN1d7Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        if config.use_alternate_vl_dit:
            self.model = AlternateVLDiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
                attend_text_every_n_blocks=config.attend_text_every_n_blocks,
            )
            logger.info("Using AlternateVLDiT for diffusion model")
        else:
            self.model = DiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
            )
            logger.info("Using DiT for diffusion model")
        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim * config.state_history_length,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )

        vl_self_attention_cfg = getattr(config, "vl_self_attention_cfg", None)
        if vl_self_attention_cfg and vl_self_attention_cfg.get("num_layers", 0) > 0:
            self.vl_self_attention = SelfAttentionTransformer(**vl_self_attention_cfg)
        else:
            self.vl_self_attention = nn.Identity()

        # P2: learnable action memory start (UVM-style). Only built when the
        # config opts in, so legacy checkpoints and the gaussian/history arms
        # are completely unaffected. Two trunk architectures share the same
        # query plumbing and zero-init contract: the UVM-faithful MLP
        # (~179M) and the horizon-structured Transformer (~3.1M).
        if getattr(config, "memory_start_enabled", False):
            if getattr(config, "memory_encoder_arch", "mlp") == "transformer":
                self.memory_encoder = ActionMemoryEncoderTransformer(
                    cond_dim=config.backbone_embedding_dim,
                    action_dim=self.action_dim,
                    action_horizon=self.action_horizon,
                    num_embodiments=config.max_num_embodiments,
                    d_model=getattr(config, "memory_encoder_d_model", 256),
                    num_layers=getattr(config, "memory_encoder_layers", 2),
                    num_queries=getattr(config, "memory_num_queries", 64),
                    bottleneck_dim=getattr(config, "memory_encoder_bottleneck_dim", 0),
                    var_head=getattr(config, "memory_encoder_var_head", False),
                    read_state=getattr(config, "memory_encoder_read_state", False),
                    state_dim=int(getattr(config, "max_state_dim", 132))
                    * int(getattr(config, "state_history_length", 1)),
                )
            elif getattr(config, "memory_encoder_arch", "mlp") == "const":
                # db15 "last cut" ablation: content-free deterministic start —
                # a single learnable (1, H, D) table, no conditioning reads.
                # See ConstActionStart docstring for the separation it buys.
                self.memory_encoder = ConstActionStart(
                    cond_dim=config.backbone_embedding_dim,
                    action_dim=self.action_dim,
                    action_horizon=self.action_horizon,
                    num_queries=getattr(config, "memory_num_queries", 64),
                )
            else:
                self.memory_encoder = ActionMemoryEncoder(
                    cond_dim=config.backbone_embedding_dim,
                    action_dim=self.action_dim,
                    action_horizon=self.action_horizon,
                    num_embodiments=config.max_num_embodiments,
                    hidden_dim=config.memory_start_hidden_dim,
                    num_queries=getattr(config, "memory_num_queries", 64),
                    query_dim=getattr(config, "memory_query_dim", 64),
                )
        else:
            self.memory_encoder = None

        # v4.1: confidence gap head (arm-independent). Its C learnable query
        # tokens are threaded through the frozen backbone after the K memory
        # queries; the readout predicts the DETACHED one-step endpoint gap
        # (ĝ_mean, ĝ_q90) and works with every flow_source_mode. T1
        # (confidence_routing_enabled) extends the readout with the per-source
        # gaps (ĝ_hist, ĝ_mem) that drive the inference-time β̂ mixture.
        if getattr(config, "confidence_tokens_enabled", False):
            self.confidence_head = ConfidenceGapHead(
                cond_dim=config.backbone_embedding_dim,
                num_queries=getattr(config, "confidence_num_queries", 4),
                num_outputs=(
                    5
                    if getattr(config, "confidence_routing_enabled", False)
                    and getattr(config, "confidence_routing_mode", "gap_ratio")
                    == "direct_beta"
                    else 4
                    if getattr(config, "confidence_routing_enabled", False)
                    else 2
                ),
            )
        else:
            self.confidence_head = None

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # State dropout parameters
        self.state_dropout_prob = config.state_dropout_prob

        # Pin the time-sampling Beta to CPU/fp32 explicitly. The action head can
        # be instantiated under a meta / no_init_weights default-device context
        # (e.g. nested from_pretrained). A Beta built from bare Python floats
        # would then place its concentration tensors on the meta device (or in
        # the active default dtype, e.g. bf16). With validate_args enabled that
        # already fails here in __init__ (Beta's internal .item() check cannot
        # run on meta); even with validation off, sample_time would later raise
        # or return garbage. Explicit device/dtype here makes the sampler depend
        # only on the config, not on the construction-time device/dtype context,
        # so the noise schedule is identical across SDPA/FA2/FA4 and meta vs.
        # real-device loads. config is the canonical source for these values.
        self.beta_dist = Beta(
            torch.tensor(float(config.noise_beta_alpha), dtype=torch.float32, device="cpu"),
            torch.tensor(float(config.noise_beta_beta), dtype=torch.float32, device="cpu"),
        )
        self.num_timestep_buckets = config.num_timestep_buckets
        self.set_trainable_parameters(
            config.tune_projector, config.tune_diffusion_model, config.tune_vlln
        )

    def zero_reinit_dit(self, seed: int) -> None:
        """db28 (from-scratch G/Z contract): re-initialise the DiT velocity
        field from scratch under the DiT/UVM zero-identity contract — every
        Linear/LayerNorm reset to a fresh random draw, then each block's
        attention output projection (attn1.to_out[0]) and the FFN output
        Linear zeroed, so every block starts as the identity map
        (block(x) = x) and the whole DiT reduces to norm_out/proj_out at
        step 0. The draw is seeded explicitly so the G and Z arms at the
        same --seed share bit-identical initial DiT weights. MUST be called
        AFTER checkpoint loading (setup.py), or the loaded pretrained
        weights silently overwrite the fresh init.
        """
        dit = self.model
        cpu_state = torch.random.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(seed)
        try:

            def _reset(module):
                if isinstance(module, (nn.Linear, nn.LayerNorm)):
                    module.reset_parameters()

            dit.apply(_reset)
            n_zero = 0
            for block in dit.transformer_blocks:
                for lin in (m for m in block.attn1.to_out if isinstance(m, nn.Linear)):
                    nn.init.zeros_(lin.weight)
                    if lin.bias is not None:
                        nn.init.zeros_(lin.bias)
                    n_zero += 1
                ff_linears = [m for m in block.ff.net if isinstance(m, nn.Linear)]
                assert ff_linears, "FeedForward has no Linear — zero-identity contract broken"
                lin = ff_linears[-1]
                nn.init.zeros_(lin.weight)
                if lin.bias is not None:
                    nn.init.zeros_(lin.bias)
                n_zero += 1
        finally:
            torch.random.set_rng_state(cpu_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
        n_params = sum(p.numel() for p in dit.parameters())
        logger.info(
            "zero_reinit_dit: DiT re-initialised from scratch (seed=%d, %.1fM params, "
            "%d output projections zeroed — per-block identity contract)",
            seed,
            n_params / 1e6,
            n_zero,
        )

    def set_trainable_parameters(
        self, tune_projector: bool, tune_diffusion_model: bool, tune_vlln: bool
    ):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_vlln = tune_vlln
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        if not tune_vlln:
            self.vlln.requires_grad_(False)
            self.vl_self_attention.requires_grad_(False)
        # P2: the memory encoder is a brand-new module; it is trainable iff the
        # memory arm is on AND tune_memory_encoder (independent of
        # tune_projector, so DiT-only 24GB finetunes can still train it).
        if self.memory_encoder is not None:
            self.memory_encoder.requires_grad_(bool(getattr(self.config, "tune_memory_encoder", True)))
        # v4.1: brand-new module — always trainable when enabled (freezing it
        # would pin the gap readout at its zero-init forever).
        if self.confidence_head is not None:
            self.confidence_head.requires_grad_(True)
        logger.debug(f"Tune action head projector: {self.tune_projector}")
        logger.debug(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        logger.debug(f"Tune action head vlln: {self.tune_vlln}")
        # Check if any parameters are still trainable. If not, log a warning.
        if not tune_projector and not tune_diffusion_model and not tune_vlln:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    logger.debug(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            logger.warning("No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()
            if not self.tune_vlln:
                self.vlln.eval()
                self.vl_self_attention.eval()
            if self.memory_encoder is not None and not getattr(
                self.config, "tune_memory_encoder", True
            ):
                self.memory_encoder.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        sample = (1 - sample) * self.config.noise_s
        return sample

    # ------------------------------------------------------------------
    # P2 anchor-lambda annealing
    # ------------------------------------------------------------------

    def set_anchor_lambda_progress(self, step: int, max_steps: int) -> None:
        """Hand the head the current optimizer step for λ annealing.

        Called once per training step by ``Gr00tTrainer.compute_loss`` (no-op
        cost for the constant schedule); stored as plain ints so the forward
        pass stays graph-free.
        """
        self._anchor_step = int(step)
        self._anchor_max_steps = max(1, int(max_steps))

    def effective_anchor_lambda(self) -> float:
        """λ of the z₀ reconstruction anchor at the current training step.

        Schedules (config ``memory_anchor_lambda_schedule``):
          * ``constant``: λ0 throughout (upstream behavior).
          * ``cosine``: λ(p) = λ_final + 0.5·(λ0−λ_final)·(1+cos(π·p)),
            p = step/max_steps clamped to [0, 1].
          * ``piecewise``: λ0 until ``memory_anchor_lambda_warmup_steps``,
            then λ_final.
        λ_final defaults to 0.1·λ0 when unset.
        """
        import math

        lam0 = float(self.config.memory_start_anchor_lambda)
        schedule = getattr(self.config, "memory_anchor_lambda_schedule", "constant")
        if schedule == "constant" or lam0 == 0:
            return lam0
        lam_final = getattr(self.config, "memory_anchor_lambda_final", None)
        lam_final = (0.1 * lam0) if lam_final is None else float(lam_final)
        if schedule == "piecewise":
            warmup = int(getattr(self.config, "memory_anchor_lambda_warmup_steps", 0))
            step = int(getattr(self, "_anchor_step", 0))
            return lam0 if step < warmup else lam_final
        if schedule == "cosine":
            step = int(getattr(self, "_anchor_step", 0))
            max_steps = max(1, int(getattr(self, "_anchor_max_steps", 1)))
            p = min(1.0, max(0.0, step / max_steps))
            return lam_final + 0.5 * (lam0 - lam_final) * (1.0 + math.cos(math.pi * p))
        raise ValueError(f"Unknown memory_anchor_lambda_schedule: {schedule!r}")

    @staticmethod
    def _infonce_anchor(
        z: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """Masked-batch InfoNCE anchor (review 2026-10-17, D4-1).

        Instead of pulling every z_i towards its own y_i independently (MSE —
        which under a soft anchor collapses to the dataset mean, B5), treat the
        batch as a retrieval problem: z_i should be closer to its own y_i than
        to any other sample's y_j.

        Per-sample masked squared distance (mask of the *query* sample i):
            d_ij = Σ_{t,d} m_i·(z_i − y_j)² / (Σ m_i + ε)
        logits = −d / τ, positives on the diagonal → F.cross_entropy.

        z must carry the gradient; y is expected to be detached by the caller.
        B == 1 has no negatives — the caller falls back to MSE.
        """
        b = z.shape[0]
        m = mask.float()  # (B, T, D)
        # Pairwise masked squared distance: (B, 1, T, D) vs (1, B, T, D) -> (B, B)
        diff = (z.float().unsqueeze(1) - y.float().unsqueeze(0)).pow(2)  # (B, B, T, D)
        m_pair = m.unsqueeze(1)  # query sample's mask broadcasts over j
        denom = m_pair.flatten(2).sum(-1) + 1e-6  # (B, 1)
        d = (diff * m_pair).flatten(2).sum(-1) / denom  # (B, B)
        logits = -d / max(float(temperature), 1e-6)
        targets = torch.arange(b, device=z.device)
        return F.cross_entropy(logits, targets)

    def _source_noise(self, like: torch.Tensor, action_input: BatchFeature) -> torch.Tensor:
        """Gaussian perturbation for the flow source, optionally masked per dim.

        T0-2 (ΔFM selective noise): with ``flow_source_noise_selective`` on and
        an ``action_noise_mask`` in the batch, gripper columns and padding stay
        clean (mask 0); otherwise this is plain ``randn_like``.
        """
        eps = torch.randn_like(like)
        if getattr(self.config, "flow_source_noise_selective", False):
            mask = action_input.get("action_noise_mask")
            if mask is not None:
                eps = eps * mask.to(device=like.device, dtype=like.dtype)
        return eps

    def _aligned_history_source(
        self,
        actions: torch.Tensor,
        action_input: BatchFeature,
        eta: float,
        rtc_offset: int | None = None,
    ) -> torch.Tensor | None:
        """Aligned previous-chunk source, or None when no history is supplied.

        With ``history_action_align="rtc"`` the stored previous chunk is first
        re-aligned by slicing off the ``history_action_offset`` prefix and
        holding the last action for the tail (``source[j] = prev[offset + j]``),
        matching the RTC in-painting convention used at inference.

        ``rtc_offset`` overrides the configured offset at inference time: the
        buffered previous chunk starts one re-plan stride in the past, so the
        slice must use offset=stride (Gr00tPolicy injects it via
        ``options["history_rtc_offset"]``; training leaves it None and keeps
        the configured offset, matching the dataset extractor).
        """
        history = action_input.get("history_action")
        if history is None:
            return None
        history = history.to(dtype=actions.dtype)
        horizon = actions.shape[1]
        if self.config.history_action_align == "rtc":
            offset = (
                self.config.history_action_offset if rtc_offset is None else int(rtc_offset)
            )
            assert offset < horizon, (
                f"history_action_offset ({offset}) must be < action horizon ({horizon}) "
                "for rtc alignment"
            )
            history = torch.cat(
                [history[:, offset:], history[:, -1:].expand(-1, offset, -1)], dim=1
            )
        if eta > 0:
            history = history + eta * self._source_noise(history, action_input)
        return history

    def _build_flow_source(
        self,
        actions: torch.Tensor,
        action_input: BatchFeature,
        base_noise: torch.Tensor,
        memory_start: torch.Tensor | None = None,
        beta_override: torch.Tensor | None = None,
        rtc_offset: int | None = None,
    ) -> torch.Tensor:
        """Build the source endpoint of the flow-matching ODE.

        Modes:
          * ``gaussian``: ``gaussian_source_noise_std * base_noise``
            (default std=1.0 reproduces the upstream source exactly).
          * ``history`` (UniWAM-style, P0-2): previous action chunk + eta
            perturbation. Samples whose ``has_action_history`` flag is 0 fall
            back to ``base_noise``.
          * ``memory`` (UVM-style, P2): ``z0 = MemoryEncoder(c)``
            (+ ``memory_start_noise_std`` perturbation). Falls back to
            ``base_noise`` when the encoder produced nothing.
          * ``memory_history_mix`` (P2 soft arm):
            ``beta * history_source + (1 - beta) * z0 + eta * eps``; samples
            without history fall back to the Gaussian part inside the history
            term, so beta=1 reproduces ``history`` and beta=0 reproduces
            ``memory`` (+eta). T1: ``beta_override`` (B,1,1) replaces the
            scalar config beta with the confidence-routed per-sample β̂.
          * ``rtc_offset``: inference-time override of the history alignment
            offset (re-plan stride); None keeps the training-time value.
        """
        mode = self.config.flow_source_mode
        if mode == "gaussian":
            noise_std = float(getattr(self.config, "gaussian_source_noise_std", 1.0))
            # Default preserves the exact legacy tensor; G-small shares the
            # semantic arm's eta=0.04 noise budget without using its z0 content.
            return base_noise if noise_std == 1.0 else noise_std * base_noise

        if mode == "history":
            history = self._aligned_history_source(
                actions,
                action_input,
                self.config.flow_source_action_noise_std,
                rtc_offset=rtc_offset,
            )
            if history is None:
                return base_noise
            has_history = action_input["has_action_history"].view(-1, 1, 1).to(
                dtype=actions.dtype
            )
            return has_history * history + (1 - has_history) * base_noise

        if mode == "memory":
            if memory_start is None:
                return base_noise
            eta = self.config.memory_start_noise_std
            source = memory_start.to(dtype=actions.dtype)
            if eta > 0:
                source = source + eta * self._source_noise(source, action_input)
            return source

        if mode == "memory_history_mix":
            eta = float(self.config.memory_start_noise_std)
            # History term reuses the history-mode alignment but with NO eta of
            # its own (eta is applied once, on the mixed result, below).
            history = self._aligned_history_source(
                actions, action_input, eta=0.0, rtc_offset=rtc_offset
            )
            if history is None:
                hist_term = base_noise
            else:
                has_history = action_input["has_action_history"].view(-1, 1, 1).to(
                    dtype=actions.dtype
                )
                hist_term = has_history * history + (1 - has_history) * base_noise
            if memory_start is None:
                source = hist_term  # degrade to history mode
            else:
                # Eval-only ablation (§13.44): zero the memory start so the
                # (1−β) component is the normalized-space constant source.
                if getattr(self.config, "memory_source_ablate_zero", False):
                    memory_start = torch.zeros_like(memory_start)
                # T1 confidence routing: per-sample β̂ (B,1,1) read out from the
                # confidence head overrides the scalar config beta (inference).
                if beta_override is not None:
                    beta = beta_override.to(device=hist_term.device, dtype=actions.dtype)
                    beta = beta.view(-1, *([1] * (hist_term.dim() - 1)))
                elif (
                    self.training
                    and getattr(self.config, "memory_start_mix_beta_sampling", "fixed")
                    == "uniform"
                ):
                    # β ~ U(min, max) per sample per step: teach the DiT the
                    # whole β-simplex so inference-time confidence routing (β̂)
                    # does not hit the fixed-β train/eval distribution shift
                    # (survey doc §13.15.2, verdict 2).
                    lo = float(getattr(self.config, "memory_start_mix_beta_min", 0.0))
                    hi = float(getattr(self.config, "memory_start_mix_beta_max", 1.0))
                    beta = (
                        torch.rand(
                            hist_term.shape[0],
                            device=hist_term.device,
                            dtype=actions.dtype,
                        )
                        * (hi - lo)
                        + lo
                    )
                    beta = beta.view(-1, *([1] * (hist_term.dim() - 1)))
                else:
                    beta = float(self.config.memory_start_mix_beta)
                # db12 (§13.61): stash the β actually used so the anchor branch
                # can gate on it (anchor-side β decoupling). No grad; eval-side
                # callers never read it (the anchor is a training-only term).
                if isinstance(beta, torch.Tensor):
                    self._last_mix_beta = beta.detach()
                else:
                    self._last_mix_beta = torch.full(
                        (hist_term.shape[0], 1, 1),
                        float(beta),
                        device=hist_term.device,
                        dtype=hist_term.dtype,
                    )
                # §13.89 (2026-10-23): stash the history term so the role-anchor
                # branch can decompose anchor_loss into the β·(h−y) structural
                # floor (dbg_anchor_mix_resid) and the z0-side excess
                # (dbg_anchor_excess). No grad; training-only diagnostics.
                self._last_hist_term = hist_term.detach()
                source = beta * hist_term + (1.0 - beta) * memory_start.to(
                    dtype=actions.dtype
                )
            if eta > 0:
                noise = self._source_noise(source, action_input)
                if (
                    self.training
                    and getattr(self.config, "memory_start_noise_sampling", "fixed")
                    == "uniform"
                ):
                    # db23a (§13.91): per-sample eta ~ U(0, eta) — isotropic
                    # source-spectrum diversity around the semantic start,
                    # testing whether noise diversity substitutes for the
                    # history corner's structured diversity (β~U, retired
                    # as a necessity by db22). Eval keeps the scalar eta.
                    eta_vec = torch.rand(
                        source.shape[0],
                        *([1] * (source.dim() - 1)),
                        device=source.device,
                        dtype=source.dtype,
                    ) * eta
                    source = source + eta_vec * noise
                else:
                    source = source + eta * noise
            return source

        raise ValueError(f"Unknown flow_source_mode: {mode!r}")

    @staticmethod
    def _closed_form_beta_star(
        actions: torch.Tensor,
        hist_source: torch.Tensor,
        mem_source: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Closed-form optimal per-sample mixture weight for
        ``source(β) = β·h + (1−β)·z0`` minimizing the masked ‖source(β) − y‖²:

            β* = <y − z0, h − z0> / ‖h − z0‖²   (masked sums over (T, D)),
            clamped to [0, 1].

        Direct-β routing supervision (survey §13.18): free target — computed
        from the two sources and the ground-truth chunk alone, no extra DiT
        forward. Samples with h ≈ z0 (degenerate, any β equivalent) default
        to 0.5.
        """
        y = actions.detach().float()
        h = hist_source.detach().float()
        z0 = mem_source.detach().float()
        m = action_mask.detach().float()
        dh = (h - z0) * m
        dy = (y - z0) * m
        dims = tuple(range(1, dh.dim()))
        num = (dy * dh).sum(dim=dims)
        den = (dh * dh).sum(dim=dims)
        beta = torch.where(
            den > 1e-6, num / den.clamp_min(1e-6), torch.full_like(num, 0.5)
        )
        return beta.clamp(0.0, 1.0)

    @torch.no_grad()
    def _one_step_endpoint_gap(
        self,
        source: torch.Tensor,
        actions: torch.Tensor,
        state_features: torch.Tensor,
        vl_embeds: torch.Tensor,
        backbone_output: BatchFeature,
        embodiment_id: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Detached per-sample masked L1 gap of the one-step endpoint
        ŷ1 = s + v_θ(s, t=0, c) against the ground-truth chunk.

        Costs one DiT forward (run under train-mode dropout — a documented
        target-noise approximation; detached, so it never perturbs the DiT
        gradient). Used as the confidence-head training target: for the mixed
        arm itself and, with T1 confidence routing, for the two PURE sources
        whose gap ratio drives the inference-time β̂.
        """
        B = actions.shape[0]
        device = vl_embeds.device
        t0 = torch.zeros(B, dtype=torch.long, device=actions.device)
        af0 = self.action_encoder(source, t0, embodiment_id)
        if self.config.add_pos_embed:
            pos_ids0 = torch.arange(af0.shape[1], dtype=torch.long, device=device)
            af0 = af0 + self.position_embedding(pos_ids0).unsqueeze(0)
        sa0 = torch.cat((state_features, af0), dim=1)
        vl_attn_mask = backbone_output.backbone_attention_mask
        if self.config.use_alternate_vl_dit:
            out0, _ = self.model(
                hidden_states=sa0,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t0,
                return_all_hidden_states=True,
                image_mask=backbone_output.image_mask,
                backbone_attention_mask=backbone_output.backbone_attention_mask,
            )
        else:
            out0, _ = self.model(
                hidden_states=sa0,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t0,
                return_all_hidden_states=True,
            )
        pred0 = self.action_decoder(out0, embodiment_id)[:, -actions.shape[1] :]
        endpoint = source + pred0  # ŷ1: one Euler step from t=0, dt=1
        gap = ((endpoint - actions).abs().float() * action_mask.float()).sum(dim=(1, 2)) / (
            action_mask.float().sum(dim=(1, 2)) + 1e-6
        )
        return gap  # (B,)

    def _func_anchor_endpoint(
        self,
        source: torch.Tensor,
        state_features: torch.Tensor,
        vl_embeds: torch.Tensor,
        backbone_output: BatchFeature,
        embodiment_id: torch.Tensor,
    ) -> torch.Tensor:
        """Grad-carrying one-step endpoint ŷ1 = s + v_θ(s, t=0, c) for the L2
        functional anchor (survey doc §13.43 mechanism ③).

        Same t=0 forward as ``_one_step_endpoint_gap`` but WITHOUT
        ``no_grad``. The caller (training forward) never backprops this
        branch through the DiT: it differentiates w.r.t. the memory-encoder
        parameters ONLY, via ``torch.autograd.grad`` + manual .grad
        accumulation (§13.44 P1-1 enforced by construction — DiT gradients
        from the anchor are never computed, let alone applied). The context
        tensors (state_features / vl_embeds) are detached purely as a
        graph-size/memory optimisation: autograd.grad targets the z0 pathway
        either way.
        """
        B = source.shape[0]
        device = vl_embeds.device
        t0 = torch.zeros(B, dtype=torch.long, device=source.device)
        af0 = self.action_encoder(source, t0, embodiment_id)
        if self.config.add_pos_embed:
            pos_ids0 = torch.arange(af0.shape[1], dtype=torch.long, device=device)
            af0 = af0 + self.position_embedding(pos_ids0).unsqueeze(0)
        sa0 = torch.cat((state_features.detach(), af0), dim=1)
        vl0 = vl_embeds.detach()
        vl_attn_mask = backbone_output.backbone_attention_mask
        if self.config.use_alternate_vl_dit:
            out0, _ = self.model(
                hidden_states=sa0,
                encoder_hidden_states=vl0,
                encoder_attention_mask=vl_attn_mask,
                timestep=t0,
                return_all_hidden_states=True,
                image_mask=backbone_output.image_mask,
                backbone_attention_mask=backbone_output.backbone_attention_mask,
            )
        else:
            out0, _ = self.model(
                hidden_states=sa0,
                encoder_hidden_states=vl0,
                encoder_attention_mask=vl_attn_mask,
                timestep=t0,
                return_all_hidden_states=True,
            )
        pred0 = self.action_decoder(out0, embodiment_id)[:, -source.shape[1] :]
        return source + pred0  # ŷ1: one Euler step from t=0, dt=1

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)

        # v4.1: the frozen backbone returns the hidden states of ALL threaded
        # queries under "memory_query_features" — K memory queries (only on the
        # "backbone" path) followed by C confidence queries. Split the tail
        # block up front so each consumer sees its own slice; on the "post"
        # path the memory features are rebuilt below from vl_self_attention,
        # and without a memory arm the key must not leak through.
        tail_queries = backbone_output.get("memory_query_features")
        if tail_queries is not None:
            K_bb = (
                self.memory_encoder.num_queries
                if self.memory_encoder is not None and self._memory_query_path() == "backbone"
                else 0
            )
            if self.confidence_head is not None:
                backbone_output["confidence_query_features"] = tail_queries[:, K_bb:]
            if K_bb > 0:
                backbone_output["memory_query_features"] = tail_queries[:, :K_bb]
            else:
                backbone_output.pop("memory_query_features", None)

        if self.memory_encoder is not None and self._memory_query_path() == "post":
            # v2 "post" path: append the K learnable query tokens (owned by the
            # memory encoder) to the frozen-backbone VL feature sequence and run
            # the combined sequence through the *trainable* vl_self_attention —
            # the frozen VLM itself is never touched. The K query outputs
            # (content-adaptive semantic views of the observation + instruction)
            # feed the memory encoder; the VL prefix is returned unchanged in
            # shape for the DiT cross-attention.
            K = self.memory_encoder.num_queries
            B = backbone_features.shape[0]
            queries = self.memory_encoder.queries.to(dtype=backbone_features.dtype)
            seq = torch.cat(
                [backbone_features, queries.unsqueeze(0).expand(B, -1, -1)], dim=1
            )
            seq = self.vl_self_attention(seq)
            backbone_features = seq[:, :-K]
            backbone_output["memory_query_features"] = seq[:, -K:]
        else:
            # Default and v3 "backbone" path: the VL features go through
            # vl_self_attention on their own. In v3 the K query hidden states
            # were already produced INSIDE the frozen backbone (see
            # Qwen3Backbone.forward) and arrive as
            # backbone_output["memory_query_features"]; they feed the encoder
            # directly, without a second pass through vl_self_attention.
            backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def _memory_query_path(self) -> str:
        return getattr(self.config, "memory_query_path", "backbone")

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """
        Forward pass through the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - action: [B, action_horizon, action_dim] (during training)
                - embodiment_id: [B] (embodiment IDs)
                - action_mask: [B, action_horizon, action_dim]

        Returns:
            BatchFeature containing:
                - loss: action prediction loss
        """
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Handle state history
        assert action_input.state.shape[1] == self.config.state_history_length
        action_input.state = action_input.state.view(action_input.state.shape[0], 1, -1)

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Dropout state features (training only): zero out dropped states.
        if self.training and self.state_dropout_prob > 0:
            do_dropout = (
                torch.rand(state_features.shape[0], device=state_features.device)
                < self.state_dropout_prob
            )
            do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
            state_features = state_features * (1 - do_dropout)

        # Embed noised action trajectory.
        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        # P2: learnable memory start z0 = MemoryEncoder(c). Query features come
        # from the frozen backbone's last-K hidden states (path="backbone", via
        # Gr00tN1d7.forward's query_embeds) or from the vl_self_attention
        # append-and-slice (path="post", see process_backbone_output).
        mem_start = None
        mem_logvar = None  # db17: log σ² companion of z₀ (NLL anchor modes only)
        if self.memory_encoder is not None:
            query_features = backbone_output.get("memory_query_features")
            if query_features is None:
                raise RuntimeError(
                    "memory_encoder is enabled but backbone_output lacks "
                    "'memory_query_features'. With memory_query_path='backbone' the "
                    "backbone must be called with query_embeds (Gr00tN1d7.forward/"
                    "get_action do this) AND the collator must append K placeholder "
                    "tokens; with 'post' process_backbone_output produces them."
                )
            # db17: NLL anchor modes consume the variance head's log σ² next to
            # z₀ = μ. Inference (get_action) never requests it — deployment
            # uses the deterministic μ only, preserving the determinism asset.
            _want_logvar = getattr(
                self.config, "memory_start_anchor_type", "mse"
            ) in ("nll", "nll_infonce") and getattr(
                self.config, "memory_encoder_var_head", False
            )
            # db18 T2: z₀ = f(H, q) — pass the flattened proprioceptive state
            # (B, max_state_dim × state_history_length) when the encoder opts
            # in; None otherwise (all encoder trunks accept the kwarg).
            _enc_state = None
            if getattr(self.config, "memory_encoder_read_state", False):
                _enc_state = action_input.state.reshape(action_input.state.shape[0], -1)
            if _want_logvar:
                mem_start, mem_logvar = self.memory_encoder(
                    query_features,
                    embodiment_id,
                    state=_enc_state,
                    return_logvar=True,
                )
            else:
                mem_start = self.memory_encoder(
                    query_features,
                    embodiment_id,
                    state=_enc_state,
                )
        # P0-2 / P2: history- and/or memory-conditioned flow source
        # (gaussian mode returns `noise`).
        source = self._build_flow_source(actions, action_input, noise, memory_start=mem_start)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * source + t * actions
        velocity = actions - source

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        # db18 T1/T2: dit_state_token=False cuts the state channel into the
        # FIELD — the DiT self-attention input becomes action tokens only.
        # (The confidence / functional-anchor auxiliary forwards keep their own
        # path and are not re-routed; the CLI rejects combining
        # --no-dit-state-token with --memory-func-anchor-enabled.)
        if getattr(self.config, "dit_state_token", True):
            sa_embs = torch.cat((state_features, action_features), dim=1)
        else:
            sa_embs = action_features
        vl_attn_mask = backbone_output.backbone_attention_mask

        if self.config.use_alternate_vl_dit:
            image_mask = backbone_output.image_mask
            backbone_attention_mask = backbone_output.backbone_attention_mask
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
            )
        else:
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
            )

        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        action_mask = action_input.action_mask
        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        # P0-1: per-sample action-loss weighting for co-training with
        # pseudo-labelled / action-free data (DreamZero-style). Real samples keep
        # weight 1.0; pseudo-labelled samples get ``pseudo_action_loss_weight``,
        # or 0 when ``drop_unlabeled_action`` is set.
        loss_weight = None
        if self.config.use_has_real_action:
            has_real_action = action_input.get("has_real_action")
            if has_real_action is not None:
                has_real_action = has_real_action.view(-1, 1, 1).to(dtype=action_loss.dtype)
                if self.config.drop_unlabeled_action:
                    loss_weight = (has_real_action > 0.5).to(dtype=action_loss.dtype)
                else:
                    loss_weight = torch.where(
                        has_real_action > 0.5,
                        torch.ones_like(has_real_action),
                        torch.full_like(
                            has_real_action, float(self.config.pseudo_action_loss_weight)
                        ),
                    )
                action_loss = action_loss * loss_weight
        if loss_weight is not None:
            loss = action_loss.sum() / ((action_mask * loss_weight).sum() + 1e-6)
        else:
            loss = action_loss.sum() / (action_mask.sum() + 1e-6)
        flow_loss = loss  # flow-matching term before the anchor is added

        # Diagnostics: detached scalars surfaced in the trainer logs (see
        # Gr00tTrainer.compute_loss) so memory-arm runs can be validated
        # without a debugger. Cheap: a handful of reductions per step.
        with torch.no_grad():
            debug_stats = {
                "flow_loss": float(flow_loss.detach().float()),
                "pred_rms": float(pred_actions.detach().float().pow(2).mean().sqrt()),
                "vel_rms": float(velocity.detach().float().pow(2).mean().sqrt()),
                "src_rms": float(source.detach().float().pow(2).mean().sqrt()),
                "act_rms": float(actions.detach().float().pow(2).mean().sqrt()),
                "mask_sum": float(action_mask.detach().float().sum()),
                "t_mean": float(t.detach().float().mean()),
                "vl_absmax": float(vl_embeds.detach().float().abs().max()),
            }
            if mem_start is not None:
                debug_stats["z0_absmax"] = float(mem_start.detach().float().abs().max())

        # P2: z0 reconstruction anchor (UVM-style): direct MSE between the
        # memory start and the ground-truth chunk, bypassing the DiT. lambda=1.0
        # is the original hard anchor; 0.05-0.1 is the soft-anchor variant; the
        # effective λ may anneal over training (cosine/piecewise, see
        # effective_anchor_lambda). Shares the action_mask and the P0-1
        # per-sample weighting so pseudo-labelled samples are not asked to
        # reconstruct fake actions.
        # P2: z0 reconstruction anchor (UVM-style): direct MSE between the
        # memory start and the ground-truth chunk, bypassing the DiT. lambda=1.0
        # is the original hard anchor; 0.05-0.1 is the soft-anchor variant; the
        # effective λ may anneal over training (cosine/piecewise, see
        # effective_anchor_lambda). Shares the action_mask and the P0-1
        # per-sample weighting so pseudo-labelled samples are not asked to
        # reconstruct fake actions.
        # D4-1 (review 2026-10-17): ``memory_start_anchor_type == "infonce"``
        # replaces the per-sample MSE with a masked-batch InfoNCE (diagonal
        # positives) — directly counteracts the soft-anchor variance collapse
        # (B5) and makes z0 discriminative across samples (B2). NOTE: the P0-1
        # per-sample loss_weight is pairwise-incompatible with InfoNCE and is
        # NOT applied in that mode — use drop_unlabeled_action instead.
        anchor_loss = None
        lam_eff = self.effective_anchor_lambda()
        # db12 (§13.61): anchor-side β decoupling gate — restrict the role/diff
        # anchors to samples whose drawn β ≤ memory_start_anchor_beta_max.
        # Under β ~ U(0,1) the role equilibrium z0* = 1.5·E[y|c] − 0.5·h lets
        # high-β samples inject h's phase lag into z0; gating at 0.5 shrinks
        # that contamination while the flow side keeps learning the full
        # simplex. Computed once, shared by the point and diff anchors.
        anchor_beta_max = float(
            getattr(self.config, "memory_start_anchor_beta_max", 1.0)
        )
        beta_gate = None
        if mem_start is not None and anchor_beta_max < 1.0:
            _drawn = getattr(self, "_last_mix_beta", None)
            if _drawn is not None and isinstance(_drawn, torch.Tensor):
                beta_gate = (_drawn.float() <= anchor_beta_max).float()  # (B,1,1)
        if mem_start is not None and lam_eff > 0:
            anchor_type = getattr(self.config, "memory_start_anchor_type", "mse")
            anchor_target = getattr(self.config, "memory_start_anchor_target", "y")
            if anchor_type == "infonce" and mem_start.shape[0] > 1:
                anchor_loss = self._infonce_anchor(
                    mem_start,
                    actions.detach(),
                    action_mask,
                    float(getattr(self.config, "memory_start_anchor_temperature", 0.1)),
                )
            elif anchor_type in ("nll", "nll_infonce"):
                # db17 (paper §8 item 11-①): explicit Gaussian NLL anchor —
                #   L = ½·Σ_mask[(μ−y)²·exp(−logvar) + logvar].
                # MSE is the σ²-fixed special case whose optimum is E[y|c]
                # (mode averaging); the learned σ(c) absorbs irreducible
                # conditional variance so μ may concentrate on ONE mode. The
                # target is always y (the role form is rejected at the config
                # layer: σ is a property of z₀, not of the mixed source s).
                # This is the STRONGEST supervision form the content dimension
                # can get — the arm closes the "z₀ was never trained well"
                # alternative explanation of the §4.2 negative result.
                if mem_logvar is None:
                    raise RuntimeError(
                        "anchor_type 'nll'/'nll_infonce' requires "
                        "memory_encoder_var_head=True (log σ² missing — set "
                        "--memory-encoder-var-head)."
                    )
                mu = mem_start.float()
                y_det = actions.detach().float()
                lv = mem_logvar.float()
                nll = 0.5 * ((mu - y_det).pow(2) * lv.neg().exp() + lv)
                nll = nll * action_mask.float()
                nll_w = action_mask.float()
                if loss_weight is not None:
                    nll = nll * loss_weight.float()
                    nll_w = nll_w * loss_weight.float()
                if beta_gate is not None:
                    nll = nll * beta_gate
                    nll_w = nll_w * beta_gate
                anchor_loss = nll.sum() / (nll_w.sum() + 1e-6)
                debug_stats["anchor_sigma_mean"] = float(
                    (0.5 * lv.detach()).exp().mean()
                )
            else:
                # L1′ (review 2026-10-18, survey doc §13.24 root cause ⑥):
                # "role" anchors the MIXED source s = βh+(1−β)z0 → y instead of
                # z0 → y. This is the multiplicative form of z0 → (y−βh)/(1−β):
                # no division, no β→1 singularity, and the anchor strength
                # (1−β)² vanishes exactly where z0 has no role in the source.
                # Grad flows only into z0 (β is a float or a no-grad uniform
                # draw during training; hist_term comes from the data). The eta
                # perturbation inside `source` is zero-mean: correct gradient
                # direction in expectation, +eta² constant. β=0 reduces to "y".
                if (
                    anchor_target == "role"
                    and self.config.flow_source_mode == "memory_history_mix"
                ):
                    anchor_ref = source
                else:
                    anchor_ref = mem_start
                anchor = F.mse_loss(
                    anchor_ref.float(), actions.detach().float(), reduction="none"
                ) * action_mask.float()
                anchor_w = action_mask.float()
                if loss_weight is not None:
                    anchor = anchor * loss_weight.float()
                    anchor_w = anchor_w * loss_weight.float()
                if beta_gate is not None:
                    anchor = anchor * beta_gate
                    anchor_w = anchor_w * beta_gate
                anchor_denom = anchor_w.sum() + 1e-6
                anchor_loss = anchor.sum() / anchor_denom
                # §13.89 (2026-10-23, sampling corrected): decompose the
                # role-anchor plateau. The db7 series trains with β~U(0,1) and
                # the anchor β-gate OFF (memory_start_anchor_beta_max=1.0);
                # the role equilibrium z0* = 1.5·E[y|c] − 0.5·h leaves the
                # residual 0.5·(3β−1)·(h−y) whose expected coefficient² under
                # U(0,1) is exactly 0.25, so the anchor floor ≈ 0.25·E‖h−y‖².
                # "anchor_mix_resid" = masked E‖β(h−y)‖² is the history-
                # mismatch term BEFORE z0 compensation (E[β²]=1/3 under
                # U(0,1), so the equilibrium floor ≈ 0.75 × mix_resid);
                # "anchor_excess" = anchor_loss − mix_resid carries the z0
                # compensation + Var(y|c) + η²·D and can go negative.
                if anchor_target == "role":
                    _h = getattr(self, "_last_hist_term", None)
                    _b = getattr(self, "_last_mix_beta", None)
                    if (
                        _h is not None
                        and _b is not None
                        and _h.shape == actions.shape
                        and _b.shape[0] == actions.shape[0]
                    ):
                        mix_elem = (
                            (_b.float() * (_h.float() - actions.detach().float()))
                            ** 2
                        ) * action_mask.float()
                        if loss_weight is not None:
                            mix_elem = mix_elem * loss_weight.float()
                        if beta_gate is not None:
                            mix_elem = mix_elem * beta_gate
                        mix_resid = mix_elem.sum() / anchor_denom
                        debug_stats["anchor_mix_resid"] = float(mix_resid.detach())
                        debug_stats["anchor_excess"] = float(
                            (anchor_loss - mix_resid).detach()
                        )
            loss = loss + lam_eff * anchor_loss.to(loss.dtype)
            # db17: nll_infonce adds the batch-contrastive sub-term under its
            # own CONSTANT lambda (not annealed) — discriminative pressure
            # against conditional-mean collapse, complementing the NLL main
            # term's distributional fit.
            if anchor_type == "nll_infonce" and mem_start.shape[0] > 1:
                con_lam = float(
                    getattr(self.config, "memory_start_anchor_con_lambda", 0.1)
                )
                if con_lam > 0:
                    con_loss = self._infonce_anchor(
                        mem_start,
                        actions.detach(),
                        action_mask,
                        float(
                            getattr(
                                self.config, "memory_start_anchor_temperature", 0.1
                            )
                        ),
                    )
                    loss = loss + con_lam * con_loss.to(loss.dtype)
                    debug_stats["anchor_con_loss"] = float(con_loss.detach().float())
                    debug_stats["anchor_con_lambda"] = con_lam
            debug_stats["anchor_loss"] = float(anchor_loss.detach().float())
            debug_stats["anchor_lambda_eff"] = float(lam_eff)
            debug_stats["anchor_type"] = anchor_type
            debug_stats["anchor_target"] = (
                anchor_target
                if anchor_type in ("mse", "nll", "nll_infonce")
                else "n/a_infonce"
            )
            debug_stats["anchor_beta_max"] = float(anchor_beta_max)
            if beta_gate is not None:
                debug_stats["anchor_gate_frac"] = float(beta_gate.mean())

        # db12 (§13.61): β=0 endpoint debias — small CONSTANT-weight pure
        # reconstruction λ·‖z0 − y‖², NOT annealed, NOT β-gated. The role
        # equilibrium z0* = 1.5·E[y|c] − 0.5·h carries a 0.5·(E[y|c] − h)
        # extrapolation bias at the β=0 deployment arm; this term pulls z0
        # back toward E[y|c] directly.
        lam_recon = float(getattr(self.config, "memory_start_z0_recon_lambda", 0.0))
        if mem_start is not None and lam_recon > 0:
            recon = F.mse_loss(
                mem_start.float(), actions.detach().float(), reduction="none"
            ) * action_mask.float()
            if loss_weight is not None:
                recon = recon * loss_weight.float()
                recon_denom = (action_mask.float() * loss_weight.float()).sum() + 1e-6
            else:
                recon_denom = action_mask.float().sum() + 1e-6
            recon_loss = recon.sum() / recon_denom
            loss = loss + lam_recon * recon_loss.to(loss.dtype)
            debug_stats["z0_recon_loss"] = float(recon_loss.detach().float())
            debug_stats["z0_recon_lambda"] = float(lam_recon)

        # db10 (2026-10-19, survey doc §13.57): differential (shape) anchor.
        # Point-value MSE anchors converge to the conditional mean E[y|c],
        # which is a flat line on mode-averaged dims (§13.41: anchor loss
        # plateaus from the first quartile — structural, not undertraining).
        # The differential form anchors FIRST DIFFERENCES along the horizon,
        # Δ(anchor_ref) → Δy, i.e. it targets slope/shape rather than level.
        # Own CONSTANT lambda (``memory_start_diff_anchor_lambda``, default
        # 0.0 = off): deliberately NOT annealed, so the shape pressure
        # survives the cosine decay of the point anchor (wall ③). Reference
        # choice mirrors the point anchor (role → mixed source s, else z0);
        # grad flows only into z0. Frame-pair weight w_t*w_{t+1} fuses the
        # action mask and the P0-1 per-sample loss_weight with correct
        # "both frames valid" semantics regardless of broadcast shape.
        lam_diff = float(getattr(self.config, "memory_start_diff_anchor_lambda", 0.0))
        if mem_start is not None and lam_diff > 0:
            _at = getattr(self.config, "memory_start_anchor_type", "mse")
            _tg = getattr(self.config, "memory_start_anchor_target", "y")
            if (
                _at == "mse"
                and _tg == "role"
                and self.config.flow_source_mode == "memory_history_mix"
            ):
                diff_ref = source
            else:
                diff_ref = mem_start
            eff_w = action_mask.float()
            if loss_weight is not None:
                eff_w = eff_w * loss_weight.float()  # broadcast to (B,H,A)
            if beta_gate is not None:
                eff_w = eff_w * beta_gate  # db12: same anchor-side β gate
            w_diff = eff_w[:, 1:, :] * eff_w[:, :-1, :]  # (B,H-1,A)
            d_ref = diff_ref.float()[:, 1:, :] - diff_ref.float()[:, :-1, :]
            d_y = actions.detach().float()[:, 1:, :] - actions.detach().float()[:, :-1, :]
            d_anchor = F.mse_loss(d_ref, d_y, reduction="none") * w_diff
            diff_anchor_loss = d_anchor.sum() / (w_diff.sum() + 1e-6)
            loss = loss + lam_diff * diff_anchor_loss.to(loss.dtype)
            debug_stats["diff_anchor_loss"] = float(diff_anchor_loss.detach().float())
            debug_stats["diff_anchor_lambda"] = float(lam_diff)

        # v4.1: confidence gap head. Training target = DETACHED one-step
        # endpoint gap ‖ŷ1 − y‖ (masked L1, per sample) with
        # ŷ1 = s + v_θ(s, t=0, c) — one extra no-grad DiT forward every
        # ``confidence_every_k`` optimizer steps. Arm-independent: the same
        # ``source`` the flow loss used is reused, so gaussian/history/memory
        # arms all train the same readout. NOTE: the target forward runs in
        # train mode (dropout active) — a small, documented target-noise
        # approximation; it is detached, so it never perturbs the DiT gradient.
        # T1 (confidence_routing_enabled, num_outputs=4): two further no-grad
        # forwards score the PURE sources (history arm / memory arm), giving
        # the per-sample gaps whose ratio drives the inference-time β̂.
        conf_loss = None
        if self.confidence_head is not None:
            conf_features = backbone_output.get("confidence_query_features")
            if conf_features is None:
                raise RuntimeError(
                    "confidence_tokens_enabled but backbone_output lacks "
                    "'confidence_query_features': the collator must append the C "
                    "placeholder tokens (processor built with confidence_num_queries "
                    "matching the model config)."
                )
            conf_weight = float(getattr(self.config, "confidence_loss_weight", 0.1))
            every_k = max(1, int(getattr(self.config, "confidence_every_k", 1)))
            if conf_weight > 0 and int(getattr(self, "_anchor_step", 0)) % every_k == 0:
                gap_target = self._one_step_endpoint_gap(
                    source, actions, state_features, vl_embeds, backbone_output,
                    embodiment_id, action_mask,
                )
                gap_pred = self.confidence_head(conf_features).float()
                g_mean, g_q90 = gap_pred[:, 0], gap_pred[:, 1]
                u90 = gap_target - g_q90
                pinball_90 = torch.maximum(0.9 * u90, -0.1 * u90)
                conf_loss = F.mse_loss(g_mean, gap_target) + pinball_90.mean()
                debug_stats["gap_target_mean"] = float(gap_target.mean())
                debug_stats["gap_pred_mean"] = float(g_mean.detach().mean())
                if self.confidence_head.num_outputs >= 4:
                    # T1 (paper §4.4): per-source endpoint gaps. The pure
                    # history arm mirrors the mix branch at beta=1 (gaussian
                    # fallback for samples without history); the pure memory
                    # arm is z0 with no eta (deterministic readout). At
                    # inference β̂ = ĝ_mem²/(ĝ_hist² + ĝ_mem² + ε).
                    hist_aligned = self._aligned_history_source(actions, action_input, eta=0.0)
                    if hist_aligned is None:
                        hist_arm_source = noise
                    else:
                        has_history = action_input["has_action_history"].view(-1, 1, 1).to(
                            dtype=actions.dtype
                        )
                        hist_arm_source = has_history * hist_aligned + (
                            1 - has_history
                        ) * noise
                    mem_arm_source = (
                        mem_start.to(dtype=actions.dtype) if mem_start is not None else noise
                    )
                    gap_hist = self._one_step_endpoint_gap(
                        hist_arm_source, actions, state_features, vl_embeds,
                        backbone_output, embodiment_id, action_mask,
                    )
                    gap_mem = self._one_step_endpoint_gap(
                        mem_arm_source, actions, state_features, vl_embeds,
                        backbone_output, embodiment_id, action_mask,
                    )
                    g_hist, g_mem = gap_pred[:, 2], gap_pred[:, 3]
                    conf_loss = conf_loss + F.mse_loss(g_hist, gap_hist) + F.mse_loss(
                        g_mem, gap_mem
                    )
                    debug_stats["gap_hist_target_mean"] = float(gap_hist.mean())
                    debug_stats["gap_mem_target_mean"] = float(gap_mem.mean())
                    debug_stats["gap_hist_pred_mean"] = float(g_hist.detach().mean())
                    debug_stats["gap_mem_pred_mean"] = float(g_mem.detach().mean())
                if self.confidence_head.num_outputs == 5:
                    # Direct-β routing (survey §13.18): the 5th output regresses
                    # the closed-form optimal per-sample mixture weight
                    # β* = <y−z0, h−z0>/‖h−z0‖² (masked, clamped) — free
                    # supervision from sources+target alone, no extra DiT pass.
                    # At inference β̂ = ĝ_β directly (no gap-ratio mapping).
                    beta_star = self._closed_form_beta_star(
                        actions, hist_arm_source, mem_arm_source, action_mask
                    )
                    beta_pred = gap_pred[:, 4]
                    conf_loss = conf_loss + F.mse_loss(beta_pred, beta_star)
                    debug_stats["beta_star_mean"] = float(beta_star.mean())
                    debug_stats["beta_star_std"] = float(beta_star.std())
                    debug_stats["beta_pred_mean"] = float(beta_pred.detach().mean())
                loss = loss + conf_weight * conf_loss.to(loss.dtype)
                debug_stats["conf_loss"] = float(conf_loss.detach())

        # L2 functional anchor (survey doc §13.43 mechanism ③, §13.44 P1
        # corrections): anchor the FUNCTION "from the same source s the flow
        # loss used, one Euler step reaches y" — fa = ‖ŷ1 − y‖² (masked),
        # ŷ1 = s + v_θ(s, t=0, c). Constant λ, no annealing — the deployment-
        # relevant pressure must persist to convergence (§13.43 bundle).
        #
        # §13.44 P1-1 (stop-grad v_θ) is enforced BY CONSTRUCTION: the anchor
        # term is NEVER added to the training loss. Its gradient w.r.t. the z0
        # pathway only is computed via torch.autograd.grad and accumulated
        # onto .grad manually — DiT-pathway gradients from the anchor are not
        # merely blocked, they are never computed. retain_graph=True keeps the
        # shared source/z0 subgraph intact for the trainer's main backward,
        # which then adds the flow/role-anchor/confidence contributions on top
        # (accumulation within the micro-batch is commutative).
        #
        # Caveats: λ_func is a per-micro-batch weight — under HF gradient
        # accumulation the returned loss is scaled but this manual
        # accumulation is not, so the effective anchor strength is ~ga× the
        # printed ratio (ga=2 → ×2); tune λ accordingly. Cost: one extra
        # grad-carrying t=0 DiT forward + its backward on anchor steps
        # (§13.44 P1-3: not zero-cost), amortised by every_k.
        func_anchor_loss = None
        lam_func = float(getattr(self.config, "memory_func_anchor_lambda", 0.0))
        if (
            getattr(self.config, "memory_func_anchor_enabled", False)
            and mem_start is not None
            and lam_func > 0
            and self.training
            and torch.is_grad_enabled()
            and int(getattr(self, "_anchor_step", 0))
            % max(1, int(getattr(self.config, "memory_func_anchor_every_k", 1)))
            == 0
        ):
            # db9 (§13.48): the anchor's source follows the DEPLOYMENT operating
            # point, not the training-time β sample. "pure_z0" (default)
            # re-draws the source as z0 + ηε — the same construction as
            # flow_source_mode="memory" — anchoring the pure-memory-source
            # function that the β=0 deployment (§13.46 discipline) actually
            # reads. db8 anchored the per-step MIXED source (β~U(0,1)): the
            # anchor pressure was dominated by β>0 samples, shaping z0 into
            # h's collaborator (routed arm won big) while starving the
            # standalone-source function (pure-z0 arm regressed +38%).
            anchor_source = source
            if (
                getattr(self.config, "memory_func_anchor_source_mode", "pure_z0")
                == "pure_z0"
            ):
                anchor_source = mem_start.to(dtype=actions.dtype)
                eta_a = float(getattr(self.config, "memory_start_noise_std", 0.0))
                if eta_a > 0:
                    anchor_source = anchor_source + eta_a * self._source_noise(
                        anchor_source, action_input
                    )
            endpoint = self._func_anchor_endpoint(
                anchor_source, state_features, vl_embeds, backbone_output, embodiment_id
            )
            fa = F.mse_loss(
                endpoint.float(), actions.detach().float(), reduction="none"
            ) * action_mask.float()
            if loss_weight is not None:
                fa = fa * loss_weight.float()
                fa_denom = (action_mask.float() * loss_weight.float()).sum() + 1e-6
            else:
                fa_denom = action_mask.float().sum() + 1e-6
            fa = fa.sum() / fa_denom
            z0_params = [
                p for p in self.memory_encoder.parameters() if p.requires_grad
            ]
            if z0_params and fa.requires_grad:
                z0_grads = torch.autograd.grad(
                    fa, z0_params, retain_graph=True, allow_unused=True
                )
                with torch.no_grad():
                    for p, g in zip(z0_params, z0_grads):
                        if g is None:
                            continue
                        g = g.to(dtype=p.dtype) * lam_func
                        p.grad = g if p.grad is None else p.grad + g
            func_anchor_loss = fa.detach()
            debug_stats["func_anchor_loss"] = float(func_anchor_loss.float())
            debug_stats["func_anchor_lambda"] = float(lam_func)
            debug_stats["func_anchor_source_mode"] = getattr(
                self.config, "memory_func_anchor_source_mode", "pure_z0"
            )

        return {
            "loss": loss,
            "action_loss": action_loss,
            "memory_anchor_loss": anchor_loss,
            "confidence_loss": conf_loss,
            "func_anchor_loss": func_anchor_loss,
            "debug_stats": debug_stats,
            "action_mask": action_mask,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }

    def _encode_features(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        """
        Encode features for the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_history_length, max_state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - state_features: [B, 1, input_embedding_dim]
        """
        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Handle state history: if we have fewer timesteps than expected, repeat to fill
        state = action_input.state
        current_T = state.shape[1]
        assert current_T == self.config.state_history_length, "current_T != state_history_length"
        # Reshape state from [B, state_history_length, max_state_dim] to [B, 1, state_history_length * max_state_dim]
        state = state.view(state.shape[0], 1, -1)

        # Embed state.
        state_features = self.state_encoder(state, embodiment_id)

        return BatchFeature(data={"backbone_features": vl_embeds, "state_features": state_features})

    @torch.no_grad()
    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_features: [B, seq_len, backbone_embedding_dim]
            state_features: [B, state_horizon, input_embedding_dim]
            embodiment_id: [B] (embodiment IDs)
            backbone_output: Output from the backbone model
        """
        vl_embeds = backbone_features

        # Set initial actions as the sampled noise, or — when the model was
        # trained with ``flow_source_mode="history"`` and the caller supplied a
        # previous action chunk — as the (perturbed) history chunk. The latter
        # matches the training-time flow source, so the model only predicts the
        # correction from the last executed chunk and 1-2 denoising steps
        # suffice. Episode starts (no history) fall back to pure noise.
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )
        # P2: episode-start memory recall — z0 = MemoryEncoder from the K query
        # outputs of vl_self_attention (no history needed, so this also covers
        # cold starts). Query features are produced by _encode_features ->
        # process_backbone_output, which mutates backbone_output in place.
        mem_start = None
        if self.memory_encoder is not None:
            query_features = backbone_output.get("memory_query_features")
            assert query_features is not None, (
                "memory_encoder is enabled but backbone_output lacks "
                "'memory_query_features' — process_backbone_output must run first"
            )
            # db18 T2: same state-KV contract as the training path.
            _enc_state = None
            if getattr(self.config, "memory_encoder_read_state", False):
                _enc_state = action_input.state.reshape(action_input.state.shape[0], -1)
            # Experiment ③ (survey §8 item 3): LeaP-spectrum adaptive-σ arm.
            # Opt-in ONLY via the eval-side post-load flag
            # ``memory_start_inference_sample``: draw z₀ ~ N(μ(c), σ(c)²) using
            # the db17 variance head instead of the deterministic μ. Default
            # stays False — deployment keeps the determinism asset.
            if getattr(self.config, "memory_start_inference_sample", False) and getattr(
                self.config, "memory_encoder_var_head", False
            ):
                _mu, _logvar = self.memory_encoder(
                    query_features, embodiment_id, state=_enc_state, return_logvar=True
                )
                mem_start = _mu + (0.5 * _logvar.float()).exp().to(_mu.dtype) * torch.randn_like(_mu)
            else:
                mem_start = self.memory_encoder(
                    query_features, embodiment_id, state=_enc_state
                )
        # v4.1: confidence readout (ĝ_mean, ĝ_q90[, ĝ_hist, ĝ_mem]) — computed
        # UP FRONT because T1 routing needs it to build the mixture source.
        confidence = None
        if self.confidence_head is not None:
            conf_features = backbone_output.get("confidence_query_features")
            if conf_features is not None:
                confidence = self.confidence_head(conf_features)
        # T1 (paper §4.4): per-sample mixture weight on the HISTORY term.
        # gap_ratio mode: β̂ = ĝ_mem²/(ĝ_hist² + ĝ_mem² + ε), clamped to [0, 1].
        # direct_beta mode (§13.18): the 5th output regresses the closed-form
        # β* directly — β̂ = ĝ_β, no gap-ratio mapping.
        beta_override = None
        if (
            confidence is not None
            and getattr(self.config, "confidence_routing_enabled", False)
            and self.config.flow_source_mode == "memory_history_mix"
        ):
            if self.confidence_head.num_outputs == 5:
                beta_override = confidence[:, 4].float().clamp(0.0, 1.0).view(-1, 1, 1)
            elif self.confidence_head.num_outputs == 4:
                g_hist = confidence[:, 2].float()
                g_mem = confidence[:, 3].float()
                if getattr(self.config, "confidence_routing_l1", False):
                    beta_hat = g_mem / (g_hist + g_mem + 1e-8)
                else:
                    beta_hat = (g_mem * g_mem) / (g_hist * g_hist + g_mem * g_mem + 1e-8)
                beta_override = beta_hat.clamp(0.0, 1.0).view(-1, 1, 1)
        # P0-2/T1 lag fix: at inference the buffered previous chunk starts one
        # re-plan stride in the past; Gr00tPolicy signals the stride via
        # options["history_rtc_offset"] (None -> training-time offset).
        rtc_offset = None
        if options is not None and options.get("history_rtc_offset") is not None:
            rtc_offset = int(options["history_rtc_offset"])
        actions = self._build_flow_source(
            actions,
            action_input,
            actions,
            memory_start=mem_start,
            beta_override=beta_override,
            rtc_offset=rtc_offset,
        )

        dt = 1.0 / self.num_inference_timesteps
        vel_strength = torch.ones_like(actions)

        if "action" in action_input:
            # If action in input when doing get action, it means we want to use RTC.
            # action_horizon is the action horizon of the input action.
            # rtc_overlap_steps is the number of steps to overlap with the previous action chunks.
            # rtc_frozen_steps is the number of steps to freeze the action, which is the latency of the policy inference.
            # rtc_ramp_rate is the rate of the ramp of denoising the actions.
            assert options is not None, "options is not None"
            assert "action_horizon" in options, "action_horizon is not in options"
            assert "rtc_overlap_steps" in options, "rtc_overlap_steps is not in options"
            assert "rtc_frozen_steps" in options, "rtc_frozen_steps is not in options"
            assert "rtc_ramp_rate" in options, "rtc_ramp_rate is not in options"

            action_horizon_before_padding = options["action_horizon"]

            # Use previous action instead of pure noise to do inpainting
            actions[:, : options["rtc_overlap_steps"], :] = action_input["action"][
                :,
                action_horizon_before_padding
                - options["rtc_overlap_steps"] : action_horizon_before_padding,
                :,
            ]
            vel_strength[:, : options["rtc_frozen_steps"], :] = 0.0
            # NOTE: use an exponential ramp strength to set the remaining unfrozen rtc_steps
            intermediate_steps = options["rtc_overlap_steps"] - options["rtc_frozen_steps"]
            # Create exponential ramp from 0 to 1 over intermediate steps
            t = torch.linspace(0.0, 1.0, intermediate_steps + 2, device=device)
            ramp = 1 - torch.exp(-options["rtc_ramp_rate"] * t)
            ramp = ramp / ramp[-1].clamp_min(1e-8)  # normalize to [0,1]
            ramp = ramp[
                1:-1
            ]  # we will only take the middle part of the ramp, ignore the 0.0 and 1.0
            # Apply ramp to the intermediate steps [batch, intermediate_steps, action_dim]
            vel_strength[
                :,
                options["rtc_frozen_steps"] : options["rtc_overlap_steps"],
                :,
            ] = ramp[None, :, None].to(device)

        # Run denoising steps.
        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # Add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            # db18 T1/T2: same cut as the training path when dit_state_token=False.
            if getattr(self.config, "dit_state_token", True):
                sa_embs = torch.cat((state_features, action_features), dim=1)
            else:
                sa_embs = action_features

            # Run model forward.
            if self.config.use_alternate_vl_dit:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                )
            else:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                )
            pred = self.action_decoder(model_output, embodiment_id)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity * vel_strength

        # v4.1: the confidence readout (ĝ_mean, ĝ_q90[, ĝ_hist, ĝ_mem]) was
        # computed up front (before the source build — T1 routing needs it).
        data = {
            "action_pred": actions,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }
        if confidence is not None:
            data["confidence"] = confidence
        if beta_override is not None:
            # T1: expose the routed per-sample mixture weight for logging /
            # stratified evaluation (weight on the HISTORY term).
            data["beta_hat"] = beta_override.view(-1)
        if mem_start is not None:
            # Q2 (review 2026-10-17): expose the raw memory-encoder output z0
            # (normalized action space, (B, T, D)) so visualization / probe
            # scripts can verify whether the encoder actually encodes the
            # recalled action chunk.
            data["memory_start"] = mem_start
        return BatchFeature(data=data)

    @torch.no_grad()
    def get_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - action_pred: [B, action_horizon, action_dim] predicted actions
        """
        features = self._encode_features(backbone_output, action_input)
        return self.get_action_with_features(
            backbone_features=features.backbone_features,
            state_features=features.state_features,
            embodiment_id=action_input.embodiment_id,
            backbone_output=backbone_output,
            action_input=action_input,
            options=options,
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def prepare_input(self, batch: dict) -> BatchFeature:
        """Prepare input batch for the action head."""
        return BatchFeature(data=batch)


def get_backbone_cls(config: Gr00tN1d7Config):
    if "nvidia/Cosmos-Reason2" in config.model_name or "Qwen/Qwen3-VL" in config.model_name:
        # We import here as Qwen3Backbone depends on newer transformers versions than the rest of the code.
        from gr00t.model.modules.qwen3_backbone import Qwen3Backbone

        return Qwen3Backbone
    else:
        raise ValueError(f"Unsupported model name: {config.model_name}")


class Gr00tN1d7(PreTrainedModel):
    """Gr00tN1d7: VLA model with Cosmos-Reason2-2B (Qwen3-VL) backbone."""

    config_class = Gr00tN1d7Config
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Gr00tN1d7Config,
        transformers_loading_kwargs: dict = {"trust_remote_code": True},
    ):
        """
        Initialize Gr00tN1d7 model.

        Args:
            config: Model configuration
            transformers_loading_kwargs: Dict with transformers loading parameters:
                - transformers_trust_remote_code: Whether to trust remote code when loading from HF Hub
                - transformers_local_files_only: Whether to only use local files
                - model_revision: Specific model revision to use
                - transformers_cache_dir: Directory to cache downloaded models
                - transformers_access_token: HuggingFace access token for gated models

        Note: During training, transformers parameters are passed from training config.
              During inference (e.g., from_pretrained), defaults are used.
        """
        super().__init__(config)
        self.config = config

        backbone_cls = get_backbone_cls(config)
        self.backbone = backbone_cls(
            model_name=config.model_name,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.select_layer,
            reproject_vision=config.reproject_vision,
            use_flash_attention=config.use_flash_attention,
            load_bf16=config.load_bf16,
            tune_top_llm_layers=config.tune_top_llm_layers,
            trainable_params_fp32=config.backbone_trainable_params_fp32,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

        # Initialize action head
        self.action_head = Gr00tN1d7ActionHead(config)
        from .processing_gr00t_n1d7 import Gr00tN1d7DataCollator

        # P2 v3 + v4.1: placeholder tokens appended at the sequence tail =
        # K memory queries (only when they pass THROUGH the backbone) + C
        # confidence queries (v4.1, arm-independent). The backbone overwrites
        # the last-(K+C) embeddings; the action head splits them back out
        # (memory first, confidence last). The "post" path and arms without
        # any threaded queries leave tokenization byte-identical to upstream.
        collator_num_queries = 0
        if getattr(config, "memory_start_enabled", False) and getattr(
            config, "memory_query_path", "backbone"
        ) == "backbone":
            collator_num_queries += config.memory_num_queries
        if getattr(config, "confidence_tokens_enabled", False):
            collator_num_queries += int(getattr(config, "confidence_num_queries", 4))
        self.collator = Gr00tN1d7DataCollator(
            model_name=config.model_name,
            model_type=config.backbone_model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
            memory_num_queries=collator_num_queries,
        )

    def _backbone_query_embeds(self, batch_size: int) -> torch.Tensor | None:
        """(B, K+C, H) learnable query embeddings threaded through the backbone.

        Concatenates, in order: the K memory queries (only on the "backbone"
        path — the v2 "post" path appends them to the VL features inside the
        action head instead) and the C confidence queries (v4.1). The collator
        has appended the same total number of placeholder tokens at the
        sequence tail; the action head splits the returned hidden states back
        out (memory first, confidence last). Returns None when neither module
        threads queries through the backbone.
        """
        parts = []
        memory_encoder = getattr(self.action_head, "memory_encoder", None)
        if memory_encoder is not None and getattr(self.config, "memory_query_path", "backbone") == "backbone":
            parts.append(memory_encoder.queries)  # (K, H) learnable parameter
        confidence_head = getattr(self.action_head, "confidence_head", None)
        if confidence_head is not None:
            parts.append(confidence_head.queries)  # (C, H) learnable parameter
        if not parts:
            return None
        queries = torch.cat(parts, dim=0)
        return queries.unsqueeze(0).expand(batch_size, -1, -1)

    def prepare_input(self, inputs: dict) -> Tuple[BatchFeature, BatchFeature]:
        """Prepare inputs for backbone and action head."""

        # NOTE -- currently the eval code doesn't use collator, so we need to add it here
        # this should ideally be fixed upstream
        if "vlm_content" in inputs:
            # Fix for n_envs > 1: Process all environments' VLM content, not just the first
            vlm_content_list = inputs["vlm_content"]
            # Ensure vlm_content_list is always a list for consistent processing
            if not isinstance(vlm_content_list, list):
                vlm_content_list = [vlm_content_list]

            # Process all VLM contents through the collator
            prep = self.collator([{"vlm_content": vlm} for vlm in vlm_content_list])["inputs"]
            inputs.pop("vlm_content")
            inputs.update(prep)

        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        # Move to device and dtype
        def to_device_with_dtype(x):
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.dtype)
            else:
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_dtype, action_inputs)

        return backbone_inputs, action_inputs

    def forward(self, inputs: dict) -> BatchFeature:
        """
        Forward pass through the complete model.

        Args:
            inputs: Dictionary containing:
                - Action inputs (state, action, embodiment_id, etc.)

        Returns:
            BatchFeature containing loss and other outputs
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        query_embeds = self._backbone_query_embeds(backbone_inputs["input_ids"].shape[0])
        backbone_outputs = self.backbone(backbone_inputs, query_embeds=query_embeds)
        action_outputs = self.action_head(backbone_outputs, action_inputs)

        return action_outputs

    def get_action(self, inputs: dict, options: dict[str, Any] | None = None) -> BatchFeature:
        """
        Generate actions using the complete model.
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)

        # Forward through backbone
        query_embeds = self._backbone_query_embeds(backbone_inputs["input_ids"].shape[0])
        backbone_outputs = self.backbone(backbone_inputs, query_embeds=query_embeds)
        action_outputs = self.action_head.get_action(backbone_outputs, action_inputs, options)

        return action_outputs

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


# Register the model with HuggingFace
AutoConfig.register("Gr00tN1d7", Gr00tN1d7Config)
AutoModel.register(Gr00tN1d7Config, Gr00tN1d7)
