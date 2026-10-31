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

from dataclasses import MISSING, asdict, dataclass, field, is_dataclass
from enum import Enum
import json
from pathlib import Path

import torch
from transformers import PretrainedConfig

from . import register_model_config


@dataclass
class Gr00tN1d7Config(PretrainedConfig):
    """Unified configuration for Gr00tN1d7 model with backbone and action head.

    Gr00tN1d7 uses the Cosmos-Reason2-2B (Qwen3-VL architecture) VLM backbone,
    replacing the Eagle backbone used in Gr00tN1d6.
    """

    # Model identification
    model_type: str = "Gr00tN1d7"
    model_dtype: str = "bfloat16"  # Use bfloat16 for Flash Attention compatibility

    # Backbone configuration
    model_name: str = "nvidia/Cosmos-Reason2-2B"
    backbone_model_type: str = "qwen"
    model_revision: str | None = None
    tune_top_llm_layers: int = 0  # Number of top LLM layers to tune
    backbone_embedding_dim: int = 2048  # project_to_dim; must match Cosmos-Reason2-2B hidden size
    tune_llm: bool = False
    tune_visual: bool = False
    select_layer: int = 12
    reproject_vision: bool = False
    use_flash_attention: bool = True
    load_bf16: bool = False  # Enable BF16 loading
    backbone_trainable_params_fp32: bool = True

    ### Processing parameters
    image_crop_size: tuple[int, int] | None = (230, 230)
    image_target_size: tuple[int, int] | None = (256, 256)

    shortest_image_edge: int | None = None
    crop_fraction: float | None = None

    random_rotation_angle: int | None = None
    color_jitter_params: dict[str, float] | None = None
    use_albumentations_transforms: bool = True
    letter_box_transform: bool = False
    # Extra augmentation config (mask-based and others).
    extra_augmentation_config: dict | None = None
    formalize_language: bool = True
    apply_sincos_state_encoding: bool = (
        False  # Global flag to enable per-embodiment sin/cos encoding
    )
    use_percentiles: bool = True
    use_relative_action: bool = False

    # Action head configuration parameters
    max_state_dim: int = 132  # Default from state_shape
    max_action_dim: int = 132  # Default from action_shape
    action_horizon: int = 40
    hidden_size: int = 1024
    input_embedding_dim: int = 1536

    # State history: number of consecutive state timesteps fed to the state encoder
    state_history_length: int = 1

    # Global parameters
    add_pos_embed: bool = True
    attn_dropout: float = 0.2
    use_vlln: bool = True
    max_seq_len: int = 1024
    use_alternate_vl_dit: bool = True  # True for AlternateVLDiT, False for DiT
    attend_text_every_n_blocks: int = 2

    diffusion_model_cfg: dict = field(
        default_factory=lambda: {
            "positional_embeddings": None,
            "num_layers": 16,
            "num_attention_heads": 32,
            "attention_head_dim": 48,
            "norm_type": "ada_norm",
            "dropout": 0.2,
            "final_dropout": True,
            "output_dim": 1024,
            "interleave_self_attention": True,
        }
    )

    # Flow matching parameters
    num_inference_timesteps: int = 4
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000

    # Training parameters
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    tune_vlln: bool = True

    # --- P0-1: co-training with pseudo-labelled / action-free data ---
    # Mirrors gr00t.configs.finetune_config.FinetuneConfig (single source of truth
    # for CLI); duplicated here so the values live in config.json and survive
    # from_pretrained, which is what the inference path reads.
    use_has_real_action: bool = False
    pseudo_action_loss_weight: float = 1.0
    drop_unlabeled_action: bool = False

    # --- P0-2: history-conditioned flow matching ---
    flow_source_mode: str = "gaussian"          # "gaussian" | "history" | "memory" | "memory_history_mix"
    gaussian_source_noise_std: float = 1.0
    """Scale of the pure Gaussian source, shared by training and inference.
    1.0 = legacy N(0,I); 0.04 = G-small noise-budget-matched control."""
    flow_source_action_noise_std: float = 0.0   # eta
    history_action_offset: int = 1
    history_action_align: str = "shift"         # "shift" | "rtc"

    # --- T0-2: ΔFM selective source-noise mask ---
    flow_source_noise_selective: bool = False
    """If True, the eta perturbation on history/memory flow sources is masked
    per dimension via the batch's ``action_noise_mask`` (gripper columns and
    padding exempt — Δ Flow Matching's b_a mask; near-discrete gripper channels
    are otherwise polluted by continuous noise). No-op when the mask is absent."""

    # --- P2: learnable action memory start (UVM-style z0 anchor) ---
    memory_start_enabled: bool = False
    """Master switch. Requires flow_source_mode in {"memory", "memory_history_mix"}.
    "memory": source = z0 = MemoryEncoder(c) + eta*eps   (UVM original recipe)
    "memory_history_mix": source = beta*history + (1-beta)*z0 + eta*eps."""
    memory_start_anchor_lambda: float = 1.0
    """lambda of the reconstruction anchor ||z0 - a0||^2. 1.0 reproduces the UVM
    hard anchor; 0.05-0.1 is the soft-anchor variant that keeps z0 a *prior*
    instead of a lookup-table answer (see survey doc §13)."""
    memory_start_diff_anchor_lambda: float = 0.0
    """lambda of the db10 differential (shape) anchor ||Δref − Δy||² over first
    differences along the horizon (0.0 = off). CONSTANT on purpose — NOT
    annealed by memory_anchor_lambda_schedule — so the shape pressure
    survives the point anchor's cosine decay. Reference mirrors the point
    anchor (role → mixed source s, else z0); grad flows only into z0."""
    memory_anchor_lambda_schedule: str = "constant"
    """Schedule for the anchor lambda over training: ``constant`` (λ0 throughout)
    | ``cosine`` (λ(p) = λ_final + 0.5·(λ0−λ_final)·(1+cos(π·p)), p=step/max_steps)
    | ``piecewise`` (λ0 until ``memory_anchor_lambda_warmup_steps``, then λ_final).
    Annealing keeps the hard anchor's early convergence signal but softens it
    late so z0 ends as a prior, not a memorised lookup (survey doc §13.8)."""
    memory_anchor_lambda_final: float | None = None
    """Final lambda for non-constant schedules; None defaults to 0.1 × λ0."""
    memory_anchor_lambda_warmup_steps: int = 0
    """Switch point (optimizer steps) of the ``piecewise`` schedule."""
    memory_start_anchor_type: str = "mse"
    """Anchor loss type: ``mse`` = λ‖z0−y‖² per-sample reconstruction (UVM
    original); ``infonce`` = masked-batch InfoNCE (review 2026-10-17, D4-1):
    logits_ij = −d(z_i, y_j)/τ, diagonal positives. Fights the soft-anchor
    variance collapse and makes z0 semantically discriminative. B==1 -> MSE."""
    memory_start_anchor_temperature: float = 0.1
    """Softmax temperature τ of the InfoNCE anchor."""
    memory_start_anchor_con_lambda: float = 0.1
    """db17 (paper §8 item 11-①): weight of the InfoNCE sub-term inside
    ``anchor_type == 'nll_infonce'`` — independent CONSTANT lambda, NOT
    annealed with the main anchor (mirrors the differential anchor's own
    lambda). >= 0."""
    memory_start_anchor_target: str = "y"
    """Anchor target for ``anchor_type == 'mse'`` (review 2026-10-18, L1′):
    ``y`` = UVM original, anchor z0 → ground-truth chunk y;
    ``role`` = role-consistent anchor, anchor the MIXED source s = βh+(1−β)z0
    → y (equivalently z0 → (y−βh)/(1−β) in multiplicative form — no division,
    no β→1 singularity, and the anchor vanishes exactly when z0 has no role).
    Eliminates the anchor-vs-role gradient conflict (survey doc §13.24 root
    cause ⑥): with target ``y`` the anchor pulls z0 → E[y|c] while the flow
    loss wants s → E[y|c], and the two collide at segment heads where h fails.
    Effective strength scales as (1−β)² (E[·]=1/3 under β~U(0,1)); β=0 reduces
    to the ``y`` anchor. Only meaningful with flow_source_mode
    ``memory_history_mix``; ignored (with a warning-free fallback to ``y``)
    for ``infonce``."""
    memory_start_mix_beta: float = 0.0
    """beta in [0, 1]: weight of the (aligned) history chunk in
    "memory_history_mix" mode. 0 = pure memory start, 1 = pure history."""
    memory_source_ablate_zero: bool = False
    """Eval-only ablation (survey doc §13.44): when True, the memory start z₀
    is replaced by zeros at source-mix time, turning the (1−β) component into
    the normalized-space constant (= dataset mean action). Decisive control
    isolating the memory encoder's CONTENT value from the 'drop history'
    effect. Never set during training."""
    memory_start_mix_beta_sampling: str = "fixed"
    """``fixed`` (scalar β everywhere, legacy) or ``uniform``: during training,
    draw β ~ U(memory_start_mix_beta_min, memory_start_mix_beta_max) per sample
    per step. Inference is unchanged (scalar β or confidence-routed β̂)."""
    memory_start_mix_beta_min: float = 0.0
    """Lower bound of the uniform β sampling range (inclusive)."""
    memory_start_mix_beta_max: float = 1.0
    """Upper bound of the uniform β sampling range (inclusive)."""
    memory_start_anchor_beta_max: float = 1.0
    """db12 (survey doc §13.61): anchor-side β decoupling. The role/diff anchors
    are computed ONLY on samples whose drawn mixture β ≤ this threshold
    (per-sample gate on the β the flow branch already drew; flow side keeps
    β ~ U(0,1) so the DiT still learns the whole simplex). Rationale: under
    β ~ U(0,1) the role-anchor equilibrium is z0* = 1.5·E[y|c] − 0.5·h — the
    high-β samples inject h's phase lag into z0 (measured on db11 oscillation
    segments: frequency nailed via text, phase inverted, RMS ratio ≈ 1.59 ≈
    |1.5 − 0.5·e^{iδ}|). Restricting the anchor to β ≤ 0.5 shrinks the h
    contamination weight while keeping the (1−β)²-vanishing property. 1.0 =
    legacy behaviour (gate off). Requires mix_beta_sampling == 'uniform' to
    have any effect (fixed-β runs have a single β — either all gated or none)."""
    memory_start_z0_recon_lambda: float = 0.0
    """db12 (survey doc §13.61): β=0 endpoint debias. Small CONSTANT-weight
    pure reconstruction term λ·‖z0 − y‖² added alongside the role anchor.
    The role equilibrium z0* = 1.5·E[y|c] − 0.5·h carries a 0.5·(E[y|c] − h)
    extrapolation bias at the β=0 deployment arm; this term pulls z0 back to
    E[y|c] directly. NOT annealed, NOT β-gated (it targets the β=0 endpoint
    by construction). 0.0 = off; db12 uses 0.05."""
    memory_start_noise_std: float = 0.0
    """eta for memory modes: Gaussian perturbation on the source to keep the
    path stochastic (hard-anchor UVM recipe uses 0.0; soft arm uses ~0.02)."""
    memory_start_noise_sampling: str = "fixed"
    """db23a (survey §13.91): ``fixed`` uses scalar ``memory_start_noise_std``
    (legacy); ``uniform`` draws per-sample eta ~ U(0, memory_start_noise_std)
    per training step (std acts as the max) — isotropic source-spectrum
    diversity around the semantic start. Applies to ``memory_history_mix``
    mode only; eval always uses the scalar eta."""
    memory_start_hidden_dim: int = 1024
    """Hidden width of the ActionMemoryEncoder trunk."""
    memory_num_queries: int = 64
    """K learnable query tokens owned by the memory encoder (UVM uses K=64).
    With memory_query_path="backbone" (default, v3) they are threaded THROUGH
    the frozen backbone (the collator appends K placeholder tokens whose
    embeddings are overwritten); with "post" (v2) they are appended to the
    frozen VL features and run through the trainable vl_self_attention."""
    memory_query_dim: int = 64
    """Per-query projection width inside ActionMemoryEncoder before the
    flatten + MLP trunk (encoder input width = memory_num_queries *
    memory_query_dim)."""
    memory_query_path: str = "backbone"
    """Where the K memory queries interact with the observation:
      * ``backbone`` (v3, UVM-faithful depth): queries pass THROUGH the frozen
        VLM — the collator appends K placeholder token ids at the sequence tail
        and a pre-hook overwrites their inputs_embeds rows with the learnable
        queries; the last-K hidden states of the retained stack feed the
        ActionMemoryEncoder. Frozen weights are never modified.
      * ``post`` (v2): queries only attend to the frozen VL features in the
        trainable vl_self_attention adapter (shallower; zero backbone contact)."""
    tune_memory_encoder: bool = True
    """Train the memory encoder even when tune_projector=False (it is a new
    randomly-initialised module; freezing it would make the memory arm a
    constant-zero source)."""

    # --- v4.1: confidence query tokens (gap predictor) ---
    confidence_tokens_enabled: bool = False
    """Thread C confidence queries through the frozen backbone (after the K
    memory queries) and read out (ĝ_mean, ĝ_q90): the predicted masked L1 gap
    of the one-step endpoint ŷ1 = s + v_θ(s, t=0, c). Arm-independent: works
    with every flow_source_mode; the collator appends C extra placeholder
    tokens, so checkpoints must be evaluated with their own processor config."""
    confidence_num_queries: int = 4
    """C confidence query tokens (v4.1)."""
    confidence_loss_weight: float = 0.1
    """Weight of MSE(ĝ_mean) + pinball_0.9(ĝ_q90) against the detached gap."""
    confidence_every_k: int = 1
    """Compute the no-grad endpoint target every k-th optimizer step (amortise
    the extra DiT forward); skipped steps omit the confidence loss."""
    confidence_routing_enabled: bool = False
    """T1 (mixed-source paper §4.4): extend the confidence head to 4 outputs
    (ĝ_mean, ĝ_q90, ĝ_hist, ĝ_mem) — the last two are the PER-SOURCE one-step
    endpoint gaps of the pure history / pure memory arms — and at inference
    derive the per-sample mixture weight on the HISTORY term as
    β̂ = ĝ_mem²/(ĝ_hist² + ĝ_mem² + ε), clamped to [0, 1] (squared-gap form,
    optimal under residual orthogonality). Requires confidence_tokens_enabled
    and flow_source_mode='memory_history_mix'."""
    confidence_routing_l1: bool = False
    """Ablation: route with the L1 ratio β̂ = ĝ_mem/(ĝ_hist + ĝ_mem + ε)
    instead of the squared-gap form."""
    confidence_routing_mode: str = "gap_ratio"
    """Routing readout variant: ``gap_ratio`` (legacy T1: 4-output head,
    β̂ = ĝ_mem²/(ĝ_hist² + ĝ_mem²)) | ``direct_beta`` (survey §13.18: 5-output
    head — the 5th output regresses the closed-form optimal mixture weight
    β* = <y−z0, h−z0>/‖h−z0‖² per sample, and at inference β̂ = ĝ_β directly).
    direct_beta removes the gap-ratio mapping that proved insensitive to
    segment structure in the long-run evaluation (§13.17, verdict 2)."""

    # --- Sub-instruction (dual text port) ---
    sub_instruction_mode: str = "off"
    """Dual text port driving the same action head: ``off`` (legacy single
    instruction) | ``concat`` (prompt = "<task>. next step: <sub>") |
    ``replace`` (the sub-instruction replaces the task text). With
    concat/replace the dataset demotes the per-frame instruction to
    ``sub_text`` and promotes the episode-level ``high_level_task``
    (episodes.jsonl) to ``text``. Propagated to the processor via the
    setup.py overrides and persisted in processor_config.json."""
    sub_instruction_dropout_prob: float = 0.0
    """Training-time CFG-style dropout on the two text ports (only when
    sub_instruction_mode != "off"): with probability p/2 the task text is
    dropped (sub-only sample), with p/2 the sub-instruction is dropped
    (task-only sample), so either port can drive the action head alone at
    inference. 0 = always feed both."""

    # --- P2 transformer memory encoder (alternative trunk) ---
    memory_encoder_arch: str = "mlp"
    """ActionMemoryEncoder trunk architecture: ``mlp`` (UVM-faithful flatten +
    2-layer MLP + per-embodiment CategorySpecificLinear head, ~179M params,
    173M in the 32 embodiment heads) | ``transformer`` (K-query self-attention
    + H learnable step-queries cross-attending to the K queries + per-
    embodiment embedding + SHARED zero-init per-step Linear, ~3.1M — ~58x
    fewer parameters and structurally horizon-general; survey doc §13.11).
    Default ``mlp`` keeps pre-existing checkpoints loadable."""
    memory_encoder_d_model: int = 256
    """d_model of the transformer memory encoder (arch="transformer" only)."""
    memory_encoder_layers: int = 2
    """Self-attention layers over the K memory queries (transformer arch)."""
    memory_encoder_bottleneck_dim: int = 0
    """db13 (paper §6.4 direction 2): hard low-dimensional content bottleneck
    for the transformer memory encoder. When > 0, z₀'s content must pass
    through an L-dim task-parameter latent. NOTE (review P0-1): with the y-MSE
    anchor unchanged this does NOT lower the Var(y|c) floor — it tests whether
    the learnable segment-head content is low-rank (z₀* = proj_M(E[y|c])).
    0 = disabled (legacy); requires arch="transformer"."""
    memory_encoder_var_head: bool = False
    """db17 (paper §8 item 11-①): heteroscedastic variance head on the
    transformer memory encoder — a second zero-init Linear(d_model →
    action_dim) emitting per-dim log σ², enabling an explicit Gaussian NLL
    anchor (anchor_type "nll"/"nll_infonce") instead of fixed-variance MSE
    (MSE = σ²-fixed special case whose optimum is E[y|c]; learned σ(c) absorbs
    irreducible conditional variance so μ can concentrate on one mode).
    Deployment still uses z₀ = μ(c) only — the deterministic-start asset is
    preserved; σ is a training-time supervision device. Requires
    arch="transformer"."""
    memory_encoder_read_state: bool = False
    """db18 T2 (paper §8 item 11-⑤): the memory encoder's cross-attention reads
    the flattened proprioceptive state as an extra KV token — z₀ = f(H, q).
    Requires arch="transformer" and memory_start_enabled. Deployment-invariant:
    z₀ stays a deterministic function of the (now fuller) condition."""
    dit_state_token: bool = True
    """db18 T1/T2 (paper §8 item 11-⑤): when False, the DiT self-attention input
    drops the state token (sa_embs = action tokens only) — the state channel
    into the FIELD is cut. Default True = legacy. Incompatible with the L2
    functional anchor (its auxiliary forward is not re-routed)."""

    zero_init_dit: bool = False
    """db28 (from-scratch G/Z arms): after checkpoint loading, re-initialise the
    DiT velocity field from scratch under the zero-identity contract (fresh
    random draw seeded by data seed + 1000, then per-block attn/FFN output
    projections zeroed). The base checkpoint still supplies the VLM backbone /
    encoders / decoders; only the DiT is trained from scratch. G and Z arms at
    the same --seed share bit-identical initial DiT weights."""

    # --- L2 functional anchor (survey doc §13.43 mechanism ③, §13.44 P1) ---
    memory_func_anchor_enabled: bool = False
    """If True, add L_func = λ_func · ‖ŷ1 − y‖² (masked) with the one-step
    endpoint ŷ1 = s + v_θ(s, t=0, c) computed on the SAME source s the flow
    loss used. The t=0 branch forward runs with the DiT pathway
    (action_encoder / position_embedding / DiT / action_decoder)
    requires_grad-frozen and the context tensors detached (§13.44 P1-1), so
    the gradient shapes ONLY the source — i.e. z0 / the memory encoder —
    never the DiT weights. Anchors the deployment-relevant function of z0
    ('few-step denoising from s reaches y') instead of its curve shape."""
    memory_func_anchor_lambda: float = 0.1
    """Constant weight of the functional anchor. Deliberately NOT annealed
    (§13.43 bundle 'λ 常量化'): unlike the reconstruction anchor, this
    pressure must persist to convergence."""
    memory_func_anchor_every_k: int = 1
    """Run the anchor every k-th optimizer step to amortise its extra
    grad-carrying t=0 DiT forward (§13.44 P1-3: not zero-cost)."""
    memory_func_anchor_source_mode: str = "pure_z0"
    """Which source the functional anchor's endpoint is computed from
    (§13.48, db9): "pure_z0" re-draws z0 + ηε — the β=0 deployment operating
    point the inference path actually reads (default; fixes the db8
    train/deploy source-distribution mismatch); "mix" keeps the db8 behavior
    of anchoring the per-step β-sampled mixed source."""

    # State augmentation parameters
    state_dropout_prob: float = 0.8  # State dropout probability
    exclude_state: bool = False  # Zero out all state inputs (ablation)
    use_mean_std: bool = False  # Use mean/std normalization instead of min/max

    # Multi-embodiment parameters
    max_num_embodiments: int = 32

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Ensures that all dataclass defaults (including those using default_factory)
        # are explicitly assigned to the instance, even if dataclasses initialization or subclassing
        # (PretrainedConfig) interferes with normal default injection.
        for f in self.__dataclass_fields__.values():
            if not hasattr(self, f.name):
                if f.default is not MISSING:
                    setattr(self, f.name, f.default)
                elif getattr(f, "default_factory", MISSING) is not MISSING:
                    setattr(self, f.name, f.default_factory())

    def to_filtered_dict(self, exclude_augment: bool = True) -> dict:
        """Return a dictionary representation of this config, optionally excluding augmentation keys."""
        if is_dataclass(self):
            cfg = asdict(self)
        else:
            cfg = dict(self.__dict__)

        if exclude_augment:
            exclude_keys = {
                "random_rotation_angle",
                "color_jitter_params",
                "use_albumentations_transforms",
                "formalize_language",
                "image_crop_size",
                "image_target_size",
                "shortest_image_edge",
                "crop_fraction",
            }
            cfg = {k: v for k, v in cfg.items() if k not in exclude_keys}

        return cfg

    def to_filtered_json(self, exclude_augment: bool = True, **kwargs) -> str:
        """Return a JSON string of this config, optionally excluding augmentation keys."""

        def default(o):
            if isinstance(o, (Path, torch.dtype, torch.device)):
                return str(o)
            if isinstance(o, Enum):
                return o.value
            return str(o)

        return json.dumps(
            self.to_filtered_dict(exclude_augment),
            indent=2,
            default=default,
            **kwargs,
        )


register_model_config("Gr00tN1d7", Gr00tN1d7Config)
