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

# Finetune config used for single node post-training.
from dataclasses import dataclass
import warnings


@dataclass
class FinetuneConfig:
    """
    Configuration for fine-tuning a Vision-Language-Action (VLA) model.

    This dataclass defines all parameters needed to launch a fine-tuning job
    on a pretrained base model using a custom dataset and embodiment-specific
    modality configuration. It controls model tuning options, data augmentation,
    and training hyperparameters.
    """

    # --- Data and Model Paths ---
    base_model_path: str
    """Path to the pretrained base model checkpoint (e.g., Hugging Face model hub or local directory)."""

    dataset_path: str
    """Path to one dataset root, or an os.pathsep-separated list of dataset roots."""

    embodiment_tag: str
    """Embodiment tag (name or value, case-insensitive). See EmbodimentTag for known tags."""

    vlm_model_path: str | None = None
    """Optional local path or hub ID for the frozen VLM. Falls back to VLM_MODEL_PATH, then the model default."""

    modality_config_path: str | None = None
    """
    Path to a Python file defining the modality configuration for the given embodiment.
    If None, use the pre-registered modality config in `gr00t/configs/data/embodiment_configs.py`.
    """

    # --- Model Tuning Flags ---
    tune_llm: bool = False
    """If True, fine-tune the language model (LLM) backbone during training."""

    tune_visual: bool = False
    """If True, fine-tune the visual encoder (e.g., ViT or CNN backbone)."""

    tune_projector: bool = True
    """If True, fine-tune the multimodal projector layers that map vision/language features to a shared space."""

    tune_diffusion_model: bool = True
    """If True, fine-tune the diffusion-based action decoder (if present in the model)."""

    tune_vlln: bool = True
    """If True, fine-tune the VLM layer-norm and vl_self_attention adapter
    (~0.2B parameters). Pass --no-tune-vlln for DiT-only finetuning on
    memory-constrained GPUs."""

    load_bf16: bool = False
    """If True, load and train the FULL model (backbone + action head) in bfloat16,
    instead of the default fp32-master-weights + bf16-autocast path. Roughly
    quarters parameter+gradient+optimizer memory, making DiT-only finetuning fit
    on ~24GB GPUs (e.g. a single RTX 4090), at some risk to long-run training
    stability (bf16 AdamW states). Keep the backbone frozen (tune_llm/tune_visual
    off) when using this flag."""

    gradient_checkpointing: bool = False
    """If True, recompute activations in backward instead of storing them
    (trades ~20-30% step time for activation memory)."""

    optim: str = "adamw_torch"
    """HF Trainer optimizer id. adamw_bnb_8bit (requires `pip install bitsandbytes`)
    keeps fp32 master weights while cutting optimizer-state memory ~4x — an
    alternative to --load-bf16 for memory-constrained GPUs."""

    state_dropout_prob: float = 0.2
    """
    Dropout probability applied to state inputs for regularization during training.
    """

    # --- Data Augmentation ---
    random_rotation_angle: int | None = None
    """Maximum rotation angle (in degrees) for random rotation augmentation of input images."""

    color_jitter_params: dict[str, float] | None = None
    """
    Parameters for color jitter augmentation on images.

    Expected keys include:
      - "brightness": float
      - "contrast": float
      - "saturation": float
      - "hue": float
    Example: {"brightness": 0.4, "contrast": 0.4, "saturation": 0.4, "hue": 0.1}

    If None, applying the default color jitter augmentation from the pretrained model.
    """

    use_percentiles: bool = True
    """
    If True, use q01/q99 percentile statistics for state/action min-max normalization.
    If False, use full min/max statistics.
    """

    shortest_image_edge: int | None = None
    """
    Resize images so the shortest edge has this size before fractional cropping.
    If set, crop_fraction must also be set and legacy image_crop_size/image_target_size
    preprocessing is disabled.
    """

    crop_fraction: float | None = None
    """
    Fraction of the resized image retained by the random/center crop.
    If set, shortest_image_edge must also be set and legacy image_crop_size/image_target_size
    preprocessing is disabled.
    """

    extra_augmentation_config: str | None = None
    """
    JSON string for extra image augmentations (mask-based and others).

    Expected keys include:
      - "background_noise_transforms": list of dicts for noise on mask regions
          - "target_mask_values": list of int (e.g., [0])
          - "p": float (probability of applying)
      - "masked_region_transforms": list of dicts for color tint on mask regions
          - "target_mask_values": list of int (e.g., [4] or [5])
          - "p": float (probability of applying)
          - "alpha_range": [min, max] for random_tint intensity

    Example: {"background_noise_transforms": [{"target_mask_values": [0], "p": 0.9}],
              "masked_region_transforms": [{"target_mask_values": [4], "p": 1.0, "alpha_range": [0, 1]}]}

    If None, no extra augmentations are applied.
    """

    # --- Training Configuration ---
    global_batch_size: int = 64
    """Total batch summed across all GPUs in one forward/backward, BEFORE
    gradient accumulation."""

    dataloader_num_workers: int = 2
    """Number of parallel worker processes used for data loading."""

    learning_rate: float = 1e-4
    """Initial learning rate for optimizer."""

    lr_scheduler_type: str = "cosine"
    """Hugging Face learning-rate scheduler (for example ``cosine`` or
    ``constant``). Use ``constant`` with ``warmup_ratio=0`` for a genuinely
    non-zero constant-LR continuation run."""

    gradient_accumulation_steps: int = 1
    """Forward passes per optimizer step. Multiplies ``global_batch_size`` to
    produce the post-accumulation per-optimizer-step batch."""

    output_dir: str = "./outputs"
    """Directory where model checkpoints, logs, and outputs are saved."""

    experiment_name: str | None = None
    """Optional experiment name used as the W&B run name. Defaults to the output directory basename."""

    wandb_project: str = "finetune-gr00t-n1d7"
    """W&B project name to log runs to."""

    save_steps: int = 1000
    """Frequency (in training steps) at which to save checkpoints."""

    save_total_limit: int = 5
    """Maximum number of checkpoints to keep before older ones are deleted."""

    num_gpus: int = 1
    """Number of GPUs available for distributed or single-node training."""

    use_wandb: bool = False
    """
    If True, log metrics and artifacts to Weights & Biases (wandb).
    The project is `finetune-gr00t-n1d7`.
    You need to login to wandb to view the logs.
    """

    max_steps: int = 10000
    """Total number of training steps to run before stopping."""

    weight_decay: float = 1e-5
    """Weight decay coefficient for optimizer (L2 regularization)."""

    warmup_ratio: float = 0.05
    """Proportion of total training steps used for learning rate warm-up."""

    ds_weights_alpha: float | None = None
    """Power-law exponent for dataset soup weighting. When set, each dataset's
    sampling weight is len(dataset)^alpha and per-dataset mix_ratio values are ignored."""

    shard_size: int = 2**10
    """Size of the shard to use for the dataset during preloading."""

    episode_sampling_rate: float = 0.1
    """Sampling rate for the episodes."""

    num_shards_per_epoch: int = int(1e5)
    """Number of shards to use for the dataset. reduce this number if vram is limited."""

    save_only_model: bool = False
    """If True, save only model weights (skip optimizer/scheduler/RNG states). Cannot resume training from these checkpoints."""

    resume_from_checkpoint: bool = False
    """If True, resume from the latest ``checkpoint-*`` in ``output_dir``. Default
    False so a rerun against an existing ``output_dir`` starts fresh instead of
    silently merging with a previous experiment. Incompatible with
    ``save_only_model=True`` (enforced by ``experiment.run``)."""

    skip_weight_loading: bool = False
    """If True, skip loading model weights from base_model_path (architecture only).
    The processor (tokenizer/config) is still loaded from base_model_path.
    Useful for CI/testing to skip the slow checkpoint shard loading."""

    # --- P0-1: co-training with pseudo-labelled / action-free data ---
    use_has_real_action: bool = False
    """If True, read a per-sample ``has_real_action`` flag from the data and weight
    the flow-matching loss by it. Real actions get weight 1.0, pseudo-labelled
    (IDM / LAPA / DreamGen) actions get ``pseudo_action_loss_weight``.

    Requires the dataset to expose the flag, either as:
      * a per-frame ``has_real_action`` column in the episode parquet, or
      * a sidecar ``<dataset>/meta/has_real_action.json``
        (``{"default": 1.0, "episodes": {"<episode_index>": 0.0, ...}}``).
    Samples without any source are treated as real (weight 1.0)."""

    pseudo_action_loss_weight: float = 1.0
    """Loss weight applied to pseudo-labelled samples when ``use_has_real_action``
    is on. 1.0 reproduces GR00T-N1 §5's 1:1 co-training; 0.5 halves their pull."""

    drop_unlabeled_action: bool = False
    """If True, pseudo-labelled samples contribute zero action loss (their
    ``action`` field is still required for shape, but is not supervised)."""

    # --- P0-2: history-conditioned flow matching ---
    flow_source_mode: str = "gaussian"
    """Source endpoint of the flow-matching ODE.
      * ``gaussian``: source = N(0, gaussian_source_noise_std² I) (default N(0,I))
      * ``history`` : source = previous action chunk (+ eta noise)
    With ``history``, the model learns the *correction* from the last executed
    chunk instead of the full action, which is what lets 1-2 denoising steps
    work at inference (UniWAM §4.3.2)."""

    gaussian_source_noise_std: float = 1.0
    """Std of the pure Gaussian flow source: s = gaussian_source_noise_std * eps.
    Used by BOTH training and deployment in flow_source_mode='gaussian'.
    1.0 preserves legacy behavior; 0.04 is the G-small noise-budget control."""

    flow_source_action_noise_std: float = 0.0
    """eta: std of the Gaussian perturbation added to the history source.
    UniWAM uses 0.02 on [0,1]-normalized actions; N1.7 normalizes to [-1,1]
    so the equivalent scale is ~0.04. 0.0 = deterministic source."""

    history_action_offset: int = 1
    """How many control steps back the "previous action chunk" is taken from.
    Also used as ``n_action_steps`` when ``history_action_align="rtc"``
    (i.e. set it to the number of steps executed per re-plan)."""

    history_action_align: str = "shift"
    """How the previous chunk is aligned to the current one:
      * ``shift``: ``source[k] = prev[k]``                        (UniWAM form)
      * ``rtc``  : ``source[j] = prev[offset + j]`` for ``j < H - offset``,
                   hold ``prev[-1]`` for the tail. This matches GR00T's RTC
                   in-painting convention (prev[n:] -> current[:H-n])."""

    use_relabeled_history: bool = False
    """E2 rolling-relabel (survey §13.51): extract training-time history windows
    from the episode's ``action_relabeled`` column — the model's OWN closed-loop
    rollout actions written by scripts/build_relabeled_rollout_dataset.py —
    instead of the pristine GT ``action`` column. DAgger-style distribution
    match between the history the model is TRAINED on and the drifted history
    it FEEDS ITSELF at inference. Episodes without the column fall back to GT
    history (legacy datasets unchanged). No-op unless flow_source_mode is
    ``history`` or ``memory_history_mix``."""

    flow_source_noise_selective: bool = False
    """T0-2 (Δ Flow Matching selective noise): mask the eta perturbation on
    history/memory flow sources per action dimension — gripper columns
    (near-discrete 0/1 channels that continuous noise would pollute) and all
    padding stay clean; only continuous dims receive the ΔFM-style source
    noise. The processor emits ``action_noise_mask`` whenever it can resolve
    the per-key column layout; without the mask this flag is a no-op."""

    # --- P2: learnable action memory start (UVM-style z0 anchor) ---
    memory_start_enabled: bool = False
    """If True, build the ActionMemoryEncoder (VL condition -> action-chunk
    prototype z0) and use memory-based flow sources. Requires
    ``flow_source_mode`` = ``memory`` (UVM hard-anchor recipe) or
    ``memory_history_mix`` (soft arm: beta*history + (1-beta)*z0 + eta*eps)."""

    memory_start_anchor_lambda: float = 1.0
    """lambda of the z0 reconstruction anchor ||z0 - a0||^2. 1.0 = UVM original
    hard anchor; 0.05-0.1 = soft anchor (z0 stays a prior, multimodality kept)."""

    memory_start_diff_anchor_lambda: float = 0.0
    """lambda of the db10 differential (shape) anchor ||Δref − Δy||² over
    horizon first-differences (0.0 = off). Constant, NOT annealed — the shape
    pressure must survive the point anchor's cosine decay (survey §13.57)."""

    memory_anchor_lambda_schedule: str = "constant"
    """Anchor-lambda schedule over training: ``constant`` | ``cosine`` | ``piecewise``.
    cosine: λ(p) = λ_final + 0.5·(λ0−λ_final)·(1+cos(π·p)), p=step/max_steps.
    piecewise: λ0 until ``--memory-anchor-lambda-warmup-steps``, then λ_final.
    Typical anneal: λ0=1.0, final=0.1 — hard anchor early for convergence
    signal, soft late to avoid the §13.2-③ variance collapse."""

    memory_anchor_lambda_final: float | None = None
    """Final lambda for non-constant schedules. None = 0.1 × memory_start_anchor_lambda."""

    memory_anchor_lambda_warmup_steps: int = 0
    """Switch point (optimizer steps) of the ``piecewise`` schedule."""

    memory_start_anchor_type: str = "mse"
    """Anchor loss type: ``mse`` = λ‖z0−y‖² per-sample reconstruction (UVM
    original); ``infonce`` = masked-batch InfoNCE (review 2026-10-17, D4-1):
    logits_ij = −d(z_i, y_j)/τ with the diagonal as positives, which directly
    fights the soft-anchor variance collapse (B5) and makes z0 semantically
    discriminative across samples (B2). B==1 falls back to MSE."""

    memory_start_anchor_temperature: float = 0.1
    """Softmax temperature τ of the InfoNCE anchor (only used when
    ``memory_start_anchor_type == 'infonce'``)."""

    memory_start_anchor_target: str = "y"
    """Anchor target for the MSE anchor (review 2026-10-18, L1′ role-consistent
    anchor): ``y`` = UVM original (anchor z0 → y); ``role`` = anchor the mixed
    source s = βh+(1−β)z0 → y, i.e. z0 → (y−βh)/(1−β) in multiplicative form
    (no division, anchor vanishes at β→1 where z0 has no role). Removes the
    anchor-vs-role gradient conflict at segment heads (survey doc §13.24 ⑥).
    Requires ``memory_start_anchor_type == 'mse'``."""

    memory_start_mix_beta: float = 0.0
    """Weight of the aligned history chunk in ``memory_history_mix`` mode.
    0 = pure memory start, 1 = pure history."""

    memory_start_mix_beta_sampling: str = "fixed"
    """Training-time β sampling in ``memory_history_mix`` mode: ``fixed`` uses
    the scalar ``memory_start_mix_beta`` (legacy behavior); ``uniform`` draws
    β ~ U(memory_start_mix_beta_min, memory_start_mix_beta_max) per sample per
    step, teaching the DiT the whole β-simplex so inference-time confidence
    routing (β̂) no longer suffers the fixed-β train/eval distribution shift
    (survey doc §13.15.2, verdict 2)."""

    memory_start_mix_beta_min: float = 0.0
    """Lower bound of the uniform β sampling range (inclusive)."""

    memory_start_mix_beta_max: float = 1.0
    """Upper bound of the uniform β sampling range (inclusive)."""

    memory_start_anchor_beta_max: float = 1.0
    """db12 (survey §13.61): compute the role/diff anchors only on samples with
    drawn β ≤ this value (flow side keeps β ~ U(0,1)). Shrinks the h
    phase-lag contamination of the z0 equilibrium (z0* = 1.5·E[y|c] − 0.5·h).
    1.0 = off (legacy). Requires uniform β sampling to take effect."""

    memory_start_z0_recon_lambda: float = 0.0
    """db12 (survey §13.61): constant weight of the β=0 endpoint debias term
    λ·‖z0 − y‖² (0.0 = off; db12 uses 0.05). Pulls z0 to E[y|c] directly,
    cancelling the role equilibrium's 0.5·(E[y|c] − h) extrapolation bias at
    the β=0 deployment arm."""

    memory_start_noise_std: float = 0.0
    """eta for memory modes. UVM recipe: 0.0 (deterministic start); soft arm:
    ~0.02 to keep the ODE path stochastic."""
    memory_start_noise_sampling: str = "fixed"
    """db23a (survey §13.91): ``fixed`` uses scalar ``memory_start_noise_std``
    (legacy); ``uniform`` draws per-sample eta ~ U(0, memory_start_noise_std)
    per training step (std acts as the max) — isotropic source-spectrum
    diversity around the semantic start. Applies to ``memory_history_mix``
    mode only; eval always uses the scalar eta."""

    memory_start_hidden_dim: int = 1024
    """Hidden width of the ActionMemoryEncoder trunk."""

    memory_num_queries: int = 64
    """K learnable query tokens (UVM: K=64). With ``memory_query_path='backbone'``
    (default) they pass THROUGH the frozen VLM (collator-appended placeholder
    tokens whose embeddings are overwritten); with ``post`` they only attend to
    the frozen VL features inside the trainable vl_self_attention adapter."""

    memory_query_dim: int = 64
    """Per-query projection width in ActionMemoryEncoder."""

    memory_query_path: str = "backbone"
    """``backbone`` (v3, UVM-faithful depth) or ``post`` (v2 shallow adapter).
    ``backbone`` also makes the data collator append K placeholder tokens, so a
    checkpoint trained with it must be evaluated with its own processor config
    (Gr00tPolicy cross-checks model vs processor K)."""

    tune_memory_encoder: bool = True
    """Train the memory encoder even when tune_projector=False."""

    # --- v4.1: confidence query tokens (gap predictor) ---
    confidence_tokens_enabled: bool = False
    """If True, thread C extra learnable *confidence* query tokens through the
    frozen backbone (same placeholder/pre-hook mechanism as the K memory
    queries; appended AFTER them) and read them out with a small MLP into
    (ĝ_mean, ĝ_q90): the predicted masked L1 gap between the one-step endpoint
    ŷ1 = s + v_θ(s, t=0, c) and the ground-truth chunk. Arm-independent: works
    with every flow_source_mode. The training target is detached and costs one
    extra no-grad DiT forward every ``confidence_every_k`` steps."""

    confidence_num_queries: int = 4
    """C confidence query tokens (v4.1). K=64 memory queries carry semantics;
    C=4 suffices for a two-scalar readout."""

    confidence_loss_weight: float = 0.1
    """Weight of the confidence loss: MSE(ĝ_mean, gap) + pinball_0.9(ĝ_q90, gap)
    against the DETACHED endpoint gap. 0 disables the head's gradient (the
    readout still runs at inference)."""

    confidence_every_k: int = 1
    """Compute the (no-grad) endpoint target only every k-th optimizer step to
    amortise its extra DiT forward; skipped steps omit the confidence loss."""

    confidence_routing_enabled: bool = False
    """T1 (mixed-source paper §4.4): 4-output confidence head
    (ĝ_mean, ĝ_q90, ĝ_hist, ĝ_mem) + per-sample inference-time mixture weight
    β̂ = ĝ_mem²/(ĝ_hist² + ĝ_mem² + ε) on the history term. Requires
    confidence_tokens_enabled and flow_source_mode='memory_history_mix'; the
    two extra no-grad DiT forwards are amortised by confidence_every_k."""

    confidence_routing_l1: bool = False
    """Ablation: L1 ratio β̂ = ĝ_mem/(ĝ_hist + ĝ_mem + ε) instead of squared."""

    confidence_routing_mode: str = "gap_ratio"
    """``gap_ratio`` (legacy: 4-output head, β̂ from the squared-gap ratio) |
    ``direct_beta`` (5-output head: the 5th output regresses the closed-form
    optimal per-sample mixture weight β* = <y−z0, h−z0>/‖h−z0‖², and inference
    uses β̂ = ĝ_β directly — removes the segment-insensitive gap-ratio mapping)."""

    # --- Sub-instruction (dual text port) ---
    sub_instruction_mode: str = "off"
    """Dual text port driving the same action head: ``off`` (legacy single
    instruction) | ``concat`` (prompt = "<task>. next step: <sub>") |
    ``replace`` (sub-instruction replaces the task text). With concat/replace
    the dataset demotes the per-frame instruction to ``sub_text`` and promotes
    the episode-level ``high_level_task`` (episodes.jsonl) to ``text``;
    episodes without the field degrade to single-port behavior."""

    sub_instruction_dropout_prob: float = 0.0
    """Training-time CFG-style dropout on the two text ports (only when
    sub_instruction_mode != "off"): p/2 drop the task (sub-only sample), p/2
    drop the sub-instruction (task-only sample), so either port can drive the
    action head alone at inference. 0 = always feed both."""

    # --- P2 transformer memory encoder (alternative trunk) ---
    memory_encoder_arch: str = "mlp"
    """ActionMemoryEncoder trunk: ``mlp`` (UVM-faithful flatten+MLP+per-
    embodiment head, ~179M) | ``transformer`` (query self-attention + step-
    query cross-attention + shared per-step head, ~3.1M; better parameter
    efficiency and cross-horizon generalisation) | ``const`` (db15 last-cut
    ablation: content-free single learnable (H, action_dim) table, no
    conditioning reads — separates the determinism asset from z0's
    conditional content). Default ``mlp`` keeps
    pre-existing checkpoints loadable."""

    memory_encoder_d_model: int = 256
    """d_model of the transformer memory encoder (arch="transformer" only)."""

    memory_encoder_layers: int = 2
    """Self-attention layers over the K memory queries (transformer arch)."""

    memory_encoder_bottleneck_dim: int = 0
    """db13 (paper §6.4 direction 2): hard low-dimensional content bottleneck
    for the transformer memory encoder. When > 0, z₀'s content must pass
    through an L-dim task-parameter latent. NOTE (review P0-1): with the y-MSE
    anchor unchanged this does NOT lower the Var(y|c) floor — it tests whether
    the learnable segment-head content is low-rank. 0 = disabled (legacy).
    Requires arch="transformer"."""

    memory_encoder_var_head: bool = False
    """db17 (paper §8 item 11-①): heteroscedastic variance head on the
    transformer memory encoder (per-dim log σ²), REQUIRED by anchor_type
    "nll"/"nll_infonce" — the explicit source-distribution supervision arm.
    Deployment uses z₀ = μ(c) only (deterministic start preserved).
    Requires arch="transformer"."""

    memory_start_anchor_con_lambda: float = 0.1
    """db17: weight of the InfoNCE sub-term inside anchor_type "nll_infonce"
    (independent CONSTANT lambda, not annealed with the main anchor — mirrors
    the differential anchor's own lambda). Must be >= 0."""

    memory_encoder_read_state: bool = False
    """db18 T2 (paper §8 item 11-⑤): the memory encoder's cross-attention reads
    the flattened proprioceptive state as an extra KV token — z₀ = f(H, q).
    Requires --memory-encoder-arch transformer and --memory-start-enabled."""

    dit_state_token: bool = True
    """db18 T1/T2: --no-dit-state-token cuts the state-conditioning channel
    into the DiT (sa_embs = action tokens only). Incompatible with
    --memory-func-anchor-enabled (its auxiliary forward is not re-routed)."""

    zero_init_dit: bool = False
    """db28 (from-scratch G/Z arms): after loading the base checkpoint, throw
    away the pretrained DiT weights and re-initialise the velocity field from
    scratch (fresh seeded draw + per-block zeroed output projections, the
    DiT/UVM zero-identity contract). The VLM backbone, encoders and decoders
    still come from the checkpoint. G/Z arms at the same --seed share
    bit-identical initial DiT weights."""

    seed: int = 42
    """db28: training seed — lands on config.data.seed (drives set_seed and
    HF TrainingArguments.seed) and, with --zero-init-dit, seeds the DiT
    re-initialisation draw (seed + 1000)."""

    extra_save_steps: str = ""
    """db28 (early-convergence probe): comma-separated extra checkpoint steps
    saved IN ADDITION to the regular --save-steps grid, e.g. "200,500,1000",
    enabling heldout mse_eff/repeat/wall-clock comparison at early
    checkpoints (review 2026-10-25 P0-2). Empty = off. Set
    --save-total-limit generously (e.g. 15) — rotation is mtime-based and
    would otherwise evict early probe checkpoints."""

    # --- L2 functional anchor (survey doc §13.43/§13.44) ---
    memory_func_anchor_enabled: bool = False
    """If True, add the L2 functional anchor: L_func = ‖ŷ1 − y‖² (masked) with
    the one-step endpoint ŷ1 = s + v_θ(s, t=0, c) on the same source s the
    flow loss used. The t=0 branch forward freezes the DiT pathway
    (requires_grad off) and detaches the context tensors, so the gradient
    shapes ONLY z0 / the memory encoder, never the DiT (§13.44 P1-1)."""

    memory_func_anchor_lambda: float = 0.1
    """Constant weight of the functional anchor (no annealing, §13.43 'λ 常量化')."""

    memory_func_anchor_every_k: int = 1
    """Run the anchor every k-th optimizer step to amortise the extra
    grad-carrying t=0 DiT forward (§13.44 P1-3: not zero-cost)."""

    memory_func_anchor_source_mode: str = "pure_z0"
    """Source the functional anchor's endpoint is computed from (§13.48 db9):
    "pure_z0" re-draws z0+ηε — anchors the β=0 deployment operating point
    (default, fixes the db8 train/deploy source-distribution mismatch);
    "mix" anchors the per-step β-sampled mixed source (db8 behavior)."""

    # --- db7 data-version: state modality extension + dataset assertions ---
    state_extra_modality_keys: str = ""
    """Comma-separated extra state modality keys appended (process-local) to
    the embodiment's hardcoded state modality (review P0-1: the pipeline reads
    state keys from MODALITY_CONFIGS, not the dataset modality.json). Use
    ``target_relative_pose`` for db7/A1 21D-state datasets. Empty = stock
    config (real DROID and db6 datasets unaffected)."""

    expected_data_version: str | None = None
    """When set, every dataset's meta/info.json must carry this
    ``data_version`` and meta/data_fingerprint.json must exist and agree with
    info.json (mtime-free identity check, review P1-2)."""

    expected_state_dim: int | None = None
    """When set, the dataset's state slices (meta/modality.json) must be
    contiguous and sum to this width (e.g. 21 for db7/A1)."""

    def __post_init__(self) -> None:
        if self.flow_source_mode not in ("gaussian", "history", "memory", "memory_history_mix"):
            raise ValueError(
                "flow_source_mode must be 'gaussian', 'history', 'memory' or "
                f"'memory_history_mix', got {self.flow_source_mode!r}"
            )
        if self.gaussian_source_noise_std <= 0:
            raise ValueError("gaussian_source_noise_std must be > 0")
        if self.gaussian_source_noise_std != 1.0 and self.flow_source_mode != "gaussian":
            raise ValueError(
                "gaussian_source_noise_std != 1.0 requires flow_source_mode='gaussian'; "
                "use memory_start_noise_std for memory/history-mix source noise"
            )
        if self.memory_start_enabled and self.flow_source_mode not in (
            "memory",
            "memory_history_mix",
        ):
            raise ValueError(
                "memory_start_enabled requires flow_source_mode 'memory' or "
                f"'memory_history_mix', got {self.flow_source_mode!r}"
            )
        if self.flow_source_mode in ("memory", "memory_history_mix") and not (
            self.memory_start_enabled
        ):
            raise ValueError(
                f"flow_source_mode={self.flow_source_mode!r} requires memory_start_enabled=True"
            )
        if self.memory_start_anchor_lambda < 0:
            raise ValueError("memory_start_anchor_lambda must be >= 0")
        if self.memory_start_diff_anchor_lambda < 0:
            raise ValueError("memory_start_diff_anchor_lambda must be >= 0")
        if not 0.0 < self.memory_start_anchor_beta_max <= 1.0:
            raise ValueError(
                "memory_start_anchor_beta_max must be in (0, 1], "
                f"got {self.memory_start_anchor_beta_max!r}"
            )
        if self.memory_start_anchor_beta_max < 1.0 and (
            self.memory_start_mix_beta_sampling != "uniform"
        ):
            raise ValueError(
                "memory_start_anchor_beta_max < 1.0 requires "
                "memory_start_mix_beta_sampling='uniform' (the β gate needs a "
                "per-sample drawn β; fixed-β runs have nothing to gate on)."
            )
        if self.memory_start_z0_recon_lambda < 0:
            raise ValueError("memory_start_z0_recon_lambda must be >= 0")
        if self.memory_anchor_lambda_schedule not in ("constant", "cosine", "piecewise"):
            raise ValueError(
                "memory_anchor_lambda_schedule must be 'constant', 'cosine' or "
                f"'piecewise', got {self.memory_anchor_lambda_schedule!r}"
            )
        if self.memory_anchor_lambda_final is not None and self.memory_anchor_lambda_final < 0:
            raise ValueError("memory_anchor_lambda_final must be >= 0 when set")
        if self.memory_anchor_lambda_warmup_steps < 0:
            raise ValueError("memory_anchor_lambda_warmup_steps must be >= 0")
        if self.memory_start_anchor_type not in ("mse", "infonce", "nll", "nll_infonce"):
            raise ValueError(
                "memory_start_anchor_type must be 'mse', 'infonce', 'nll' or "
                f"'nll_infonce', got {self.memory_start_anchor_type!r}"
            )
        if self.memory_start_anchor_target not in ("y", "role"):
            raise ValueError(
                "memory_start_anchor_target must be 'y' or 'role', "
                f"got {self.memory_start_anchor_target!r}"
            )
        if (
            self.memory_start_anchor_target == "role"
            and self.memory_start_anchor_type != "mse"
        ):
            raise ValueError(
                "memory_start_anchor_target='role' requires "
                "memory_start_anchor_type='mse' (role-consistent anchoring is "
                "a per-sample target; InfoNCE is a batch-contrastive loss)."
            )
        if self.memory_start_anchor_temperature <= 0:
            raise ValueError("memory_start_anchor_temperature must be > 0")
        if self.memory_start_anchor_type in ("nll", "nll_infonce"):
            if not self.memory_encoder_var_head:
                raise ValueError(
                    "anchor_type 'nll'/'nll_infonce' requires "
                    "--memory-encoder-var-head (the NLL anchor reads per-dim "
                    "log σ² from the variance head)."
                )
            if self.memory_encoder_arch != "transformer":
                raise ValueError(
                    "anchor_type 'nll'/'nll_infonce' requires "
                    "memory_encoder_arch='transformer' (the variance head is "
                    "only implemented on the transformer trunk)."
                )
        if self.memory_start_anchor_con_lambda < 0:
            raise ValueError("memory_start_anchor_con_lambda must be >= 0")
        # db18 (paper §8 item 11-⑤): state routing ablation guards
        if self.memory_encoder_read_state:
            if not self.memory_start_enabled:
                raise ValueError(
                    "memory_encoder_read_state requires --memory-start-enabled"
                )
            if self.memory_encoder_arch != "transformer":
                raise ValueError(
                    "memory_encoder_read_state requires "
                    "memory_encoder_arch='transformer' (state KV token is only "
                    "implemented on the transformer trunk)."
                )
        if not self.dit_state_token and self.memory_func_anchor_enabled:
            raise ValueError(
                "--no-dit-state-token is incompatible with "
                "--memory-func-anchor-enabled (the functional anchor's "
                "auxiliary forward is not re-routed; disable one of them)."
            )
        if self.zero_init_dit and not self.tune_diffusion_model:
            raise ValueError(
                "--zero-init-dit requires --tune-diffusion-model (default True): "
                "re-initialising the DiT and then freezing it would deploy a "
                "frozen random velocity field."
            )
        if not 0.0 <= self.memory_start_mix_beta <= 1.0:
            raise ValueError("memory_start_mix_beta must be in [0, 1]")
        if self.memory_start_mix_beta_sampling not in ("fixed", "uniform"):
            raise ValueError(
                "memory_start_mix_beta_sampling must be 'fixed' or 'uniform', "
                f"got {self.memory_start_mix_beta_sampling!r}"
            )
        if not (
            0.0 <= self.memory_start_mix_beta_min <= self.memory_start_mix_beta_max <= 1.0
        ):
            raise ValueError(
                "require 0 <= memory_start_mix_beta_min <= "
                f"memory_start_mix_beta_max <= 1, got [{self.memory_start_mix_beta_min}, "
                f"{self.memory_start_mix_beta_max}]"
            )
        if self.memory_start_noise_std < 0:
            raise ValueError("memory_start_noise_std must be >= 0")
        if self.memory_start_noise_sampling not in ("fixed", "uniform"):
            raise ValueError(
                "memory_start_noise_sampling must be 'fixed' or 'uniform', "
                f"got {self.memory_start_noise_sampling!r}"
            )
        if self.memory_num_queries < 1:
            raise ValueError("memory_num_queries must be >= 1")
        if self.memory_query_dim < 1:
            raise ValueError("memory_query_dim must be >= 1")
        if self.memory_query_path not in ("backbone", "post"):
            raise ValueError(
                f"memory_query_path must be 'backbone' or 'post', got {self.memory_query_path!r}"
            )
        if self.history_action_align not in ("shift", "rtc"):
            raise ValueError(
                f"history_action_align must be 'shift' or 'rtc', got {self.history_action_align!r}"
            )
        if self.flow_source_action_noise_std < 0:
            raise ValueError("flow_source_action_noise_std must be >= 0")
        if self.history_action_offset < 1:
            raise ValueError("history_action_offset must be >= 1")
        if self.use_relabeled_history and self.flow_source_mode not in (
            "history",
            "memory_history_mix",
        ):
            raise ValueError(
                "use_relabeled_history only affects the history-chunk extraction and "
                "therefore requires flow_source_mode 'history' or 'memory_history_mix', "
                f"got {self.flow_source_mode!r}"
            )
        if self.confidence_num_queries < 1:
            raise ValueError("confidence_num_queries must be >= 1")
        if self.confidence_loss_weight < 0:
            raise ValueError("confidence_loss_weight must be >= 0")
        if self.confidence_every_k < 1:
            raise ValueError("confidence_every_k must be >= 1")
        if self.confidence_routing_enabled:
            if not self.confidence_tokens_enabled:
                raise ValueError(
                    "confidence_routing_enabled requires confidence_tokens_enabled=True"
                )
            if self.flow_source_mode != "memory_history_mix":
                raise ValueError(
                    "confidence_routing_enabled routes between history and memory "
                    "sources and therefore requires flow_source_mode='memory_history_mix', "
                    f"got {self.flow_source_mode!r}"
                )
        if self.confidence_routing_mode not in ("gap_ratio", "direct_beta"):
            raise ValueError(
                "confidence_routing_mode must be 'gap_ratio' or 'direct_beta', "
                f"got {self.confidence_routing_mode!r}"
            )
        if self.sub_instruction_mode not in ("off", "concat", "replace"):
            raise ValueError(
                "sub_instruction_mode must be 'off', 'concat' or 'replace', got "
                f"{self.sub_instruction_mode!r}"
            )
        if not 0.0 <= self.sub_instruction_dropout_prob <= 1.0:
            raise ValueError(
                f"sub_instruction_dropout_prob must be in [0, 1], got "
                f"{self.sub_instruction_dropout_prob}"
            )
        if self.memory_encoder_arch not in ("mlp", "transformer", "const"):
            raise ValueError(
                f"memory_encoder_arch must be 'mlp', 'transformer' or 'const', "
                f"got {self.memory_encoder_arch!r}"
            )
        if self.memory_encoder_d_model < 8:
            raise ValueError("memory_encoder_d_model must be >= 8")
        if self.memory_encoder_layers < 1:
            raise ValueError("memory_encoder_layers must be >= 1")
        if self.memory_encoder_bottleneck_dim < 0:
            raise ValueError("memory_encoder_bottleneck_dim must be >= 0")
        if self.memory_encoder_bottleneck_dim > 0 and self.memory_encoder_arch != "transformer":
            raise ValueError(
                "memory_encoder_bottleneck_dim requires "
                "memory_encoder_arch='transformer'"
            )
        if self.memory_func_anchor_lambda < 0:
            raise ValueError("memory_func_anchor_lambda must be >= 0")
        if self.memory_func_anchor_every_k < 1:
            raise ValueError("memory_func_anchor_every_k must be >= 1")
        if self.memory_func_anchor_enabled and not self.memory_start_enabled:
            raise ValueError(
                "memory_func_anchor_enabled requires memory_start_enabled "
                "(the functional anchor shapes the memory start z0)"
            )
        if self.memory_func_anchor_source_mode not in ("mix", "pure_z0"):
            raise ValueError(
                f"memory_func_anchor_source_mode must be 'mix' or 'pure_z0', "
                f"got {self.memory_func_anchor_source_mode!r}"
            )
        if self.expected_state_dim is not None and self.expected_state_dim < 1:
            raise ValueError("expected_state_dim must be >= 1")
        if self.state_extra_modality_keys and self.expected_state_dim is None:
            warnings.warn(
                "state_extra_modality_keys set without --expected-state-dim: "
                "the launch-time dataset width check is skipped (key presence "
                "is still asserted). Pass --expected-state-dim 21 for db7/A1.",
                stacklevel=2,
            )
        if self.gradient_accumulation_steps < 1:
            raise ValueError(
                f"gradient_accumulation_steps must be >= 1, got {self.gradient_accumulation_steps}"
            )
        if self.gradient_accumulation_steps > 1:
            accumulated_batch_size = self.global_batch_size * self.gradient_accumulation_steps
            warnings.warn(
                f"global_batch_size={self.global_batch_size} is pre-accumulation; "
                f"accumulated_batch_size={accumulated_batch_size} "
                f"(× gradient_accumulation_steps={self.gradient_accumulation_steps}).",
                stacklevel=2,
            )
