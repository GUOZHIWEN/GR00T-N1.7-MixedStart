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

import json
import logging
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

from gr00t.configs.base_config import Config
from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.data.dataset.factory import DatasetFactory
from gr00t.model.base.model_pipeline import ModelPipeline
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
from gr00t.model.registry import register_model
from gr00t.utils.dist_utils import run_or_wait_on_rank0


# Convert tensors to lists for JSON serialization
def convert_tensors_to_lists(obj):
    """Recursively convert tensors to lists in nested dictionaries/lists."""
    if torch.is_tensor(obj) or isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_tensors_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_tensors_to_lists(item) for item in obj]
    else:
        return obj


_MODULE_GROUP_PREFIXES = [
    # (parameter-name prefix, display name) — order matters: first match wins.
    ("backbone.model.visual", "backbone.visual (ViT)"),
    ("backbone.model.language_model", "backbone.language_model (frozen VLM)"),
    ("backbone.lm_head", "backbone.lm_head"),
    ("action_head.model", "action_head.DiT"),
    ("action_head.vl_self_attention", "action_head.vl_self_attention"),
    ("action_head.memory_encoder", "action_head.memory_encoder (z0)"),
    ("action_head.confidence_head", "action_head.confidence_head (gap)"),
    ("action_head.state_encoder", "action_head.state_encoder"),
    ("action_head.action_encoder", "action_head.action_encoder"),
    ("action_head.action_decoder", "action_head.action_decoder"),
    ("action_head.vlln", "action_head.vlln"),
    ("action_head.position_embedding", "action_head.position_embedding"),
]


def _format_module_param_table(model) -> str:
    """Per-module trainable/total parameter table for the training log."""
    groups: dict[str, list[int]] = {}  # display name -> [trainable, total]
    for name, p in model.named_parameters():
        group = None
        for prefix, display in _MODULE_GROUP_PREFIXES:
            if name.startswith(prefix):
                group = display
                break
        if group is None:
            group = name.split(".")[0] + " (other)"
        stats = groups.setdefault(group, [0, 0])
        stats[1] += p.numel()
        if p.requires_grad:
            stats[0] += p.numel()
    lines = []
    for group, (train, total) in groups.items():
        frozen_mark = "" if train else "  [FROZEN]"
        lines.append(f"  {group:44s} {train:>14,} / {total:>14,}{frozen_mark}")
    return "\n".join(lines)


class Gr00tN1d7Pipeline(ModelPipeline):
    model_class = Gr00tN1d7
    processor_class = Gr00tN1d7Processor

    def __init__(self, config: Config, save_cfg_dir: Path):
        super().__init__(config)
        self.save_cfg_dir = save_cfg_dir

        # Build transformers loading kwargs from training config
        transformers_loading_kwargs = {
            "trust_remote_code": self.config.training.transformers_trust_remote_code,
            "local_files_only": self.config.training.transformers_local_files_only,
        }
        if self.model_config.model_revision is not None:
            transformers_loading_kwargs["revision"] = self.model_config.model_revision
        if self.config.training.transformers_cache_dir is not None:
            transformers_loading_kwargs["cache_dir"] = self.config.training.transformers_cache_dir
        if self.config.training.transformers_access_token is not None:
            transformers_loading_kwargs["token"] = self.config.training.transformers_access_token

        self.transformers_loading_kwargs = transformers_loading_kwargs

    @property
    def model_config(self):
        return self.config.model

    def setup(self):
        self.model = self._create_model()
        self.train_dataset, self.eval_dataset = self._create_dataset(self.save_cfg_dir)
        self.data_collator = self._create_collator()

    def _create_model(self):
        """Setup model with proper vocabulary expansion."""
        skip_weight_loading = getattr(self.config.training, "skip_weight_loading", False)
        if self.config.training.start_from_checkpoint is not None and not skip_weight_loading:
            model, loading_info = AutoModel.from_pretrained(
                self.config.training.start_from_checkpoint,
                tune_llm=self.config.model.tune_llm,
                tune_visual=self.config.model.tune_visual,
                tune_projector=self.config.model.tune_projector,
                tune_diffusion_model=self.config.model.tune_diffusion_model,
                tune_vlln=self.config.model.tune_vlln,
                state_dropout_prob=self.config.model.state_dropout_prob,
                backbone_trainable_params_fp32=self.config.model.backbone_trainable_params_fp32,
                load_bf16=self.config.model.load_bf16,
                # P0 switches: land in config.json so the inference path reads
                # the same values the model was trained with.
                use_has_real_action=self.config.model.use_has_real_action,
                pseudo_action_loss_weight=self.config.model.pseudo_action_loss_weight,
                drop_unlabeled_action=self.config.model.drop_unlabeled_action,
                flow_source_mode=self.config.model.flow_source_mode,
                gaussian_source_noise_std=getattr(
                    self.config.model, "gaussian_source_noise_std", 1.0
                ),
                flow_source_action_noise_std=self.config.model.flow_source_action_noise_std,
                history_action_offset=self.config.model.history_action_offset,
                history_action_align=self.config.model.history_action_align,
                memory_start_enabled=self.config.model.memory_start_enabled,
                memory_start_anchor_lambda=self.config.model.memory_start_anchor_lambda,
                memory_anchor_lambda_schedule=getattr(
                    self.config.model, "memory_anchor_lambda_schedule", "constant"
                ),
                memory_anchor_lambda_final=getattr(
                    self.config.model, "memory_anchor_lambda_final", None
                ),
                memory_anchor_lambda_warmup_steps=getattr(
                    self.config.model, "memory_anchor_lambda_warmup_steps", 0
                ),
                memory_start_mix_beta=self.config.model.memory_start_mix_beta,
                memory_start_noise_std=self.config.model.memory_start_noise_std,
                memory_start_noise_sampling=getattr(
                    self.config.model, "memory_start_noise_sampling", "fixed"
                ),
                memory_start_hidden_dim=self.config.model.memory_start_hidden_dim,
                memory_num_queries=getattr(self.config.model, "memory_num_queries", 64),
                memory_query_dim=getattr(self.config.model, "memory_query_dim", 64),
                memory_query_path=getattr(self.config.model, "memory_query_path", "backbone"),
                memory_encoder_arch=getattr(self.config.model, "memory_encoder_arch", "mlp"),
                memory_encoder_d_model=getattr(
                    self.config.model, "memory_encoder_d_model", 256
                ),
                memory_encoder_layers=getattr(self.config.model, "memory_encoder_layers", 2),
                # db13 (paper §6.4 direction 2): low-dim content bottleneck —
                # same kwargs chokepoint discipline as below.
                memory_encoder_bottleneck_dim=getattr(
                    self.config.model, "memory_encoder_bottleneck_dim", 0
                ),
                tune_memory_encoder=self.config.model.tune_memory_encoder,
                # T0-2 / v4.1 switches: land in config.json so the inference
                # path reads the same values the model was trained with.
                flow_source_noise_selective=getattr(
                    self.config.model, "flow_source_noise_selective", False
                ),
                confidence_tokens_enabled=getattr(
                    self.config.model, "confidence_tokens_enabled", False
                ),
                confidence_num_queries=getattr(self.config.model, "confidence_num_queries", 4),
                confidence_loss_weight=getattr(self.config.model, "confidence_loss_weight", 0.1),
                confidence_every_k=getattr(self.config.model, "confidence_every_k", 1),
                confidence_routing_enabled=getattr(
                    self.config.model, "confidence_routing_enabled", False
                ),
                confidence_routing_l1=getattr(self.config.model, "confidence_routing_l1", False),
                confidence_routing_mode=getattr(
                    self.config.model, "confidence_routing_mode", "gap_ratio"
                ),
                # smoke_db3 incident (2026-10-17): THIS kwarg list is the real
                # chokepoint — a field missing here silently keeps the
                # checkpoint's config.json value (or the dataclass default),
                # and the launch_finetune round-trip assertion cannot see it
                # (it checks one hop upstream). Keep in sync with
                # finetune_config.py whenever a new flag is added.
                memory_start_anchor_type=getattr(
                    self.config.model, "memory_start_anchor_type", "mse"
                ),
                memory_start_anchor_target=getattr(
                    self.config.model, "memory_start_anchor_target", "y"
                ),
                memory_start_anchor_temperature=getattr(
                    self.config.model, "memory_start_anchor_temperature", 0.1
                ),
                # db10 (survey §13.57): differential (shape) anchor lambda —
                # must ride this kwargs list or from_pretrained silently
                # keeps the base checkpoint's value (smoke_db3 chokepoint).
                memory_start_diff_anchor_lambda=getattr(
                    self.config.model, "memory_start_diff_anchor_lambda", 0.0
                ),
                memory_start_mix_beta_sampling=getattr(
                    self.config.model, "memory_start_mix_beta_sampling", "fixed"
                ),
                memory_start_mix_beta_min=getattr(
                    self.config.model, "memory_start_mix_beta_min", 0.0
                ),
                memory_start_mix_beta_max=getattr(
                    self.config.model, "memory_start_mix_beta_max", 1.0
                ),
                # db12 (§13.61): anchor β gate + β=0 debias — same kwargs
                # chokepoint as memory_start_diff_anchor_lambda above; must
                # ride this list or from_pretrained keeps dataclass defaults.
                memory_start_anchor_beta_max=getattr(
                    self.config.model, "memory_start_anchor_beta_max", 1.0
                ),
                memory_start_z0_recon_lambda=getattr(
                    self.config.model, "memory_start_z0_recon_lambda", 0.0
                ),
                # Provenance: the processor consumes these (see below), but
                # landing them on the model config keeps config.json
                # self-describing.
                sub_instruction_mode=getattr(
                    self.config.model, "sub_instruction_mode", "off"
                ),
                sub_instruction_dropout_prob=getattr(
                    self.config.model, "sub_instruction_dropout_prob", 0.0
                ),
                # L2 functional anchor (§13.47) — smoke_db8_func incident #3
                # (2026-10-18): these three were passed launch_finetune →
                # config.model (hop 1, asserted) but MISSING here, so
                # from_pretrained rebuilt the config from the base checkpoint's
                # config.json with dataclass defaults and the anchor silently
                # stayed off for the whole 200-step smoke (twice). The hop-2
                # landing assertion below now covers them.
                memory_func_anchor_enabled=getattr(
                    self.config.model, "memory_func_anchor_enabled", False
                ),
                memory_func_anchor_lambda=getattr(
                    self.config.model, "memory_func_anchor_lambda", 0.1
                ),
                memory_func_anchor_every_k=getattr(
                    self.config.model, "memory_func_anchor_every_k", 1
                ),
                memory_func_anchor_source_mode=getattr(
                    self.config.model, "memory_func_anchor_source_mode", "pure_z0"
                ),
                # db17 (paper §8 11-①): NLL var head + InfoNCE sub-term weight —
                # same kwargs chokepoint discipline as every flag above.
                memory_encoder_var_head=getattr(
                    self.config.model, "memory_encoder_var_head", False
                ),
                memory_start_anchor_con_lambda=getattr(
                    self.config.model, "memory_start_anchor_con_lambda", 0.1
                ),
                # db18 (paper §8 11-⑤): state routing ablation (T1/T2).
                memory_encoder_read_state=getattr(
                    self.config.model, "memory_encoder_read_state", False
                ),
                dit_state_token=getattr(
                    self.config.model, "dit_state_token", True
                ),
                # db28: from-scratch DiT re-init (G/Z arms) — rides the same
                # kwargs chokepoint; the actual re-init runs AFTER loading
                # below, or the checkpoint weights would overwrite it.
                zero_init_dit=getattr(
                    self.config.model, "zero_init_dit", False
                ),
                transformers_loading_kwargs=self.transformers_loading_kwargs,
                output_loading_info=True,
                **self.transformers_loading_kwargs,
            )

            # P2-1b (smoke_db3 / direct_beta incidents): verify the switches
            # above actually LANDED on the instantiated model config. HF
            # from_pretrained setattr's kwargs onto the config only when the
            # kwarg is present in the call above — a missing entry fails
            # silently, one hop downstream of the launch_finetune assertion.
            for _f in (
                "flow_source_mode",
                "gaussian_source_noise_std",
                "memory_start_enabled",
                "memory_start_anchor_lambda",
                "memory_start_anchor_type",
                "memory_start_anchor_target",
                "memory_start_diff_anchor_lambda",
                "memory_start_mix_beta",
                "memory_start_mix_beta_sampling",
                "memory_start_noise_std",
                "memory_start_noise_sampling",
                "memory_encoder_arch",
                "memory_encoder_bottleneck_dim",
                "confidence_tokens_enabled",
                "confidence_routing_enabled",
                "confidence_routing_mode",
                "history_action_offset",
                "history_action_align",
                "sub_instruction_mode",
                "memory_func_anchor_enabled",
                "memory_func_anchor_lambda",
                "memory_func_anchor_every_k",
                "memory_func_anchor_source_mode",
                "memory_encoder_var_head",
                "memory_start_anchor_con_lambda",
                "memory_encoder_read_state",
                "dit_state_token",
                "zero_init_dit",
                "memory_start_anchor_beta_max",
                "memory_start_z0_recon_lambda",
            ):
                _want = getattr(self.config.model, _f, None)
                _got = getattr(model.config, _f, None)
                assert _got == _want, (
                    f"from_pretrained landing mismatch on {_f!r}: "
                    f"model.config={_got!r} vs ExperimentConfig.model={_want!r} — "
                    "add the missing kwarg to the from_pretrained call in "
                    "gr00t/model/gr00t_n1d7/setup.py"
                )

            missing_keys = loading_info.get("missing_keys", [])
            mask_token_missing = any("mask_token" in key for key in missing_keys)
            if mask_token_missing and model.action_head.mask_token is not None:
                with torch.no_grad():
                    model.action_head.mask_token.data.copy_(
                        0.02 * torch.randn_like(model.action_head.mask_token)
                    )
                logging.info("mask_token not in checkpoint - initialized")

            unexpected_keys = loading_info.get("unexpected_keys", [])
            mismatched_keys = loading_info.get("mismatched_keys", [])
            # P2: the memory encoder is absent from pre-P2 checkpoints (missing
            # when enabled) and absent from the model when disabled (unexpected
            # when loading a memory-trained ckpt with the arm switched off).
            # Both directions are fine: the head is zero-init / unused resp.
            memory_enabled = bool(getattr(self.config.model, "memory_start_enabled", False))
            mem_reinit = False
            if memory_enabled:
                # Missing: the checkpoint predates the memory encoder (or was
                # trained with the other ``memory_encoder_arch``). Mismatched:
                # shared-name keys whose shapes differ between the mlp and
                # transformer trunks. Either way re-initialise with real
                # tensors (meta-device instantiation leaves UNINITIALISED
                # GARBAGE ~1e31 -> z0 explodes, flow source/velocity hit inf,
                # loss prints 0.0 with NaN/inf grads — the smoke_mem_hard bug).
                mem_bad = [
                    k
                    for k in (list(missing_keys) + list(mismatched_keys))
                    if "memory_encoder" in k
                ]
                if mem_bad:
                    model.action_head.memory_encoder.reset_parameters()
                    # Zero-init contract: the output head must be exactly zero
                    # after reset (CategorySpecificLinear exposes .W, the
                    # transformer trunk's shared Linear exposes .weight).
                    # db15 const arch: no separate head module — the table
                    # itself is the zero-init output.
                    enc = model.action_head.memory_encoder
                    head = getattr(enc, "head", None)
                    if head is None:
                        w = enc.table
                    else:
                        w = getattr(head, "W", None)
                        if w is None:
                            w = head.weight
                    assert torch.isfinite(w.float()).all() and (w == 0).all(), (
                        "memory_encoder head must be zero-initialised after reset"
                    )
                    logging.info(
                        "memory_encoder missing/mismatched in checkpoint - zero-init fresh (%d keys)",
                        len(mem_bad),
                    )
                    mem_reinit = True
                # The re-init covers every memory_encoder tensor; drop its
                # keys from the strict-mismatch report.
                mismatched_keys = [k for k in mismatched_keys if "memory_encoder" not in k]
            # v4.1: identical story for the confidence gap head (bare Parameter
            # ``queries`` + zero-init readout; no CategorySpecificLinear here).
            confidence_enabled = bool(
                getattr(self.config.model, "confidence_tokens_enabled", False)
            )
            if confidence_enabled:
                conf_bad = [
                    k
                    for k in (list(missing_keys) + list(mismatched_keys))
                    if "confidence_head" in k
                ]
                if conf_bad:
                    # Missing: the checkpoint predates the head. Mismatched: a
                    # routing-enabled 4-output head loading a legacy 2-output
                    # readout. Either way re-initialise with real tensors
                    # (meta-device instantiation leaves garbage otherwise).
                    model.action_head.confidence_head.reset_parameters()
                    q = model.action_head.confidence_head.queries
                    assert torch.isfinite(q.float()).all(), (
                        "confidence_head queries must be finite after reset"
                    )
                    logging.info(
                        "confidence_head missing/mismatched in checkpoint - zero-init fresh (%d keys)",
                        len(conf_bad),
                    )
            other_missing = [
                k
                for k in missing_keys
                if "mask_token" not in k
                and not (memory_enabled and "memory_encoder" in k)
                and not (confidence_enabled and "confidence_head" in k)
            ]
            if not memory_enabled or mem_reinit:
                # Arm off, or arch-switched/fresh encoder re-initialised above:
                # leftover other-arch memory_encoder keys in the checkpoint are
                # expected and must not fail the strict load.
                unexpected_keys = [k for k in unexpected_keys if "memory_encoder" not in k]
            if not confidence_enabled:
                unexpected_keys = [k for k in unexpected_keys if "confidence_head" not in k]
            else:
                # Routing-enabled head (4 outputs) loading a legacy 2-output
                # checkpoint: the readout mismatch was re-initialised above.
                mismatched_keys = [k for k in mismatched_keys if "confidence_head" not in k]
            errors = []
            if other_missing:
                errors.append(f"Missing keys ({len(other_missing)}): {other_missing}")
            if unexpected_keys:
                errors.append(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys}")
            if mismatched_keys:
                errors.append(f"Mismatched keys ({len(mismatched_keys)}): {mismatched_keys}")
            if errors:
                raise RuntimeError(
                    "Checkpoint weight mismatch for "
                    f"{self.config.training.start_from_checkpoint}:\n" + "\n".join(errors)
                )

        else:
            model = self.model_class(
                self.config.model,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
            )

        # db28 (from-scratch G/Z arms): AFTER the checkpoint is fully loaded
        # and validated, re-initialise the DiT under the zero-identity
        # contract (fresh random draw + per-block zeroed attn/FFN output
        # projections). Seed = data seed + 1000, so the G and Z arms at the
        # same --seed start from bit-identical DiT weights. Runs before the
        # load_bf16 cast so the re-init happens in fp32.
        if getattr(model.config, "zero_init_dit", False):
            _dit_seed = int(getattr(self.config.data, "seed", 42)) + 1000
            model.action_head.zero_reinit_dit(_dit_seed)

        if self.config.model.load_bf16:
            # Extend load_bf16 from backbone-only to the FULL model: the action
            # head is otherwise always fp32, and even DiT-only finetuning in
            # fp32 master weights does not fit on ~24GB GPUs
            # (3.1B fp32 params + fp32 grads + AdamW states ≈ 26GB).
            # NOTE: with tune_llm/tune_visual enabled this would also override
            # the backbone's trainable_params_fp32 restoration — keep the
            # backbone frozen when training in full bf16.
            model = model.to(torch.bfloat16)
            logging.info(
                "load_bf16=True: cast FULL model (backbone + action head) to bfloat16; "
                "gradients and AdamW states will also be bf16"
            )

        logging.debug(f"Model Config: {model.config}")
        with run_or_wait_on_rank0(label="final_model_config.json write") as is_rank0:
            if is_rank0:
                with open(self.save_cfg_dir / "final_model_config.json", "w") as f:
                    f.write(model.config.to_filtered_json())
        # Print parameter statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(
            f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)"
        )
        # Per-module breakdown: makes freeze-flag mistakes (e.g. an accidentally
        # trainable backbone on a 24GB GPU, or a frozen memory encoder) visible
        # at a glance in the training log.
        logging.info("Per-module parameter breakdown (trainable / total):\n%s", _format_module_param_table(model))
        logging.debug(f"Model architecture: {model}")

        return model

    def _get_statistics(
        self,
    ) -> dict[str, dict[str, dict[str, dict[str, list[float]]]]] | None:
        return None

    def _get_embodiment_id_mapping(self) -> dict[str, int]:
        return None

    def _create_dataset(self, save_cfg_dir: Path):
        """Create appropriate dataset based on task and mode."""
        letter_box_transform = self.model_config.letter_box_transform
        logging.info("N1.7 letter_box_transform=%s", letter_box_transform)
        # P2 v3 + v4.1: placeholder tokens appended by the collator =
        # K memory queries (only when they pass THROUGH the backbone) + C
        # confidence queries (arm-independent). The backbone overwrites the
        # last-(K+C) embeddings; the action head splits them back out.
        collator_num_queries = 0
        if getattr(self.model_config, "memory_start_enabled", False) and getattr(
            self.model_config, "memory_query_path", "backbone"
        ) == "backbone":
            collator_num_queries += self.model_config.memory_num_queries
        if getattr(self.model_config, "confidence_tokens_enabled", False):
            collator_num_queries += int(getattr(self.model_config, "confidence_num_queries", 4))
        if self.config.training.start_from_checkpoint is not None:
            processor = AutoProcessor.from_pretrained(
                self.config.training.start_from_checkpoint,
                # Overrides
                modality_configs=self.config.data.modality_configs,
                use_percentiles=self.model_config.use_percentiles,
                image_crop_size=self.model_config.image_crop_size,
                image_target_size=self.model_config.image_target_size,
                random_rotation_angle=self.model_config.random_rotation_angle,
                color_jitter_params=self.model_config.color_jitter_params,
                model_name=self.model_config.model_name,
                model_type=self.model_config.backbone_model_type,
                formalize_language=self.model_config.formalize_language,
                apply_sincos_state_encoding=self.model_config.apply_sincos_state_encoding,
                max_action_horizon=self.model_config.action_horizon,
                use_albumentations=self.model_config.use_albumentations_transforms,
                extra_augmentation_config=self.model_config.extra_augmentation_config,
                shortest_image_edge=self.model_config.shortest_image_edge,
                crop_fraction=self.model_config.crop_fraction,
                letter_box_transform=letter_box_transform,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
                use_alternate_vl_dit=self.model_config.use_alternate_vl_dit,
                use_relative_action=self.model_config.use_relative_action,
                # State augmentation overrides
                exclude_state=self.model_config.exclude_state,
                state_dropout_prob=self.model_config.state_dropout_prob,
                use_mean_std=self.model_config.use_mean_std,
                # P2 v3 collator placeholder tokens
                memory_num_queries=collator_num_queries,
                # Sub-instruction (dual text port): must reach the processor,
                # otherwise the dataset demotes text->sub_text while the
                # processor stays in "off" mode and silently drops sub_text.
                sub_instruction_mode=getattr(
                    self.model_config, "sub_instruction_mode", "off"
                ),
                sub_instruction_dropout_prob=getattr(
                    self.model_config, "sub_instruction_dropout_prob", 0.0
                ),
                **self.transformers_loading_kwargs,
            )
        else:
            processor = self.processor_class(
                modality_configs=self.config.data.modality_configs,
                use_percentiles=self.model_config.use_percentiles,
                statistics=self._get_statistics(),  # By default is None, so this will be computed and set later.
                embodiment_id_mapping=self._get_embodiment_id_mapping(),  # By default is None, so this will be set later.
                image_crop_size=self.model_config.image_crop_size,
                image_target_size=self.model_config.image_target_size,
                random_rotation_angle=self.model_config.random_rotation_angle,
                color_jitter_params=self.model_config.color_jitter_params,
                model_name=self.model_config.model_name,
                model_type=self.model_config.backbone_model_type,
                formalize_language=self.model_config.formalize_language,
                max_state_dim=self.model_config.max_state_dim,
                max_action_dim=self.model_config.max_action_dim,
                apply_sincos_state_encoding=self.model_config.apply_sincos_state_encoding,
                max_action_horizon=self.model_config.action_horizon,
                use_albumentations=self.model_config.use_albumentations_transforms,
                extra_augmentation_config=self.model_config.extra_augmentation_config,
                shortest_image_edge=self.model_config.shortest_image_edge,
                crop_fraction=self.model_config.crop_fraction,
                letter_box_transform=letter_box_transform,
                use_relative_action=self.model_config.use_relative_action,
                # State augmentation
                exclude_state=self.model_config.exclude_state,
                state_dropout_prob=self.model_config.state_dropout_prob,
                use_mean_std=self.model_config.use_mean_std,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
                # P2 v3 collator placeholder tokens
                memory_num_queries=collator_num_queries,
                # Sub-instruction (dual text port), same rationale as above.
                sub_instruction_mode=getattr(
                    self.model_config, "sub_instruction_mode", "off"
                ),
                sub_instruction_dropout_prob=getattr(
                    self.model_config, "sub_instruction_dropout_prob", 0.0
                ),
            )

        logging.debug(
            f"Processor configs for training: {json.dumps({k: str(v) for k, v in vars(processor).items()}, indent=2)}"
        )
        with run_or_wait_on_rank0(label="final_processor_config.json write") as is_rank0:
            if is_rank0:
                with open(self.save_cfg_dir / "final_processor_config.json", "w") as f:
                    json.dump({k: str(v) for k, v in vars(processor).items()}, f, indent=2)

        self.processor = processor
        dataset_factory = DatasetFactory(config=self.config)
        train_dataset, eval_dataset = dataset_factory.build(processor=self.processor)

        with run_or_wait_on_rank0(label="dataset_statistics.json write") as is_rank0:
            if is_rank0:
                stats = train_dataset.get_dataset_statistics()
                stats_dict = convert_tensors_to_lists(stats)
                with open(save_cfg_dir / "dataset_statistics.json", "w") as f:
                    json.dump(stats_dict, f, indent=2)
                logging.info("Saved dataset statistics for inference")

        return train_dataset, eval_dataset

    def _create_collator(self):
        data_collator = self.processor.collator
        return data_collator


register_model(Gr00tN1d7Config, Gr00tN1d7Pipeline)
