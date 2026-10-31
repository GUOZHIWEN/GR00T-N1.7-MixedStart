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

# Launch finetuning for N1.7 on "single node".
# This script tries to provide a similar user experience as current OSS.

import json
import os
from pathlib import Path
import sys

import tyro

from gr00t.configs.base_config import get_default_config
from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.experiment.experiment import run


class _Tee:
    """Stream proxy that writes to both the original stream and a log file."""

    def __init__(self, stream, log_file):
        self._stream = stream
        self._log = log_file

    def write(self, data):
        self._stream.write(data)
        self._log.write(data)
        self._log.flush()
        return len(data)

    def flush(self):
        self._stream.flush()
        self._log.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _archive_console_log(output_dir: str) -> None:
    """Tee stdout/stderr into <output_dir>/log.txt (append mode).

    Skipped inside torchrun worker processes (RANK is set) so multi-GPU
    launches only log once, from the launcher process.
    """
    if "RANK" in os.environ:
        return
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(log_dir / "log.txt", "a", buffering=1, encoding="utf-8", errors="replace")
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
    print(f"[launch_finetune] console log archived to {log_dir / 'log.txt'}")


# Make sure the user provided modality config is registered.
def load_modality_config(modality_config_path: str):
    import importlib
    import sys

    path = Path(modality_config_path)
    if path.exists() and path.suffix == ".py":
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
        print(f"Loaded modality config: {path}")
    else:
        raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")


if __name__ == "__main__":
    # Set LOGURU_LEVEL environment variable if not already set (default: INFO)
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"
    # Use tyro for clean CLI
    ft_config = tyro.cli(FinetuneConfig, description=__doc__)
    # Archive the full console output (trainer prints, tqdm, warnings) to
    # <output_dir>/log.txt alongside the checkpoints.
    _archive_console_log(ft_config.output_dir)
    from gr00t.data.embodiment_tags import EmbodimentTag

    ft_config.embodiment_tag = EmbodimentTag.resolve(ft_config.embodiment_tag)
    embodiment_tag = ft_config.embodiment_tag.value

    # all rank workers should register for the modality config
    if ft_config.modality_config_path is not None:
        load_modality_config(ft_config.modality_config_path)

    dataset_paths = [path for path in ft_config.dataset_path.split(os.pathsep) if path]

    # db7 data-version: state modality extension + dataset assertion chain.
    # The pipeline resolves state keys from MODALITY_CONFIGS (not the dataset
    # modality.json — review P0-1), so the extra keys must be appended
    # process-locally BEFORE the config/dataset factory reads them; the
    # dataset-side assertion then fails LOUDLY if the keys are absent there
    # (silent-drop guard). Real-DROID runs leave the flag empty and are
    # completely unaffected.
    from gr00t.configs.data.state_extra_keys import (
        apply_state_extra_keys,
        check_dataset_state_keys,
        parse_extra_keys,
    )

    _extra_state_keys = parse_extra_keys(ft_config.state_extra_modality_keys)
    if _extra_state_keys:
        apply_state_extra_keys(embodiment_tag, _extra_state_keys)
    if _extra_state_keys or ft_config.expected_state_dim is not None or ft_config.expected_data_version is not None:
        for _ds in dataset_paths:
            check_dataset_state_keys(
                _ds,
                required_state_keys=_extra_state_keys,
                expected_state_dim=ft_config.expected_state_dim,
                expected_data_version=ft_config.expected_data_version,
            )

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": dataset_paths,
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment_tag,
                    }
                ],
            }
        }
    )
    config.load_config_path = None

    # overwrite with finetune config supplied by the user
    config.model.tune_llm = ft_config.tune_llm
    config.model.tune_visual = ft_config.tune_visual
    config.model.tune_projector = ft_config.tune_projector
    config.model.tune_diffusion_model = ft_config.tune_diffusion_model
    config.model.tune_vlln = ft_config.tune_vlln
    config.model.state_dropout_prob = ft_config.state_dropout_prob
    config.model.random_rotation_angle = ft_config.random_rotation_angle
    config.model.color_jitter_params = ft_config.color_jitter_params
    config.model.use_percentiles = ft_config.use_percentiles
    if (ft_config.shortest_image_edge is None) != (ft_config.crop_fraction is None):
        raise ValueError("shortest_image_edge and crop_fraction must be set together")
    if ft_config.shortest_image_edge is not None:
        config.model.shortest_image_edge = ft_config.shortest_image_edge
        config.model.crop_fraction = ft_config.crop_fraction
        config.model.image_crop_size = None
        config.model.image_target_size = None
    if ft_config.extra_augmentation_config:
        config.model.extra_augmentation_config = json.loads(ft_config.extra_augmentation_config)
    else:
        config.model.extra_augmentation_config = None

    config.model.load_bf16 = ft_config.load_bf16
    config.model.reproject_vision = False
    config.model.model_name = (
        ft_config.vlm_model_path
        or os.environ.get("VLM_MODEL_PATH")
        or config.model.model_name
    )
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    # P0-1: co-training with pseudo-labelled / action-free data
    config.model.use_has_real_action = ft_config.use_has_real_action
    config.model.pseudo_action_loss_weight = ft_config.pseudo_action_loss_weight
    config.model.drop_unlabeled_action = ft_config.drop_unlabeled_action
    # P0-2: history-conditioned flow matching
    config.model.flow_source_mode = ft_config.flow_source_mode
    config.model.gaussian_source_noise_std = ft_config.gaussian_source_noise_std
    config.model.flow_source_action_noise_std = ft_config.flow_source_action_noise_std
    config.model.history_action_offset = ft_config.history_action_offset
    config.model.history_action_align = ft_config.history_action_align
    # P2: learnable action memory start (UVM-style)
    config.model.memory_start_enabled = ft_config.memory_start_enabled
    config.model.memory_start_anchor_lambda = ft_config.memory_start_anchor_lambda
    config.model.memory_start_diff_anchor_lambda = ft_config.memory_start_diff_anchor_lambda
    config.model.memory_anchor_lambda_schedule = ft_config.memory_anchor_lambda_schedule
    config.model.memory_anchor_lambda_final = ft_config.memory_anchor_lambda_final
    config.model.memory_anchor_lambda_warmup_steps = ft_config.memory_anchor_lambda_warmup_steps
    config.model.memory_start_anchor_type = ft_config.memory_start_anchor_type
    config.model.memory_start_anchor_target = ft_config.memory_start_anchor_target
    config.model.memory_start_anchor_temperature = ft_config.memory_start_anchor_temperature
    config.model.memory_start_mix_beta = ft_config.memory_start_mix_beta
    config.model.memory_start_mix_beta_sampling = ft_config.memory_start_mix_beta_sampling
    config.model.memory_start_mix_beta_min = ft_config.memory_start_mix_beta_min
    config.model.memory_start_mix_beta_max = ft_config.memory_start_mix_beta_max
    # db12 (§13.61): anchor-side β decoupling + β=0 endpoint debias
    config.model.memory_start_anchor_beta_max = ft_config.memory_start_anchor_beta_max
    config.model.memory_start_z0_recon_lambda = ft_config.memory_start_z0_recon_lambda
    config.model.memory_start_noise_std = ft_config.memory_start_noise_std
    config.model.memory_start_noise_sampling = ft_config.memory_start_noise_sampling
    config.model.memory_start_hidden_dim = ft_config.memory_start_hidden_dim
    config.model.memory_num_queries = ft_config.memory_num_queries
    config.model.memory_query_dim = ft_config.memory_query_dim
    config.model.memory_query_path = ft_config.memory_query_path
    config.model.tune_memory_encoder = ft_config.tune_memory_encoder
    # T0-2: ΔFM selective source-noise mask
    config.model.flow_source_noise_selective = ft_config.flow_source_noise_selective
    # v4.1: confidence query tokens (gap predictor)
    config.model.confidence_tokens_enabled = ft_config.confidence_tokens_enabled
    config.model.confidence_num_queries = ft_config.confidence_num_queries
    config.model.confidence_loss_weight = ft_config.confidence_loss_weight
    config.model.confidence_every_k = ft_config.confidence_every_k
    # T1: confidence routing (β̂ from per-source gaps)
    config.model.confidence_routing_enabled = ft_config.confidence_routing_enabled
    config.model.confidence_routing_l1 = ft_config.confidence_routing_l1
    config.model.confidence_routing_mode = ft_config.confidence_routing_mode
    # Sub-instruction (dual text port): model side lands in config.json and
    # feeds the processor overrides in setup.py; data side drives the
    # high-level/sub demotion in ShardedSingleStepDataset.
    config.model.sub_instruction_mode = ft_config.sub_instruction_mode
    config.model.sub_instruction_dropout_prob = ft_config.sub_instruction_dropout_prob
    config.data.sub_instruction_mode = ft_config.sub_instruction_mode
    # P2 transformer memory encoder (alternative trunk)
    config.model.memory_encoder_arch = ft_config.memory_encoder_arch
    config.model.memory_encoder_d_model = ft_config.memory_encoder_d_model
    config.model.memory_encoder_layers = ft_config.memory_encoder_layers
    # db13 (paper §6.4 direction 2): low-dim content bottleneck
    config.model.memory_encoder_bottleneck_dim = ft_config.memory_encoder_bottleneck_dim
    # db17 (paper §8 11-①): explicit source-distribution supervision (NLL var head)
    config.model.memory_encoder_var_head = ft_config.memory_encoder_var_head
    config.model.memory_start_anchor_con_lambda = ft_config.memory_start_anchor_con_lambda
    # db18 (paper §8 11-⑤): state routing ablation (T1/T2)
    config.model.memory_encoder_read_state = ft_config.memory_encoder_read_state
    config.model.dit_state_token = ft_config.dit_state_token
    # db28: from-scratch DiT re-init (G/Z arms)
    config.model.zero_init_dit = ft_config.zero_init_dit
    # L2 functional anchor (§13.43/§13.44)
    config.model.memory_func_anchor_enabled = ft_config.memory_func_anchor_enabled
    config.model.memory_func_anchor_lambda = ft_config.memory_func_anchor_lambda
    config.model.memory_func_anchor_every_k = ft_config.memory_func_anchor_every_k
    config.model.memory_func_anchor_source_mode = ft_config.memory_func_anchor_source_mode

    # P2-1 (review 2026-10-17): propagation round-trip assertion. The
    # direct_beta incident (§13.18-§13.19: flag silently stayed at its default
    # because the upload lagged) showed that a broken link in this pass-through
    # fails silently. Verify every critical flag actually landed.
    for _f in (
        "flow_source_mode",
        "gaussian_source_noise_std",
        "memory_start_enabled",
        "memory_start_mix_beta",
        "memory_start_mix_beta_sampling",
        "memory_start_noise_sampling",
        "memory_start_anchor_type",
        "memory_start_anchor_target",
        "memory_start_diff_anchor_lambda",
        "memory_start_anchor_beta_max",
        "memory_start_z0_recon_lambda",
        "memory_encoder_arch",
        "memory_encoder_bottleneck_dim",
        "memory_encoder_var_head",
        "memory_start_anchor_con_lambda",
        "memory_encoder_read_state",
        "dit_state_token",
        "zero_init_dit",
        "memory_func_anchor_enabled",
        "memory_func_anchor_source_mode",
        "confidence_tokens_enabled",
        "confidence_routing_enabled",
        "confidence_routing_mode",
        "sub_instruction_mode",
    ):
        if not hasattr(config.model, _f):
            raise RuntimeError(
                f"config propagation broken: model config lacks {_f!r} — "
                "add the field to gr00t/configs/model/gr00t_n1d7.py"
            )
        assert getattr(config.model, _f) == getattr(ft_config, _f), (
            f"config propagation mismatch on {_f!r}: "
            f"model={getattr(config.model, _f)!r} vs ft_config={getattr(ft_config, _f)!r}"
        )

    # db7: the appended state keys must reach the data config the factory
    # actually reads (factory.py consumes config.data.modality_configs[tag]).
    _landed = config.data.modality_configs[embodiment_tag]["state"].modality_keys
    for _k in _extra_state_keys:
        assert _k in _landed, (
            f"state modality propagation broken: {_k!r} applied but not in "
            f"config.data.modality_configs[{embodiment_tag!r}]['state'] ({_landed})"
        )

    config.training.experiment_name = ft_config.experiment_name
    config.training.start_from_checkpoint = ft_config.base_model_path
    config.training.optim = ft_config.optim
    config.training.gradient_checkpointing = ft_config.gradient_checkpointing
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate = ft_config.learning_rate
    config.training.lr_scheduler_type = ft_config.lr_scheduler_type
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir = ft_config.output_dir
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.num_gpus = ft_config.num_gpus
    config.training.use_wandb = ft_config.use_wandb
    config.training.max_steps = ft_config.max_steps
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.wandb_project = ft_config.wandb_project

    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch
    config.data.ds_weights_alpha = ft_config.ds_weights_alpha
    # db28: training seed (drives set_seed + TrainingArguments.seed; also seeds
    # the zero_init_dit re-init draw via seed + 1000)
    config.data.seed = ft_config.seed
    # db28: extra early-probe checkpoint steps (early-convergence comparison)
    config.training.extra_save_steps = ft_config.extra_save_steps

    # P0 switches, data side: drive per-sample flag resolution and history-chunk
    # extraction in ShardedSingleStepDataset (see DatasetFactory).
    config.data.use_has_real_action = ft_config.use_has_real_action
    config.data.flow_source_mode = ft_config.flow_source_mode
    config.data.history_action_offset = ft_config.history_action_offset
    config.data.history_action_align = ft_config.history_action_align
    config.data.use_relabeled_history = ft_config.use_relabeled_history
    # Landing assertion (v1.66 propagation discipline, data side): a silently
    # dropped relabeled-history flag would retrain on pristine GT history and
    # invalidate the DAgger comparison without any error (§13.51 E2).
    assert getattr(config.data, "use_relabeled_history", None) == (
        ft_config.use_relabeled_history
    ), (
        "config propagation broken: use_relabeled_history did not land on "
        "config.data — check launch_finetune/factory wiring"
    )

    config.training.save_only_model = ft_config.save_only_model
    config.training.resume_from_checkpoint = ft_config.resume_from_checkpoint
    config.training.skip_weight_loading = ft_config.skip_weight_loading

    run(config)
