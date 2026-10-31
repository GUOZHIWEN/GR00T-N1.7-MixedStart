# MoS reproduction guide

This is the public entry point for reproducing the released models, paper tables, and statistical figures. Historical experiment commands are retained only in [`history/EXPERIMENTS.md`](history/EXPERIMENTS.md); they are not the current recipe.

## 1. Scope and evidence lines

All quantitative results below are **open-loop replay** results, not closed-loop task success or real-robot performance.

| Evidence line | Action DiT initialization | Training data | Training seeds | Output |
|---|---|---|---|---|
| Official release pair | Pretrained GR00T-N1.7 Action DiT | 100 synthetic + 300 DROID episodes | 2 | `mos_best`, `mos_best_no_mem` |
| Paper main comparison | Zero-initialized Action DiT | Same 400 episodes | 0, 1, 2 | `runs/mos_paper_main.*` |
| Paper efficiency curves | Zero-initialized Action DiT | Same 400 episodes | 2, 3 | Early-checkpoint deployment metrics |

The semantic arm uses an MLP semantic-start encoder, strict past-only shift/1 history, `β ~ U(0, 0.3)`, `η = 0.04`, and the mixed-source consistency loss (`λ = 0.1`, cosine schedule). Its Gaussian counterpart has no semantic encoder or history source and uses unit-standard-deviation Gaussian noise. Both arms otherwise share the training budget. Do not compare numbers across the two DiT-initialization settings as if they were the same experiment.

A single RTX 4090 with 24 GB was used for the recorded GPU experiments. Runtime depends on storage, video decoding, and hardware; step-to-threshold efficiency is not wall-clock acceleration.

## 2. Environment and immutable inputs

Run commands from the repository root. Linux and a CUDA-capable GPU are required for training and the full evaluation suite.

```bash
pip install uv
uv sync --locked
uv pip install modelscope

# Record the exact source revision used for the run.
git rev-parse HEAD
uv run python -c "import gr00t.configs.model.gr00t_n1d7 as m; print(m.__file__)"
```

The second command must point inside this checkout, not an unrelated installed copy. Download the upstream base weights separately; the MoS release does not redistribute them.

```bash
modelscope download --model nvidia/GR00T-N1.7-3B --local_dir GR00T-N1.7-3B
# Alternatively, use an existing local GR00T-N1.7-3B directory.
export BASE_MODEL_PATH="$PWD/GR00T-N1.7-3B"
# Optional: use an existing local VLM instead of the default hub model.
export VLM_MODEL_PATH=/path/to/Cosmos-Reason2-2B
export CUDA_VISIBLE_DEVICES=0
```

If this base repository is unavailable through ModelScope in your region, download `nvidia/GR00T-N1.7-3B` with Hugging Face tooling instead. Use the same base revision throughout a controlled comparison.

## 3. Data preparation and verification

### 3.1 Pinned public artifacts

```bash
uv run python scripts/prepare_mos_data.py --output-dir data
```

This command downloads all four archives from the single pinned dataset revision recorded in [`DATA_MANIFEST.json`](DATA_MANIFEST.json), verifies archive SHA256 and size, rejects archive links and path traversal, and validates episode/frame counts and state/action dimensions. It never substitutes local staging files for missing public artifacts.

Ordinary users only need the command above. Maintainers changing the public dataset must also regenerate the Parquet preview when staged archives are present, then publish with `MODELSCOPE_API_TOKEN` set. The publisher uploads the declared changed files and preview, resolves the new immutable commit, downloads and verifies all four archives and the preview, and updates the manifest and sample provenance. Staging directories are removed only when `--remove-staged` is explicitly supplied after successful verification. Do not bypass a failed hash check or replace an input with a different dataset.

```bash
# Maintainer-only dataset publication; not required for reproduction.
uv pip install -U modelscope
read -rsp "ModelScope SDK token: " MODELSCOPE_API_TOKEN; echo
export MODELSCOPE_API_TOKEN
uv run python scripts/build_mos_dataset_preview.py
uv run python scripts/publish_mos_datasets.py --remove-staged
unset MODELSCOPE_API_TOKEN

# Anonymous fixed-revision download and full archive/data validation.
uv run python scripts/prepare_mos_data.py --output-dir data
```

Expected datasets:

| Directory | Purpose | Episodes | State/action |
|---|---|---:|---|
| `data/mix_droid_synth_train` | Joint training | 400 | 17D / 17D |
| `data/mix_droid_synth_eval` | Mixed-domain held-out data | 40 | 17D / 17D |
| `data/synth_mixture_eval_db7_17d` | Synthetic evaluation | 10 | 17D / 17D |
| `data/droid_subset_eval` | Independent DROID evaluation | 30 | 17D / 17D |

The 17D layout is EEF 9D + gripper 1D + joint 7D. Camera keys are `exterior_image_1_left` and `wrist_image_left`; the action horizon is 40.

### 3.2 Rebuild instead of downloading

To rebuild the public mixture from source, use the fixed DROID source revision `b697b419ca258bf377cbc0c3279e5348fc5c7c6f`. This path downloads substantially more raw data than the pinned archives.

```bash
uv pip install jsonlines
# ffmpeg must be installed and available on PATH.
uv run python scripts/download_droid_subset.py \
  --num-episodes 300 --num-eval-episodes 30 \
  --cache-dir /path/to/droid_cache \
  --output-dir data/droid_subset_train \
  --eval-output-dir data/droid_subset_eval

uv run python scripts/build_synthetic_mixture_dataset.py \
  --output-dir data/synth_mixture_train_db7 --num-episodes 100 \
  --eval-output-dir data/synth_mixture_eval_db7 --num-eval-episodes 10 \
  --segments-per-episode 4 --steps-per-segment 50 --teleport-prob 0.5 \
  --seed 0 --high-level-task-pool --hl-conditional \
  --relative-pose-state --phase-in-instruction \
  --randomize-zones --coord-in-text --data-version db7

uv run python scripts/merge_lerobot_datasets.py \
  --inputs data/droid_subset_train data/synth_mixture_train_db7 \
  --drop-state-keys target_relative_pose --output-dir data/mix_droid_synth_train
uv run python scripts/merge_lerobot_datasets.py \
  --inputs data/droid_subset_eval data/synth_mixture_eval_db7 \
  --drop-state-keys target_relative_pose --output-dir data/mix_droid_synth_eval
uv run python scripts/merge_lerobot_datasets.py \
  --inputs data/synth_mixture_eval_db7 --drop-state-keys target_relative_pose \
  --output-dir data/synth_mixture_eval_db7_17d
```

Do not pass `target_relative_pose` as an extra state channel to the final 17D models. See [`DATASET_CARD.md`](DATASET_CARD.md) for provenance, held-out boundaries, and limitations.

## 4. Official release pair: pretrained-DiT fine-tuning

Always run the smoke pair before the long run. Training scripts refuse to overwrite non-empty output directories and validate the saved checkpoint configuration after training.

```bash
bash scripts/train_mos_release.sh both smoke 2
# Inspect both smoke logs for finite loss and successful checkpoint-200 writes.

bash scripts/train_mos_release.sh both final 2
bash scripts/eval_mos_release.sh all 2
```

The full evaluation produces early-checkpoint curves and final comparisons. Synthetic final evaluation uses seeds 0–3; DROID uses seed 0. Each result has 10 trajectories and two repeat passes. Repeat controls are built into `compare_flow_sources.py`; there is no `--repeats` option.

To aggregate again without rerunning inference:

```bash
uv run python scripts/aggregate_mos_release.py \
  --run-dir runs --run-suffix _v2 --training-seed 2 \
  --out-prefix runs/mos_release_comparison_s2
```

Outputs: `runs/mos_release_comparison_s2.{json,png,csv}`. The JSON records model identity, evaluation protocol, metric means, paired ratios, and trajectory-cluster bootstrap confidence intervals.

Expected release-pair reference values, regenerated from the same protocol rather than copied from historical arms:

| Domain | Metric ↓ | Semantic | Gaussian | Gaussian/semantic paired ratio [95% CI] |
|---|---|---:|---:|---:|
| Synthetic | Effective action MSE | 0.0471 | 0.0560 | 1.33 [1.04, 1.60] |
| Synthetic | Repeat divergence | 0.0220 | 0.0644 | 3.00 [2.71, 3.31] |
| DROID | Effective action MSE | 0.0174 | 0.0165 | 0.97 [0.81, 1.14], not significant |
| DROID | Repeat divergence | 0.0206 | 0.0432 | 2.10 [1.90, 2.27] |

Small numerical differences can arise from software and hardware. Investigate material differences by comparing the per-trajectory source JSON files, not by changing the estimator.

### 4.1 Export, checksum, and actual-load acceptance

```bash
# Use an empty output directory; export intentionally refuses overwrites.
uv run python scripts/export_mos_release.py \
  --run-dir runs --run-suffix _v2 --output-dir release_reproduced

# Structural validation plus generation and verification of SHA256SUMS.
uv run python scripts/validate_mos_release.py \
  --release-dir release_reproduced --write-sums \
  --out-json runs/release_structure_validation.json

# Separate GPU acceptance step with the real weights.
uv run python scripts/validate_mos_release.py \
  --release-dir release_reproduced --load --device cuda \
  --out-json runs/release_real_load_validation.json
```

Structural checks or tiny fake tensors do not establish that the public weights load. A successful release requires the second command and a real-NPZ single-observation inference run; see [`INFERENCE.md`](INFERENCE.md).

### 4.2 Optional: reaggregate historical release outputs

Some historical JSON files name `data/synth_mixture_eval_db7`, a 22D-state container, although the evaluated model read only the 17D channels. New evaluations must always use `data/synth_mixture_eval_db7_17d`. Do not edit historical JSON paths and do not enable a blanket dataset alias.

To audit an archived result set, first verify that all 10 episodes have identical model-visible inputs: the 17D state prefix, complete actions, task indices, both video files, and task/segment metadata. The evaluation uses checkpoint statistics, not the dataset's recomputed `meta/stats.json`.

```bash
uv run python scripts/verify_legacy_synth17d.py \
  --legacy data/synth_mixture_eval_db7 \
  --canonical data/synth_mixture_eval_db7_17d \
  --out-json runs/synth17d_equivalence.json

uv run python scripts/aggregate_mos_release.py \
  --run-dir runs --run-suffix _v2 --training-seed 2 \
  --legacy-synth-equivalence runs/synth17d_equivalence.json \
  --out-prefix runs/mos_release_reaggregated_s2
```

For the archived G-small result directory layout, additionally pass `--legacy-run-layout` to `aggregate_gsmall_noise_budget.py`. Both compatibility choices are recorded in the output JSON; neither is required for new experiments.

## 5. Paper main comparison: from-scratch Action DiT

### 5.1 Final checkpoints and paired final evaluation

Run the following for each training seed `0`, `1`, and `2`. The script uses `--zero-init-dit`, retains the frozen VLM, and saves checkpoints at 200/500/1000/2500/5000 steps.

```bash
for SEED in 0 1 2; do
  bash scripts/train_mos_from_scratch.sh both final "$SEED"
  bash scripts/eval_mos_from_scratch.sh final "$SEED"
done
```

The run names are `db28g_s{seed}` and `db28z_s{seed}`. They are compatibility identifiers for canonical result paths, not different model variants. Start with `both smoke 2` when validating a new environment.

### 5.2 Early-checkpoint efficiency

Efficiency is measured with held-out effective action MSE and repeat divergence, not training loss. Complete seed 3 and run early evaluation for both efficiency seeds:

```bash
bash scripts/train_mos_from_scratch.sh both final 3
bash scripts/eval_mos_from_scratch.sh early 2
bash scripts/eval_mos_from_scratch.sh early 3
```

The seed-3 semantic checkpoint is named `db28z_s3_soft` to preserve the published result layout; it uses the same soft mixed-source consistency recipe as the other semantic arms.

### 5.3 Main aggregation

```bash
uv run python scripts/aggregate_mos_from_scratch.py \
  --run-dir runs --training-seeds 0 1 2 --efficiency-seeds 2 3 \
  --n-boot 20000 --out-prefix runs/mos_paper_main
```

Outputs:

- `runs/mos_paper_main.json`: final metrics, paired ratios, early curves, threshold-crossing steps, and available training runtimes.
- `runs/mos_paper_main.png`: final paired ratios.
- `runs/mos_paper_main_efficiency.png`: deployment metrics at all five checkpoints.

Paired ratios are `mean(Gaussian_i / Semantic_i)`, not ratios of aggregate means; values above 1 favor the semantic source. Final intervals resample whole `(training_seed, trajectory)` clusters. A confidence interval containing 1 does not establish a significant accuracy improvement.

## 6. Noise-budget control: G vs G-small vs Z

G-small removes the semantic encoder and trains from scratch with `s = 0.04ε`. It isolates reduced source variance from conditional semantic content. Complete the canonical seed-2 G/Z pair in §5 first.

```bash
bash scripts/run_db29_gsmall_s2.sh smoke
bash scripts/run_db29_gsmall_s2.sh train
bash scripts/run_db29_gsmall_s2.sh eval
uv run python scripts/aggregate_gsmall_noise_budget.py \
  --run-dir runs --out-prefix runs/db29_gsmall_s2_noise_budget
```

The JSON and PNG compare unit Gaussian noise, small Gaussian noise, and semantic start plus small noise under the same replay protocol. This is an open-loop control, not evidence that arbitrary fixed-noise retraining will improve closed-loop behavior.

## 7. Source-pricing, perturbation, and long-horizon controls

### 7.1 Standard controls

```bash
bash scripts/eval_mos_from_scratch.sh controls 2
```

This evaluates 30 DROID trajectories, observation perturbations, semantic deployment with `η = 0`, and a Gaussian fixed-noise-at-deployment control. The latter freezes noise only after ordinary stochastic Gaussian training; it is not a fixed-noise training experiment.

### 7.2 Long-horizon data and evaluation

The long set has eight 50-step segments per episode. Evaluation must use 400 steps; truncating it to 200 steps does not reproduce the long-horizon result.

```bash
uv run python scripts/build_synthetic_mixture_dataset.py \
  --output-dir data/synth_mixture_long_unused \
  --eval-output-dir data/synth_mixture_eval_long \
  --num-eval-episodes 10 --segments-per-episode 8 --steps-per-segment 50 \
  --teleport-prob 0.5 --seed 20261027 --train-pool-as-eval \
  --high-level-task-pool --hl-conditional --relative-pose-state \
  --phase-in-instruction --randomize-zones --coord-in-text --data-version long

uv run python scripts/merge_lerobot_datasets.py \
  --inputs data/synth_mixture_eval_long --drop-state-keys target_relative_pose \
  --output-dir data/synth_mixture_eval_long_17d

LONG_EVAL_DATASET=data/synth_mixture_eval_long_17d \
  bash scripts/eval_mos_from_scratch.sh long 2

uv run python scripts/aggregate_mos_controls.py \
  --run-dir runs --training-seed 2 --long-dataset data/synth_mixture_eval_long_17d \
  --out-prefix runs/mos_paper_controls
```

`aggregate_mos_controls.py` validates dataset path, checkpoint identity, horizon, seed, noise, and perturbation options. It requires the 17D long dataset path; keep that path when publishing result JSON files or adjust both evaluation and validation together. The generated `mos_paper_controls.json` and `mos_paper_controls_source_pricing.png` cover source content/randomness and long-horizon controls. A zero candidate repeat divergence yields an undefined ratio (`null`), so interpret the absolute values instead of reporting an infinite improvement.

## 8. Encoder/history-window factor cells

Use independently named runs for each cell. Reusing output directories between different protocols would make the comparison invalid.

```bash
# MLP + strict past-only shift/1.
Z_RUN_NAME=mos_mlp_shift_s2 ENCODER_ARCH=mlp \
  HISTORY_ALIGN=shift HISTORY_OFFSET=1 \
  bash scripts/train_mos_from_scratch.sh semantic final 2
Z_RUN_NAME=mos_mlp_shift_s2 \
  bash scripts/eval_mos_from_scratch.sh final 2

# Transformer + shift/1.
Z_RUN_NAME=mos_tx_shift_s2 ENCODER_ARCH=transformer \
  HISTORY_ALIGN=shift HISTORY_OFFSET=1 \
  bash scripts/train_mos_from_scratch.sh semantic final 2
Z_RUN_NAME=mos_tx_shift_s2 \
  bash scripts/eval_mos_from_scratch.sh final 2

# MLP + rtc/16; this is a distinct history-alignment condition.
Z_RUN_NAME=mos_mlp_rtc16_s2 ENCODER_ARCH=mlp \
  HISTORY_ALIGN=rtc HISTORY_OFFSET=16 \
  bash scripts/train_mos_from_scratch.sh semantic final 2
Z_RUN_NAME=mos_mlp_rtc16_s2 \
  bash scripts/eval_mos_from_scratch.sh final 2

# Transformer + rtc/16.
Z_RUN_NAME=mos_tx_rtc16_s2 ENCODER_ARCH=transformer \
  HISTORY_ALIGN=rtc HISTORY_OFFSET=16 \
  bash scripts/train_mos_from_scratch.sh semantic final 2
Z_RUN_NAME=mos_tx_rtc16_s2 \
  bash scripts/eval_mos_from_scratch.sh final 2

uv run python scripts/aggregate_mos_factors.py \
  --cell mlp_shift=mos_mlp_shift_s2 --cell tx_shift=mos_tx_shift_s2 \
  --cell mlp_rtc16=mos_mlp_rtc16_s2 --cell tx_rtc16=mos_tx_rtc16_s2 \
  --training-seed 2 --out-json runs/mos_factor_cells.json
```

Treat rtc/16 as a factor under test, not as a drop-in equivalent of strict past-only shift/1. Inspect the actual alignment semantics before interpreting a difference.

## 9. Direct-target vs mixed-source consistency supervision

These are semantic-arm, pretrained-DiT fine-tuning controls. Use distinct suffixes so neither run nor evaluation output is overwritten.

```bash
RUN_SUFFIX=_y_anchor_v2 ANCHOR_TARGET=y \
  bash scripts/train_mos_release.sh semantic final 2
RUN_SUFFIX=_role_anchor_v2 ANCHOR_TARGET=role \
  bash scripts/train_mos_release.sh semantic final 2

uv run python scripts/visualize_memory_start.py \
  --dataset-path data/synth_mixture_eval_db7_17d \
  --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
  --model-path runs/mos_best_y_anchor_v2/checkpoint-5000 \
  --traj-ids 0 1 2 3 4 5 6 7 8 9 --steps 200 --execution-horizon 16 \
  --denoising-steps 4 --segment-meta-path data/synth_mixture_eval_db7_17d/meta/segment_schedule.jsonl \
  --high-level-task-from-meta --fixed-beta 0.3 --seed 0 \
  --out-dir runs/mos_y_anchor_tracking

uv run python scripts/visualize_memory_start.py \
  --dataset-path data/synth_mixture_eval_db7_17d \
  --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
  --model-path runs/mos_best_role_anchor_v2/checkpoint-5000 \
  --traj-ids 0 1 2 3 4 5 6 7 8 9 --steps 200 --execution-horizon 16 \
  --denoising-steps 4 --segment-meta-path data/synth_mixture_eval_db7_17d/meta/segment_schedule.jsonl \
  --high-level-task-from-meta --fixed-beta 0.3 --seed 0 \
  --out-dir runs/mos_role_anchor_tracking

uv run python scripts/aggregate_anchor_supervision.py \
  --direct runs/mos_y_anchor_tracking/memory_start_metrics.json \
  --mixed runs/mos_role_anchor_tracking/memory_start_metrics.json \
  --out-json runs/mos_anchor_supervision.json
```

This diagnostic evaluates semantic-start tracking NMSE at segment boundaries and within segments. It is not a substitute for the paired G/Z deployment comparison.

## 10. Table-to-JSON and aggregate-command map

Do not type statistics into plotting scripts. Regenerate aggregates from canonical `compare_metrics.json` files first, then render figures from those aggregates.

| Paper output | Required machine-readable inputs | Aggregate entry point |
|---|---|---|
| Release-pair result table and training-efficiency curves | `runs/eval_mos_release_*/compare_metrics.json` | `aggregate_mos_release.py` (§4) |
| Main multi-seed result table | `runs/eval_mos_scratch_*/compare_metrics.json` | `aggregate_mos_from_scratch.py` (§5.3) |
| Training-efficiency curves and step-to-threshold table | `runs/mos_paper_main.json`, including `early` and `efficiency_thresholds` | `aggregate_mos_from_scratch.py` (§5.3) |
| Noise-budget table | `runs/eval_db29_*/compare_metrics.json` | `aggregate_gsmall_noise_budget.py` (§6) |
| Source-pricing table | `compare_metrics.json` files from the §7 control evaluations | `aggregate_mos_controls.py` (§7) |
| Consistency-loss table | `runs/mos_*_anchor_tracking/memory_start_metrics.json` | `aggregate_anchor_supervision.py` (§9) |
| Encoder/window factor table | `compare_metrics.json` files from the §8 factor cells | `aggregate_mos_factors.py` (§8) |

Only the English paper PDF is included under `paper/`; paper-authoring scripts and the paper's figure directory are intentionally excluded. The aggregate JSON and CSV outputs remain available for reproducing tables and rendering new plots. README's training-efficiency figure is included separately as `assets/fig_train_efficiency.png`.

### 10.1 README trajectory figures

First regenerate the aggregate performance figure:

```bash
uv run python scripts/generate_mos_release_readme_assets.py \
  --run-dir runs --input-json runs/mos_release_comparison_s2.json \
  --output-dir release/assets
```

For paired trajectory panels, run **both checkpoints in the same evaluation process** and save the raw arrays. This is an additional visualization evaluation; it does not replace the canonical single-arm protocol used for the reported confidence intervals. Run these commands from the repository root, where `release/` and `runs/` contain the real weights:

```bash
# Four paired evaluations, followed by four shared-axis figures.
# release: pretrained-DiT fine-tuning; from-scratch: seed-2 zero-initialized DiT.
bash scripts/generate_mos_trajectory_comparisons.sh all

# Run only one evidence line if GPU time or disk is limited.
bash scripts/generate_mos_trajectory_comparisons.sh release
bash scripts/generate_mos_trajectory_comparisons.sh from-scratch
```

The wrapper uses 10 held-out trajectories, at most 200 environment steps (shorter DROID episodes end at their actual length), execution horizon 16, four denoising steps, `history_source=pred`, `β=0`, evaluation seed 0, and `--save-trajectory-arrays`. It uses `release/mos_best` and `release/mos_best_no_mem` for the release pair, and `runs/db28z_s2/checkpoint-5000` and `runs/db28g_s2/checkpoint-5000` for the from-scratch pair. Outputs go to `runs/trajectory_{release,db28}_{synth,droid}_es0/`, `release/assets/mos_action_trajectories_*.{png,json}`, and `assets/db28_action_trajectories_*.{png,json}`.

To redraw one panel without inference:

```bash
uv run python scripts/generate_mos_release_trajectory_assets.py \
  --evaluation-dir runs/trajectory_release_synth_es0 \
  --semantic-label mos_best --semantic-model "$PWD/release/mos_best" \
  --gaussian-label mos_best_no_mem --gaussian-model "$PWD/release/mos_best_no_mem" \
  --domain synth --dims 0 1 3 4 \
  --output-dir release/assets --output-prefix mos_action_trajectories
```

Ground truth (black), semantic (blue), and Gaussian (gold) share each action-dimension axis. The common legend is outside the plotting area. The representative trajectory is the deterministic upper-median paired-MSE ratio; dimensions are fixed, never selected for visual appeal. Existing PNG-only evaluations cannot reconstruct these curves exactly: rerun with `--save-trajectory-arrays`. The JSON next to each figure records checkpoint identity, trajectory selection, and provenance. The two DiT-initialization evidence lines must not be presented as the same model experiment.

The wrapper additionally writes `assets/db28_action_trajectories_clear_gap_synth.{png,json}` using `--selection max-ratio`. This explicitly labeled example selects the largest Gaussian/semantic paired-MSE ratio over the same 10 synthetic seed-0 trajectories, with the same fixed dimensions. It demonstrates a clear-gap case, not the typical effect; aggregate conclusions still come from the paired multi-seed statistics. To redraw it without inference, reuse the command above with `--evaluation-dir runs/trajectory_db28_synth_es0`, the from-scratch checkpoint paths, `--output-dir assets --output-prefix db28_action_trajectories_clear_gap`, and `--selection max-ratio`.

## 11. Local checks and release gate

Offline checks that do not require model weights:

```bash
uv run python -m unittest discover -s tests -v
uv run python scripts/test_export_mos_release.py
```

Before publication, separately complete all of the following:

1. Both real release directories pass `validate_mos_release.py --load` on the intended runtime.
2. `examples/mos_inference.py` successfully processes a real NPZ observation and writes finite 40-step actions.
3. Both arms pass 200-step GPU smoke training; full reproductions pass the checkpoint/protocol assertions in the aggregation scripts.
4. The exact source revision, data manifest, model checksums, and validation JSON are archived with the release.
5. Remote upload, anonymous download, and load-from-downloaded-copy are tested independently. A local structural test cannot prove those steps.

## 12. Troubleshooting

| Symptom | Check |
|---|---|
| A new option is unrecognized | Confirm `import gr00t...` resolves inside this checkout and the argument is forwarded through the model setup chain. |
| `Unrecognized options: --repeats` | Remove it. The evaluator performs two repeat passes by default. |
| Boolean value `False` is rejected | Tyro booleans use flags: omit the flag or use `--no-<option>`; do not append `False`. |
| Aggregation rejects a result | Compare its recorded model path, dataset path, seed, horizon, noise, and perturbation options with the canonical protocol. Do not bypass validation. |
| A checkpoint path is treated as a remote repository | Pass the existing checkpoint as an absolute path. |
| An output directory already exists | Use a new `OUTPUT_ROOT`, `RUN_SUFFIX`, or explicit run name. Do not mix results from different recipes. |
| OOM on a 24 GB GPU | Retain the default global batch 4, accumulation 2, BF16 loading, and gradient checkpointing; lowering batch size changes the effective batch unless accumulation is adjusted. |
| DROID accuracy ratio includes 1 | Report no statistically significant accuracy difference; do not convert a repeatability result into an accuracy or task-success claim. |
