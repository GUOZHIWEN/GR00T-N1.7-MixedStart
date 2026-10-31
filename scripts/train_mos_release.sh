#!/usr/bin/env bash
# Train the official pretrained-DiT MoS release pair.
# Usage: bash scripts/train_mos_release.sh <semantic|gaussian|both> <smoke|final> [seed]
# Optional env: BASE_MODEL_PATH, TRAIN_DATASET, OUTPUT_ROOT, RUN_SUFFIX,
#               CUDA_VISIBLE_DEVICES, GLOBAL_BATCH_SIZE, GRAD_ACCUM, SAVE_TOTAL_LIMIT.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ARM="${1:-both}"
MODE="${2:-final}"
SEED="${3:-2}"
[[ "$ARM" =~ ^(semantic|gaussian|both)$ ]] || { echo "Usage: $0 <semantic|gaussian|both> <smoke|final> [seed]" >&2; exit 2; }
[[ "$MODE" =~ ^(smoke|final)$ ]] || { echo "MODE must be smoke or final" >&2; exit 2; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "seed must be a non-negative integer" >&2; exit 2; }

BASE_MODEL_PATH="${BASE_MODEL_PATH:-$ROOT/GR00T-N1.7-3B}"
TRAIN_DATASET="${TRAIN_DATASET:-data/mix_droid_synth_train}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs}"
RUN_SUFFIX="${RUN_SUFFIX:-_v2}"
ANCHOR_TARGET="${ANCHOR_TARGET:-role}"
[[ "$ANCHOR_TARGET" =~ ^(role|y)$ ]] || { echo "ANCHOR_TARGET must be role or y" >&2; exit 2; }
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-15}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NO_ALBUMENTATIONS_UPDATE=1

[[ -f "$BASE_MODEL_PATH/config.json" ]] || { echo "Missing base config: $BASE_MODEL_PATH/config.json" >&2; exit 1; }
[[ -f "$TRAIN_DATASET/meta/info.json" ]] || { echo "Missing dataset: $TRAIN_DATASET" >&2; exit 1; }
uv run python - "$TRAIN_DATASET" <<'PY'
import json, sys
p = sys.argv[1] + '/meta/info.json'
c = json.load(open(p, encoding='utf-8'))
assert c['total_episodes'] == 400, (p, c['total_episodes'])
assert c['total_frames'] == 108292, (p, c['total_frames'])
assert c['features']['observation.state']['shape'] == [17], (p, c['features']['observation.state']['shape'])
assert c['features']['action']['shape'] == [17], (p, c['features']['action']['shape'])
print('PASS dataset: 400 episodes, 108292 frames, 17D state/action')
PY

if [[ "$MODE" == smoke ]]; then
  MAX_STEPS=200
  SAVE_STEPS=200
  EXTRA_SAVE_STEPS=""
  SAVE_TOTAL_LIMIT=1
else
  MAX_STEPS=5000
  SAVE_STEPS=5000
  EXTRA_SAVE_STEPS="200,500,1000,2500,3500"
fi

train_one() {
  local arm="$1" out log
  if [[ "$MODE" == smoke ]]; then
    out="$OUTPUT_ROOT/smoke_mos_release_${arm}_s${SEED}"
  elif [[ "$arm" == semantic ]]; then
    out="$OUTPUT_ROOT/mos_best${RUN_SUFFIX}"
  else
    out="$OUTPUT_ROOT/mos_best_no_mem${RUN_SUFFIX}"
  fi
  if [[ -e "$out" && -n "$(ls -A "$out" 2>/dev/null)" ]]; then
    echo "Refusing to overwrite non-empty $out; choose another OUTPUT_ROOT/RUN_SUFFIX." >&2
    exit 1
  fi
  mkdir -p "$out"
  log="${out}_log.txt"
  local args=(
    --base-model-path "$BASE_MODEL_PATH"
    --dataset-path "$TRAIN_DATASET"
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT
    --num-gpus 1 --output-dir "$out" --seed "$SEED"
    --max-steps "$MAX_STEPS" --global-batch-size "$GLOBAL_BATCH_SIZE"
    --gradient-accumulation-steps "$GRAD_ACCUM"
    --save-steps "$SAVE_STEPS" --save-total-limit "$SAVE_TOTAL_LIMIT" --save-only-model
    --learning-rate 1e-4 --weight-decay 1e-5 --warmup-ratio 0.05
    --load-bf16 --gradient-checkpointing
    --no-tune-llm --no-tune-visual --tune-projector
    --tune-diffusion-model --tune-vlln
    --history-action-align shift --history-action-offset 1
    --sub-instruction-mode off
  )
  if [[ -n "$EXTRA_SAVE_STEPS" ]]; then args+=(--extra-save-steps "$EXTRA_SAVE_STEPS"); fi
  if [[ "$arm" == semantic ]]; then
    args+=(
      --flow-source-mode memory_history_mix --memory-start-enabled
      --memory-encoder-arch mlp --memory-start-hidden-dim 1024
      --memory-num-queries 64 --memory-query-dim 64 --memory-query-path backbone
      --tune-memory-encoder
      --memory-start-mix-beta-sampling uniform
      --memory-start-mix-beta-min 0.0 --memory-start-mix-beta-max 0.3
      --memory-start-noise-std 0.04 --memory-start-noise-sampling fixed
      --memory-start-anchor-type mse --memory-start-anchor-target "$ANCHOR_TARGET"
      --memory-start-anchor-lambda 0.1 --memory-anchor-lambda-schedule cosine
      --memory-start-diff-anchor-lambda 0.0 --memory-start-z0-recon-lambda 0.0
    )
  else
    # Pure upstream GR00T-N1.7 Gaussian-source fine-tuning. No history source,
    # semantic encoder, consistency loss, confidence head, or DiT reinitialization.
    args+=(--flow-source-mode gaussian --gaussian-source-noise-std 1.0 --no-memory-start-enabled)
  fi
  # Deliberately no --zero-init-dit: load the Action DiT trained in GR00T-N1.7-3B.
  uv run python gr00t/experiment/launch_finetune.py "${args[@]}" 2>&1 | tee "$log"
  uv run python - "$out/checkpoint-${MAX_STEPS}/config.json" \
    "$arm" "$SEED" "$ANCHOR_TARGET" <<'PY'
import json, sys
path, arm, seed, anchor_target = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
c = json.load(open(path, encoding='utf-8'))
common = {
    'history_action_align': 'shift', 'history_action_offset': 1,
    'zero_init_dit': False, 'sub_instruction_mode': 'off',
    'confidence_tokens_enabled': False, 'confidence_routing_enabled': False,
    'tune_projector': True, 'tune_diffusion_model': True, 'tune_vlln': True,
}
semantic = {
    'flow_source_mode': 'memory_history_mix', 'memory_start_enabled': True,
    'memory_encoder_arch': 'mlp', 'memory_start_mix_beta_sampling': 'uniform',
    'memory_start_mix_beta_min': 0.0, 'memory_start_mix_beta_max': 0.3,
    'memory_start_noise_std': 0.04, 'memory_start_anchor_target': anchor_target,
    'memory_start_anchor_lambda': 0.1, 'memory_anchor_lambda_schedule': 'cosine',
}
gaussian = {
    'flow_source_mode': 'gaussian', 'memory_start_enabled': False,
    'gaussian_source_noise_std': 1.0,
}
for key, want in (common | (semantic if arm == 'semantic' else gaussian)).items():
    assert c.get(key) == want, f'{key}: got {c.get(key)!r}, expected {want!r}'
print(f'PASS {arm}: final recipe landed; pretrained DiT retained; seed={seed}')
PY
}

if [[ "$ARM" == semantic || "$ARM" == both ]]; then train_one semantic; fi
if [[ "$ARM" == gaussian || "$ARM" == both ]]; then train_one gaussian; fi
