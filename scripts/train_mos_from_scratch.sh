#!/usr/bin/env bash
# Reproduce the paper's from-scratch Action DiT G/Z arms.
# Usage: bash scripts/train_mos_from_scratch.sh <gaussian|semantic|both> <smoke|final> [seed]
# Optional: BASE_MODEL_PATH, TRAIN_DATASET, OUTPUT_ROOT, G_RUN_NAME, Z_RUN_NAME,
#           ENCODER_ARCH (mlp|transformer), HISTORY_ALIGN (shift|rtc), HISTORY_OFFSET.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ARM="${1:-both}"
MODE="${2:-final}"
SEED="${3:-2}"
[[ "$ARM" =~ ^(gaussian|semantic|both)$ ]] || { echo "Invalid arm: $ARM" >&2; exit 2; }
[[ "$MODE" =~ ^(smoke|final)$ ]] || { echo "Invalid mode: $MODE" >&2; exit 2; }
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "Seed must be non-negative" >&2; exit 2; }

BASE_MODEL_PATH="${BASE_MODEL_PATH:-$ROOT/GR00T-N1.7-3B}"
TRAIN_DATASET="${TRAIN_DATASET:-data/mix_droid_synth_train}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs}"
ENCODER_ARCH="${ENCODER_ARCH:-mlp}"
HISTORY_ALIGN="${HISTORY_ALIGN:-shift}"
HISTORY_OFFSET="${HISTORY_OFFSET:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NO_ALBUMENTATIONS_UPDATE=1

[[ -f "$BASE_MODEL_PATH/config.json" ]] || { echo "Missing base model: $BASE_MODEL_PATH" >&2; exit 1; }
uv run python - "$TRAIN_DATASET" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "meta/info.json"
c = json.loads(p.read_text(encoding="utf-8"))
assert c["total_episodes"] == 400, c["total_episodes"]
assert c["total_frames"] == 108292, c["total_frames"]
assert c["features"]["observation.state"]["shape"] == [17]
assert c["features"]["action"]["shape"] == [17]
print("PASS: 400 episodes, 108292 frames, 17D state/action")
PY

if [[ "$MODE" == smoke ]]; then
  MAX_STEPS=200; SAVE_STEPS=200; EXTRA_SAVE_STEPS=""; SAVE_TOTAL_LIMIT=1
else
  MAX_STEPS=5000; SAVE_STEPS=5000; EXTRA_SAVE_STEPS="200,500,1000,2500"; SAVE_TOTAL_LIMIT=10
fi

run_name() {
  local arm="$1"
  if [[ "$arm" == gaussian ]]; then
    echo "${G_RUN_NAME:-db28g_s${SEED}}"
  elif [[ -n "${Z_RUN_NAME:-}" ]]; then
    echo "$Z_RUN_NAME"
  elif [[ "$SEED" == 3 && "$ENCODER_ARCH" == mlp && "$HISTORY_ALIGN" == shift ]]; then
    echo "db28z_s3_soft"
  else
    echo "db28z_s${SEED}"
  fi
}

train_one() {
  local arm="$1" name out log
  name="$(run_name "$arm")"
  out="$OUTPUT_ROOT/$name"
  if [[ "$MODE" == smoke ]]; then out="$OUTPUT_ROOT/smoke_${name}"; fi
  if [[ -e "$out" && -n "$(ls -A "$out" 2>/dev/null)" ]]; then
    echo "Refusing non-empty output: $out" >&2; exit 1
  fi
  mkdir -p "$out"
  log="${out}_log.txt"
  local args=(
    --base-model-path "$BASE_MODEL_PATH" --dataset-path "$TRAIN_DATASET"
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT
    --num-gpus 1 --output-dir "$out" --seed "$SEED"
    --max-steps "$MAX_STEPS" --global-batch-size "$GLOBAL_BATCH_SIZE"
    --gradient-accumulation-steps "$GRAD_ACCUM"
    --save-steps "$SAVE_STEPS" --save-total-limit "$SAVE_TOTAL_LIMIT" --save-only-model
    --learning-rate 1e-4 --weight-decay 1e-5 --warmup-ratio 0.05
    --load-bf16 --gradient-checkpointing
    --no-tune-llm --no-tune-visual --tune-projector
    --tune-diffusion-model --tune-vlln --zero-init-dit
    --history-action-align "$HISTORY_ALIGN" --history-action-offset "$HISTORY_OFFSET"
    --sub-instruction-mode off
  )
  if [[ -n "$EXTRA_SAVE_STEPS" ]]; then args+=(--extra-save-steps "$EXTRA_SAVE_STEPS"); fi
  if [[ "$arm" == semantic ]]; then
    args+=(
      --flow-source-mode memory_history_mix --memory-start-enabled
      --memory-encoder-arch "$ENCODER_ARCH" --memory-start-hidden-dim 1024
      --memory-num-queries 64 --memory-query-dim 64 --memory-query-path backbone
      --tune-memory-encoder
      --memory-start-mix-beta-sampling uniform
      --memory-start-mix-beta-min 0.0 --memory-start-mix-beta-max 0.3
      --memory-start-noise-std 0.04 --memory-start-noise-sampling fixed
      --memory-start-anchor-type mse --memory-start-anchor-target role
      --memory-start-anchor-lambda 0.1 --memory-anchor-lambda-schedule cosine
      --memory-start-diff-anchor-lambda 0.0 --memory-start-z0-recon-lambda 0.0
    )
  else
    args+=(--flow-source-mode gaussian --gaussian-source-noise-std 1.0 --no-memory-start-enabled)
  fi
  uv run python gr00t/experiment/launch_finetune.py "${args[@]}" 2>&1 | tee "$log"
  uv run python - "$out/checkpoint-${MAX_STEPS}/config.json" "$arm" "$SEED" \
    "$ENCODER_ARCH" "$HISTORY_ALIGN" "$HISTORY_OFFSET" <<'PY'
import json, sys
path, arm, seed, encoder, align, offset = sys.argv[1:]
c = json.load(open(path, encoding="utf-8"))
common = {
    "zero_init_dit": True, "select_layer": 16, "sub_instruction_mode": "off",
    "history_action_align": align, "history_action_offset": int(offset),
    "confidence_tokens_enabled": False, "confidence_routing_enabled": False,
    "tune_projector": True, "tune_diffusion_model": True, "tune_vlln": True,
}
semantic = {
    "flow_source_mode": "memory_history_mix", "memory_start_enabled": True,
    "memory_encoder_arch": encoder, "memory_start_mix_beta_sampling": "uniform",
    "memory_start_mix_beta_min": 0.0, "memory_start_mix_beta_max": 0.3,
    "memory_start_noise_std": 0.04, "memory_start_anchor_target": "role",
    "memory_start_anchor_lambda": 0.1, "memory_anchor_lambda_schedule": "cosine",
}
gaussian = {
    "flow_source_mode": "gaussian", "memory_start_enabled": False,
    "gaussian_source_noise_std": 1.0,
}
for key, want in (common | (semantic if arm == "semantic" else gaussian)).items():
    assert c.get(key) == want, f"{key}: got {c.get(key)!r}, expected {want!r}"
print(f"PASS {arm}: from-scratch DiT; encoder={encoder}; history={align}/{offset}; seed={seed}")
PY
}

if [[ "$ARM" == gaussian || "$ARM" == both ]]; then train_one gaussian; fi
if [[ "$ARM" == semantic || "$ARM" == both ]]; then train_one semantic; fi
