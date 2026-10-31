#!/usr/bin/env bash
# G-small: train and deploy s=0.04*eps; otherwise mirror db28g_s2.
# Usage: bash scripts/run_db29_gsmall_s2.sh {smoke|train|eval}
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-$ROOT/GR00T-N1.7-3B}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export NO_ALBUMENTATIONS_UPDATE=1

case "$MODE" in
  smoke)
    OUT="${SMOKE_OUT:-runs/smoke_db29_gsmall_s2}"
    MAX_STEPS=200
    SAVE_STEPS=200
    ;;
  train)
    OUT="${OUT:-runs/db29_gsmall_s2}"
    MAX_STEPS=5000
    SAVE_STEPS=5000
    ;;
  eval)
    exec bash "$ROOT/scripts/eval_db29_gsmall_s2.sh" all
    ;;
  *)
    echo "Usage: bash scripts/run_db29_gsmall_s2.sh {smoke|train|eval}" >&2
    exit 2
    ;;
esac

[[ -f "$BASE_MODEL_PATH/config.json" ]] || { echo "Missing base model: $BASE_MODEL_PATH" >&2; exit 1; }
[[ -d data/mix_droid_synth_train ]] || { echo "Missing data/mix_droid_synth_train" >&2; exit 1; }
if [[ -d "$OUT" && -n "$(ls -A "$OUT")" ]]; then
  echo "Refusing non-empty output directory: $OUT (choose a new OUT/SMOKE_OUT)" >&2
  exit 1
fi
mkdir -p runs

uv run python scripts/verify_gaussian_source_noise.py
uv run python gr00t/experiment/launch_finetune.py \
  --base-model-path "$BASE_MODEL_PATH" \
  --dataset-path data/mix_droid_synth_train \
  --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
  --num-gpus 1 \
  --output-dir "$OUT" \
  --max-steps "$MAX_STEPS" \
  --global-batch-size 4 \
  --gradient-accumulation-steps 2 \
  --save-steps "$SAVE_STEPS" \
  --save-total-limit 1 \
  --save-only-model \
  --no-tune-llm --no-tune-visual \
  --tune-projector --tune-diffusion-model --tune-vlln \
  --load-bf16 --gradient-checkpointing \
  --learning-rate 1e-4 --weight-decay 1e-5 --warmup-ratio 0.05 \
  --zero-init-dit --seed 2 \
  --flow-source-mode gaussian \
  --gaussian-source-noise-std 0.04 \
  2>&1 | tee "runs/$(basename "$OUT")_log.txt"

# Do not accept a run merely because training exited: prove the new parameter
# reached BOTH the deployable checkpoint and the complete training config.
uv run python - "$OUT/checkpoint-$MAX_STEPS/config.json" "$OUT/experiment_cfg/conf.yaml" "$MAX_STEPS" <<'PY'
import json
from pathlib import Path
import sys
import yaml

ckpt_path, conf_path, expected_step = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
cfg = json.loads(ckpt_path.read_text(encoding="utf-8"))
expected_model = {
    "flow_source_mode": "gaussian",
    "gaussian_source_noise_std": 0.04,
    "memory_start_enabled": False,
    "zero_init_dit": True,
    "tune_llm": False,
    "tune_visual": False,
    "select_layer": 16,
}
for key, want in expected_model.items():
    got = cfg.get(key)
    assert got == want, f"checkpoint {key}: got {got!r}, expected {want!r}"
conf = yaml.safe_load(conf_path.read_text(encoding="utf-8"))
assert conf["data"]["seed"] == 2, conf["data"]["seed"]
assert conf["training"]["global_batch_size"] == 4
assert conf["training"]["gradient_accumulation_steps"] == 2
assert conf["training"]["max_steps"] == expected_step
print(f"PASS {ckpt_path.parent}: std=0.04, seed=2, batch=4, accum=2, steps={expected_step}")
PY

if [[ "$MODE" == smoke ]]; then
  echo "Smoke complete. Inspect runs/$(basename "$OUT")_log.txt for finite/decreasing loss before train."
else
  echo "Training complete. Next: bash scripts/run_db29_gsmall_s2.sh eval"
fi
