#!/usr/bin/env bash
# rtc/16 early-checkpoint heldout evaluation. checkpoint-5000 is NOT rerun.
# Usage: bash scripts/eval_f2early_checkpoints.sh [synth|droid|all]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-all}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NO_ALBUMENTATIONS_UPDATE=1
mkdir -p runs
[[ "$MODE" =~ ^(synth|droid|all)$ ]] || { echo "Usage: $0 [synth|droid|all]" >&2; exit 2; }

# Preflight every required checkpoint before starting the first GPU rollout.
for ARM in db28z2b_s2 db28z2c_s2; do
  for STEP in 200 500 1000 2500; do
    CKPT="runs/${ARM}/checkpoint-${STEP}"
    [[ -f "$CKPT/config.json" ]] || { echo "Missing $CKPT/config.json" >&2; exit 1; }
    [[ -f "$CKPT/model.safetensors.index.json" || -f "$CKPT/model.safetensors" ]] || {
      echo "Missing model weights in $CKPT" >&2; exit 1;
    }
    uv run python - "$ARM" "$CKPT/config.json" <<'PY'
import json, sys
arm, path = sys.argv[1:]
c = json.load(open(path, encoding="utf-8"))
expected = {
    "flow_source_mode": "memory_history_mix",
    "memory_start_enabled": True,
    "memory_encoder_arch": "mlp" if arm == "db28z2b_s2" else "transformer",
    "history_action_align": "rtc",
    "history_action_offset": 16,
    "memory_start_mix_beta_sampling": "uniform",
    "memory_start_mix_beta_max": 0.3,
    "memory_start_noise_std": 0.04,
    "memory_start_anchor_lambda": 0.1,
    "memory_anchor_lambda_schedule": "cosine",
    "memory_start_anchor_target": "role",
    "zero_init_dit": True,
}
for key, want in expected.items():
    assert c.get(key) == want, f"{arm}: {key}={c.get(key)!r}, expected {want!r}"
print(f"PASS {arm}: rtc/16 recipe and encoder architecture")
PY
  done
done

COMMON=(
  --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT
  --traj-ids 0 1 2 3 4 5 6 7 8 9
  --steps 200 --execution-horizon 16 --denoising-steps 4
  --history-source pred
  --force-fixed-beta --fixed-beta-value 0.0
  --seed 0
)
for ARM in db28z2b_s2 db28z2c_s2; do
  for STEP in 200 500 1000 2500; do
    CKPT="runs/${ARM}/checkpoint-${STEP}"
    if [[ "$MODE" == synth || "$MODE" == all ]]; then
      OUT="runs/eval_f2early_${ARM}_ckpt${STEP}_synth_es0"
      if [[ -f "$OUT/compare_metrics.json" ]]; then
        echo "Reusing completed result: $OUT (aggregation will verify its protocol)"
      else
        uv run python gr00t/eval/compare_flow_sources.py \
          --dataset-path data/synth_mixture_eval_db7_17d \
          "${COMMON[@]}" --model-paths "$CKPT" \
          --segment-meta-path data/synth_mixture_eval_db7_17d/meta/segment_schedule.jsonl \
          --high-level-task-from-meta \
          --out-dir "$OUT" 2>&1 | tee "${OUT}_log.txt"
      fi
    fi
    if [[ "$MODE" == droid || "$MODE" == all ]]; then
      OUT="runs/eval_f2early_${ARM}_ckpt${STEP}_droid"
      if [[ -f "$OUT/compare_metrics.json" ]]; then
        echo "Reusing completed result: $OUT (aggregation will verify its protocol)"
      else
        uv run python gr00t/eval/compare_flow_sources.py \
          --dataset-path data/droid_subset_eval \
          "${COMMON[@]}" --model-paths "$CKPT" \
          --out-dir "$OUT" 2>&1 | tee "${OUT}_log.txt"
      fi
    fi
  done
done

echo "Early evaluation complete; existing 5000-step results were not modified."
echo "Next: uv run python scripts/aggregate_f2early.py --include-shift-reference"
