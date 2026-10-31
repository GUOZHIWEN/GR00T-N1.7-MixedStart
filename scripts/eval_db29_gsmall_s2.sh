#!/usr/bin/env bash
# G-small deployment evaluation: fresh 0.04*eps on every replan and repeat pass.
# Usage: bash scripts/eval_db29_gsmall_s2.sh [synth|droid|all]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-all}"
CKPT="${CKPT:-runs/db29_gsmall_s2/checkpoint-5000}"
PREFIX="${EVAL_PREFIX:-eval_s2_db29_gsmall_s2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NO_ALBUMENTATIONS_UPDATE=1
mkdir -p runs

[[ "$MODE" =~ ^(synth|droid|all)$ ]] || { echo "Usage: $0 [synth|droid|all]" >&2; exit 2; }
[[ -f "$CKPT/config.json" ]] || { echo "Missing checkpoint: $CKPT" >&2; exit 1; }
uv run python - "$CKPT/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1], encoding="utf-8"))
assert c.get("flow_source_mode") == "gaussian"
assert c.get("gaussian_source_noise_std") == 0.04, c.get("gaussian_source_noise_std")
assert c.get("zero_init_dit") is True
print("PASS deployment checkpoint: gaussian_source_noise_std=0.04")
PY

COMMON=(
  --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT
  --model-paths "$CKPT"
  --traj-ids 0 1 2 3 4 5 6 7 8 9
  --steps 200 --execution-horizon 16 --denoising-steps 4
  --history-source pred
  --force-fixed-beta --fixed-beta-value 0.0
  --seed 0
)
# repeat_probe=True is the evaluator default: R=2 per (trajectory, eval seed).
# Do NOT add --repeats; that option does not exist in this evaluator.
if [[ "$MODE" == synth || "$MODE" == all ]]; then
  for ES in 0 1 2 3; do
    OUT="runs/${PREFIX}_synth_es${ES}"
    if [[ -f "$OUT/compare_metrics.json" ]]; then
      echo "Reusing completed result: $OUT (aggregation will verify its protocol)"
    else
      CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" uv run python gr00t/eval/compare_flow_sources.py \
        --dataset-path data/synth_mixture_eval_db7_17d \
        "${COMMON[@]}" --seed "$ES" \
        --segment-meta-path data/synth_mixture_eval_db7_17d/meta/segment_schedule.jsonl \
        --high-level-task-from-meta \
        --out-dir "$OUT" 2>&1 | tee "${OUT}_log.txt"
    fi
  done
fi
if [[ "$MODE" == droid || "$MODE" == all ]]; then
  OUT="runs/${PREFIX}_droid"
  if [[ -f "$OUT/compare_metrics.json" ]]; then
    echo "Reusing completed result: $OUT (aggregation will verify its protocol)"
  else
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" uv run python gr00t/eval/compare_flow_sources.py \
      --dataset-path data/droid_subset_eval \
      "${COMMON[@]}" \
      --out-dir "$OUT" 2>&1 | tee "${OUT}_log.txt"
  fi
fi

echo "Evaluation complete. Next: uv run python scripts/aggregate_gsmall_noise_budget.py"
