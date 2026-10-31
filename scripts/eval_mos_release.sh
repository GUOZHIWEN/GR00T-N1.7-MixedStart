#!/usr/bin/env bash
# Held-out open-loop evaluation for the official pretrained-DiT release pair.
# Usage: bash scripts/eval_mos_release.sh <baseline|early|final|all> [seed]
# baseline evaluates only checkpoint-5000 and deliberately skips curve aggregation.
# Optional env: RUN_SUFFIX, OUTPUT_ROOT, SYNTH_EVAL_DATASET, DROID_EVAL_DATASET,
#               CUDA_VISIBLE_DEVICES.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-all}"
SEED="${2:-2}"
RUN_SUFFIX="${RUN_SUFFIX:-_v2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs}"
SYNTH_EVAL_DATASET="${SYNTH_EVAL_DATASET:-data/synth_mixture_eval_db7_17d}"
DROID_EVAL_DATASET="${DROID_EVAL_DATASET:-data/droid_subset_eval}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NO_ALBUMENTATIONS_UPDATE=1
[[ "$MODE" =~ ^(baseline|early|final|all)$ ]] || { echo "Usage: $0 <baseline|early|final|all> [seed]" >&2; exit 2; }
Z="$OUTPUT_ROOT/mos_best${RUN_SUFFIX}"
G="$OUTPUT_ROOT/mos_best_no_mem${RUN_SUFFIX}"
for ARM_DIR in "$Z" "$G"; do
  [[ -f "$ARM_DIR/checkpoint-5000/config.json" ]] || { echo "Train both arms first; missing $ARM_DIR/checkpoint-5000" >&2; exit 1; }
done

run_eval() {
  local arm_dir="$1" arm="$2" step="$3" domain="$4" es="$5"
  local out="$OUTPUT_ROOT/eval_mos_release_${arm}_ckpt${step}_${domain}_es${es}_s${SEED}"
  if [[ -f "$out/compare_metrics.json" ]]; then echo "Reusing $out"; return; fi
  local dataset="$DROID_EVAL_DATASET" extra=()
  if [[ "$domain" == synth ]]; then
    dataset="$SYNTH_EVAL_DATASET"
    extra=(--segment-meta-path "$dataset/meta/segment_schedule.jsonl" --high-level-task-from-meta)
  fi
  [[ -f "$dataset/meta/info.json" ]] || { echo "Missing evaluation dataset: $dataset" >&2; exit 1; }
  uv run python gr00t/eval/compare_flow_sources.py \
    --dataset-path "$dataset" \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
    --model-paths "$arm_dir/checkpoint-${step}" --labels "$arm" \
    --traj-ids 0 1 2 3 4 5 6 7 8 9 --steps 200 --execution-horizon 16 \
    --denoising-steps 4 --history-source pred \
    --force-fixed-beta --fixed-beta-value 0.0 --seed "$es" \
    "${extra[@]}" --out-dir "$out" 2>&1 | tee "${out}_log.txt"
}

for spec in "mos_best:$Z" "mos_best_no_mem:$G"; do
  arm="${spec%%:*}"; arm_dir="${spec#*:}"
  if [[ "$MODE" == early || "$MODE" == all ]]; then
    for STEP in 200 500 1000 2500 3500 5000; do
      for DOMAIN in synth droid; do run_eval "$arm_dir" "$arm" "$STEP" "$DOMAIN" 0; done
    done
  fi
  if [[ "$MODE" == baseline || "$MODE" == final || "$MODE" == all ]]; then
    for ES in 0 1 2 3; do run_eval "$arm_dir" "$arm" 5000 synth "$ES"; done
    run_eval "$arm_dir" "$arm" 5000 droid 0
  fi
done

if [[ "$MODE" != baseline ]]; then
  uv run python scripts/aggregate_mos_release.py \
    --run-dir "$OUTPUT_ROOT" --run-suffix "$RUN_SUFFIX" --training-seed "$SEED" \
    --out-prefix "$OUTPUT_ROOT/mos_release_comparison_s${SEED}"
fi
