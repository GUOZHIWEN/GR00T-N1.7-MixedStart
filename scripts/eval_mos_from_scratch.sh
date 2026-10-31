#!/usr/bin/env bash
# Held-out evaluation for paper from-scratch G/Z checkpoints.
# Usage: bash scripts/eval_mos_from_scratch.sh <final|early|controls|long|all> [training-seed]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-final}"
SEED="${2:-2}"
[[ "$MODE" =~ ^(final|early|controls|long|all)$ ]] || { echo "Invalid mode: $MODE" >&2; exit 2; }
OUTPUT_ROOT="${OUTPUT_ROOT:-runs}"
SYNTH_EVAL_DATASET="${SYNTH_EVAL_DATASET:-data/synth_mixture_eval_db7_17d}"
DROID_EVAL_DATASET="${DROID_EVAL_DATASET:-data/droid_subset_eval}"
LONG_EVAL_DATASET="${LONG_EVAL_DATASET:-data/synth_mixture_eval_long}"
G_NAME="${G_RUN_NAME:-db28g_s${SEED}}"
if [[ -n "${Z_RUN_NAME:-}" ]]; then Z_NAME="$Z_RUN_NAME";
elif [[ "$SEED" == 3 ]]; then Z_NAME="db28z_s3_soft"; else Z_NAME="db28z_s${SEED}"; fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NO_ALBUMENTATIONS_UPDATE=1
mkdir -p "$OUTPUT_ROOT"

ckpt() {
  local arm="$1" step="$2" name
  if [[ "$arm" == mos_best_no_mem ]]; then name="$G_NAME"; else name="$Z_NAME"; fi
  echo "$OUTPUT_ROOT/$name/checkpoint-$step"
}

run_eval() {
  local arm="$1" step="$2" domain="$3" es="$4" suffix="$5" steps="$6" traj_count="$7"
  shift 7
  local model; model="$(ckpt "$arm" "$step")"
  [[ -f "$model/config.json" ]] || { echo "Missing checkpoint: $model" >&2; exit 1; }
  local dataset="$DROID_EVAL_DATASET" extra=()
  if [[ "$domain" == synth || "$domain" == long ]]; then
    dataset="$SYNTH_EVAL_DATASET"
    [[ "$domain" == long ]] && dataset="$LONG_EVAL_DATASET"
    extra=(--segment-meta-path "$dataset/meta/segment_schedule.jsonl" --high-level-task-from-meta)
  fi
  [[ -f "$dataset/meta/info.json" ]] || { echo "Missing dataset: $dataset" >&2; exit 1; }
  local out="$OUTPUT_ROOT/eval_mos_scratch_${arm}_ckpt${step}_${domain}_es${es}_s${SEED}${suffix}"
  if [[ -f "$out/compare_metrics.json" ]]; then echo "Reusing $out"; return; fi
  local trajectories=()
  for ((i=0; i<traj_count; i++)); do trajectories+=("$i"); done
  uv run python gr00t/eval/compare_flow_sources.py \
    --dataset-path "$dataset" \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
    --model-paths "$model" --labels "$arm" \
    --traj-ids "${trajectories[@]}" --steps "$steps" --execution-horizon 16 \
    --denoising-steps 4 --history-source pred \
    --force-fixed-beta --fixed-beta-value 0.0 --seed "$es" \
    "${extra[@]}" "$@" --out-dir "$out" 2>&1 | tee "${out}_log.txt"
}

for arm in mos_best_no_mem mos_best; do
  if [[ "$MODE" == final || "$MODE" == all ]]; then
    for es in 0 1 2 3; do run_eval "$arm" 5000 synth "$es" "" 200 10; done
    run_eval "$arm" 5000 droid 0 "" 200 10
  fi
  if [[ "$MODE" == early || "$MODE" == all ]]; then
    for step in 200 500 1000 2500 5000; do
      run_eval "$arm" "$step" synth 0 "" 200 10
      run_eval "$arm" "$step" droid 0 "" 200 10
    done
  fi
  if [[ "$MODE" == controls || "$MODE" == all ]]; then
    run_eval "$arm" 5000 droid 0 "_all30" 200 30
    run_eval "$arm" 5000 synth 0 "_perturb" 200 10 \
      --obs-image-noise-std 8.0 --obs-state-noise-std 0.02
    if [[ "$arm" == mos_best ]]; then
      run_eval "$arm" 5000 synth 0 "_eta0" 200 10 --fixed-noise-std 0.0
      run_eval "$arm" 5000 droid 0 "_eta0" 200 30 --fixed-noise-std 0.0
      run_eval "$arm" 5000 synth 0 "_perturb_eta0" 200 10 \
        --fixed-noise-std 0.0 --obs-image-noise-std 8.0 --obs-state-noise-std 0.02
    else
      run_eval "$arm" 5000 synth 0 "_fixed_noise" 200 10 --fixed-noise-draw
    fi
  fi
  if [[ "$MODE" == long || "$MODE" == all ]]; then
    run_eval "$arm" 5000 long 0 "" 400 10
    run_eval "$arm" 5000 long 0 "_perturb" 400 10 \
      --obs-image-noise-std 8.0 --obs-state-noise-std 0.02
    if [[ "$arm" == mos_best ]]; then
      run_eval "$arm" 5000 long 0 "_eta0" 400 10 --fixed-noise-std 0.0
      run_eval "$arm" 5000 long 0 "_perturb_eta0" 400 10 \
        --fixed-noise-std 0.0 --obs-image-noise-std 8.0 --obs-state-noise-std 0.02
    fi
  fi
done

echo "Evaluation complete. Aggregate main/early results with scripts/aggregate_mos_from_scratch.py"
