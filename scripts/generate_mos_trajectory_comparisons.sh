#!/usr/bin/env bash
# Real paired inference and shared-axis trajectory figures for both evidence lines.
# Usage: bash scripts/generate_mos_trajectory_comparisons.sh [release|from-scratch|all]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-all}"
[[ "$MODE" =~ ^(release|from-scratch|all)$ ]] || {
  echo "Usage: $0 [release|from-scratch|all]" >&2; exit 2;
}
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NO_ALBUMENTATIONS_UPDATE=1
SYNTH="${SYNTH_EVAL_DATASET:-data/synth_mixture_eval_db7_17d}"
DROID="${DROID_EVAL_DATASET:-data/droid_subset_eval}"
for dataset in "$SYNTH" "$DROID"; do
  [[ -f "$dataset/meta/info.json" ]] || {
    echo "Missing $dataset; run: uv run python scripts/prepare_mos_data.py --output-dir data" >&2; exit 1;
  }
done

run_pair() {
  local semantic="$1" gaussian="$2" domain="$3" out="$4"
  local dataset="$DROID" extra=()
  [[ -f "$semantic/config.json" && -f "$gaussian/config.json" ]] || {
    echo "Missing real checkpoint: $semantic or $gaussian" >&2; exit 1;
  }
  if [[ "$domain" == synth ]]; then
    dataset="$SYNTH"
    extra=(--segment-meta-path "$dataset/meta/segment_schedule.jsonl" --high-level-task-from-meta)
  fi
  local complete=1
  [[ -f "$out/compare_metrics.json" ]] || complete=0
  for tid in {0..9}; do [[ -f "$out/traj_${tid}_trajectory.npz" ]] || complete=0; done
  if [[ "$complete" == 0 ]]; then
    uv run python gr00t/eval/compare_flow_sources.py \
      --dataset-path "$dataset" \
      --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
      --model-paths "$semantic" "$gaussian" --labels mos_best mos_best_no_mem \
      --traj-ids 0 1 2 3 4 5 6 7 8 9 --steps 200 --execution-horizon 16 \
      --denoising-steps 4 --history-source pred \
      --force-fixed-beta --fixed-beta-value 0.0 --seed 0 \
      --save-trajectory-arrays "${extra[@]}" --out-dir "$out" 2>&1 | tee "${out}_log.txt"
  else
    echo "Reusing complete paired evaluation: $out"
  fi
}

draw_pair() {
  local semantic="$1" gaussian="$2" domain="$3" out="$4" asset_dir="$5" prefix="$6"
  uv run python scripts/generate_mos_release_trajectory_assets.py \
    --evaluation-dir "$out" \
    --semantic-label mos_best --semantic-model "$semantic" \
    --gaussian-label mos_best_no_mem --gaussian-model "$gaussian" \
    --domain "$domain" --dims 0 1 3 4 \
    --output-dir "$asset_dir" --output-prefix "$prefix"
}

if [[ "$MODE" == release || "$MODE" == all ]]; then
  Z="$PWD/release/mos_best"
  G="$PWD/release/mos_best_no_mem"
  for domain in synth droid; do
    OUT="runs/trajectory_release_${domain}_es0"
    run_pair "$Z" "$G" "$domain" "$OUT"
    draw_pair "$Z" "$G" "$domain" "$OUT" release/assets mos_action_trajectories
  done
fi

if [[ "$MODE" == from-scratch || "$MODE" == all ]]; then
  Z="$PWD/runs/db28z_s2/checkpoint-5000"
  G="$PWD/runs/db28g_s2/checkpoint-5000"
  for domain in synth droid; do
    OUT="runs/trajectory_db28_${domain}_es0"
    run_pair "$Z" "$G" "$domain" "$OUT"
    draw_pair "$Z" "$G" "$domain" "$OUT" assets db28_action_trajectories
    if [[ "$domain" == synth ]]; then
      # Explicit clear-gap example; never replaces representative or aggregate results.
      uv run python scripts/generate_mos_release_trajectory_assets.py \
        --evaluation-dir "$OUT" \
        --semantic-label mos_best --semantic-model "$Z" \
        --gaussian-label mos_best_no_mem --gaussian-model "$G" \
        --domain synth --dims 0 1 3 4 --selection max-ratio \
        --output-dir assets --output-prefix db28_action_trajectories_clear_gap
    fi
  done
fi
