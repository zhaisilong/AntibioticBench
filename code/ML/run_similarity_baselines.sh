#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_similarity_baselines.sh path/to/benchmark_data_20260706 path/to/output_dir
# Optional environment variables:
#   TASKS="classification regression"              # or only one task
#   MODELS="1nn 5nn"
#   SPLIT_TYPES="random scaffold repeated_stratified_perimeter strict_perimeter"
#   SEEDS="0 1 2 3 4"
#   DATASETS="Staphylococcus Escherichia"          # optional; empty means all datasets
#   WEIGHTING="similarity"                         # similarity or uniform
#   CLASSIFICATION_TARGET_COL="active"
#   REGRESSION_TARGET_COL="log10_mic"
#   VALUE_COL="mic_numeric"
#   ACTIVE_COL="active"
#   CLASSIFICATION_RANK_METRIC="test_pr_auc"
#   REGRESSION_RANK_METRIC="test_rmse"

DATA_ROOT=${1:-${DATA_ROOT:-}}
: "${DATA_ROOT:?Provide DATA_ROOT or the first argument}"
[[ -d "${DATA_ROOT}" ]] || { echo "ERROR: DATA_ROOT does not exist" >&2; exit 1; }
DATA_ROOT="$(cd "${DATA_ROOT}" && pwd)"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
OUT_ROOT=${2:-${OUT_ROOT:-${REPO_ROOT}/outputs/ML/similarity}}

TASKS=${TASKS:-"classification regression"}
MODELS=${MODELS:-"1nn 5nn"}
SPLIT_TYPES=${SPLIT_TYPES:-"random scaffold repeated_stratified_perimeter strict_perimeter"}
SEEDS=${SEEDS:-"0 1 2 3 4"}
DATASETS=${DATASETS:-""}
WEIGHTING=${WEIGHTING:-"similarity"}
CLASSIFICATION_TARGET_COL=${CLASSIFICATION_TARGET_COL:-""}
REGRESSION_TARGET_COL=${REGRESSION_TARGET_COL:-""}
VALUE_COL=${VALUE_COL:-""}
ACTIVE_COL=${ACTIVE_COL:-""}
CLASSIFICATION_RANK_METRIC=${CLASSIFICATION_RANK_METRIC:-"test_pr_auc"}
REGRESSION_RANK_METRIC=${REGRESSION_RANK_METRIC:-"test_rmse"}

mkdir -p "$OUT_ROOT/logs"

echo "TASKS=$TASKS"
echo "MODELS=$MODELS"
echo "SPLIT_TYPES=$SPLIT_TYPES"
echo "SEEDS=$SEEDS"
echo "DATASETS=${DATASETS:-ALL}"
echo "WEIGHTING=$WEIGHTING"
echo "CLASSIFICATION_TARGET_COL=${CLASSIFICATION_TARGET_COL:-AUTO}"
echo "REGRESSION_TARGET_COL=${REGRESSION_TARGET_COL:-AUTO}"
echo "VALUE_COL=${VALUE_COL:-AUTO}"
echo "ACTIVE_COL=${ACTIVE_COL:-AUTO}"

for TASK in $TASKS; do
  echo "===== Running similarity baselines task: $TASK ====="
  CMD=(python "$SCRIPT_DIR/train_similarity_baselines.py"
    --data-root "$DATA_ROOT"
    --outdir "$OUT_ROOT"
    --task "$TASK"
    --models $MODELS
    --split-types $SPLIT_TYPES
    --seeds $SEEDS
    --weighting "$WEIGHTING")

  if [[ -n "$DATASETS" ]]; then
    CMD+=(--datasets $DATASETS)
  fi
  if [[ -n "$CLASSIFICATION_TARGET_COL" ]]; then
    CMD+=(--classification-target-col "$CLASSIFICATION_TARGET_COL")
  fi
  if [[ -n "$REGRESSION_TARGET_COL" ]]; then
    CMD+=(--regression-target-col "$REGRESSION_TARGET_COL")
  fi
  if [[ -n "$VALUE_COL" ]]; then
    CMD+=(--value-col "$VALUE_COL")
  fi
  if [[ -n "$ACTIVE_COL" ]]; then
    CMD+=(--active-col "$ACTIVE_COL")
  fi

  "${CMD[@]}" 2>&1 | tee "$OUT_ROOT/logs/${TASK}_similarity_baselines.log"
done

python "$SCRIPT_DIR/summarize_similarity_baselines.py" \
  --results-root "$OUT_ROOT" \
  --classification-rank-metric "$CLASSIFICATION_RANK_METRIC" \
  --regression-rank-metric "$REGRESSION_RANK_METRIC" \
  2>&1 | tee "$OUT_ROOT/logs/summary.log"

echo "All similarity baseline runs done."
