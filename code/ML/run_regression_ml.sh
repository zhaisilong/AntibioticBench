#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_regression_ml.sh path/to/benchmark_data_20260706 path/to/output_dir
# Optional environment variables:
#   MODELS="rf xgb lightgbm"                 # svm is slow; recommended to run separately
#   SPLIT_TYPES="random scaffold repeated_stratified_perimeter strict_perimeter"
#   SEEDS="0 1 2 3 4"
#   DATASETS="Staphylococcus Escherichia"   # optional; empty means all datasets
#   TARGET_COL="log10_mic"                  # optional; auto-detected if empty
#   VALUE_COL="mic_numeric"                 # optional; used to compute log10MIC if no log10 target exists
#   ACTIVE_COL="active"                     # optional; used for EF/Hit/Recall; otherwise inferred by MIC <= threshold
#   RANK_METRIC="test_rmse"                 # lower is better by default

DATA_ROOT=${1:-${DATA_ROOT:-}}
: "${DATA_ROOT:?Provide DATA_ROOT or the first argument}"
[[ -d "${DATA_ROOT}" ]] || { echo "ERROR: DATA_ROOT does not exist" >&2; exit 1; }
DATA_ROOT="$(cd "${DATA_ROOT}" && pwd)"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
OUT_ROOT=${2:-${OUT_ROOT:-${REPO_ROOT}/outputs/ML/regression}}

SPLIT_TYPES=${SPLIT_TYPES:-"random scaffold repeated_stratified_perimeter strict_perimeter"}
SEEDS=${SEEDS:-"0 1 2 3 4"}
DATASETS=${DATASETS:-""}
MODELS=${MODELS:-"rf xgb lightgbm"}
TARGET_COL=${TARGET_COL:-""}
VALUE_COL=${VALUE_COL:-""}
ACTIVE_COL=${ACTIVE_COL:-""}
RANK_METRIC=${RANK_METRIC:-"test_rmse"}

mkdir -p "$OUT_ROOT/logs"

echo "MODELS=$MODELS"
echo "SPLIT_TYPES=$SPLIT_TYPES"
echo "SEEDS=$SEEDS"
echo "DATASETS=${DATASETS:-ALL}"
echo "TARGET_COL=${TARGET_COL:-AUTO}"
echo "VALUE_COL=${VALUE_COL:-AUTO}"
echo "ACTIVE_COL=${ACTIVE_COL:-AUTO}"
echo "RANK_METRIC=$RANK_METRIC"

for MODEL in $MODELS; do
  echo "===== Running regression model: $MODEL ====="

  CMD=(python "$SCRIPT_DIR/train_regression_ml.py"
    --data-root "$DATA_ROOT"
    --outdir "$OUT_ROOT"
    --model "$MODEL"
    --split-types $SPLIT_TYPES
    --seeds $SEEDS)

  if [[ -n "$DATASETS" ]]; then
    CMD+=(--datasets $DATASETS)
  fi
  if [[ -n "$TARGET_COL" ]]; then
    CMD+=(--target-col "$TARGET_COL")
  fi
  if [[ -n "$VALUE_COL" ]]; then
    CMD+=(--value-col "$VALUE_COL")
  fi
  if [[ -n "$ACTIVE_COL" ]]; then
    CMD+=(--active-col "$ACTIVE_COL")
  fi

  "${CMD[@]}" 2>&1 | tee "$OUT_ROOT/logs/${MODEL}.log"
done

python "$SCRIPT_DIR/summarize_regression_ml.py" \
  --results-root "$OUT_ROOT" \
  --rank-metric "$RANK_METRIC" \
  2>&1 | tee "$OUT_ROOT/logs/summary.log"

if [[ "${COMPUTE_LANDSCAPE:-0}" == "1" ]]; then
  echo "===== Running regression molecular landscape roughness analysis ====="

  ROUGHNESS_CMD=(python "$SCRIPT_DIR/roughness_regression.py"
    --data-root "$DATA_ROOT"
    --outdir "$OUT_ROOT/_regression_landscape_roughness"
    --split-types $SPLIT_TYPES
    --seeds $SEEDS)

  if [[ -n "$DATASETS" ]]; then
    ROUGHNESS_CMD+=(--datasets $DATASETS)
  fi

  if [[ "${SAVE_MOLECULE_LEVEL_ROUGHNESS:-0}" == "1" ]]; then
    ROUGHNESS_CMD+=(--save-molecule-level)
  fi

  "${ROUGHNESS_CMD[@]}" 2>&1 | tee "$OUT_ROOT/logs/regression_landscape_roughness.log"
fi

echo "All regression runs done."
