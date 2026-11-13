#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_ml.sh path/to/benchmark_data20260703 path/to/output_dir
# Optional environment variables:
#   SPLIT_TYPES="random scaffold repeated_stratified_perimeter strict_perimeter"
#   SEEDS="0 1 2 3 4"
#   DATASETS="Staphylococcus Escherichia"   # optional; empty means all datasets
#   COMPUTE_LANDSCAPE=1                       # run model-independent MODI/logMIC roughness
#
# Summary outputs include:
#   _combined_summary/leaderboard_mean_sd.csv
#   _combined_summary/ood_drop_summary.csv
#   _combined_summary/perimeter_ood_error_analysis.csv  # dedicated diagnosis 12.4

DATA_ROOT=${1:-${DATA_ROOT:-}}
: "${DATA_ROOT:?Provide DATA_ROOT or the first argument}"
[[ -d "${DATA_ROOT}" ]] || { echo "ERROR: DATA_ROOT does not exist" >&2; exit 1; }
DATA_ROOT="$(cd "${DATA_ROOT}" && pwd)"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
OUT_ROOT=${2:-${OUT_ROOT:-${REPO_ROOT}/outputs/ML/classification}}

SPLIT_TYPES=${SPLIT_TYPES:-"random scaffold repeated_stratified_perimeter strict_perimeter"}
SEEDS=${SEEDS:-"0 1 2 3 4"}
DATASETS=${DATASETS:-""}
MODELS=${MODELS:-"rf svm xgb lightgbm"}

mkdir -p "$OUT_ROOT/logs"

echo "MODELS=$MODELS"
echo "SPLIT_TYPES=$SPLIT_TYPES"
echo "SEEDS=$SEEDS"

for MODEL in $MODELS; do
  echo "===== Running model: $MODEL ====="
  if [[ -n "$DATASETS" ]]; then
    python "$SCRIPT_DIR/train_ml.py" \
      --data-root "$DATA_ROOT" \
      --outdir "$OUT_ROOT" \
      --model "$MODEL" \
      --split-types $SPLIT_TYPES \
      --seeds $SEEDS \
      --datasets $DATASETS \
      2>&1 | tee "$OUT_ROOT/logs/${MODEL}.log"
  else
    python "$SCRIPT_DIR/train_ml.py" \
      --data-root "$DATA_ROOT" \
      --outdir "$OUT_ROOT" \
      --model "$MODEL" \
      --split-types $SPLIT_TYPES \
      --seeds $SEEDS \
      2>&1 | tee "$OUT_ROOT/logs/${MODEL}.log"
  fi
done

python "$SCRIPT_DIR/summarize_ml.py" \
  --results-root "$OUT_ROOT" \
  --rank-metric test_pr_auc \
  2>&1 | tee "$OUT_ROOT/logs/summary.log"

if [[ "${COMPUTE_LANDSCAPE:-0}" == "1" ]]; then
  python "$SCRIPT_DIR/roughness.py" \
    --data-root "$DATA_ROOT" \
    --outdir "$OUT_ROOT/_landscape_roughness" \
    --split-types $SPLIT_TYPES \
    --seeds $SEEDS \
    2>&1 | tee "$OUT_ROOT/logs/landscape_roughness.log"
fi

echo "All done."
