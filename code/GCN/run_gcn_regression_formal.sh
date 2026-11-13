#!/usr/bin/env bash
set -euo pipefail

# AntibioticBench GCN regression: 12 datasets × 4 split types
# random/scaffold/repeated_stratified_perimeter: seeds 0-4
# strict_perimeter: seed 0 only
# Default physical GPU: 1. Override with: GPU_ID=3 bash run_gcn_regression_formal.sh

GPU_ID="${GPU_ID:-1}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PROJECT_DIR="$(cd "${PROJECT_DIR}" && pwd)"
DATA_ROOT="${DATA_ROOT:-${BENCHMARK_ROOT:+${BENCHMARK_ROOT}/benchmark_data_20260706}}"
: "${DATA_ROOT:?Set DATA_ROOT to the directory containing prepared benchmark splits}"
[[ -d "$DATA_ROOT" ]] || { echo "ERROR: DATA_ROOT does not exist" >&2; exit 1; }
DATA_ROOT="$(cd "$DATA_ROOT" && pwd)"
OUT_ROOT="${OUT_ROOT:-${BENCHMARK_ROOT:-${PROJECT_DIR}/../..}/outputs/GCN/output_regression}"
[[ "$OUT_ROOT" = /* ]] || OUT_ROOT="$PWD/$OUT_ROOT"
CONFIG="${CONFIG:-$PROJECT_DIR/dataset_config.csv}"
[[ "${CONFIG}" = /* ]] || CONFIG="${PWD}/${CONFIG}"

mkdir -p "$OUT_ROOT"
cd "$PROJECT_DIR"

if [[ -n "${CONDA_SH:-}" ]]; then
  source "${CONDA_SH}"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
else
  echo "ERROR: conda not found." >&2
  exit 1
fi
conda activate "${CONDA_ENV:-gcn_ab}"

LOG_FILE="$OUT_ROOT/gcn_regression_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log"

echo "GPU_ID=$GPU_ID"

CUDA_VISIBLE_DEVICES="$GPU_ID" python train_gcn_antibioticbench.py \
  --data-root "$DATA_ROOT" \
  --out-root "$OUT_ROOT" \
  --model-name GCN \
  --dataset-config "$CONFIG" \
  --tasks regression \
  --split-types random,scaffold,repeated_stratified_perimeter,strict_perimeter \
  --seeds 0,1,2,3,4 \
  --strict-perimeter-seeds 0 \
  --epochs 300 \
  --patience 40 \
  --batch-size 128 \
  --eval-batch-size 512 \
  --num-layers 4 \
  --hidden-dim 256 \
  --mlp-hidden-dim 256 \
  --dropout 0.15 \
  --pooling mean \
  --lr 1e-3 \
  --min-lr 1e-6 \
  --weight-decay 1e-5 \
  --grad-clip 5.0 \
  --select-metric val_rmse \
  --log-every-batches 50 \
  2>&1 | tee "$LOG_FILE"
