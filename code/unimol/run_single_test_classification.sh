#!/bin/bash
set -Eeuo pipefail

SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SCRIPT_DIR="$(cd "${SCRIPT_DIR}" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-unimol}"
: "${DATA_ROOT:?Set DATA_ROOT to the prepared dataset-split directory}"
DATA_ROOT="$(cd "${DATA_ROOT}" && pwd)"
SHARED_CACHE_ROOT="${SHARED_CACHE_ROOT:-${PROJECT_ROOT}/outputs/unimol/shared_3d_cache/UniMol_v1_noH_conf11_seed0}"
[[ "${SHARED_CACHE_ROOT}" = /* ]] || SHARED_CACHE_ROOT="${PWD}/${SHARED_CACHE_ROOT}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/unimol/test_classification}"
[[ "${OUT_ROOT}" = /* ]] || OUT_ROOT="${PWD}/${OUT_ROOT}"
DATASET="${DATASET:-1_Staphylococcus_median_16_MolWt_50_1000_value_le_8192_multitask}"
SPLIT_TYPE="${SPLIT_TYPE:-random}"
SEED="${SEED:-0}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniforge3/etc/profile.d/conda.sh"
else
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV}"

RUN_DIR="${DATA_ROOT}/${DATASET}/classification/${SPLIT_TYPE}/seed_${SEED}"
SHARED_CACHE_DIR="${SHARED_CACHE_ROOT}/${DATASET}"
OUT_DIR="${OUT_ROOT}/Uni-Mol/${DATASET}/classification/${SPLIT_TYPE}/seed_${SEED}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
python "${SCRIPT_DIR}/train_unimol_classification_antibioticbench.py" \
  --train-csv "${RUN_DIR}/train.csv" \
  --validation-csv "${RUN_DIR}/validation.csv" \
  --test-csv "${RUN_DIR}/test.csv" \
  --out-dir "${OUT_DIR}" \
  --shared-cache-dir "${SHARED_CACHE_DIR}" \
  --dataset-name "${DATASET}" \
  --split-type "${SPLIT_TYPE}" \
  --seed "${SEED}" \
  --epochs 3 \
  --patience 2 \
  --batch-size 8 \
  --eval-batch-size 16 \
  --conf-size 11 \
  --conformer-seed 0 \
  --num-workers 2 \
  --cpu-threads 1
