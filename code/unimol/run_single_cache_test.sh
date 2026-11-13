#!/bin/bash
set -Eeuo pipefail

SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SCRIPT_DIR="$(cd "${SCRIPT_DIR}" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-unimol}"
UNIMOL_REPO="${UNIMOL_REPO:-}"
if [[ -n "${UNIMOL_REPO}" ]]; then
  UNIMOL_REPO="$(cd "${UNIMOL_REPO}" && pwd)"
fi
: "${DATA_ROOT:?Set DATA_ROOT to the prepared dataset-split directory}"
DATA_ROOT="$(cd "${DATA_ROOT}" && pwd)"
SHARED_CACHE_ROOT="${SHARED_CACHE_ROOT:-${PROJECT_ROOT}/outputs/unimol/shared_3d_cache/UniMol_v1_noH_conf11_seed0}"
[[ "${SHARED_CACHE_ROOT}" = /* ]] || SHARED_CACHE_ROOT="${PWD}/${SHARED_CACHE_ROOT}"
DATASET="${DATASET:-1_Staphylococcus_median_16_MolWt_50_1000_value_le_8192_multitask}"
CPU_WORKERS="${CPU_WORKERS:-8}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniforge3/etc/profile.d/conda.sh"
else
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV}"

export PYTHONPATH="${SCRIPT_DIR}${UNIMOL_REPO:+:${UNIMOL_REPO}}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python "${SCRIPT_DIR}/precompute_unimol_shared_3d_cache.py" \
  --dataset-dir "${DATA_ROOT}/${DATASET}" \
  --cache-dir "${SHARED_CACHE_ROOT}/${DATASET}" \
  --conf-size 11 \
  --conformer-seed 0 \
  --cpu-workers "${CPU_WORKERS}" \
  --mode fast \
  --max-atoms 256
