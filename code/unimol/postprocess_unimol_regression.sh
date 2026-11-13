#!/bin/bash
set -Eeuo pipefail
SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SCRIPT_DIR="$(cd "${SCRIPT_DIR}" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/unimol/regression}"
[[ "${OUT_ROOT}" = /* ]] || OUT_ROOT="${PWD}/${OUT_ROOT}"
CONDA_ENV="${CONDA_ENV:-unimol}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniforge3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV}"

python "${SCRIPT_DIR}/summarize_unimol_outputs.py" \
  --out-root "${OUT_ROOT}" \
  --task regression \
  --expected-runs 192
