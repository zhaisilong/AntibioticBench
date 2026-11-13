#!/usr/bin/env bash
set -euo pipefail

activate_conda_env() {
  local env_name="$1"
  local conda_sh="${CONDA_SH:-}"
  if [[ -z "${conda_sh}" ]]; then
    for candidate in \
      "${HOME}/miniforge3/etc/profile.d/conda.sh" \
      "${HOME}/miniconda3/etc/profile.d/conda.sh" \
      "${HOME}/anaconda3/etc/profile.d/conda.sh"; do
      if [[ -f "${candidate}" ]]; then
        conda_sh="${candidate}"
        break
      fi
    done
  fi
  if [[ -z "${conda_sh}" || ! -f "${conda_sh}" ]]; then
    echo "ERROR: conda.sh not found. Set CONDA_SH to your conda initialization script." >&2
    exit 1
  fi
  source "${conda_sh}"
  conda activate "${env_name}"
}

# Resolve code from this launcher; user-supplied relative paths use the caller's directory.
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
PROJECT_DIR="$(cd -- "${PROJECT_DIR}" && pwd)"
REPO_ROOT="$(cd -- "${PROJECT_DIR}/../.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${BENCHMARK_ROOT:-${REPO_ROOT}}/outputs/AttentiveFP/classification}"
case "${OUT_ROOT}" in /*) ;; *) OUT_ROOT="${PWD}/${OUT_ROOT}" ;; esac
MODEL_NAME="${MODEL_NAME:-AttentiveFP}"; CONDA_ENV="${CONDA_ENV:-AttentiveFP}"
cd "${PROJECT_DIR}"; activate_conda_env "${CONDA_ENV}"
python summarize_attentivefp_outputs.py --out-root "${OUT_ROOT}" --model-name "${MODEL_NAME}"
python activity_cliff_pair_analysis.py --out-root "${OUT_ROOT}" --model-name "${MODEL_NAME}" --scopes test_test
python validate_attentivefp_outputs.py --out-root "${OUT_ROOT}" --task classification --strict
echo "AttentiveFP classification postprocess finished."
