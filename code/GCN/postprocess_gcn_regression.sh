#!/usr/bin/env bash
set -euo pipefail

# Summarize regression-only GCN outputs.
BENCHMARK_ROOT="${BENCHMARK_ROOT:-}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PROJECT_DIR="$(cd "${PROJECT_DIR}" && pwd)"
OUT_ROOT="${OUT_ROOT:-${BENCHMARK_ROOT:-${PROJECT_DIR}/../..}/outputs/GCN/output_regression}"
[[ "${OUT_ROOT}" = /* ]] || OUT_ROOT="${PWD}/${OUT_ROOT}"
MODEL_NAME="${MODEL_NAME:-GCN}"
CONDA_ENV="${CONDA_ENV:-gcn_ab}"
RUN_AC_PAIR_ANALYSIS="${RUN_AC_PAIR_ANALYSIS:-1}"
MAX_PAIRS_PER_RUN="${MAX_PAIRS_PER_RUN:-0}"
SKIP_PAIR_FILES="${SKIP_PAIR_FILES:-0}"

activate_conda() {
  if [[ -n "${CONDA_SH:-}" ]]; then
    source "${CONDA_SH}"
  elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/miniforge3/etc/profile.d/conda.sh"
  elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
  else
    echo "ERROR: conda initialization script not found." >&2
    exit 1
  fi
  conda activate "${CONDA_ENV}"
}

cd "${PROJECT_DIR}"
activate_conda

python summarize_gcn_outputs.py \
  --out-root "${OUT_ROOT}" \
  --model-name "${MODEL_NAME}"

if [[ "${RUN_AC_PAIR_ANALYSIS}" == "1" ]]; then
  AC_ARGS=(--out-root "${OUT_ROOT}" --model-name "${MODEL_NAME}" --scopes all_internal,test_test --sim-cutoff 0.7 --fold-change 16 --max-pairs-per-run "${MAX_PAIRS_PER_RUN}")
  if [[ "${SKIP_PAIR_FILES}" == "1" ]]; then
    AC_ARGS+=(--skip-pair-files)
  fi
  python activity_cliff_pair_analysis.py "${AC_ARGS[@]}"
fi

echo "Regression postprocess finished.."
