#!/usr/bin/env bash
set -euo pipefail

BENCHMARK_ROOT="${BENCHMARK_ROOT:-}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PROJECT_DIR="$(cd "${PROJECT_DIR}" && pwd)"
OUT_ROOT="${OUT_ROOT:-${BENCHMARK_ROOT:-${PROJECT_DIR}/../..}/outputs/GIN/output_classification}"
[[ "${OUT_ROOT}" = /* ]] || OUT_ROOT="${PWD}/${OUT_ROOT}"
MODEL_NAME="${MODEL_NAME:-GINE}"
CONDA_ENV="${CONDA_ENV:-chemprop-repro}"

cd "${PROJECT_DIR}"
if [[ -n "${CONDA_SH:-}" ]]; then
  source "${CONDA_SH}"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "ERROR: configure CONDA_SH or make conda available on PATH" >&2; exit 1
fi
conda activate "${CONDA_ENV}"


# Preserve existing diagnostics; this postprocess only writes its own outputs.
python summarize_gine_outputs.py \
  --out-root "${OUT_ROOT}" \
  --model-name "${MODEL_NAME}"

python activity_cliff_pair_analysis.py \
  --out-root "${OUT_ROOT}" \
  --model-name "${MODEL_NAME}" \
  --scopes test_test \
  --sim-cutoff 0.7 \
  --fold-change 16


python validate_gine_outputs.py \
  --out-root "${OUT_ROOT}" \
  --task classification \
  --strict

echo "GINE classification postprocess finished.."
