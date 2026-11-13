#!/bin/bash
set -euo pipefail
SCRIPT_DIR="${SCRIPT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
[[ "${SCRIPT_DIR}" = /* ]] || SCRIPT_DIR="${PWD}/${SCRIPT_DIR}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
[[ "${BENCHMARK_ROOT}" = /* ]] || BENCHMARK_ROOT="${PWD}/${BENCHMARK_ROOT}"
OUT_ROOT="${OUT_ROOT:-${BENCHMARK_ROOT}/outputs/KPGT/classification}"
[[ "${OUT_ROOT}" = /* ]] || OUT_ROOT="${PWD}/${OUT_ROOT}"
CONDA_ENV="${CONDA_ENV:-KPGT}"
ALLOW_INCOMPLETE="${ALLOW_INCOMPLETE:-0}"
COMPUTE_LANDSCAPE="${COMPUTE_LANDSCAPE:-0}"
DATA_ROOT="${DATA_ROOT:-}"
if [[ "${COMPUTE_LANDSCAPE}" == "1" && -z "${DATA_ROOT}" ]]; then
  echo "Set DATA_ROOT to the prepared split-data directory when COMPUTE_LANDSCAPE=1" >&2; exit 2
fi
[[ -z "${DATA_ROOT}" || "${DATA_ROOT}" = /* ]] || DATA_ROOT="${PWD}/${DATA_ROOT}"
DATASET_CONFIG="${DATASET_CONFIG:-${SCRIPT_DIR}/dataset_config.csv}"
[[ "${DATASET_CONFIG}" = /* ]] || DATASET_CONFIG="${PWD}/${DATASET_CONFIG}"
if [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniforge3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
else
  eval "$(conda shell.bash hook)"
fi
conda activate "${CONDA_ENV}"
cd "${SCRIPT_DIR}"
cmd=(python -u summarize_kpgt_outputs.py --task classification --out-root "${OUT_ROOT}" --expected-runs 192)
[[ "${ALLOW_INCOMPLETE}" == "1" ]] && cmd+=(--allow-incomplete)
"${cmd[@]}"

if [[ "${COMPUTE_LANDSCAPE}" == "1" ]]; then
  python -u compute_landscape_roughness.py \
    --task classification \
    --data-root "${DATA_ROOT}" \
    --dataset-config "${DATASET_CONFIG}" \
    --outdir "${OUT_ROOT}/KPGT/_landscape_roughness"
fi
