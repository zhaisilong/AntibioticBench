#!/bin/bash
set -Eeuo pipefail
SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SCRIPT_DIR="$(cd "${SCRIPT_DIR}" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
: "${DATA_ROOT:?Set DATA_ROOT to the prepared split-data directory}"
[[ -d "${DATA_ROOT}" ]] || { echo "ERROR: DATA_ROOT does not exist" >&2; exit 1; }
DATA_ROOT="$(cd "${DATA_ROOT}" && pwd)"
WORK_DIR="${WORK_DIR:-${PROJECT_ROOT}}"
[[ -d "${WORK_DIR}" ]] || { echo "ERROR: WORK_DIR does not exist" >&2; exit 1; }
WORK_DIR="$(cd "${WORK_DIR}" && pwd)"
CONDA_SH="${CONDA_SH:-}"
CONDA_ENV="${CONDA_ENV:-chemprop-repro}"
CHEMPROP_EXECUTABLE="${CHEMPROP_EXECUTABLE:-chemprop}"

GPU_ID="${GPU_ID:-0}"

if [[ -n "${CONDA_SH}" ]]; then
  [[ -f "${CONDA_SH}" ]] || { echo "ERROR: CONDA_SH does not exist" >&2; exit 1; }
  source "${CONDA_SH}"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "ERROR: configure CONDA_SH or make conda available on PATH" >&2; exit 1
fi
conda activate "${CONDA_ENV}"
python --version
command -v chemprop >/dev/null
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
python - <<'PY'
import chemprop, torch, rdkit, numpy, pandas, scipy, sklearn
print("chemprop:", getattr(chemprop, "__version__", "unknown"))
print("torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("visible GPUs:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable")
print("cuda:0:", torch.cuda.get_device_name(0))
PY

echo -n "classification run count: "
python "${SCRIPT_DIR}/train_chemprop_classification_benchmark.py" \
  --data-root "${DATA_ROOT}" \
  --out-root "${PROJECT_ROOT}/outputs/chemprop/classification" \
  --print-run-count

echo -n "regression run count: "
python "${SCRIPT_DIR}/train_chemprop_regression_benchmark.py" \
  --data-root "${DATA_ROOT}" \
  --out-root "${PROJECT_ROOT}/outputs/chemprop/regression" \
  --print-run-count

echo "PRE-FLIGHT CHECK PASSED"
