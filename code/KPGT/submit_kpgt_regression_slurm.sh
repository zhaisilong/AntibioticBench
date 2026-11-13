#!/bin/bash
set -euo pipefail

SCRIPT_DIR="${SCRIPT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
[[ "${SCRIPT_DIR}" = /* ]] || SCRIPT_DIR="${PWD}/${SCRIPT_DIR}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
[[ "${BENCHMARK_ROOT}" = /* ]] || BENCHMARK_ROOT="${PWD}/${BENCHMARK_ROOT}"
OUT_ROOT="${OUT_ROOT:-${BENCHMARK_ROOT}/outputs/KPGT/regression}"
[[ "${OUT_ROOT}" = /* ]] || OUT_ROOT="${PWD}/${OUT_ROOT}"
RUN_MODE="${RUN_MODE:-batch}"
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"
POSTPROCESS="${POSTPROCESS:-1}"
ARRAY_SPEC="${ARRAY_SPEC:-}"
EXCLUDE_NODES="${EXCLUDE_NODES:-}"

DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the prepared split-data directory}"
[[ "${DATA_ROOT}" = /* ]] || DATA_ROOT="${PWD}/${DATA_ROOT}"
KPGT_ROOT="${KPGT_ROOT:?Set KPGT_ROOT to the external KPGT source directory}"
[[ "${KPGT_ROOT}" = /* ]] || KPGT_ROOT="${PWD}/${KPGT_ROOT}"
if [[ -n "${KPGT_PRETRAINED_MODEL:-}" && "${KPGT_PRETRAINED_MODEL}" != /* ]]; then
  KPGT_PRETRAINED_MODEL="${PWD}/${KPGT_PRETRAINED_MODEL}"
  export KPGT_PRETRAINED_MODEL
fi
export KPGT_ROOT
export BENCHMARK_ROOT SCRIPT_DIR OUT_ROOT DATA_ROOT
mkdir -p "${OUT_ROOT}/slurm_logs"
cd "${SCRIPT_DIR}"

if [[ -z "${ARRAY_SPEC}" ]]; then
  case "${RUN_MODE}" in
    test)  ARRAY_SPEC="0-0" ;;
    batch) ARRAY_SPEC="0-191%${MAX_CONCURRENT}" ;;
    *) echo "ERROR: RUN_MODE must be test or batch" >&2; exit 2 ;;
  esac
fi

SBATCH_EXTRA=()
[[ -n "${EXCLUDE_NODES}" ]] && SBATCH_EXTRA+=(--exclude="${EXCLUDE_NODES}")

echo "Submitting KPGT regression array: ${ARRAY_SPEC}"
JOBID=$(sbatch --parsable "${SBATCH_EXTRA[@]}" \
  --array="${ARRAY_SPEC}" \
  --output="${OUT_ROOT}/slurm_logs/%x_%A_%a.out" \
  --error="${OUT_ROOT}/slurm_logs/%x_%A_%a.err" \
  --export=ALL \
  kpgt_regression_array.slurm)
JOBID="${JOBID%%;*}"
echo "Submitted regression array job: ${JOBID}"

if [[ "${POSTPROCESS}" == "1" && "${RUN_MODE}" == "batch" ]]; then
  printf -v summary_cmd 'bash %q' "${SCRIPT_DIR}/postprocess_kpgt_regression.sh"
  SUMMARY_JOBID=$(sbatch --parsable "${SBATCH_EXTRA[@]}" \
    --job-name=kpgt_reg_sum \
    --dependency="afterok:${JOBID}" \
    --cpus-per-task=4 --mem=16G --time=04:00:00 \
    --output="${OUT_ROOT}/slurm_logs/kpgt_reg_summary_%j.out" \
    --error="${OUT_ROOT}/slurm_logs/kpgt_reg_summary_%j.err" \
    --export=ALL \
    --wrap="${summary_cmd}")
  SUMMARY_JOBID="${SUMMARY_JOBID%%;*}"
  echo "Submitted dependent regression summary job: ${SUMMARY_JOBID}"
fi
