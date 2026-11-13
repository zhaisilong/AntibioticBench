#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="${SCRIPT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
[[ "${SCRIPT_DIR}" = /* ]] || SCRIPT_DIR="${PWD}/${SCRIPT_DIR}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
[[ "${BENCHMARK_ROOT}" = /* ]] || BENCHMARK_ROOT="${PWD}/${BENCHMARK_ROOT}"
OUT_ROOT="${OUT_ROOT:-${BENCHMARK_ROOT}/outputs/MolFormer/regression}"
[[ "${OUT_ROOT}" = /* ]] || OUT_ROOT="${PWD}/${OUT_ROOT}"
MOLFORMER_ROOT="${MOLFORMER_ROOT:-${BENCHMARK_ROOT}/.cache/molformer}"
[[ "${MOLFORMER_ROOT}" = /* ]] || MOLFORMER_ROOT="${PWD}/${MOLFORMER_ROOT}"
CONDA_ENV="${CONDA_ENV:-molformer}"
MODEL_ID="${MODEL_ID:-ibm-research/MoLFormer-XL-both-10pct}"
MODEL_REVISION="${MODEL_REVISION:-compat-v4}"
RUN_MODE="${RUN_MODE:-batch}"
MAX_CONCURRENT="${MAX_CONCURRENT:-1}"
POSTPROCESS="${POSTPROCESS:-1}"
PREFETCH_MODEL="${PREFETCH_MODEL:-1}"
ARRAY_SPEC="${ARRAY_SPEC:-}"
EXCLUDE_NODES="${EXCLUDE_NODES:-}"
export HF_HOME="${HF_HOME:-${MOLFORMER_ROOT}/hf_cache}"
[[ "${HF_HOME}" = /* ]] || export HF_HOME="${PWD}/${HF_HOME}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
[[ "${HF_HUB_CACHE}" = /* ]] || export HF_HUB_CACHE="${PWD}/${HF_HUB_CACHE}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the prepared split-data directory}"
[[ "${DATA_ROOT}" = /* ]] || DATA_ROOT="${PWD}/${DATA_ROOT}"
export BENCHMARK_ROOT SCRIPT_DIR OUT_ROOT DATA_ROOT MOLFORMER_ROOT
mkdir -p "${OUT_ROOT}/slurm_logs" "${HF_HUB_CACHE}"
cd "${SCRIPT_DIR}"
if [[ -z "${ARRAY_SPEC}" ]]; then
  case "${RUN_MODE}" in
    test) ARRAY_SPEC="0-0" ;;
    batch) ARRAY_SPEC="0-191%${MAX_CONCURRENT}" ;;
    *) echo "ERROR: RUN_MODE must be test or batch" >&2; exit 2 ;;
  esac
fi
if [[ "${PREFETCH_MODEL}" == "1" ]]; then
  if [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then source "${HOME}/miniforge3/etc/profile.d/conda.sh";
  elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then source "${HOME}/miniconda3/etc/profile.d/conda.sh";
  else eval "$(conda shell.bash hook)"; fi
  conda activate "${CONDA_ENV}"
  python -u prefetch_molformer_model.py --model-id "${MODEL_ID}" --model-revision "${MODEL_REVISION}"
fi
SBATCH_EXTRA=()
[[ -n "${EXCLUDE_NODES}" ]] && SBATCH_EXTRA+=(--exclude="${EXCLUDE_NODES}")
echo "Submitting MoLFormer regression array: ${ARRAY_SPEC}"
JOBID=$(sbatch --parsable "${SBATCH_EXTRA[@]}" \
  --array="${ARRAY_SPEC}" \
  --output="${OUT_ROOT}/slurm_logs/%x_%A_%a.out" \
  --error="${OUT_ROOT}/slurm_logs/%x_%A_%a.err" \
  --export=ALL \
  molformer_regression_array.slurm)
JOBID="${JOBID%%;*}"
echo "Submitted regression array job: ${JOBID}"
if [[ "${POSTPROCESS}" == "1" && "${RUN_MODE}" == "batch" ]]; then
  printf -v summary_cmd 'bash %q' "${SCRIPT_DIR}/postprocess_molformer_regression.sh"
  SUMMARY_JOBID=$(sbatch --parsable "${SBATCH_EXTRA[@]}" \
    --job-name=molformer_reg_sum --dependency="afterok:${JOBID}" \
    --cpus-per-task=4 --mem=16G --time=04:00:00 \
    --output="${OUT_ROOT}/slurm_logs/molformer_reg_summary_%j.out" \
    --error="${OUT_ROOT}/slurm_logs/molformer_reg_summary_%j.err" \
    --export=ALL \
    --wrap="${summary_cmd}")
  echo "Submitted dependent regression summary job: ${SUMMARY_JOBID%%;*}"
fi
