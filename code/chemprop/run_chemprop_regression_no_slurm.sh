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

# NUM_WORKERS only controls DataLoader subprocesses.
# CPU_THREADS controls OpenMP/BLAS/NumExpr/PyTorch CPU thread pools.
NUM_WORKERS="${NUM_WORKERS:-0}"
CPU_THREADS="${CPU_THREADS:-32}"

export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS}"
export BLIS_NUM_THREADS="${CPU_THREADS}"
export VECLIB_MAXIMUM_THREADS="${CPU_THREADS}"
export RAYON_NUM_THREADS="${CPU_THREADS}"
export TOKENIZERS_PARALLELISM=false

RUN_MODE="${RUN_MODE:-batch}"
TEST_RUN_INDEX="${TEST_RUN_INDEX:-0}"
POSTPROCESS="${POSTPROCESS:-1}"
RESUME="${RESUME:-1}"
FORCE="${FORCE:-0}"
SPLIT_TYPES="${SPLIT_TYPES:-random,scaffold,repeated_stratified_perimeter,strict_perimeter}"
SEEDS="${SEEDS:-0,1,2,3,4}"

activate_env() {
    if [[ -n "${CONDA_SH}" ]]; then
        [[ -f "${CONDA_SH}" ]] || { echo "ERROR: CONDA_SH does not exist" >&2; exit 1; }
        source "${CONDA_SH}"
    elif command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
    else
        echo "ERROR: configure CONDA_SH or make conda available on PATH" >&2; exit 1
    fi
    conda activate "${CONDA_ENV}"
}

preflight() {
    [[ -d "${DATA_ROOT}" ]] || { echo "ERROR: DATA_ROOT missing" >&2; exit 1; }
    [[ -d "${WORK_DIR}" ]] || { echo "ERROR: WORK_DIR missing" >&2; exit 1; }
    [[ -f "${SCRIPT_DIR}/${TRAIN_SCRIPT}" ]] || { echo "ERROR: missing ${TRAIN_SCRIPT}" >&2; exit 1; }
    [[ -f "${SCRIPT_DIR}/${SUMMARY_SCRIPT}" ]] || { echo "ERROR: missing ${SUMMARY_SCRIPT}" >&2; exit 1; }
    command -v "${CHEMPROP_EXECUTABLE}" >/dev/null 2>&1 || {
        echo "ERROR: Chemprop executable not found" >&2
        exit 1
    }

    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    python - <<'PY'
import importlib, os, torch
for name in ["chemprop","rdkit","numpy","pandas","scipy","sklearn"]:
    importlib.import_module(name)
print("torch:", torch.__version__)
print("torch CPU threads:", torch.get_num_threads())
print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))
print("MKL_NUM_THREADS:", os.environ.get("MKL_NUM_THREADS"))
print("OPENBLAS_NUM_THREADS:", os.environ.get("OPENBLAS_NUM_THREADS"))
print("CUDA available:", torch.cuda.is_available())
print("visible GPU count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")
print("logical cuda:0:", torch.cuda.get_device_name(0))
print(torch.zeros(1, device="cuda"))
PY

    # Verify and display the metric direction patch before launching any run.
    python "${SCRIPT_DIR}/chemprop_cli_direction_guard.py" --check-only
}

make_plan() {
    local plan_output

    CURRENT_STAGE="build_run_plan"
    CURRENT_RUN_INDEX="NA"

    plan_output=$(python "${SCRIPT_DIR}/${TRAIN_SCRIPT}" \
        --data-root "${DATA_ROOT}" \
        --out-root "${OUT_ROOT}" \
        --split-types "${SPLIT_TYPES}" \
        --seeds "${SEEDS}" \
        --print-run-count 2>&1)

    printf '%s\n' "${plan_output}"

    RUN_COUNT=$(printf '%s\n' "${plan_output}" | awk '/^[[:space:]]*[0-9]+[[:space:]]*$/ {gsub(/[[:space:]]/, "", $0); n=$0} END {print n}')

    if ! [[ "${RUN_COUNT}" =~ ^[0-9]+$ ]] || [[ "${RUN_COUNT}" -lt 1 ]]; then
        echo "ERROR: could not parse a valid run count from the output above." >&2
        exit 1
    fi

    case "${RUN_MODE}" in
        test)
            [[ "${TEST_RUN_INDEX}" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid TEST_RUN_INDEX"; exit 1; }
            [[ "${TEST_RUN_INDEX}" -ge 0 && "${TEST_RUN_INDEX}" -lt "${RUN_COUNT}" ]] || {
                echo "ERROR: TEST_RUN_INDEX outside 0..$((RUN_COUNT-1))" >&2
                exit 1
            }
            RUN_START="${TEST_RUN_INDEX}"
            RUN_END="${TEST_RUN_INDEX}"
            ;;
        batch)
            RUN_START=0
            RUN_END=$((RUN_COUNT-1))
            ;;
        *)
            echo "ERROR: RUN_MODE must be test or batch" >&2
            exit 1
            ;;
    esac

    RESUME_ARG=()
    if [[ "${RESUME}" == "1" ]]; then
        RESUME_ARG+=(--resume)
    fi

    FORCE_ARG=()
    if [[ "${FORCE}" == "1" ]]; then
        FORCE_ARG+=(--force)
    fi
}

TASK="regression"
TRAIN_SCRIPT="train_chemprop_regression_benchmark.py"
SUMMARY_SCRIPT="summarize_chemprop_regression.py"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/outputs/chemprop/regression}"
mkdir -p "${OUT_ROOT}/local_logs"

STAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="${OUT_ROOT}/local_logs/regression_${STAMP}.log"
FAILURE_FILE="${OUT_ROOT}/FAILED_LAST_RUN.txt"
SUCCESS_FILE="${OUT_ROOT}/ALL_RUNS_AND_SUMMARY_COMPLETED.txt"
exec > >(tee -a "${MASTER_LOG}") 2>&1

CURRENT_STAGE="initialization"
CURRENT_RUN_INDEX="NA"
on_error() {
    local rc=$?
    local failed_line="${BASH_LINENO[0]:-NA}"

    {
        echo "task=${TASK}"
        echo "stage=${CURRENT_STAGE}"
        echo "run_index=${CURRENT_RUN_INDEX}"
        echo "exit_code=${rc}"
        echo "failed_line=${failed_line}"
        echo "time=$(date --iso-8601=seconds 2>/dev/null || date)"
    } > "${FAILURE_FILE}"

    echo
    echo "ERROR: workflow stopped."
    echo "stage=${CURRENT_STAGE}"
    echo "run_index=${CURRENT_RUN_INDEX}"
    echo "exit_code=${rc}"
    echo "failed_line=${failed_line}"
    echo "Later runs and summary were not executed."
    echo "See the local failure record for details."
    exit "${rc}"
}
trap on_error ERR

activate_env
preflight
make_plan
rm -f "${SUCCESS_FILE}"

echo "===== REGRESSION, NO SLURM ====="
echo "GPU_ID=${GPU_ID}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "CPU_THREADS=${CPU_THREADS}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "MKL_NUM_THREADS=${MKL_NUM_THREADS}"
echo "OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS}"
echo "RUN_RANGE=${RUN_START}-${RUN_END}"
echo "INPUT=smiles log10_mic"
echo "EARLY_STOP=min val RMSE; patience=20; min_delta=0.0"

for ((RUN_INDEX=RUN_START; RUN_INDEX<=RUN_END; RUN_INDEX++)); do
    CURRENT_STAGE="regression_training"
    CURRENT_RUN_INDEX="${RUN_INDEX}"
    echo "===== REGRESSION RUN ${RUN_INDEX}/${RUN_END} START ====="
    python -u "${SCRIPT_DIR}/${TRAIN_SCRIPT}" \
        --data-root "${DATA_ROOT}" \
        --out-root "${OUT_ROOT}" \
        --work-dir "${WORK_DIR}" \
        --chemprop-executable "${CHEMPROP_EXECUTABLE}" \
        --single-run-index "${RUN_INDEX}" \
        --split-types "${SPLIT_TYPES}" \
        --seeds "${SEEDS}" \
        --message-hidden-dim 300 \
        --depth 3 \
        --dropout 0.0 \
        --ffn-hidden-dim 300 \
        --ffn-num-layers 1 \
        --aggregation norm \
        --aggregation-norm 100 \
        --activation relu \
        --epochs 600 \
        --patience 20 \
        --min-delta 0.0 \
        --warmup-epochs 2 \
        --batch-size 64 \
        --eval-batch-size 256 \
        --init-lr 1e-5 \
        --max-lr 5e-4 \
        --final-lr 1e-5 \
        --grad-clip 5.0 \
        --num-workers "${NUM_WORKERS}" \
        --accelerator gpu \
        --devices 1 \
        --local-k 10 \
        "${RESUME_ARG[@]}" \
        "${FORCE_ARG[@]}" \
        2>&1 | tee "${OUT_ROOT}/local_logs/regression_run_${RUN_INDEX}.log"
done

if [[ "${POSTPROCESS}" == "1" ]]; then
    CURRENT_STAGE="regression_summary"
    CURRENT_RUN_INDEX="NA"
    python -u "${SCRIPT_DIR}/${SUMMARY_SCRIPT}" \
        --out-root "${OUT_ROOT}" \
        2>&1 | tee "${OUT_ROOT}/local_logs/regression_summary.log"
fi

CURRENT_STAGE="completed"
rm -f "${FAILURE_FILE}"
printf "task=%s\ncompleted=%s\nrun_range=%s-%s\n" \
"${TASK}" "$(date --iso-8601=seconds 2>/dev/null || date)" "${RUN_START}" "${RUN_END}" \
> "${SUCCESS_FILE}"
echo "===== REGRESSION WORKFLOW COMPLETED ====="
