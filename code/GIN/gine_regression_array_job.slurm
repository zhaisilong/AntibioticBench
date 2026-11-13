#!/usr/bin/env bash
#SBATCH --job-name=gine_reg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
# 如需排除某个节点，可在 submit 脚本里设置 EXCLUDE_NODES，例如：EXCLUDE_NODES=gpu06 bash submit_gine_regression_slurm.sh

set -euo pipefail

: "${PROJECT_DIR:?Set PROJECT_DIR via the submit script}"
: "${DATA_ROOT:?Set DATA_ROOT to the prepared split directory}"
: "${OUT_ROOT:?Set OUT_ROOT to the output directory}"
cd "${PROJECT_DIR}"
RUN_ROOT="${OUT_ROOT}"
mkdir -p "${RUN_ROOT}/slurm_logs"
exec > >(tee -a "${RUN_ROOT}/slurm_logs/gine_reg_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out") 2> >(tee -a "${RUN_ROOT}/slurm_logs/gine_reg_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" >&2)

if [[ -n "${CONDA_SH:-}" ]]; then
  source "${CONDA_SH}"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "ERROR: configure CONDA_SH or make conda available on PATH" >&2; exit 1
fi
conda activate "${CONDA_ENV}"

python --version
python - <<'PY'
import torch
print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('cuda_device_count', torch.cuda.device_count(), 'current_name', torch.cuda.get_device_name(0))
PY

python train_gine_regression_antibioticbench.py \
  --data-root "${DATA_ROOT}" \
  --out-root "${OUT_ROOT}" \
  --model-name "${MODEL_NAME}" \
  --dataset-config "${CONFIG}" \
  --tasks regression \
  --split-types "${SPLIT_TYPES}" \
  --seeds "${SEEDS}" \
  --single-run-index "${SLURM_ARRAY_TASK_ID}" \
  --epochs "${EPOCHS}" \
  --patience "${PATIENCE}" \
  --batch-size "${BATCH_SIZE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --dropout "${DROPOUT}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --num-layers "${NUM_LAYERS}" \
  --mlp-hidden-dim "${MLP_HIDDEN_DIM}" \
  --pooling "${POOLING}" \
  --select-metric val_rmse \
  --log-every-batches "${LOG_EVERY_BATCHES}"
