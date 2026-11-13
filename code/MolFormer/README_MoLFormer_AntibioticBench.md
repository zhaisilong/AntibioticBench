# MoLFormer AntibioticBench scripts

This package fine-tunes `ibm-research/MoLFormer-XL-both-10pct` on the 12 AntibioticBench datasets for separate classification and regression tasks.

## Fixed protocol

- Model revision: `compat-v4` (for Transformers 4.x)
- Input: RDKit canonical non-isomeric SMILES; no silent truncation
- Maximum sequence length: 202 tokens
- Fine-tuning: full encoder + official-style two-layer residual head
- Pooling: MoLFormer `pooler_output` (masked mean pooler); masked mean fallback is included
- Optimizer: AdamW, betas `(0.9, 0.99)`, epsilon `1e-8`
- Learning rate: constant `3e-5`
- Weight decay: `0.0`
- Train/evaluation batch size: `32/64`
- Head dropout: `0.10`
- Gradient clipping: `5.0`
- Maximum epochs / early-stopping patience: `300/20`
- Classification: BCEWithLogitsLoss, fixed threshold 0.5, checkpoint by `val_pr_auc`, no `pos_weight`
- Regression: raw `log10_mic`, MSELoss, no target standardization, checkpoint by `val_rmse`
- Precision: FP32
- Splits: random/scaffold/repeated_stratified_perimeter seeds 0-4; strict_perimeter seed 0 only
- Total runs: 192 classification + 192 regression

## Paths and setup

Run the scripts from this package after preparing the train, validation, and test split directory (the full tables in `data/` are not precomputed splits). From the repository root:

```bash
cd code/MolFormer
export DATA_ROOT="relative/or/absolute/path/to/prepared-splits"
```

`SCRIPT_DIR` is detected automatically. `DATA_ROOT` must be set; `BENCHMARK_ROOT`, `OUT_ROOT`, `DATASET_CONFIG`, `HF_HOME`, and `HF_HUB_CACHE` can be overridden. Output defaults to the repository's `outputs/MolFormer/` and the Hugging Face cache to `.cache/molformer/hf_cache/`. The optional `MOLFORMER_ROOT` and `--molformer-repo` values are retained for compatibility; model weights are obtained through Hugging Face. Launch Slurm jobs through the supplied submit scripts so the source directory is passed to workers.

The scripts are provided as-is and execution or reproduction is not guaranteed. Check local dependencies and cluster configuration before using them.

## Validate the environment and cache the model

```bash
cd code/MolFormer
source ~/miniforge3/etc/profile.d/conda.sh
conda activate molformer

export HF_HOME="relative/or/absolute/path/to/hf-cache"
export HF_HUB_CACHE="$HF_HOME/hub"

python prefetch_molformer_model.py \
  --model-id ibm-research/MoLFormer-XL-both-10pct \
  --model-revision compat-v4
```

After the model is cached, `LOCAL_FILES_ONLY=1` may be used for completely offline training.

## No-Slurm server: classification test

Use a separate test output root so test run 0 cannot be mistaken for a formal result:

```bash
cd code/MolFormer

RUN_MODE=test \
GPUS="0" \
MAX_JOBS_PER_GPU=1 \
EPOCHS=3 \
PATIENCE=2 \
OUT_ROOT="../../outputs/MolFormer/test_classification" \
POSTPROCESS=0 \
RESUME=0 \
FORCE=1 \
VALIDATE_INPUTS=1 \
PREFETCH_MODEL=1 \
SHOW_RUN_OUTPUT=1 \
bash run_molformer_classification_no_slurm.sh
```

## No-Slurm server: regression test

```bash
cd code/MolFormer

RUN_MODE=test \
GPUS="1" \
MAX_JOBS_PER_GPU=1 \
EPOCHS=3 \
PATIENCE=2 \
OUT_ROOT="../../outputs/MolFormer/test_regression" \
POSTPROCESS=0 \
RESUME=0 \
FORCE=1 \
VALIDATE_INPUTS=1 \
PREFETCH_MODEL=1 \
SHOW_RUN_OUTPUT=1 \
bash run_molformer_regression_no_slurm.sh
```

Replace GPU 0/1 with currently free physical GPU IDs.

## No-Slurm server: formal classification

Run inside tmux:

```bash
cd code/MolFormer

env \
  CONDA_ENV=molformer \
  GPUS="0" \
  MAX_JOBS_PER_GPU=1 \
  EPOCHS=300 \
  PATIENCE=20 \
  BATCH_SIZE=32 \
  EVAL_BATCH_SIZE=64 \
  LR=3e-5 \
  POSTPROCESS=1 \
  RESUME=1 \
  FORCE=0 \
  STOP_ON_ERROR=1 \
  VALIDATE_INPUTS=1 \
  PREFETCH_MODEL=1 \
  SHOW_RUN_OUTPUT=1 \
  bash run_molformer_classification_no_slurm.sh
```

## No-Slurm server: formal regression

```bash
cd code/MolFormer

env \
  CONDA_ENV=molformer \
  GPUS="1" \
  MAX_JOBS_PER_GPU=1 \
  EPOCHS=300 \
  PATIENCE=20 \
  BATCH_SIZE=32 \
  EVAL_BATCH_SIZE=64 \
  LR=3e-5 \
  POSTPROCESS=1 \
  RESUME=1 \
  FORCE=0 \
  STOP_ON_ERROR=1 \
  VALIDATE_INPUTS=1 \
  PREFETCH_MODEL=1 \
  SHOW_RUN_OUTPUT=1 \
  bash run_molformer_regression_no_slurm.sh
```

Classification and regression can be run in separate tmux sessions on different GPUs. `MAX_JOBS_PER_GPU=1` is recommended for the formal benchmark.

## Partial run range / recovery

```bash
START_INDEX=40 END_INDEX=80 GPUS="0" POSTPROCESS=0 RESUME=1 \
  bash run_molformer_classification_no_slurm.sh
```

`RESUME=1` skips run directories containing `_SUCCESS`. A failed run contains `_FAILED`; use `FORCE=1` to delete and rebuild a selected run directory.

## Slurm alternative

Classification:

```bash
RUN_MODE=test POSTPROCESS=0 bash submit_molformer_classification_slurm.sh
RUN_MODE=batch MAX_CONCURRENT=1 POSTPROCESS=1 bash submit_molformer_classification_slurm.sh
```

Regression:

```bash
RUN_MODE=test POSTPROCESS=0 bash submit_molformer_regression_slurm.sh
RUN_MODE=batch MAX_CONCURRENT=1 POSTPROCESS=1 bash submit_molformer_regression_slurm.sh
```

Change the `#SBATCH --partition`, memory, time, and GPU directives if the cluster policy differs.

## Per-run output

```text
output_classification/MoLFormer/<dataset>/classification/<split>/seed_<seed>/
output_regression/MoLFormer/<dataset>/regression/<split>/seed_<seed>/
```

Each successful run contains:

```text
_SUCCESS
best_model.pth
metrics_full.csv
metrics_train.csv
metrics_validation.csv
metrics_test.csv
predictions_train.csv
predictions_validation.csv
predictions_test.csv
diagnostics_metrics.csv
diagnostic_subgroup_metrics.csv
train_log.csv
training_curve.csv
train_log_long.csv
feature_summary.json
model_params.json
run_info.json
molformer_input_audit.csv
```

The launcher also keeps the full console output for every epoch:

```text
output_*/no_slurm_logs/<RUN_TAG>/run_<INDEX>_gpu<GPU>.log
```

`train_log.csv` and `training_curve.csv` include train loss, validation loss, learning rate, overall train/validation metrics, best epoch so far, patience state, and epoch time.

## Split-role policy

- Train: fitting, predictions, and overall metrics only.
- Validation: checkpoint selection, predictions, and overall metrics only.
- Test: evaluated once after loading the best validation checkpoint; predictions, overall metrics, Morgan/Tanimoto train-reference NN diagnostics, MIC-boundary/range diagnostics, activity-cliff diagnostics, and regression local-variance diagnostics.

## Summary output

After all 192 runs finish, postprocessing creates:

```text
MoLFormer/all_run_metrics.csv
MoLFormer/aggregate_mean_sd.csv
MoLFormer/ood_change_random_vs_ood_splits.csv
MoLFormer/all_diagnostics_metrics.csv
MoLFormer/_combined_summary/classification_summary_like_ml/...
MoLFormer/_combined_summary/regression_summary_like_ml/...
```

Model-independent landscape roughness is optional because it is identical across models. To calculate it here:

```bash
COMPUTE_LANDSCAPE=1 bash postprocess_molformer_classification.sh
COMPUTE_LANDSCAPE=1 bash postprocess_molformer_regression.sh
```

## Important input audit behavior

The script canonicalizes SMILES with RDKit and removes isomeric information to match the public MoLFormer pretraining representation. It writes `molformer_input_audit.csv`. Invalid SMILES or token sequences longer than 202 cause the run to fail; they are never silently dropped or truncated.
