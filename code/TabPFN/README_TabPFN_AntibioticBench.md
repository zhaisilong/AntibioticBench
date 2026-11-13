# TabPFN-3 AntibioticBench scripts

## Fixed benchmark design

- Model: TabPFN-3, local OSS inference.
- Fine-tuning: **No**.
- Hyperparameter search: **No**.
- TabPFN core inference settings remain fixed (n_estimators=8, auto_scale_n_estimators=True, softmax_temperature=0.9, inference_precision=auto, random_state=0, n_preprocessing_jobs=1; classifier balance_probabilities=False). For runtime efficiency, `fit_mode=fit_with_cache` is explicitly used so validation/test prediction batches reuse the fitted train-context KV cache.
- Device: CUDA.
- Molecular input representation: **Morgan/ECFP4 radius=2, 2048 bits + the full RDKit `Descriptors._descList` 2D descriptor set**.
- Classification threshold: 0.5.
- NN diagnostics: test -> train, **Morgan radius=2, 2048 bits only**, Tanimoto; RDKit2D descriptors are not used for NN similarity; bins >=0.7, 0.4-0.7, <0.4.
- Activity cliff: test-train pair, **Morgan radius=2, 2048-bit Tanimoto only**, Tanimoto >=0.90 and |Δlog10MIC| >= log10(16).
- strict_perimeter: seed_0 only. Other splits: seeds 0-4.
- Expected runs per task with 12 datasets: 12 x (5 + 5 + 5 + 1) = 192.

### Molecular feature handling

The TabPFN model input is the concatenation:

```text
Morgan ECFP4 (radius=2, 2048 bits) + RDKit 2D descriptors
```

RDKit2D uses the descriptor functions exposed by the installed RDKit version through `Descriptors._descList`. No external StandardScaler is applied; standard TabPFN's native preprocessing remains responsible for feature scaling/transformation. Descriptor calculation failures and non-finite values are converted to `NaN` and passed to TabPFN's native missing-value preprocessing. `feature_summary.json` records the descriptor names, dimensions, and missingness.

**Important separation:** NN and Activity Cliff diagnostics do not use the concatenated TabPFN feature matrix. They remain fixed to Morgan radius=2, nBits=2048 + Tanimoto with train as the reference library.

### Important: standard TabPFN has no downstream epoch training

`TabPFNClassifier.fit()` / `TabPFNRegressor.fit()` in this benchmark do not update pretrained weights. Therefore there is no true epoch-by-epoch training loss curve. To avoid self-context evaluation, this version **does not predict the labeled train set**. `X_train + y_train` is used only as TabPFN's in-context labeled context. Therefore:

- `predictions_train.csv` is **not generated**.
- Train performance fields in `metrics_full.csv` are `NA`, except descriptive train-set statistics such as `train_n`, class counts/ratio, or train target mean/std.
- `train_log.csv` and `training_curve.csv` contain exactly one final diagnostic row.
- Classification `train_loss` is NA; `val_loss` is the final validation log loss.
- Regression `train_loss` is NA; `val_loss` is the final validation MSE.
- `epoch`, `learning_rate`, `best_epoch` are NA.
- The `note` field explicitly records that train prediction was intentionally omitted to avoid self-context evaluation.

Before running a launcher, set `DATA_ROOT` to the prepared split directory. `SCRIPT_DIR`, `OUT_ROOT`, `FEATURE_CACHE_ROOT`, `KV_CACHE_ROOT`, and `CONDA_SH` can be overridden; the default output paths below are relative to the repository root.

## Files to place in

`code/TabPFN`

1. `tabpfn_benchmark_common.py`
2. `train_tabpfn_classification_antibioticbench.py`
3. `train_tabpfn_regression_antibioticbench.py`
4. `run_tabpfn_classification_no_slurm.sh`
5. `run_tabpfn_regression_no_slurm.sh`

Then:

```bash
cd code/TabPFN
chmod +x run_tabpfn_classification_no_slurm.sh run_tabpfn_regression_no_slurm.sh
```

## Input paths

The scripts auto-discover datasets under:

```text
${DATA_ROOT}/
  <dataset_name>/
    classification/
      random/seed_0..4/
      scaffold/seed_0..4/
      repeated_stratified_perimeter/seed_0..4/
      strict_perimeter/seed_0/
    regression/
      ...same split layout...
```

Each run directory must contain `train.csv`, `validation.csv`, and `test.csv`.

The loader recognizes common names including `smiles`, `SMILES`, `smiles_std`, `SMILES_std`; classification target `active`; regression target `log10_mic`.

## Output paths

Classification:

```text
outputs/TabPFN/classification/
  TabPFN/<dataset>/classification/<split>/seed_<n>/...
```

Regression:

```text
outputs/TabPFN/regression/
  TabPFN/<dataset>/regression/<split>/seed_<n>/...
```

Task-specific cache locations:

```text
output_classification/
  feature_cache/<dataset>/...
  kv_cache/<dataset>/<split>/seed_<n>/kv_cache_runtime_info.json

output_regression/
  feature_cache/<dataset>/...
  kv_cache/<dataset>/<split>/seed_<n>/kv_cache_runtime_info.json
```

`SAVE_FITTED_KV_CACHE=1` additionally writes `fitted_model.tabpfn_fit` inside each run's task-specific `kv_cache` directory. The live `fit_with_cache` KV cache used during prediction remains process-local GPU memory.

Per-run outputs include:

- `metrics_full.csv`
- `predictions_validation.csv`
- `predictions_test.csv`
- `diagnostics_metrics.csv`
- `diagnostic_subgroup_metrics.csv`
- `activity_cliff_pairs_test.csv`
- `train_log.csv`
- `training_curve.csv`
- `train_log_long.csv`
- `feature_summary.json`
- `model_params.json`
- `run_info.json`
- `model.joblib` (small reference metadata; no duplicated 200MB checkpoint)
- `run.log`
- `SUCCESS` or `FAILED.txt`

Train is used only as the labeled in-context reference and is not predicted. Validation receives overall model predictions/metrics. Test receives model predictions plus NN/activity-cliff/MIC-boundary/MIC-range diagnostics; regression additionally receives local-roughness diagnostics.

## 1. Classification test run

Use one physical GPU, for example GPU 0:

```bash
cd code/TabPFN

GPU_ID=0 \
RUN_MODE=test \
POSTPROCESS=0 \
RESUME=0 \
FORCE=1 \
bash run_tabpfn_classification_no_slurm.sh
```

`RUN_MODE=test` executes only run index 0.

## 2. Regression test run

For example physical GPU 1:

```bash
cd code/TabPFN

GPU_ID=1 \
RUN_MODE=test \
POSTPROCESS=0 \
RESUME=0 \
FORCE=1 \
bash run_tabpfn_regression_no_slurm.sh
```

If the TabPFN-3 regressor checkpoint has not yet been downloaded, the first regression fit can trigger the licensed checkpoint download. Existing cached authentication can be used.

## 3. Formal classification

```bash
cd code/TabPFN

GPU_ID=0 \
RUN_MODE=batch \
POSTPROCESS=1 \
RESUME=1 \
FORCE=0 \
bash run_tabpfn_classification_no_slurm.sh
```

## 4. Formal regression

```bash
cd code/TabPFN

GPU_ID=1 \
RUN_MODE=batch \
POSTPROCESS=1 \
RESUME=1 \
FORCE=0 \
bash run_tabpfn_regression_no_slurm.sh
```

Classification and regression can be launched in separate tmux windows on different physical GPUs.

## 5. Partial/recovery runs

Run a specific inclusive range:

```bash
GPU_ID=0 RUN_MODE=batch RUN_RANGE=20-39 RESUME=1 bash run_tabpfn_classification_no_slurm.sh
```

Run one exact global run index:

```bash
GPU_ID=0 RUN_MODE=batch SINGLE_RUN_INDEX=25 RESUME=0 FORCE=1 bash run_tabpfn_classification_no_slurm.sh
```

The Python script prints the complete `RUN_INDEX` map at startup.

## 6. GPU OOM handling

`PRED_BATCH_SIZE` controls only how many query molecules are predicted per call; it is a computational/memory setting, not model HPO. Default is 256. If an A800 run is OOM, reduce it, for example:

```bash
PRED_BATCH_SIZE=128 GPU_ID=0 RUN_MODE=test FORCE=1 bash run_tabpfn_classification_no_slurm.sh
```

Do not change `n_estimators`, `softmax_temperature`, or internal inference config just to fix OOM; that would change the model setting.

## 7. Summary files

After formal runs, the scripts generate:

```text
output_*/TabPFN/all_run_metrics.csv
output_*/TabPFN/aggregate_mean_sd.csv
output_*/TabPFN/ood_change_random_vs_ood_splits.csv
output_*/_combined_summary/<task>_summary_like_ml/...
```

For strict_perimeter, only seed 0 is scheduled by construction.


## 2026-08-10 update: persistent feature cache + prediction progress + KV cache + no train prediction

Four runtime/methodological changes are enabled in this version:

1. **Prediction batch progress logging**: validation/test prediction writes progress approximately every 5% with batch count, row count, elapsed time and ETA to `run.log` and the launcher terminal log.
2. **Task-isolated persistent per-dataset feature caches**: Morgan radius=2/2048-bit fingerprints and the full RDKit2D descriptor matrix are computed once per dataset **within each task**. Classification and regression no longer share the same feature-cache files:

```text
outputs/TabPFN/classification/
  feature_cache/
    <dataset_name>/
      molecular_features.npz
      metadata.json
      .cache.lock

outputs/TabPFN/regression/
  feature_cache/
    <dataset_name>/
      molecular_features.npz
      metadata.json
      .cache.lock
```

Within classification, all classification splits/seeds reuse the classification cache. Within regression, all regression splits/seeds reuse the regression cache. The cache metadata now records the task, and each cache manager scans only its own task directory. Therefore concurrently launched classification/regression jobs never contend for or extend the same molecular-feature cache file.

The cache is automatically rebuilt if its feature definition, task, or RDKit version is incompatible. If a later run contains a SMILES not present in that task's cache, only the missing rows are appended/recomputed.

3. **Fitted KV cache**: both classifier and regressor use `fit_mode=fit_with_cache` with `keep_cache_on_device=True`. The **live KV cache used for prediction is process-local GPU memory, not a shared disk file**. Classification and regression processes therefore already have separate live KV caches even when they run simultaneously on different GPUs.

For organization/reproducibility, each task now also has its own KV-cache directory:

```text
output_classification/kv_cache/<dataset>/<split>/seed_<n>/
output_regression/kv_cache/<dataset>/<split>/seed_<n>/
```

Every run writes `kv_cache_runtime_info.json` there. By default the large fitted KV state itself is **not** written to disk. If persistent fitted state is explicitly required, launch with `SAVE_FITTED_KV_CACHE=1`; TabPFN then writes `fitted_model.tabpfn_fit` into the corresponding task-specific KV-cache directory. This is disabled by default because a fitted state can be large and 192 runs per task can consume substantial disk space.

The fitted train-context Transformer KV representation is built during `fit()` and reused by validation/test prediction batches. This changes the computation/memory trade-off, not the dataset-specific model weights; there is still no fine-tuning.

4. **No train prediction**: the labeled train set is already the TabPFN context. Predicting the same rows with their labels still present in context would be self-context evaluation, so `predictions_train.csv` and train performance metrics are intentionally omitted/NA. This also removes the largest query-prediction block for large datasets.

Because no train predictions are computed, prediction-dependent train-neighbor fields are intentionally unavailable: classification `nn_train_score`, regression `nn_train_pred_log10mic`, and activity-cliff pair directional-accuracy fields that require predictions for both test and train endpoints are written as `NaN`. Structural NN similarity, NN bins, AC flags/counts, MIC differences, local roughness, and all test error/performance diagnostics remain available.

Because Morgan 2048 + RDKit2D exceeds TabPFN's 2000-feature validation guardrail on the current RDKit installation (2265 total features with 217 descriptors), this version explicitly uses `ignore_pretraining_limits=True`. This is an input-limit override, not HPO or fine-tuning.
