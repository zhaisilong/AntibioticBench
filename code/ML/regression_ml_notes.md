# AntibioticBench 传统机器学习回归任务脚本说明

本脚本体系用于 `regression/` 文件夹中的 log10MIC 回归任务，和二分类脚本分开保存，避免覆盖已有分类结果。

## 输入数据目录

期望目录结构：

```text
DATA_ROOT/
  1_Staphylococcus_median_16_MolWt_50_1000_value_le_8192_multitask/
    regression/
      random/seed_0/train.csv validation.csv test.csv
      scaffold/seed_0/train.csv validation.csv test.csv
      repeated_stratified_perimeter/seed_0/...
      strict_perimeter/seed_0/...
```

## 输入列

固定分子输入列：

```text
smiles
```

回归标签已写死为：

```text
log10_mic
```

脚本会强制检查每个 `train.csv`、`validation.csv` 和 `test.csv` 是否存在 `log10_mic` 列；如果缺失会直接报错。脚本不会自动识别 `log10MIC`、`log10_value`、`target`、`label` 等其他列，也不会从原始 MIC 数值列临时计算 log10MIC。

这样做的目的：

```text
避免 log2MIC、log10MIC、原始 MIC 或二分类标签 active 混用。
```

原始 MIC 数值列仍可用于辅助诊断，例如 MIC boundary、active/inactive 推断、predicted MIC 反变换检查等，但不作为训练标签来源。

## 模型集合

正式支持：

```text
rf        RandomForestRegressor
svm       SVR, RBF kernel
xgb       XGBRegressor, objective=reg:squarederror
lightgbm  LGBMRegressor, objective=regression
```

## 输出文件

每个 run 输出到：

```text
OUT_ROOT/model/dataset_name/regression/split_type/seed_x/
```

主要文件：

```text
metrics_full.csv
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
model.joblib
```

## 训练过程输出

`training_curve.csv` 和 `train_log.csv` 包含：

```text
iteration
train_loss
val_loss
train_mse
val_mse
train_rmse
val_rmse
train_mae
val_mae
train_spearman
val_spearman
train_within_1_dilution
val_within_1_dilution
train_within_2_dilutions
val_within_2_dilutions
note
```

其中 `train_loss` 和 `val_loss` 对回归任务定义为 MSE。

- XGBoost / LightGBM：如果软件包返回 eval history，则输出每一轮 boosting iteration。
- RF / SVR：sklearn 没有 epoch/iteration 训练历史，因此输出训练完成后的最终 train/validation MSE、RMSE、MAE 等一行，并标记 `final_only_no_epoch_loss_for_this_model`。

## 回归主指标

```text
MAE
RMSE
MedAE
R²
Pearson
Spearman
within 1 dilution
within 2 dilutions
median fold error
```

within-dilution 使用 log10MIC 尺度：

```text
within 1 dilution: |predicted log10MIC - observed log10MIC| <= log10(2) ≈ 0.301
within 2 dilutions: |predicted log10MIC - observed log10MIC| <= log10(4) ≈ 0.602
```

## 虚拟筛选排序指标

回归模型用于虚拟筛选时：

```text
score_reg = - predicted_log10MIC
```

分数越高，表示预测 MIC 越低、活性越强。

脚本会尽量根据 `active` 列或 `MIC <= threshold` 推断活性标签，然后输出：

```text
EF1%, EF5%, EF10%
Hit_rate@Top1%, Hit_rate@Top5%, Hit_rate@Top10%
Recall@Top50, Recall@Top100, Recall@Top500
```

## 诊断分析输出

脚本支持以下诊断：

### 12.1 MIC-boundary regression error analysis

输出：

```text
boundary_mae
non_boundary_mae
boundary_rmse
non_boundary_rmse
boundary_regression_error_enrichment
boundary_within_1_dilution
non_boundary_within_1_dilution
```

### 12.2 Nearest-neighbor error analysis

按训练集最近邻 Tanimoto similarity 分组：

```text
>=0.7
0.4-0.7
<0.4
```

每组输出 MAE、RMSE、Spearman、within 1/2 dilution、EF/Hit 等。

### 12.3 MIC-specific activity cliff diagnosis

定义：

```text
Tanimoto similarity >= 0.7
|log10MIC_i - log10MIC_j| >= log10(16) ≈ 1.204
```

输出：

```text
ac_mae
non_ac_mae
ac_rmse
non_ac_rmse
ac_error_enrichment
ac_pair_directional_accuracy
```

### 12.5 Molecular landscape roughness proxy

每个 test 分子计算训练集 k 个近邻的 log10MIC 方差：

```text
local_train_log10mic_variance_k10
```

并比较 high local variance 和 low local variance 分子的误差差异。

### 12.6 MIC range-specific regression error analysis

按中心 threshold 分组：

```text
MIC <= T/2
T/2 < MIC <= 2T
MIC > 2T
```

分别输出 MAE、RMSE、Spearman、within 1/2 dilution 等。

## 推荐提交顺序

先测试：

```bash
RUN_MODE=test \
RUN_TAG=Staphylococcus_rf_regression_random_seed0_test \
sbatch --parsable run_regression_ml_cluster_dataset_named.slurm
```

测试 XGBoost 训练曲线：

```bash
RUN_MODE=test \
RUN_TAG=Staphylococcus_xgb_regression_random_seed0_test \
TEST_MODEL=xgb \
sbatch --parsable run_regression_ml_cluster_dataset_named.slurm
```

正式跑 RF / XGB / LightGBM：

```bash
RUN_MODE=batch \
RUN_TAG=all_datasets_rf_xgb_lightgbm_regression_4splits_5seeds \
MODELS="rf xgb lightgbm" \
SPLIT_TYPES="random scaffold repeated_stratified_perimeter strict_perimeter" \
SEEDS="0 1 2 3 4" \
sbatch --parsable run_regression_ml_cluster_dataset_named.slurm
```

最后单独跑 SVR：

```bash
RUN_MODE=svm \
RUN_TAG=all_datasets_svr_regression_4splits_5seeds \
SPLIT_TYPES="random scaffold repeated_stratified_perimeter strict_perimeter" \
SEEDS="0 1 2 3 4" \
sbatch --parsable run_regression_ml_cluster_dataset_named.slurm
```

## 回归任务的 Molecular landscape roughness analysis

回归任务建议新增并使用 `roughness_regression.py`，它与分类版 `roughness.py` 分开，避免 `active`、`log2MIC` 和 `log10_mic` 混用。

### 输入目录

```text
DATA_ROOT/dataset_name/regression/split_type/seed_x/train.csv
DATA_ROOT/dataset_name/regression/split_type/seed_x/validation.csv
DATA_ROOT/dataset_name/regression/split_type/seed_x/test.csv
```

### 写死输入列

```text
smiles
log10_mic
```

如果任何 split 文件缺少 `log10_mic`，脚本会直接报错。

### 主要输出

```text
_regression_landscape_roughness/regression_landscape_roughness_by_split_seed.csv
_regression_landscape_roughness/regression_landscape_roughness_mean_sd.csv
_regression_landscape_roughness/per_run/.../test_to_train_nn_roughness.csv
```

可选输出：

```text
_regression_landscape_roughness/regression_landscape_roughness_molecule_level.csv
```

### 核心指标

```text
mean_knn_log10mic_variance
median_knn_log10mic_variance
mean_knn_log10mic_absdiff
mean_nearest_neighbor_log10mic_absdiff
activity_cliff_density
test_to_train_nn_sim_mean
test_to_train_nn_lt_0p4_ratio
test_to_train_nn_log10mic_absdiff_mean
test_to_train_ac_relative_flag_ratio
```

Activity cliff 定义：

```text
Tanimoto similarity >= 0.7
|log10_mic_i - log10_mic_j| >= log10(16) ≈ 1.204
```

### 单独运行

```bash
python roughness_regression.py \
  --data-root ${DATA_ROOT} \
  --outdir ${OUT_ROOT}/test_roughness \
  --datasets Staphylococcus \
  --split-types random \
  --seeds 0
```

### 和回归训练一起运行

```bash
COMPUTE_LANDSCAPE=1 \
bash run_regression_ml.sh \
  ${DATA_ROOT} \
  ${OUT_ROOT}/test_with_roughness
```

如果需要保存每个分子的局部 roughness 诊断：

```bash
COMPUTE_LANDSCAPE=1 SAVE_MOLECULE_LEVEL_ROUGHNESS=1 bash run_regression_ml.sh DATA_ROOT OUT_ROOT
```


## Output location

The regression and similarity launchers and Python entry points default to
directories under the repository's ignored `outputs/ML/` directory. Slurm
launchers use separate task-specific locations there. Set `OUT_ROOT`, pass the
output argument to the launcher, or use `--outdir` to override the default.
Configured paths outside the repository are not protected by its `.gitignore`.
