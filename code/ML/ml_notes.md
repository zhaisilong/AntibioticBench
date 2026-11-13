# AntibioticBench 传统机器学习模型固定超参数策略

本 benchmark 不采用超参数搜索。所有模型使用预先声明的固定超参数，在所有 genus、split type 和 split seed 上保持一致。这样做的目的不是追求单个数据集上的最高分，而是保证模型比较的公平性、可复现性和可解释性。

## 为什么不直接使用“最新版本默认超参数”

不建议在论文中写“使用最新版默认参数”。原因是不同软件包版本的默认参数可能变化，导致别人复现时结果不一致。更稳妥的做法是：明确写出所有关键参数，并保存 `model_params.json`、`run_info.json`、软件版本、训练命令和 split seed。

## 输入列与输入特征

本 benchmark 的传统机器学习训练脚本固定使用：

- 分子输入列：`smiles`
- 二分类活性标签列：`active`

训练脚本会强制检查每个 `train.csv`、`validation.csv` 和 `test.csv` 中是否同时存在 `smiles` 和 `active` 两列；如果缺失，会直接报错。其他 SMILES-like 列，例如 `smiles_std`、`standardized_smiles` 或 `canonical_smiles`，不会被用于模型训练。

所有传统机器学习模型统一使用由 `smiles` 列计算得到的：

- Morgan fingerprint：radius = 2, nBits = 2048
- RDKit 2D descriptors：RDKit 内置二维理化描述符
- 缺失值处理：`SimpleImputer(strategy="median")`
- SVM 额外使用 `StandardScaler()`；树模型不进行标准化


本脚本正式支持的传统机器学习模型为：

```text
rf
svm
xgb
lightgbm
```


## 固定超参数

### Random Forest

```text
RandomForestClassifier(
    n_estimators=500,
    max_features="sqrt",
    class_weight="balanced",
    n_jobs=-1,
    random_state=0
)
```

选择理由：500 棵树通常能提供稳定的集成估计；`max_features="sqrt"` 是分类随机森林常用设置，可增强树之间的去相关性；`class_weight="balanced"` 用于处理不同菌属 active/inactive 比例不一致的问题。

### SVM

```text
SVC(
    C=1.0,
    kernel="rbf",
    gamma="scale",
    probability=True,
    class_weight="balanced",
    random_state=0
)
```

选择理由：RBF kernel 是小分子指纹/描述符任务中常用的非线性基线；`C=1.0` 和 `gamma="scale"` 是稳健的通用配置；`probability=True` 用于输出 PR-AUC、ROC-AUC、EF 和 Hit rate 所需的排序分数；`class_weight="balanced"` 用于类别不均衡。

### XGBoost

```text
XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="binary:logistic",
    eval_metric=["logloss", "auc", "aucpr"],
    tree_method="hist",
    scale_pos_weight=neg/pos,
    random_state=0,
    n_jobs=-1
)
```

选择理由：`learning_rate=0.05` 和 `n_estimators=500` 是较稳健的中等学习率/中等树数配置；`max_depth=6` 提供足够非线性表达但避免过深；`subsample` 和 `colsample_bytree` 增加随机性并降低过拟合；`scale_pos_weight=neg/pos` 用于类别不均衡。

### LightGBM

```text
LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.9,
    colsample_bytree=0.9,
    class_weight="balanced",
    random_state=0,
    n_jobs=-1
)
```

选择理由：`num_leaves=31` 是 LightGBM 的经典稳健配置；`learning_rate=0.05` 和 `n_estimators=500` 保持与 XGBoost 相近的模型容量；`subsample` 和 `colsample_bytree` 用于降低过拟合；`class_weight="balanced"` 用于处理 active/inactive 不均衡。

## 阈值策略

主结果默认使用固定阈值 0.5 计算 Accuracy、F1 和 MCC。ROC-AUC、PR-AUC、EF 和 Hit rate 不依赖该阈值，而是直接使用预测概率排序。

脚本也保留 `--threshold-mode val_f1` 选项，用于敏感性分析：在 validation set 上选择 F1 最优阈值，再应用到 test set。但主 benchmark 建议使用固定 0.5，避免不同模型通过阈值选择获得额外优势。

## 训练过程输出

训练脚本现在会在每个 run 的输出目录中保存以下训练过程文件：

```text
train_log.csv
training_curve.csv
train_log_long.csv
```

其中 `train_log.csv` 和 `training_curve.csv` 为宽表格式，核心列包括：

```text
iteration
train_loss
val_loss
train_auc
val_auc
train_pr_auc
val_pr_auc
note
```

这里的 `train_loss` 和 `val_loss` 指二分类 log loss。

- XGBoost：输出每一轮 boosting iteration 的 train/validation logloss、AUC 和 AUC-PR。
- LightGBM：输出每一轮 boosting iteration 的 train/validation binary_logloss 和 AUC。
- RF 和 SVM：这类 sklearn 模型没有 epoch/iteration 式训练过程，因此只输出训练结束后的最终 train/validation log_loss，并在 `note` 中标记为 `final_only_no_epoch_loss_for_this_model`。

`train_log_long.csv` 保存库原始 evals_result 风格的长表，主要用于排查不同软件包输出字段名的差异。


## Perimeter split 两种方案的训练约定

当前数据目录中 Perimeter / Max-distance split 包含两种方案：

- `repeated_stratified_perimeter`：重复分层 Perimeter split，通常包含 `seed_0` 到 `seed_4`，用于 5 套 split 的 mean ± SD 和 OOD 稳定性评价。
- `strict_perimeter`：确定性 strict Perimeter split，通常只包含 `seed_0`，用于强 OOD stress test。该 split 不应为了凑 5 个 seed 人为加入随机性。

训练脚本默认同时读取 `random`、`scaffold`、`repeated_stratified_perimeter` 和 `strict_perimeter`。如果某个 split 缺少某些 seed 文件夹，脚本会自动跳过，不会报错。

OOD Drop 的计算方式为：

```text
Scaffold OOD Drop = Metric_random - Metric_scaffold
Repeated Perimeter OOD Drop = Metric_random - Metric_repeated_stratified_perimeter
Strict Perimeter OOD Drop = Metric_random - Metric_strict_perimeter
```

其中 `strict_perimeter` 通常只有一套确定性划分，因此不应与 `random/scaffold` 的 5-seed 稳定性含义混淆。

## 模型失效诊断输出约定（12.1–12.5）

本脚本体系按五类诊断组织模型失败原因：

| 编号 | 诊断名称 | 实现位置 | 输出文件 |
|---|---|---|---|
| 12.1 | MIC-boundary error analysis | `train_eval_diagnose_classification_ml_v2_perimeter2.py` | 每个 run 的 `diagnostics_metrics.csv`、`diagnostic_subgroup_metrics.csv`、`metrics_full.csv` |
| 12.2 | Nearest-neighbor error analysis | `train_eval_diagnose_classification_ml_v2_perimeter2.py` | 每个 run 的 `predictions_test.csv`、`diagnostics_metrics.csv`、`diagnostic_subgroup_metrics.csv` |
| 12.3 | Activity cliff diagnosis | `train_eval_diagnose_classification_ml_v2_perimeter2.py` | 每个 run 的 `predictions_test.csv`、`diagnostics_metrics.csv`、`diagnostic_subgroup_metrics.csv` |
| 12.4 | Perimeter OOD error analysis | `train_eval_diagnose_classification_ml_v2_perimeter2.py` + `summarize_classification_ml_results_v2_perimeter2.py` | `_combined_summary/perimeter_ood_error_analysis.csv` |
| 12.5 | Molecular landscape roughness analysis | `compute_landscape_roughness_v2_perimeter2.py` | `_landscape_roughness/landscape_roughness_by_split_seed.csv`、`landscape_roughness_mean_sd.csv` |

### 12.4 Perimeter OOD error analysis

为使 12.4 与论文计划完全对应，汇总脚本额外输出：

```text
_combined_summary/perimeter_ood_error_analysis.csv
```

该表以 `model + dataset_name + genus + target_split` 为一行，重点比较 `random` 与两类 Perimeter split：

- `repeated_stratified_perimeter`
- `strict_perimeter`

核心字段包括：

```text
random_test_pr_auc
perimeter_test_pr_auc
perimeter_drop_test_pr_auc
random_test_EF1%
perimeter_test_EF1%
perimeter_drop_test_EF1%
perimeter_nn_lt_0p4_n
perimeter_nn_lt_0p4_error_rate
perimeter_nn_lt_0p4_pr_auc
perimeter_nn_lt_0p4_EF1%
perimeter_overall_error_rate
low_nn_error_enrichment_vs_overall
strict_minus_repeated_test_pr_auc
strict_minus_repeated_test_EF1%
```

其中：

```text
perimeter_drop_test_pr_auc = random_test_pr_auc - perimeter_test_pr_auc
perimeter_drop_test_EF1% = random_test_EF1% - perimeter_test_EF1%
```

对于 PR-AUC、EF1%、EF5%、Hit rate 等越高越好的指标，正值表示 Perimeter OOD 场景下性能下降。

`perimeter_nn_lt_0p4_*` 表示 Perimeter test set 中与训练集最近邻相似度低于 0.4 的远距离化学空间样本表现，用于区分普通新骨架泛化失败和更强的远距离化学空间外推失败。

## Output location

By default, the local classification launcher and Python training entry point
write under the repository's ignored `outputs/ML/classification/` directory.
The Slurm launcher defaults to `outputs/ML/output/`. Set `OUT_ROOT`, pass the
output argument to the launcher, or use `--outdir` to select another location.
Configured paths outside the repository are not protected by its `.gitignore`.
