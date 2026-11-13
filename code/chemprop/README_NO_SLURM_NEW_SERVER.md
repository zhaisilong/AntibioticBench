# Chemprop AntibioticBench：新服务器无 Slurm 版

## 配置与环境检查

从仓库根目录进入 `code/chemprop`。指定已准备好的数据划分目录；仓库数据压缩包内的原始表不包含预先划分的 train、validation 和 test 文件。`WORK_DIR`、`OUT_ROOT`、`CONDA_SH` 和 `CONDA_ENV` 可按需覆盖。默认使用 PATH 上的 conda，输出位于仓库的 `outputs/chemprop/` 下。

```bash
cd code/chemprop
export DATA_ROOT="path/to/prepared/splits"
# 如 conda 不在 PATH：export CONDA_SH="path/to/conda.sh"
GPU_ID=0 CONDA_ENV=chemprop-repro bash check_no_slurm_environment.sh
```

独立的 Chemprop、checkpoint 或预训练模型可能受各自的许可约束；需自行准备依赖。执行记录尽量省略本地路径，但外部工具生成的原生日志和模型文件仍应在分享前检查。

12 个完整数据集时，分类和回归应各打印 192 个 run。

## 分类测试

```bash
GPU_ID=0 \
CONDA_ENV=chemprop-repro \
RUN_MODE=test \
TEST_RUN_INDEX=0 \
POSTPROCESS=0 \
RESUME=0 \
FORCE=1 \
bash run_chemprop_classification_no_slurm.sh
```

## 回归测试

```bash
GPU_ID=0 \
CONDA_ENV=chemprop-repro \
RUN_MODE=test \
TEST_RUN_INDEX=0 \
POSTPROCESS=0 \
RESUME=0 \
FORCE=1 \
bash run_chemprop_regression_no_slurm.sh
```

## 正式分类

```bash
GPU_ID=0 \
CONDA_ENV=chemprop-repro \
RUN_MODE=batch \
POSTPROCESS=1 \
RESUME=1 \
FORCE=0 \
bash run_chemprop_classification_no_slurm.sh
```

## 正式回归

```bash
GPU_ID=0 \
CONDA_ENV=chemprop-repro \
RUN_MODE=batch \
POSTPROCESS=1 \
RESUME=1 \
FORCE=0 \
bash run_chemprop_regression_no_slurm.sh
```

建议在 tmux 中运行。使用 tmux 时不需要 nohup，也不需要 `&`。

执行逻辑：

```text
逐个 run 串行训练
→ 任一 run 失败，立即停止
→ 后续 run 和汇总均不执行
→ 全部 run 成功后自动汇总
```

固定协议：

```text
分类输入：smiles active
分类早停：最大 val PR-AUC

回归输入：smiles log10_mic
回归早停：最小 val RMSE
```

日志：

```text
outputs/chemprop/classification/local_logs/
outputs/chemprop/regression/local_logs/
```

失败时生成 `FAILED_LAST_RUN.txt`；全部成功后生成
`ALL_RUNS_AND_SUMMARY_COMPLETED.txt`。

中断后使用 `RESUME=1 FORCE=0` 重新运行，已有完整
`metrics_full.csv` 的 run 会跳过。

## 注意

- 分类和回归读取已有数据划分，不重新生成 train、validation 和 test 文件。
- 修改工作目录与数据位置后，先检查环境和任务数量；脚本执行和论文结果复现不作保证。
