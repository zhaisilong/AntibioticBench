# Chemprop AntibioticBench v11：CPU 32线程 + 早停方向修复

## 本版修改

1. 分类和回归脚本默认 `CPU_THREADS=32`，`NUM_WORKERS=0`。
2. 分类严格按 **validation PR-AUC 最大化** 早停和选择 checkpoint。
3. 回归严格按 **validation RMSE 最小化** 早停和选择 checkpoint。
4. 新增 `chemprop_cli_direction_guard.py`：在同一 Python 进程中、Chemprop 创建 Lightning callback 之前，显式设置：
   - `prc.higher_is_better = True`
   - `rmse.higher_is_better = False`
5. `patience=20`、`min_delta=0.0` 均显式传给 Chemprop。
6. 训练完成后执行双重校验：
   - checkpoint 文件中的 epoch 必须与训练曲线最佳 epoch 一致；
   - 该 checkpoint 在 validation 上重新预测得到的 PR-AUC/RMSE 必须与训练曲线最佳值一致（分类按同一梯形积分 AUPRC 定义复算，数值容差为 0.01）。
7. 校验失败时立即停止，不写入 `metrics_full.csv`，避免错误模型进入正式汇总。
8. v9/v10 结果没有 v11 校验标记；即使使用 `RESUME=1`，也不会被当作可信完成结果跳过。旧结果会移动到：

```text
OUT_ROOT/_untrusted_run_backups/
```

因此不会被汇总脚本误读。

运行前请参考 `README_NO_SLURM_NEW_SERVER.md` 设置 `DATA_ROOT` 和环境。

## 分类正式任务

```bash
cd code/chemprop

taskset -c 0-31 env \
GPU_ID=1 \
NUM_WORKERS=0 \
CPU_THREADS=32 \
CONDA_ENV=chemprop-repro \
RUN_MODE=batch \
POSTPROCESS=1 \
RESUME=1 \
FORCE=0 \
bash run_chemprop_classification_no_slurm.sh
```

## 回归正式任务

GPU 2 同属 NUMA node 0，可将两个任务各分配 24 个物理核心，避免争抢：

```bash
cd code/chemprop

taskset -c 24-47 env \
GPU_ID=2 \
NUM_WORKERS=0 \
CPU_THREADS=24 \
CONDA_ENV=chemprop-repro \
RUN_MODE=batch \
POSTPROCESS=1 \
RESUME=1 \
FORCE=0 \
bash run_chemprop_regression_no_slurm.sh
```

若分类和回归不同时运行，回归也可使用 `CPU_THREADS=32` 和 `taskset -c 0-31`。

## 先做单个 run 测试

```bash
GPU_ID=1 \
NUM_WORKERS=0 \
CPU_THREADS=32 \
CONDA_ENV=chemprop-repro \
RUN_MODE=test \
TEST_RUN_INDEX=0 \
POSTPROCESS=0 \
RESUME=0 \
FORCE=1 \
bash run_chemprop_classification_no_slurm.sh
```

成功日志必须包含：

```text
EARLY_STOP_DIRECTION_GUARD=...
SELECTED_CHECKPOINT: ...
CHECKPOINT_VALIDATION: ... "early_stopping_validated": true
FINISHED; outputs saved to ...
```

`metrics_full.csv` 必须包含：

```text
runner_version=v11_cpu32_earlystop_guard
early_stopping_validated=True
early_stopping_direction=max   # classification
# 或 min                         # regression
```
