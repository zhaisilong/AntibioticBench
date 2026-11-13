# v10 CPU 限制版

## 修改内容

分类和回归无 Slurm 脚本新增：

```bash
NUM_WORKERS="${NUM_WORKERS:-0}"
CPU_THREADS="${CPU_THREADS:-4}"

export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS}"
export BLIS_NUM_THREADS="${CPU_THREADS}"
export VECLIB_MAXIMUM_THREADS="${CPU_THREADS}"
export RAYON_NUM_THREADS="${CPU_THREADS}"
export TOKENIZERS_PARALLELISM=false
```

`NUM_WORKERS` 只控制 Chemprop/PyTorch DataLoader 子进程数量，不等于总 CPU 核数。
`CPU_THREADS` 用于限制常见数学库和 OpenMP 线程池。

## 推荐提交

```bash
GPU_ID=1 NUM_WORKERS=0 CPU_THREADS=4 CONDA_ENV=chemprop-repro RUN_MODE=batch POSTPROCESS=1 RESUME=1 FORCE=0 bash run_chemprop_classification_no_slurm.sh
```

回归只需把最后一行脚本名改成：

```text
run_chemprop_regression_no_slurm.sh
```

## 硬限制 CPU 核心

环境变量属于线程池限制，不是操作系统级硬限制。需要严格限定到 4 个逻辑核时，可使用：

```bash
taskset -c 0-3 env GPU_ID=1 NUM_WORKERS=0 CPU_THREADS=4 CONDA_ENV=chemprop-repro RUN_MODE=batch POSTPROCESS=1 RESUME=1 FORCE=0 bash run_chemprop_classification_no_slurm.sh
```

请根据服务器上空闲的 CPU 核心范围调整 `0-3`。
