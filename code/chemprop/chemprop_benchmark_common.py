#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared runner, evaluator, diagnostics and summarizer for AntibioticBench Chemprop.

Designed for Chemprop v2 CLI and an already-created directory structure:

DATA_ROOT/
  dataset_name/
    classification|regression/
      random|scaffold|repeated_stratified_perimeter|strict_perimeter/
        seed_0/
          train.csv
          validation.csv
          test.csv

The script NEVER creates a new data split. The three existing CSV files are passed
into ``chemprop train`` as train/validation/test in that exact order.
"""

from __future__ import annotations

from privacy_utils import sanitize_metadata

import argparse
import datetime as dt
import json
import math
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
except Exception as exc:  # pragma: no cover
    raise RuntimeError("RDKit is required in the Chemprop environment.") from exc

try:
    from scipy.stats import pearsonr, spearmanr
except Exception as exc:  # pragma: no cover
    raise RuntimeError("scipy is required in the Chemprop environment.") from exc

try:
    from sklearn.metrics import (
        accuracy_score,
        auc,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        log_loss,
        mean_absolute_error,
        mean_squared_error,
        median_absolute_error,
        matthews_corrcoef,
        precision_recall_curve,
        precision_score,
        r2_score,
        recall_score,
        roc_auc_score,
    )
except Exception as exc:  # pragma: no cover
    raise RuntimeError("scikit-learn is required in the Chemprop environment.") from exc

RDLogger.DisableLog("rdApp.*")

MODEL_NAME = "Chemprop"
MODEL_FAMILY = "D-MPNN"
RUNNER_VERSION = "v11_cpu32_earlystop_guard"
EARLY_STOP_POLICY_VERSION = "antibioticbench_v11_metric_direction_guard"
DEFAULT_SPLITS = ["random", "scaffold", "repeated_stratified_perimeter", "strict_perimeter"]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
SMILES_COL = "smiles"
CLASS_TARGET_COL = "active"
REG_TARGET_COL = "log10_mic"
MIC_CANDIDATES = [
    "mic_numeric", "value", "MIC", "mic", "mic_value", "median_value",
    "value_ug_ml", "mic_ug_ml",
]
LOG2_MIC_CANDIDATES = ["log2_mic", "log2_MIC", "log2_value", "logMIC", "logmic"]


def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def early_stopping_policy(task: str) -> Dict[str, Any]:
    if task == "classification":
        tracking_metric, direction = "prc", "max"
    elif task == "regression":
        tracking_metric, direction = "rmse", "min"
    else:
        raise ValueError(f"Unsupported task for early stopping policy: {task}")
    return {
        "policy_version": EARLY_STOP_POLICY_VERSION,
        "runner_version": RUNNER_VERSION,
        "task": task,
        "tracking_metric": tracking_metric,
        "direction": direction,
        "patience_semantics": "stop after patience consecutive validation epochs without improvement",
    }


def archive_untrusted_run(out_dir: Path, out_root: str | Path, reason: str) -> Optional[Path]:
    """Move a legacy/partial run outside ``OUT_ROOT/Chemprop`` so summaries cannot ingest it."""
    if not out_dir.exists() or not any(out_dir.iterdir()):
        return None
    model_root = Path(out_root).resolve() / MODEL_NAME
    try:
        rel = out_dir.resolve().relative_to(model_root)
    except ValueError:
        rel = Path(out_dir.name)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup = Path(out_root).resolve() / "_untrusted_run_backups" / rel / f"{stamp}_{reason}"
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(out_dir), str(backup))
    return backup


def completed_metrics_are_trusted(metrics_path: Path, task: str) -> bool:
    try:
        row = pd.read_csv(metrics_path).iloc[0]
    except Exception:
        return False
    validated = str(row.get("early_stopping_validated", "")).strip().lower() in {"true", "1", "yes"}
    return (
        validated
        and str(row.get("runner_version", "")) == RUNNER_VERSION
        and str(row.get("early_stopping_policy_version", "")) == EARLY_STOP_POLICY_VERSION
        and str(row.get("task", "")) == task
    )


def native_training_is_trusted(native_dir: Path, task: str) -> bool:
    marker = native_dir / "TRAINING_COMPLETED.json"
    if not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return False
    expected = early_stopping_policy(task)
    return all(payload.get(k) == expected.get(k) for k in ("policy_version", "runner_version", "task", "tracking_metric", "direction"))


def parse_csv_list(value: str | Sequence[str] | None, cast=str) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        raw = [x for x in re.split(r"[,\s]+", str(value).strip()) if x]
    return [cast(x) for x in raw]


def numeric_sort_key(path: Path) -> Tuple[int, str]:
    m = re.match(r"^(\d+)_", path.name)
    return (int(m.group(1)) if m else 10**9, path.name)


def parse_genus(dataset_name: str) -> str:
    parts = dataset_name.split("_")
    return parts[1] if len(parts) >= 2 and parts[0].isdigit() else dataset_name


def parse_threshold(dataset_name: str, fallback: float = 32.0) -> float:
    m = re.search(r"median_([0-9]+(?:\.[0-9]+)?)", dataset_name)
    return float(m.group(1)) if m else float(fallback)


def first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b not in (0, 0.0) and np.isfinite(b) else np.nan


def safe_corr(y_true: np.ndarray, y_pred: np.ndarray, method: str) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 3:
        return np.nan
    x = y_true[mask]
    y = y_pred[mask]
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    try:
        return float(spearmanr(x, y).statistic if method == "spearman" else pearsonr(x, y).statistic)
    except Exception:
        return np.nan


def safe_roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y, score)) if len(np.unique(y)) >= 2 else np.nan
    except Exception:
        return np.nan


def safe_pr_auc(y: np.ndarray, score: np.ndarray) -> float:
    try:
        return float(average_precision_score(y, score)) if len(np.unique(y)) >= 2 else np.nan
    except Exception:
        return np.nan


@dataclass(frozen=True)
class RunSpec:
    dataset_name: str
    genus: str
    task: str
    split_type: str
    seed: int
    threshold_for_boundary: float
    seed_dir: str
    train_csv: str
    validation_csv: str
    test_csv: str


def discover_run_specs(
    data_root: str | Path,
    task: str,
    split_types: Sequence[str] = DEFAULT_SPLITS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    datasets: Optional[Sequence[str]] = None,
    threshold_fallback: float = 32.0,
) -> List[RunSpec]:
    root = Path(data_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError("DATA_ROOT does not exist")
    keep = set(datasets or [])
    dataset_dirs = sorted(
        [p for p in root.iterdir() if p.is_dir() and (p / task).is_dir()],
        key=numeric_sort_key,
    )
    if keep:
        dataset_dirs = [p for p in dataset_dirs if p.name in keep or parse_genus(p.name) in keep]

    specs: List[RunSpec] = []
    missing: List[str] = []
    for ds in dataset_dirs:
        for split_type in split_types:
            split_seeds = [0] if split_type == "strict_perimeter" else list(seeds)
            for seed in split_seeds:
                seed_dir = ds / task / split_type / f"seed_{seed}"
                train = seed_dir / "train.csv"
                val = seed_dir / "validation.csv"
                test = seed_dir / "test.csv"
                if all(x.is_file() for x in (train, val, test)):
                    specs.append(
                        RunSpec(
                            dataset_name=ds.name,
                            genus=parse_genus(ds.name),
                            task=task,
                            split_type=split_type,
                            seed=int(seed),
                            threshold_for_boundary=parse_threshold(ds.name, threshold_fallback),
                            seed_dir=str(seed_dir),
                            train_csv=str(train),
                            validation_csv=str(val),
                            test_csv=str(test),
                        )
                    )
                else:
                    missing.append(str(seed_dir))
    if not specs:
        raise RuntimeError(
            f"No complete runs found in the configured data root for task={task}. "
            "Each run must contain train.csv, validation.csv and test.csv."
        )
    if missing:
        print(f"[{now()}] NOTE: skipped {len(missing)} incomplete/missing run folders.", flush=True)
    return specs


def run_output_dir(out_root: str | Path, spec: RunSpec) -> Path:
    return Path(out_root).resolve() / MODEL_NAME / spec.dataset_name / spec.task / spec.split_type / f"seed_{spec.seed}"


def log_message(log_path: Path, text: str) -> None:
    line = f"[{now()}] {text}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_command(cmd: Sequence[str], log_path: Path, cwd: Optional[Path] = None) -> None:
    log_message(log_path, "Starting Chemprop command (arguments omitted from log)")
    private_paths = {str(Path.home()), str(log_path.parent), socket.gethostname()}
    if cwd:
        private_paths.add(str(cwd))
    for arg in cmd:
        value = str(arg)
        if Path(value).is_absolute():
            private_paths.add(value)
            parent = Path(value).parent
            if len(parent.parts) > 2:
                private_paths.add(str(parent))
    private_paths.discard("/")

    def redact(line: str) -> str:
        for value in sorted(private_paths, key=len, reverse=True):
            if value:
                line = line.replace(value, "<local-path>")
        return line

    cli_log = None
    if "--logfile" in cmd:
        cli_log = Path(cmd[cmd.index("--logfile") + 1])
    with log_path.open("a", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            list(map(str, cmd)), cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            line = redact(line)
            print(line, end="", flush=True)
            logf.write(line)
            logf.flush()
        rc = proc.wait()
    # Chemprop writes its own logfile separately. Preserve metric text used by
    # the curve parser while removing paths after the child process closes it.
    if cli_log is not None and cli_log.is_file():
        content = cli_log.read_text(encoding="utf-8", errors="replace")
        cli_log.write_text(redact(content), encoding="utf-8")
    if rc != 0:
        raise RuntimeError(f"Chemprop process failed with exit code {rc}") from None


def validate_input_csv(path: str | Path, task: str) -> pd.DataFrame:
    """Validate the fixed AntibioticBench input contract.

    Classification always uses exactly ``smiles`` as the molecular input column
    and ``active`` as the only training target. Regression always uses exactly
    ``smiles`` and ``log10_mic``. Extra metadata/diagnostic columns are allowed,
    but are never passed to Chemprop as targets.
    """
    df = pd.read_csv(path)
    target_col = CLASS_TARGET_COL if task == "classification" else REG_TARGET_COL
    required = [SMILES_COL, target_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{Path(path).name}: missing required columns {missing}. "
            f"Required columns for task={task}: {required}; columns={list(df.columns)}"
        )
    if df.empty:
        raise ValueError(f"{Path(path).name}: file is empty")

    smiles = df[SMILES_COL]
    if smiles.isna().any() or smiles.astype(str).str.strip().eq("").any():
        raise ValueError(f"{Path(path).name}: smiles contains missing or empty values")

    target = pd.to_numeric(df[target_col], errors="coerce")
    if target.isna().any():
        bad_n = int(target.isna().sum())
        raise ValueError(f"{Path(path).name}: {target_col} contains {bad_n} missing/non-numeric values")

    if task == "classification":
        values = set(target.astype(int).unique().tolist())
        if not values.issubset({0, 1}) or not np.allclose(target, target.astype(int)):
            raise ValueError(
                f"{Path(path).name}: active must contain only binary numeric labels 0/1; "
                f"observed values={sorted(set(target.unique().tolist()))[:20]}"
            )
        df[target_col] = target.astype(int)
    else:
        if not np.isfinite(target.to_numpy(dtype=float)).all():
            raise ValueError(f"{Path(path).name}: log10_mic contains non-finite values")
        df[target_col] = target.astype(float)

    return df


def checkpoint_epoch_1based(path: Path) -> Optional[int]:
    """Parse Lightning's zero-based ``epoch=N`` filename field as a one-based epoch."""
    m = re.search(r"(?:^|[-_])epoch[=_-](\d+)(?:[-_.]|$)", path.name, flags=re.I)
    return int(m.group(1)) + 1 if m else None


def find_best_model(native_dir: Path, expected_best_epoch: Optional[int] = None) -> Path:
    # Prefer an epoch-bearing checkpoint matching the best validation epoch.
    ckpts = sorted(native_dir.rglob("*.ckpt"))
    if expected_best_epoch is not None:
        matches = [p for p in ckpts if checkpoint_epoch_1based(p) == int(expected_best_epoch)]
        if matches:
            named_best = [p for p in matches if "best" in p.name.lower()]
            return (named_best or matches)[0]
        epoch_bearing = [(p, checkpoint_epoch_1based(p)) for p in ckpts if checkpoint_epoch_1based(p) is not None]
        if epoch_bearing:
            observed = ", ".join(f"{p.name}->epoch{e}" for p, e in epoch_bearing[:20])
            raise RuntimeError(
                "Checkpoint/validation-curve mismatch: "
                f"expected best epoch={expected_best_epoch}, available checkpoints={observed}. "
                "Stopping before test evaluation so an incorrect checkpoint cannot enter the benchmark."
            )

    preferred = sorted(native_dir.rglob("best.pt"))
    if preferred:
        return preferred[0]
    preferred_ckpt = sorted(native_dir.rglob("best*.ckpt"))
    if preferred_ckpt:
        return preferred_ckpt[0]
    pts = sorted(native_dir.rglob("*.pt"))
    if pts:
        return pts[0]
    if ckpts:
        return ckpts[0]
    raise FileNotFoundError("No Chemprop .pt/.ckpt model found in the configured model directory")


def validate_selected_model_score(
    task: str,
    best_val_metric: float,
    val_df: pd.DataFrame,
    val_pred: np.ndarray,
) -> Dict[str, Any]:
    """Cross-check the selected checkpoint against the logged best validation metric."""
    if task == "classification":
        y = pd.to_numeric(val_df[CLASS_TARGET_COL], errors="coerce").to_numpy(dtype=int)
        score = np.asarray(val_pred, dtype=float)
        precision, recall, _ = precision_recall_curve(y, score)
        observed = float(auc(recall, precision))
        # Match Chemprop's trapezoidal AUPRC definition; allow only small numerical differences.
        tolerance = 0.01
        direction = "max"
        metric_name = "val_pr_auc"
    else:
        y = pd.to_numeric(val_df[REG_TARGET_COL], errors="coerce").to_numpy(dtype=float)
        observed = float(np.sqrt(mean_squared_error(y, np.asarray(val_pred, dtype=float))))
        tolerance = max(1e-4, 0.01 * abs(float(best_val_metric))) if np.isfinite(best_val_metric) else np.nan
        direction = "min"
        metric_name = "val_rmse"

    delta = abs(float(observed) - float(best_val_metric)) if np.isfinite(observed) and np.isfinite(best_val_metric) else np.nan
    passed = bool(np.isfinite(delta) and delta <= tolerance)
    if not passed:
        raise RuntimeError(
            "Selected checkpoint failed validation-metric consistency check: "
            f"task={task}, metric={metric_name}, direction={direction}, "
            f"training_curve_best={best_val_metric}, checkpoint_prediction_metric={observed}, "
            f"absolute_delta={delta}, tolerance={tolerance}. "
            "Test metrics were not written. Remove/archive the run and retrain with the v11 metric-direction guard."
        )
    return {
        "early_stopping_validated": True,
        "early_stopping_direction": direction,
        "checkpoint_validation_metric": metric_name,
        "checkpoint_validation_observed": observed,
        "checkpoint_validation_reference": best_val_metric,
        "checkpoint_validation_abs_delta": delta,
        "checkpoint_validation_tolerance": tolerance,
    }


def copy_best_model(model_path: Path, out_dir: Path) -> Path:
    suffix = model_path.suffix if model_path.suffix in {".pt", ".ckpt"} else ".pt"
    dst = out_dir / f"best{suffix}"
    if model_path.resolve() != dst.resolve():
        shutil.copy2(model_path, dst)
    return dst


def _events_to_curve(native_dir: Path, task: str) -> pd.DataFrame:
    """Read all available Chemprop/Lightning scalar histories from TensorBoard."""
    events = sorted(native_dir.rglob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
    if not events:
        return pd.DataFrame()
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return pd.DataFrame()

    ea = EventAccumulator(str(events[-1]), size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if not tags:
        return pd.DataFrame()

    values: Dict[str, List[float]] = {}
    steps: Dict[str, List[int]] = {}
    for tag in tags:
        try:
            evs = ea.Scalars(tag)
        except Exception:
            continue
        values[tag] = [float(e.value) for e in evs]
        steps[tag] = [int(e.step) for e in evs]

    def choose_tag(candidates: Sequence[str], contains: Sequence[str] = ()) -> Optional[str]:
        for c in candidates:
            if c in values:
                return c
        for t in values:
            low = t.lower()
            if all(x.lower() in low for x in contains):
                return t
        return None

    train_loss_tag = choose_tag(["train_loss_epoch", "train_loss"], contains=("train", "loss"))
    val_loss_tag = choose_tag(["val_loss", "val/loss"], contains=("val", "loss"))
    tag_map: Dict[str, Optional[str]] = {
        "train_loss": train_loss_tag,
        "val_loss": val_loss_tag,
        "learning_rate": choose_tag(["lr-Adam", "lr-Adam/pg1", "lr"], contains=("lr",)),
    }
    if task == "classification":
        tag_map.update({
            "val_pr_auc": choose_tag(["val/prc", "val_prc", "val/auprc"], contains=("val", "prc")),
            "val_auc": choose_tag(["val/roc", "val_roc", "val/auroc"], contains=("val", "roc")),
            "val_accuracy": choose_tag(["val/accuracy", "val_accuracy"], contains=("val", "accuracy")),
            "val_f1": choose_tag(["val/f1", "val_f1"], contains=("val", "f1")),
            "val_mcc": choose_tag(["val/binary-mcc", "val_binary_mcc"], contains=("val", "mcc")),
        })
    else:
        tag_map.update({
            "val_rmse": choose_tag(["val/rmse", "val_rmse"], contains=("val", "rmse")),
            "val_mae": choose_tag(["val/mae", "val_mae"], contains=("val", "mae")),
            "val_mse": choose_tag(["val/mse", "val_mse"], contains=("val", "mse")),
            "val_r2": choose_tag(["val/r2", "val_r2"], contains=("val", "r2")),
        })

    # Epoch-level tags normally have one value per epoch. Use the longest non-LR series.
    lengths = [len(values[t]) for key, t in tag_map.items() if t and key != "learning_rate"]
    n = max(lengths) if lengths else 0
    if n == 0:
        return pd.DataFrame()
    out = pd.DataFrame({"iteration": np.arange(1, n + 1), "epoch": np.arange(1, n + 1)})
    for col, tag in tag_map.items():
        arr = values.get(tag, []) if tag else []
        # LR can be logged per batch. Sample it at approximately one value per epoch.
        if col == "learning_rate" and len(arr) > n:
            idx = np.linspace(0, len(arr) - 1, n).astype(int)
            arr = [arr[i] for i in idx]
        if len(arr) < n:
            arr = list(arr) + [np.nan] * (n - len(arr))
        out[col] = np.asarray(arr[:n], dtype=float)
    return out


def _csv_metrics_to_curve(native_dir: Path, task: str) -> pd.DataFrame:
    files = sorted(native_dir.rglob("metrics.csv"), key=lambda p: p.stat().st_mtime)
    for f in reversed(files):
        try:
            raw = pd.read_csv(f)
        except Exception:
            continue
        if raw.empty:
            continue
        # Lightning CSV logs one scalar event per row. Collapse by epoch using last non-null.
        if "epoch" in raw.columns:
            grouped = raw.groupby("epoch", dropna=True).last(numeric_only=False).reset_index()
        else:
            grouped = raw.copy()
            grouped["epoch"] = np.arange(1, len(grouped) + 1)
        rename = {
            "train_loss_epoch": "train_loss",
            "val/loss": "val_loss",
            "val/prc": "val_pr_auc",
            "val/roc": "val_auc",
            "val/accuracy": "val_accuracy",
            "val/f1": "val_f1",
            "val/binary-mcc": "val_mcc",
            "val/rmse": "val_rmse",
            "val/mae": "val_mae",
            "val/mse": "val_mse",
            "val/r2": "val_r2",
        }
        grouped = grouped.rename(columns=rename)
        grouped["epoch"] = pd.to_numeric(grouped["epoch"], errors="coerce") + (0 if grouped["epoch"].min() >= 1 else 1)
        grouped["iteration"] = grouped["epoch"]
        return grouped
    return pd.DataFrame()




def _cli_log_to_curve(out_dir: Path) -> pd.DataFrame:
    """Best-effort fallback parser for Lightning progress-bar text logs."""
    candidates = [out_dir / "chemprop_cli.log", out_dir / "run.log"]
    epoch_rows: Dict[int, Dict[str, float]] = {}
    for path in candidates:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").replace("\r", "\n")
        except Exception:
            continue
        for line in text.splitlines():
            em = re.search(r"Epoch\s+(\d+)", line, flags=re.I)
            if not em:
                em = re.search(r"epoch[=:\s]+(\d+)", line, flags=re.I)
            if not em:
                continue
            epoch0 = int(em.group(1))
            row = epoch_rows.setdefault(epoch0, {"epoch": epoch0 + 1, "iteration": epoch0 + 1})
            patterns = {
                "train_loss": r"train_loss(?:_epoch)?[=:\s]+([-+0-9.eE]+)",
                "val_loss": r"val_loss[=:\s]+([-+0-9.eE]+)",
                "val_pr_auc": r"(?:val[/_](?:prc|pr_auc|auprc))[=:\s]+([-+0-9.eE]+)",
                "val_auc": r"(?:val[/_](?:roc|auc|auroc))[=:\s]+([-+0-9.eE]+)",
                "val_rmse": r"val[/_]rmse[=:\s]+([-+0-9.eE]+)",
                "val_mae": r"val[/_]mae[=:\s]+([-+0-9.eE]+)",
            }
            for col, pat in patterns.items():
                m = re.search(pat, line, flags=re.I)
                if m:
                    try:
                        row[col] = float(m.group(1))
                    except Exception:
                        pass
    if not epoch_rows:
        return pd.DataFrame()
    return pd.DataFrame([epoch_rows[k] for k in sorted(epoch_rows)])

def parse_training_curve(native_dir: Path, task: str, out_dir: Path) -> Tuple[pd.DataFrame, int, float, str]:
    curve = _events_to_curve(native_dir, task)
    source = "tensorboard_event"
    if curve.empty:
        curve = _csv_metrics_to_curve(native_dir, task)
        source = "lightning_metrics_csv"
    if curve.empty:
        curve = _cli_log_to_curve(out_dir)
        source = "chemprop_cli_log"
    if curve.empty:
        curve = pd.DataFrame([{"iteration": 1, "epoch": 1, "train_loss": np.nan, "val_loss": np.nan}])
        source = "unavailable_fallback"

    required = ["train_loss", "val_loss", "learning_rate"]
    if task == "classification":
        required += ["train_log_loss", "val_log_loss", "train_auc", "val_auc", "train_pr_auc", "val_pr_auc",
                     "train_accuracy", "val_accuracy", "train_f1", "val_f1", "train_mcc", "val_mcc"]
        selection_col, mode = ("val_pr_auc", "max") if "val_pr_auc" in curve and curve["val_pr_auc"].notna().any() else ("val_loss", "min")
    else:
        required += ["train_mse", "val_mse", "train_rmse", "val_rmse", "train_mae", "val_mae",
                     "train_spearman", "val_spearman", "train_pearson", "val_pearson",
                     "train_within_1_dilution", "val_within_1_dilution",
                     "train_within_2_dilution", "val_within_2_dilution"]
        selection_col, mode = ("val_rmse", "min") if "val_rmse" in curve and curve["val_rmse"].notna().any() else ("val_loss", "min")
    for c in required:
        if c not in curve:
            curve[c] = np.nan

    vals = pd.to_numeric(curve[selection_col], errors="coerce")
    if vals.notna().any():
        idx = vals.idxmax() if mode == "max" else vals.idxmin()
        best_epoch = int(curve.loc[idx, "epoch"])
        best_value = float(vals.loc[idx])
        running = vals.cummax() if mode == "max" else vals.cummin()
    else:
        best_epoch = int(curve["epoch"].iloc[-1])
        best_value = np.nan
        running = pd.Series(np.nan, index=curve.index)
    best_epoch_so_far = []
    for i in curve.index:
        prefix = pd.to_numeric(curve.loc[:i, selection_col], errors="coerce")
        if not prefix.notna().any():
            best_epoch_so_far.append(np.nan)
            continue
        best_idx = prefix.idxmax() if mode == "max" else prefix.idxmin()
        best_epoch_so_far.append(int(curve.loc[best_idx, "epoch"]))
    curve["best_epoch_so_far"] = best_epoch_so_far
    curve["best_val_metric_so_far"] = running
    curve["note"] = (
        f"source={source}; Chemprop CLI natively logs train_loss and validation metrics. "
        "Final train/validation metrics from the selected best model are stored in metrics_full.csv."
    )
    curve.to_csv(out_dir / "train_log.csv", index=False)
    curve.to_csv(out_dir / "training_curve.csv", index=False)

    id_cols = [c for c in ["iteration", "epoch"] if c in curve.columns]
    value_cols = [c for c in curve.columns if c not in id_cols + ["note"]]
    long_rows: List[Dict[str, Any]] = []
    for _, row in curve.iterrows():
        for c in value_cols:
            if pd.notna(row[c]):
                split = "train" if c.startswith("train_") else "validation" if c.startswith("val_") else "training"
                long_rows.append({
                    "iteration": row.get("iteration"), "epoch": row.get("epoch"),
                    "split": split, "metric": c, "value": row[c],
                })
    pd.DataFrame(long_rows).to_csv(out_dir / "train_log_long.csv", index=False)
    return curve, best_epoch, best_value, selection_col


def build_chemprop_train_command(args: argparse.Namespace, spec: RunSpec, native_dir: Path, cli_log: Path) -> List[str]:
    task = spec.task
    metrics = ["roc", "prc", "accuracy", "f1", "binary-mcc"] if task == "classification" else ["rmse", "mae", "mse", "r2"]
    # Classification selects and early-stops by maximum validation PR-AUC; regression by minimum validation RMSE.
    tracking = "prc" if task == "classification" else "rmse"
    loss = "bce" if task == "classification" else "mse"
    guard_script = Path(__file__).resolve().with_name("chemprop_cli_direction_guard.py")
    if not guard_script.is_file():
        raise FileNotFoundError(f"Missing Chemprop metric-direction guard: {guard_script}")
    cmd = [
        sys.executable, str(guard_script), "train",
        "-i", spec.train_csv, spec.validation_csv, spec.test_csv,
        "-t", task,
        "--smiles-columns", SMILES_COL,
        "--target-columns", CLASS_TARGET_COL if task == "classification" else REG_TARGET_COL,
        "--loss-function", loss,
        "--metrics", *metrics,
        "--tracking-metric", tracking,
        "--ensemble-size", "1",
        "--message-hidden-dim", str(args.message_hidden_dim),
        "--depth", str(args.depth),
        "--dropout", str(args.dropout),
        "--ffn-hidden-dim", str(args.ffn_hidden_dim),
        "--ffn-num-layers", str(args.ffn_num_layers),
        "--aggregation", args.aggregation,
        "--aggregation-norm", str(args.aggregation_norm),
        "--activation", args.activation,
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--min-delta", str(args.min_delta),
        "--warmup-epochs", str(args.warmup_epochs),
        "--batch-size", str(args.batch_size),
        "--init-lr", str(args.init_lr),
        "--max-lr", str(args.max_lr),
        "--final-lr", str(args.final_lr),
        "--grad-clip", str(args.grad_clip),
        "--data-seed", str(spec.seed),
        "--pytorch-seed", str(spec.seed),
        "--num-workers", str(args.num_workers),
        "--accelerator", args.accelerator,
        "--devices", str(args.devices),
        "--output-dir", str(native_dir),
        "--save-data-splits",
        "--logfile", str(cli_log),
        "-v",
    ]
    if task == "classification" and args.class_balance:
        cmd.append("--class-balance")
    if args.no_cache:
        cmd.append("--no-cache")
    return cmd


def run_prediction(
    args: argparse.Namespace,
    model_path: Path,
    source_csv: str | Path,
    split_name: str,
    raw_dir: Path,
    run_log: Path,
) -> np.ndarray:
    src = pd.read_csv(source_csv)
    pred_input = raw_dir / f"{split_name}_smiles_only.csv"
    raw_output = raw_dir / f"{split_name}_chemprop_raw.csv"
    src[[SMILES_COL]].to_csv(pred_input, index=False)
    cmd = [
        args.chemprop_executable, "predict",
        "-i", str(pred_input),
        "--model-paths", str(model_path),
        "-o", str(raw_output),
        "--smiles-columns", SMILES_COL,
        "--batch-size", str(args.eval_batch_size),
        "--num-workers", str(args.num_workers),
        "--accelerator", args.accelerator,
        "--devices", str(args.devices),
    ]
    run_command(cmd, run_log)
    raw = pd.read_csv(raw_output)
    if len(raw) != len(src):
        raise RuntimeError(
            f"Prediction row count mismatch for {split_name}: input={len(src)} output={len(raw)}. "
            "This usually indicates invalid SMILES or dropped rows; stop to avoid misaligned metrics."
        )
    candidate_cols = [c for c in raw.columns if c != SMILES_COL]
    numeric_candidates: List[str] = []
    for c in candidate_cols:
        x = pd.to_numeric(raw[c], errors="coerce")
        if x.notna().sum() == len(raw):
            numeric_candidates.append(c)
    if not numeric_candidates:
        raise ValueError(f"Cannot identify prediction column in {raw_output}; columns={list(raw.columns)}")
    pred_col = numeric_candidates[-1]
    pred = pd.to_numeric(raw[pred_col], errors="coerce").to_numpy(dtype=float)
    return pred


# ------------------------- molecular diagnostics -------------------------

def smiles_to_fp(smiles: str):
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    except Exception:
        return None


def bulk_tanimoto(fp, train_fps: Sequence[Any]) -> np.ndarray:
    sims = np.full(len(train_fps), np.nan, dtype=float)
    if fp is None:
        return sims
    valid = [i for i, x in enumerate(train_fps) if x is not None]
    if not valid:
        return sims
    vals = DataStructs.BulkTanimotoSimilarity(fp, [train_fps[i] for i in valid])
    sims[np.asarray(valid, dtype=int)] = np.asarray(vals, dtype=float)
    return sims


def nn_bin_label(sim: float) -> str:
    if not np.isfinite(sim):
        return "NA"
    if sim >= 0.7:
        return ">=0.7"
    if sim >= 0.4:
        return "0.4-0.7"
    return "<0.4"


def infer_mic_fields(df: pd.DataFrame, threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mic_col = first_existing_col(df, MIC_CANDIDATES)
    log2_col = first_existing_col(df, LOG2_MIC_CANDIDATES)
    if mic_col is not None:
        mic = pd.to_numeric(df[mic_col], errors="coerce").to_numpy(dtype=float)
        log10mic = np.where(mic > 0, np.log10(mic), np.nan)
        log2mic = np.where(mic > 0, np.log2(mic), np.nan)
    elif REG_TARGET_COL in df.columns:
        log10mic = pd.to_numeric(df[REG_TARGET_COL], errors="coerce").to_numpy(dtype=float)
        mic = np.power(10.0, log10mic)
        log2mic = log10mic / math.log10(2.0)
    elif log2_col is not None:
        log2mic = pd.to_numeric(df[log2_col], errors="coerce").to_numpy(dtype=float)
        mic = np.power(2.0, log2mic)
        log10mic = np.log10(mic)
    else:
        n = len(df)
        mic = log10mic = log2mic = np.full(n, np.nan, dtype=float)
    boundary = np.where(np.isfinite(mic), ((mic >= threshold / 2) & (mic <= threshold * 2)).astype(int), np.nan)
    ranges = np.full(len(df), "NA", dtype=object)
    finite = np.isfinite(mic)
    ranges[finite & (mic <= threshold / 2)] = "strong_MIC_le_T_over_2"
    ranges[finite & (mic > threshold / 2) & (mic <= threshold * 2)] = "boundary_T_over_2_to_2T"
    ranges[finite & (mic > threshold * 2)] = "weak_MIC_gt_2T"
    return mic, log10mic, log2mic, boundary, ranges


def add_test_diagnostics_classification(
    test_pred: pd.DataFrame,
    train_pred: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    out = test_pred.copy()
    train = train_pred.copy()
    train_fps = [smiles_to_fp(s) for s in train[SMILES_COL].astype(str)]
    test_fps = [smiles_to_fp(s) for s in out[SMILES_COL].astype(str)]
    train_mic, train_log10, train_log2, _, _ = infer_mic_fields(train, threshold)
    test_mic, test_log10, test_log2, boundary, ranges = infer_mic_fields(out, threshold)
    out["mic_numeric"] = test_mic
    out["log2_mic_for_diagnosis"] = test_log2
    out["log10_mic_for_diagnosis"] = test_log10
    out["boundary_flag"] = boundary
    out["threshold_for_boundary"] = threshold
    out["mic_range_group"] = ranges

    rows: List[Dict[str, Any]] = []
    train_scores = pd.to_numeric(train["pred_score"], errors="coerce").to_numpy(dtype=float)
    train_active = pd.to_numeric(train["true_active"], errors="coerce").to_numpy(dtype=float)
    test_scores = pd.to_numeric(out["pred_score"], errors="coerce").to_numpy(dtype=float)
    for i, fp in enumerate(test_fps):
        sims = bulk_tanimoto(fp, train_fps)
        if np.isfinite(sims).any():
            j = int(np.nanargmax(sims))
            nn_sim = float(sims[j])
        else:
            j, nn_sim = -1, np.nan
        cliffs: np.ndarray = np.array([], dtype=int)
        max_diff = np.nan
        dir_acc = np.nan
        if np.isfinite(test_log2[i]) and np.isfinite(sims).any():
            diffs = np.abs(train_log2 - test_log2[i])
            cliffs = np.where((sims >= 0.7) & np.isfinite(diffs) & (diffs >= 4.0))[0]
            if len(cliffs):
                max_diff = float(np.nanmax(diffs[cliffs]))
                true_direction = train_log2[cliffs] - test_log2[i]
                pred_direction = test_scores[i] - train_scores[cliffs]
                valid = np.isfinite(true_direction) & np.isfinite(pred_direction) & (true_direction != 0)
                if valid.any():
                    dir_acc = float(np.mean(pred_direction[valid] * true_direction[valid] > 0))
        rows.append({
            "nn_train_row": j if j >= 0 else np.nan,
            "nn_sim": nn_sim,
            "nn_bin": nn_bin_label(nn_sim),
            "nn_train_smiles": train.iloc[j][SMILES_COL] if j >= 0 else np.nan,
            "nn_train_active": train_active[j] if j >= 0 else np.nan,
            "nn_train_score": train_scores[j] if j >= 0 else np.nan,
            "ac_flag": int(len(cliffs) > 0),
            "ac_neighbor_count": int(len(cliffs)),
            "ac_max_abs_log2mic_diff": max_diff,
            "ac_pair_directional_accuracy_for_molecule": dir_acc,
            "ac_pair_count_for_molecule": int(len(cliffs)),
        })
    return pd.concat([out.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def add_test_diagnostics_regression(
    test_pred: pd.DataFrame,
    train_pred: pd.DataFrame,
    threshold: float,
    local_k: int = 10,
) -> pd.DataFrame:
    out = test_pred.copy()
    train = train_pred.copy()
    train_fps = [smiles_to_fp(s) for s in train[SMILES_COL].astype(str)]
    test_fps = [smiles_to_fp(s) for s in out[SMILES_COL].astype(str)]
    train_true = pd.to_numeric(train["true_log10MIC"], errors="coerce").to_numpy(dtype=float)
    train_pred_values = pd.to_numeric(train["pred_log10MIC"], errors="coerce").to_numpy(dtype=float)
    test_true = pd.to_numeric(out["true_log10MIC"], errors="coerce").to_numpy(dtype=float)
    test_pred_values = pd.to_numeric(out["pred_log10MIC"], errors="coerce").to_numpy(dtype=float)
    mic, _, _, boundary, ranges = infer_mic_fields(out, threshold)
    out["mic_numeric"] = mic
    out["boundary_flag"] = boundary
    out["threshold_for_boundary"] = threshold
    out["mic_range_group"] = ranges

    rows: List[Dict[str, Any]] = []
    cutoff = math.log10(16.0)
    for i, fp in enumerate(test_fps):
        sims = bulk_tanimoto(fp, train_fps)
        if np.isfinite(sims).any():
            j = int(np.nanargmax(sims))
            nn_sim = float(sims[j])
            valid_idx = np.where(np.isfinite(sims))[0]
            order = valid_idx[np.argsort(-sims[valid_idx])]
            top = order[: min(local_k, len(order))]
        else:
            j, nn_sim, top = -1, np.nan, np.array([], dtype=int)
        local_vals = train_true[top]
        local_vals = local_vals[np.isfinite(local_vals)]
        local_var = float(np.var(local_vals, ddof=1)) if len(local_vals) >= 2 else np.nan
        local_sd = float(np.std(local_vals, ddof=1)) if len(local_vals) >= 2 else np.nan

        cliffs: np.ndarray = np.array([], dtype=int)
        max_diff = np.nan
        dir_acc = np.nan
        if np.isfinite(test_true[i]) and np.isfinite(sims).any():
            diffs = np.abs(train_true - test_true[i])
            cliffs = np.where((sims >= 0.7) & np.isfinite(diffs) & (diffs >= cutoff))[0]
            if len(cliffs):
                max_diff = float(np.nanmax(diffs[cliffs]))
                true_direction = train_true[cliffs] - test_true[i]
                pred_direction = train_pred_values[cliffs] - test_pred_values[i]
                valid = np.isfinite(true_direction) & np.isfinite(pred_direction) & (true_direction != 0)
                if valid.any():
                    dir_acc = float(np.mean(pred_direction[valid] * true_direction[valid] > 0))
        rows.append({
            "nn_train_row": j if j >= 0 else np.nan,
            "nn_sim": nn_sim,
            "nn_bin": nn_bin_label(nn_sim),
            "nn_train_smiles": train.iloc[j][SMILES_COL] if j >= 0 else np.nan,
            "nn_train_log10mic": train_true[j] if j >= 0 else np.nan,
            "nn_train_pred_log10mic": train_pred_values[j] if j >= 0 else np.nan,
            "local_train_log10mic_variance_k10": local_var,
            "local_train_log10mic_sd_k10": local_sd,
            "ac_flag": int(len(cliffs) > 0),
            "ac_neighbor_count": int(len(cliffs)),
            "ac_max_abs_log10mic_diff": max_diff,
            "ac_pair_directional_accuracy_for_molecule": dir_acc,
            "ac_pair_count_for_molecule": int(len(cliffs)),
        })
    out = pd.concat([out.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    median_var = pd.to_numeric(out["local_train_log10mic_variance_k10"], errors="coerce").median()
    out["local_variance_group"] = np.where(
        pd.to_numeric(out["local_train_log10mic_variance_k10"], errors="coerce") >= median_var,
        "high_local_variance", "low_local_variance",
    )
    out.loc[out["local_train_log10mic_variance_k10"].isna(), "local_variance_group"] = "NA"
    return out


# ------------------------- metrics -------------------------

def screening_metrics(y_true: np.ndarray, score: np.ndarray, prefix: str = "") -> Dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    mask = np.isfinite(s) & np.isfinite(y)
    y, s = y[mask], s[mask]
    n = len(y)
    total_pos = int(np.sum(y == 1))
    base_rate = total_pos / n if n else np.nan
    order = np.argsort(-s) if n else np.array([], dtype=int)
    out: Dict[str, float] = {}
    for pct in (1, 5, 10):
        k = max(1, int(math.ceil(n * pct / 100))) if n else 0
        hits = int(np.sum(y[order[:k]] == 1)) if k else 0
        hit_rate = hits / k if k else np.nan
        ef = hit_rate / base_rate if base_rate and np.isfinite(base_rate) else np.nan
        out[f"{prefix}top_{pct}pct_n"] = k
        out[f"{prefix}top_{pct}pct_hits"] = hits
        out[f"{prefix}hit_rate_top_{pct}pct"] = hit_rate
        out[f"{prefix}ef_{pct}pct"] = ef
        out[f"{prefix}Hit_rate@Top{pct}%"] = hit_rate
        out[f"{prefix}EF{pct}%"] = ef
    for k0 in (50, 100, 500):
        k = min(k0, n)
        hits = int(np.sum(y[order[:k]] == 1)) if k else 0
        recall = hits / total_pos if total_pos else np.nan
        out[f"{prefix}Top{k0}_hits"] = hits
        out[f"{prefix}Recall@Top{k0}"] = recall
    return out


def classification_metrics(df: pd.DataFrame, prefix: str) -> Dict[str, float]:
    y = pd.to_numeric(df["true_active"], errors="coerce").to_numpy(dtype=float)
    p = pd.to_numeric(df["pred_score"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask].astype(int)
    p = np.clip(p[mask], 1e-7, 1 - 1e-7)
    pred = (p >= 0.5).astype(int)
    out: Dict[str, float] = {
        f"{prefix}_n": int(len(y)),
        f"{prefix}_pos": int(np.sum(y == 1)),
        f"{prefix}_neg": int(np.sum(y == 0)),
        f"{prefix}_pos_ratio": float(np.mean(y == 1)) if len(y) else np.nan,
        f"{prefix}_roc_auc": safe_roc_auc(y, p),
        f"{prefix}_pr_auc": safe_pr_auc(y, p),
        f"{prefix}_accuracy": float(accuracy_score(y, pred)) if len(y) else np.nan,
        f"{prefix}_balanced_accuracy": float(balanced_accuracy_score(y, pred)) if len(y) else np.nan,
        f"{prefix}_precision": float(precision_score(y, pred, zero_division=0)) if len(y) else np.nan,
        f"{prefix}_recall": float(recall_score(y, pred, zero_division=0)) if len(y) else np.nan,
        f"{prefix}_f1": float(f1_score(y, pred, zero_division=0)) if len(y) else np.nan,
        f"{prefix}_mcc": float(matthews_corrcoef(y, pred)) if len(y) else np.nan,
        f"{prefix}_log_loss": float(log_loss(y, p, labels=[0, 1])) if len(y) else np.nan,
        f"{prefix}_brier": float(brier_score_loss(y, p)) if len(y) else np.nan,
    }
    if len(y):
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        out[f"{prefix}_specificity"] = safe_div(tn, tn + fp)
        out[f"{prefix}_sensitivity"] = safe_div(tp, tp + fn)
    else:
        out[f"{prefix}_specificity"] = out[f"{prefix}_sensitivity"] = np.nan
    out.update(screening_metrics(y, p, prefix=f"{prefix}_"))
    return out


def regression_metrics(df: pd.DataFrame, prefix: str) -> Dict[str, float]:
    y = pd.to_numeric(df["true_log10MIC"], errors="coerce").to_numpy(dtype=float)
    pred = pd.to_numeric(df["pred_log10MIC"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[mask], pred[mask]
    err = pred - y
    abs_err = np.abs(err)
    out: Dict[str, float] = {
        f"{prefix}_n": int(len(y)),
        f"{prefix}_log10mic_mean": float(np.mean(y)) if len(y) else np.nan,
        f"{prefix}_log10mic_std": float(np.std(y, ddof=1)) if len(y) >= 2 else np.nan,
        f"{prefix}_mae": float(mean_absolute_error(y, pred)) if len(y) else np.nan,
        f"{prefix}_mse": float(mean_squared_error(y, pred)) if len(y) else np.nan,
        f"{prefix}_rmse": float(math.sqrt(mean_squared_error(y, pred))) if len(y) else np.nan,
        f"{prefix}_medae": float(median_absolute_error(y, pred)) if len(y) else np.nan,
        f"{prefix}_r2": float(r2_score(y, pred)) if len(y) >= 2 else np.nan,
        f"{prefix}_pearson": safe_corr(y, pred, "pearson"),
        f"{prefix}_spearman": safe_corr(y, pred, "spearman"),
        f"{prefix}_within_1_dilution": float(np.mean(abs_err <= math.log10(2))) if len(y) else np.nan,
        f"{prefix}_within_2_dilutions": float(np.mean(abs_err <= math.log10(4))) if len(y) else np.nan,
        f"{prefix}_median_fold_error": float(np.median(np.power(10.0, abs_err))) if len(y) else np.nan,
        f"{prefix}_mean_fold_error": float(np.mean(np.power(10.0, abs_err))) if len(y) else np.nan,
        f"{prefix}_max_abs_error_log10": float(np.max(abs_err)) if len(y) else np.nan,
    }
    if "true_active_for_screening" in df:
        active = pd.to_numeric(df.loc[mask, "true_active_for_screening"], errors="coerce").fillna(0).astype(int).to_numpy()
        out.update(screening_metrics(active, -pred, prefix=f"{prefix}_"))
    return out


def make_classification_prediction_df(
    source: pd.DataFrame,
    pred: np.ndarray,
    split_name: str,
    spec: RunSpec,
) -> pd.DataFrame:
    out = source.copy().reset_index(drop=True)
    y = pd.to_numeric(out[CLASS_TARGET_COL], errors="coerce").astype(int)
    p = np.clip(np.asarray(pred, dtype=float), 0.0, 1.0)
    out["split"] = split_name
    out["dataset_name"] = spec.dataset_name
    out["genus"] = spec.genus
    out["task"] = spec.task
    out["split_type"] = spec.split_type
    out["seed"] = spec.seed
    out["model"] = MODEL_NAME
    out["model_family"] = MODEL_FAMILY
    out["active"] = y
    out["true_active"] = y
    out["pred_score"] = p
    out["pred_prob"] = p
    out["pred_label"] = (p >= 0.5).astype(int)
    out["pred_label_0p5"] = out["pred_label"]
    out["threshold_used"] = 0.5
    out["val_best_f1_threshold"] = np.nan
    out["error"] = (out["pred_label"].to_numpy() != y.to_numpy()).astype(int)
    return out


def make_regression_prediction_df(
    source: pd.DataFrame,
    pred: np.ndarray,
    split_name: str,
    spec: RunSpec,
) -> pd.DataFrame:
    out = source.copy().reset_index(drop=True)
    true = pd.to_numeric(out[REG_TARGET_COL], errors="coerce").to_numpy(dtype=float)
    pred = np.asarray(pred, dtype=float)
    abs_error = np.abs(pred - true)
    mic, _, _, boundary, ranges = infer_mic_fields(out, spec.threshold_for_boundary)
    if CLASS_TARGET_COL in out:
        active = pd.to_numeric(out[CLASS_TARGET_COL], errors="coerce").fillna(0).astype(int).to_numpy()
    else:
        active = np.where(np.isfinite(mic), (mic <= spec.threshold_for_boundary).astype(int), 0)
    out["split"] = split_name
    out["dataset_name"] = spec.dataset_name
    out["genus"] = spec.genus
    out["task"] = spec.task
    out["split_type"] = spec.split_type
    out["seed"] = spec.seed
    out["model"] = MODEL_NAME
    out["model_family"] = MODEL_FAMILY
    out["log10_mic"] = true
    out["true_log10MIC"] = true
    out["true_MIC"] = np.power(10.0, true)
    out["mic_numeric"] = mic
    out["pred_log10MIC"] = pred
    out["pred_MIC"] = np.power(10.0, pred)
    out["score_reg"] = -pred
    out["abs_error_log10"] = abs_error
    out["sq_error_log10"] = np.square(pred - true)
    out["fold_error"] = np.power(10.0, abs_error)
    out["within_1_dilution"] = (abs_error <= math.log10(2)).astype(int)
    out["within_2_dilutions"] = (abs_error <= math.log10(4)).astype(int)
    out["true_active_for_screening"] = active
    out["threshold_for_boundary"] = spec.threshold_for_boundary
    # Per benchmark rule, train/validation receive prediction + overall metrics only.
    # boundary/NN/activity-cliff/local-roughness fields are added only to predictions_test.csv.
    return out


def _classification_subgroup(g: pd.DataFrame) -> Dict[str, float]:
    if g.empty:
        return {"n": 0}
    m = classification_metrics(g, "tmp")
    return {k.replace("tmp_", ""): v for k, v in m.items()}


def _regression_subgroup(g: pd.DataFrame) -> Dict[str, float]:
    if g.empty:
        return {"n": 0}
    m = regression_metrics(g, "tmp")
    return {k.replace("tmp_", ""): v for k, v in m.items()}


def diagnostics_classification(test: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
    diag: Dict[str, Any] = {}
    subgroup_rows: List[Dict[str, Any]] = []
    groups = [
        ("boundary_flag", "MIC_boundary"),
        ("nn_bin", "NN_similarity"),
        ("ac_flag", "Activity_cliff"),
        ("mic_range_group", "MIC_range"),
    ]
    for col, group_type in groups:
        if col not in test:
            continue
        for val, g in test.groupby(col, dropna=False):
            r = {"group_type": group_type, "group_col": col, "group_value": val}
            r.update(_classification_subgroup(g))
            subgroup_rows.append(r)

    def error_rate(mask: np.ndarray) -> float:
        g = test.loc[mask]
        return float(g["error"].mean()) if len(g) else np.nan

    if "boundary_flag" in test:
        b = pd.to_numeric(test["boundary_flag"], errors="coerce") == 1
        diag["boundary_n"] = int(b.sum())
        diag["non_boundary_n"] = int((~b).sum())
        diag["boundary_error_rate"] = error_rate(b.to_numpy())
        diag["non_boundary_error_rate"] = error_rate((~b).to_numpy())
        diag["boundary_error_enrichment"] = safe_div(diag["boundary_error_rate"], diag["non_boundary_error_rate"])
    nn_map = {">=0.7": "nn_ge_0p7", "0.4-0.7": "nn_0p4_0p7", "<0.4": "nn_lt_0p4"}
    if "nn_bin" in test:
        for label, stem in nn_map.items():
            g = test[test["nn_bin"] == label]
            vals = _classification_subgroup(g)
            for k, v in vals.items():
                diag[f"{stem}_{k}"] = v
    if "ac_flag" in test:
        ac = pd.to_numeric(test["ac_flag"], errors="coerce") == 1
        diag["ac_n"] = int(ac.sum())
        diag["non_ac_n"] = int((~ac).sum())
        diag["ac_error_rate"] = error_rate(ac.to_numpy())
        diag["non_ac_error_rate"] = error_rate((~ac).to_numpy())
        diag["ac_error_enrichment"] = safe_div(diag["ac_error_rate"], diag["non_ac_error_rate"])
        diag["ac_subgroup_pr_auc"] = safe_pr_auc(
            pd.to_numeric(test.loc[ac, "true_active"], errors="coerce").to_numpy(dtype=int),
            pd.to_numeric(test.loc[ac, "pred_score"], errors="coerce").to_numpy(dtype=float),
        ) if ac.sum() else np.nan
        counts = pd.to_numeric(test.get("ac_pair_count_for_molecule"), errors="coerce").fillna(0).to_numpy(dtype=float)
        accs = pd.to_numeric(test.get("ac_pair_directional_accuracy_for_molecule"), errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(accs) & (counts > 0)
        diag["ac_pair_total_count"] = int(np.sum(counts))
        diag["ac_pair_directional_accuracy"] = float(np.average(accs[valid], weights=counts[valid])) if valid.any() else np.nan
    return diag, pd.DataFrame(subgroup_rows)


def diagnostics_regression(test: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
    diag: Dict[str, Any] = {}
    subgroup_rows: List[Dict[str, Any]] = []
    groups = [
        ("boundary_flag", "MIC_boundary"),
        ("nn_bin", "NN_similarity"),
        ("ac_flag", "Activity_cliff"),
        ("mic_range_group", "MIC_range"),
        ("local_variance_group", "Local_variance"),
    ]
    for col, group_type in groups:
        if col not in test:
            continue
        for val, g in test.groupby(col, dropna=False):
            r = {"group_type": group_type, "group_col": col, "group_value": val}
            r.update(_regression_subgroup(g))
            subgroup_rows.append(r)

    def subgroup(mask: np.ndarray) -> Dict[str, float]:
        return _regression_subgroup(test.loc[mask])

    if "boundary_flag" in test:
        b = pd.to_numeric(test["boundary_flag"], errors="coerce") == 1
        bm, nm = subgroup(b.to_numpy()), subgroup((~b).to_numpy())
        diag.update({
            "boundary_n": bm.get("n", 0), "non_boundary_n": nm.get("n", 0),
            "boundary_mae": bm.get("mae"), "non_boundary_mae": nm.get("mae"),
            "boundary_rmse": bm.get("rmse"), "non_boundary_rmse": nm.get("rmse"),
            "boundary_regression_error_enrichment": safe_div(bm.get("mae", np.nan), nm.get("mae", np.nan)),
            "boundary_within_1_dilution": bm.get("within_1_dilution"),
            "non_boundary_within_1_dilution": nm.get("within_1_dilution"),
        })
    nn_map = {">=0.7": "nn_ge_0p7", "0.4-0.7": "nn_0p4_0p7", "<0.4": "nn_lt_0p4"}
    if "nn_bin" in test:
        for label, stem in nn_map.items():
            vals = _regression_subgroup(test[test["nn_bin"] == label])
            for k, v in vals.items():
                diag[f"{stem}_{k}"] = v
    if "ac_flag" in test:
        ac = pd.to_numeric(test["ac_flag"], errors="coerce") == 1
        am, nm = subgroup(ac.to_numpy()), subgroup((~ac).to_numpy())
        diag.update({
            "ac_n": am.get("n", 0), "non_ac_n": nm.get("n", 0),
            "ac_mae": am.get("mae"), "non_ac_mae": nm.get("mae"),
            "ac_rmse": am.get("rmse"), "non_ac_rmse": nm.get("rmse"),
            "ac_error_enrichment": safe_div(am.get("mae", np.nan), nm.get("mae", np.nan)),
        })
        counts = pd.to_numeric(test.get("ac_pair_count_for_molecule"), errors="coerce").fillna(0).to_numpy(dtype=float)
        accs = pd.to_numeric(test.get("ac_pair_directional_accuracy_for_molecule"), errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(accs) & (counts > 0)
        diag["ac_pair_total_count"] = int(np.sum(counts))
        diag["ac_pair_directional_accuracy"] = float(np.average(accs[valid], weights=counts[valid])) if valid.any() else np.nan
    if "local_train_log10mic_variance_k10" in test:
        var = pd.to_numeric(test["local_train_log10mic_variance_k10"], errors="coerce")
        diag["mean_local_train_log10mic_variance_k10"] = float(var.mean())
        diag["median_local_train_log10mic_variance_k10"] = float(var.median())
        high = test["local_variance_group"] == "high_local_variance"
        hm, lm = subgroup(high.to_numpy()), subgroup((~high).to_numpy())
        diag["high_local_variance_mae"] = hm.get("mae")
        diag["low_local_variance_mae"] = lm.get("mae")
        diag["local_variance_error_enrichment"] = safe_div(hm.get("mae", np.nan), lm.get("mae", np.nan))
    return diag, pd.DataFrame(subgroup_rows)


# ------------------------- one-run orchestration -------------------------

def run_one(args: argparse.Namespace, spec: RunSpec) -> Dict[str, Any]:
    out_dir = ensure_dir(run_output_dir(args.out_root, spec))
    metrics_path = out_dir / "metrics_full.csv"

    if args.resume and metrics_path.is_file() and not args.force:
        if completed_metrics_are_trusted(metrics_path, spec.task):
            run_log = out_dir / "run.log"
            log_message(run_log, "RESUME: trusted v11 metrics_full.csv exists; skipping completed run.")
            return pd.read_csv(metrics_path).iloc[0].to_dict()
        backup = archive_untrusted_run(out_dir, args.out_root, "legacy_or_unvalidated_metrics")
        out_dir = ensure_dir(run_output_dir(args.out_root, spec))
        metrics_path = out_dir / "metrics_full.csv"
        print(f"[{now()}] Archived legacy/unvalidated run locally.", flush=True)

    if args.force and out_dir.exists():
        for p in out_dir.iterdir():
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
        ensure_dir(out_dir)

    # Do not silently resume from a checkpoint left by v9/v10 or an interrupted v11 training process.
    native_probe = out_dir / "chemprop_native"
    existing_probe = list(native_probe.rglob("*.pt")) + list(native_probe.rglob("*.ckpt")) if native_probe.exists() else []
    if existing_probe and not native_training_is_trusted(native_probe, spec.task):
        backup = archive_untrusted_run(out_dir, args.out_root, "untrusted_or_partial_checkpoint")
        out_dir = ensure_dir(run_output_dir(args.out_root, spec))
        metrics_path = out_dir / "metrics_full.csv"
        print(f"[{now()}] Archived untrusted/partial checkpoint run locally.", flush=True)

    run_log = out_dir / "run.log"
    log_message(run_log, f"START {spec.dataset_name} | {spec.task} | {spec.split_type} | seed={spec.seed}")
    log_message(run_log, "EARLY_STOP_POLICY: " + json.dumps(early_stopping_policy(spec.task), ensure_ascii=False))
    train_df = validate_input_csv(spec.train_csv, spec.task)
    val_df = validate_input_csv(spec.validation_csv, spec.task)
    test_df = validate_input_csv(spec.test_csv, spec.task)
    log_message(run_log, f"data sizes: train={len(train_df)} validation={len(val_df)} test={len(test_df)}")

    native_dir = ensure_dir(out_dir / "chemprop_native")
    raw_dir = ensure_dir(out_dir / "raw_predictions")
    cli_log = out_dir / "chemprop_cli.log"
    existing_models = list(native_dir.rglob("*.pt")) + list(native_dir.rglob("*.ckpt"))
    start = time.time()
    if not existing_models:
        cmd = build_chemprop_train_command(args, spec, native_dir, cli_log)
        run_command(cmd, run_log, cwd=Path(args.work_dir) if args.work_dir else None)
        completion_payload = early_stopping_policy(spec.task) | {
            "completed_at": now(),
            "command": None,  # Arguments may contain local paths or credentials.
        }
        save_json(completion_payload, native_dir / "TRAINING_COMPLETED.json")
    else:
        if not native_training_is_trusted(native_dir, spec.task):
            raise RuntimeError("Existing checkpoint is not trusted by the v11 early-stopping policy.")
        log_message(run_log, "Found trusted completed v11 native checkpoint; training skipped and post-processing resumed.")
    train_seconds = time.time() - start

    curve, best_epoch, best_val_metric, best_metric_name = parse_training_curve(native_dir, spec.task, out_dir)
    native_best = find_best_model(native_dir, expected_best_epoch=best_epoch)
    best_model = copy_best_model(native_best, out_dir)
    log_message(run_log, f"SELECTED_CHECKPOINT: {native_best.name}; best_epoch={best_epoch}; {best_metric_name}={best_val_metric}")

    pred_train = run_prediction(args, best_model, spec.train_csv, "train", raw_dir, run_log)
    pred_val = run_prediction(args, best_model, spec.validation_csv, "validation", raw_dir, run_log)
    # Chemprop训练日志中的回归RMSE位于标准化目标尺度，
    # chemprop predict输出则位于原始log10_mic尺度。
    # 一致性检查前必须把训练RMSE转换到原始尺度。
    best_val_metric_for_validation = float(best_val_metric)
    if spec.task == "regression":
        train_target_for_scale = pd.to_numeric(
            train_df["log10_mic"], errors="coerce"
        ).to_numpy(dtype=float)
        train_target_for_scale = train_target_for_scale[
            np.isfinite(train_target_for_scale)
        ]
        if train_target_for_scale.size == 0:
            raise RuntimeError(
                "No finite training log10_mic values for target scaling."
            )
        target_scale = float(np.std(train_target_for_scale, ddof=0))
        if not np.isfinite(target_scale) or target_scale <= 0:
            raise RuntimeError(
                f"Invalid regression target scale: {target_scale}"
            )
        best_val_metric_for_validation *= target_scale
        print(
            f"CHECKPOINT_VALIDATION_SCALE: "
            f"training_val_rmse_scaled={best_val_metric:.12g}, "
            f"target_scale={target_scale:.12g}, "
            f"training_val_rmse_raw="
            f"{best_val_metric_for_validation:.12g}"
        )
    checkpoint_validation = validate_selected_model_score(
        spec.task,
        best_val_metric_for_validation,
        val_df,
        pred_val,
    )
    log_message(run_log, "CHECKPOINT_VALIDATION: " + json.dumps(checkpoint_validation, ensure_ascii=False, default=str))
    pred_test = run_prediction(args, best_model, spec.test_csv, "test", raw_dir, run_log)

    identity = {
        "model": MODEL_NAME,
        "model_family": MODEL_FAMILY,
        "runner_version": RUNNER_VERSION,
        "early_stopping_policy_version": EARLY_STOP_POLICY_VERSION,
        "task": spec.task,
        "dataset_name": spec.dataset_name,
        "genus": spec.genus,
        "split_type": spec.split_type,
        "seed": spec.seed,
        "feature_type": "Chemprop_v2_D-MPNN_graph_only",
        "threshold_from_dataset_name": spec.threshold_for_boundary,
        "threshold_for_boundary": spec.threshold_for_boundary,
        "train_seconds": train_seconds,
        "best_epoch": best_epoch,
        "best_val_metric": best_val_metric,
        "best_val_metric_name": best_metric_name,
        "last_epoch": int(pd.to_numeric(curve["epoch"], errors="coerce").max()),
        "epochs_requested": int(args.epochs),
        "early_stopped": bool(int(pd.to_numeric(curve["epoch"], errors="coerce").max()) < int(args.epochs)),
        "threshold_mode": "fixed_0p5" if spec.task == "classification" else "NA",
        "threshold_used": 0.5 if spec.task == "classification" else np.nan,
        "early_stopping_tracking_metric": "prc" if spec.task == "classification" else "rmse",
        "early_stopping_patience": int(args.patience),
        "early_stopping_min_delta": float(args.min_delta),
    }
    identity.update(checkpoint_validation)
    best_row = curve[curve["epoch"] == best_epoch]
    if not best_row.empty:
        identity["train_loss"] = pd.to_numeric(best_row.iloc[0].get("train_loss"), errors="coerce")
        identity["val_loss"] = pd.to_numeric(best_row.iloc[0].get("val_loss"), errors="coerce")
    else:
        identity["train_loss"] = identity["val_loss"] = np.nan

    if spec.task == "classification":
        train_pred = make_classification_prediction_df(train_df, pred_train, "train", spec)
        val_pred = make_classification_prediction_df(val_df, pred_val, "validation", spec)
        test_pred = make_classification_prediction_df(test_df, pred_test, "test", spec)
        # As requested, only test receives NN and activity-cliff diagnostics.
        test_pred = add_test_diagnostics_classification(test_pred, train_pred, spec.threshold_for_boundary)
        metrics = dict(identity)
        metrics.update(classification_metrics(train_pred, "train"))
        metrics.update(classification_metrics(val_pred, "val"))
        metrics.update(classification_metrics(test_pred, "test"))
        diag, subgroup = diagnostics_classification(test_pred)
    else:
        train_pred = make_regression_prediction_df(train_df, pred_train, "train", spec)
        val_pred = make_regression_prediction_df(val_df, pred_val, "validation", spec)
        test_pred = make_regression_prediction_df(test_df, pred_test, "test", spec)
        test_pred = add_test_diagnostics_regression(test_pred, train_pred, spec.threshold_for_boundary, args.local_k)
        metrics = dict(identity)
        metrics.update(regression_metrics(train_pred, "train"))
        metrics.update(regression_metrics(val_pred, "val"))
        metrics.update(regression_metrics(test_pred, "test"))
        diag, subgroup = diagnostics_regression(test_pred)

    train_pred.to_csv(out_dir / "predictions_train.csv", index=False)
    val_pred.to_csv(out_dir / "predictions_validation.csv", index=False)
    test_pred.to_csv(out_dir / "predictions_test.csv", index=False)
    metrics.update(diag)
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    pd.DataFrame([diag]).to_csv(out_dir / "diagnostics_metrics.csv", index=False)
    subgroup.to_csv(out_dir / "diagnostic_subgroup_metrics.csv", index=False)

    feature_summary = {
        "model": MODEL_NAME,
        "model_family": MODEL_FAMILY,
        "feature_type": "Chemprop v2 directed-message-passing neural network (D-MPNN), graph-only",
        "model_input": "SMILES -> RDKit molecular graph -> Chemprop atom/bond features -> directed bond message passing",
        "extra_morgan_features_used_for_training": False,
        "extra_rdkit_2d_descriptors_used_for_training": False,
        "nn_diagnostic_fingerprint": {"type": "Morgan", "radius": 2, "nBits": 2048, "similarity": "Tanimoto"},
        "important_note": "Morgan fingerprints are used only for NN/activity-cliff diagnostics, not as Chemprop training inputs.",
    }
    model_params = {
        "message_hidden_dim": args.message_hidden_dim,
        "depth": args.depth,
        "dropout": args.dropout,
        "ffn_hidden_dim": args.ffn_hidden_dim,
        "ffn_num_layers": args.ffn_num_layers,
        "aggregation": args.aggregation,
        "aggregation_norm": args.aggregation_norm,
        "activation": args.activation,
        "epochs_max": args.epochs,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "warmup_epochs": args.warmup_epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "init_lr": args.init_lr,
        "max_lr": args.max_lr,
        "final_lr": args.final_lr,
        "grad_clip": args.grad_clip,
        "ensemble_size": 1,
        "loss": "BCE" if spec.task == "classification" else "MSE",
        "checkpoint_selection": "maximum validation PR-AUC" if spec.task == "classification" else "minimum validation RMSE",
        "checkpoint_selection_direction": "max" if spec.task == "classification" else "min",
        "metric_direction_guard": EARLY_STOP_POLICY_VERSION,
        "class_balance": bool(args.class_balance) if spec.task == "classification" else False,
    }
    save_json(feature_summary, out_dir / "feature_summary.json")
    save_json(model_params, out_dir / "model_params.json")
    save_json(sanitize_metadata({
        "start_or_finish_time": now(),
        "runner_version": RUNNER_VERSION,
        "early_stopping_policy": early_stopping_policy(spec.task),
        "checkpoint_validation": checkpoint_validation,
        "script": str(Path(sys.argv[0]).resolve()),
        "argv": sys.argv,
        "run_spec": asdict(spec),
        "data_root": str(Path(args.data_root).resolve()),
        "out_root": str(Path(args.out_root).resolve()),
        "output_dir": str(out_dir),
        "best_model_source": str(native_best),
        "best_model_copy": str(best_model),
        "chemprop_executable": shutil.which(args.chemprop_executable),
        "python": sys.version,
        "platform": platform.platform(),
        "environment": {k: os.environ.get(k) for k in [
            "CONDA_DEFAULT_ENV", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID"
        ]},
        "train_seconds": train_seconds,
        "columns": {"smiles": SMILES_COL, "target": CLASS_TARGET_COL if spec.task == "classification" else REG_TARGET_COL},
    }), out_dir / "run_info.json")
    log_message(run_log, "FINISHED; outputs saved")
    return metrics


# ------------------------- summarization -------------------------

def _flatten_agg_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = ["_".join([str(x) for x in col if str(x)]) if isinstance(col, tuple) else str(col) for col in df.columns]
    return df


def summarize_results(out_root: str | Path, task: str) -> Dict[str, str]:
    root = Path(out_root).resolve()
    model_root = root / MODEL_NAME
    files = sorted(model_root.rglob("metrics_full.csv"))
    if not files:
        raise FileNotFoundError(f"No metrics_full.csv found under {model_root}")
    frames = []
    for f in files:
        try:
            d = pd.read_csv(f)
            if not d.empty:
                d["metrics_file"] = str(f.relative_to(root))
                frames.append(d)
        except Exception as exc:
            print(f"Warning: failed to read {f.name}: {type(exc).__name__}", file=sys.stderr)
    all_df = pd.concat(frames, ignore_index=True, sort=False)
    all_df = all_df[all_df["task"] == task].copy()
    if all_df.empty:
        raise RuntimeError(f"No {task} rows found in collected metrics")
    model_root.mkdir(parents=True, exist_ok=True)
    all_path = model_root / "all_run_metrics.csv"
    all_df.to_csv(all_path, index=False)

    id_cols = ["model", "model_family", "task", "dataset_name", "genus", "split_type"]
    numeric_cols = [c for c in all_df.columns if pd.api.types.is_numeric_dtype(all_df[c]) and c != "seed"]
    agg = all_df.groupby(id_cols, dropna=False)[numeric_cols].agg(["mean", "std", "count"]).reset_index()
    agg = _flatten_agg_columns(agg)
    agg_path = model_root / "aggregate_mean_sd.csv"
    agg.to_csv(agg_path, index=False)

    combined = root / "_combined_summary" / ("classification_summary_like_ml" if task == "classification" else "regression_summary_like_ml")
    combined.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(combined / "combined_all_run_metrics.csv", index=False)

    primary = "test_pr_auc" if task == "classification" else "test_rmse"
    leaderboard = all_df.groupby(["model", "dataset_name", "genus", "split_type"], dropna=False)[primary].agg(["mean", "std", "count"]).reset_index()
    leaderboard["rank"] = leaderboard.groupby(["dataset_name", "split_type"])["mean"].rank(ascending=(task == "regression"), method="min")
    leaderboard.to_csv(combined / "leaderboard_mean_sd.csv", index=False)

    # OOD changes are calculated on split-level means, so strict_perimeter seed_0 remains valid.
    mean_by_split = all_df.groupby(["model", "dataset_name", "genus", "split_type"], dropna=False).mean(numeric_only=True).reset_index()
    rows: List[Dict[str, Any]] = []
    metrics_to_compare = ["test_pr_auc", "test_EF1%", "test_roc_auc"] if task == "classification" else ["test_rmse", "test_mae", "test_spearman"]
    for (model, dataset, genus), g in mean_by_split.groupby(["model", "dataset_name", "genus"]):
        random_rows = g[g["split_type"] == "random"]
        if random_rows.empty:
            continue
        base = random_rows.iloc[0]
        for target_split in ["scaffold", "repeated_stratified_perimeter", "strict_perimeter"]:
            target_rows = g[g["split_type"] == target_split]
            if target_rows.empty:
                continue
            target = target_rows.iloc[0]
            r: Dict[str, Any] = {"model": model, "dataset_name": dataset, "genus": genus, "target_split": target_split}
            for m in metrics_to_compare:
                if m not in g:
                    continue
                r[f"random_{m}"] = base.get(m)
                r[f"ood_{m}"] = target.get(m)
                if task == "regression" and m in {"test_rmse", "test_mae"}:
                    r[f"ood_increase_{m}"] = target.get(m) - base.get(m)
                else:
                    r[f"ood_drop_{m}"] = base.get(m) - target.get(m)
            diagnostic_cols = (
                ["nn_lt_0p4_n", "nn_lt_0p4_error_rate", "nn_lt_0p4_pr_auc", "nn_lt_0p4_EF1%",
                 "boundary_error_rate", "ac_error_rate"]
                if task == "classification" else
                ["nn_lt_0p4_n", "nn_lt_0p4_rmse", "nn_lt_0p4_mae", "nn_lt_0p4_spearman",
                 "boundary_mae", "ac_mae"]
            )
            for c in diagnostic_cols:
                if c in target.index:
                    r[f"perimeter_{c}"] = target.get(c)
            rows.append(r)
    ood = pd.DataFrame(rows)
    ood.to_csv(model_root / "ood_change_random_vs_ood_splits.csv", index=False)
    ood.to_csv(combined / ("ood_drop_summary.csv" if task == "classification" else "ood_change_summary.csv"), index=False)

    perimeter = ood[ood["target_split"].isin(["repeated_stratified_perimeter", "strict_perimeter"])].copy() if not ood.empty else ood
    perimeter_name = "perimeter_ood_error_analysis.csv" if task == "classification" else "perimeter_regression_ood_error_analysis.csv"
    perimeter.to_csv(combined / perimeter_name, index=False)

    ranking = all_df[["model", "dataset_name", "genus", "split_type", "seed", primary]].copy()
    ranking["rank"] = ranking.groupby(["dataset_name", "split_type", "seed"])[primary].rank(ascending=(task == "regression"), method="min")
    ranking.to_csv(combined / "model_ranking_per_task_seed.csv", index=False)
    ranking.groupby("model")["rank"].agg(["mean", "std", "count"]).reset_index().to_csv(combined / "model_ranking_by_metric.csv", index=False)
    ranking[ranking["rank"] == 1].groupby("model").size().reset_index(name="top_model_count").to_csv(combined / "top_model_frequency.csv", index=False)
    return {
        "all_run_metrics": str(all_path),
        "aggregate_mean_sd": str(agg_path),
        "combined_summary": str(combined),
    }


# ------------------------- CLI -------------------------

def add_common_args(p: argparse.ArgumentParser, task: str) -> argparse.ArgumentParser:
    p.add_argument("--data-root", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--work-dir", default=None, help="Optional working directory; normally chemprop-main.")
    p.add_argument("--chemprop-executable", default="chemprop")
    p.add_argument("--split-types", default=",".join(DEFAULT_SPLITS))
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--datasets", default="", help="Comma/space separated full dataset names or genus names; empty means all.")
    p.add_argument("--threshold-fallback", type=float, default=32.0)
    p.add_argument("--single-run-index", type=int, default=None)
    p.add_argument("--print-run-plan", action="store_true")
    p.add_argument("--print-run-count", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force", action="store_true")

    # Fixed, official-default-centered Chemprop architecture.
    p.add_argument("--message-hidden-dim", type=int, default=300)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--ffn-hidden-dim", type=int, default=300)
    p.add_argument("--ffn-num-layers", type=int, default=1)
    p.add_argument("--aggregation", default="norm", choices=["mean", "sum", "norm"])
    p.add_argument("--aggregation-norm", type=float, default=100.0)
    p.add_argument("--activation", default="relu")
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--min-delta", type=float, default=0.0)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--init-lr", type=float, default=1e-5)
    p.add_argument("--max-lr", type=float, default=5e-4)
    p.add_argument("--final-lr", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--accelerator", default="gpu")
    p.add_argument("--devices", default="1")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--class-balance", action="store_true", help="Classification only; disabled by default for the main benchmark.")
    p.add_argument("--local-k", type=int, default=10)
    return p


def task_main(task: str) -> None:
    parser = add_common_args(argparse.ArgumentParser(description=f"Run AntibioticBench Chemprop {task} benchmark."), task)
    args = parser.parse_args()
    split_types = parse_csv_list(args.split_types, str)
    seeds = parse_csv_list(args.seeds, int)
    datasets = parse_csv_list(args.datasets, str)
    specs = discover_run_specs(args.data_root, task, split_types, seeds, datasets or None, args.threshold_fallback)
    if args.print_run_count:
        print(len(specs))
        return
    if args.print_run_plan:
        print(json.dumps([sanitize_metadata(asdict(s)) for s in specs], indent=2, ensure_ascii=False))
        return
    if args.single_run_index is not None:
        idx = int(args.single_run_index)
        if idx < 0 or idx >= len(specs):
            raise IndexError(f"single-run-index={idx} outside 0..{len(specs)-1}")
        specs = [specs[idx]]
    print(f"[{now()}] Expanded {len(specs)} run(s) for task={task}.", flush=True)
    for i, spec in enumerate(specs, 1):
        print(f"[{now()}] RUN {i}/{len(specs)}: {spec.dataset_name} | {spec.split_type} | seed_{spec.seed}", flush=True)
        run_one(args, spec)


def summary_main(task: str) -> None:
    p = argparse.ArgumentParser(description=f"Summarize AntibioticBench Chemprop {task} outputs.")
    p.add_argument("--out-root", required=True)
    args = p.parse_args()
    outputs = summarize_results(args.out_root, task)
    print(json.dumps(sanitize_metadata(outputs), indent=2, ensure_ascii=False))
