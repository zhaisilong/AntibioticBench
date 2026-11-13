#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uni-Mol v1 AntibioticBench common training/evaluation code.

Design:
- Uses the exact SMILES strings in the supplied split CSV files.
- Does not perform salt removal, neutralization, tautomer standardization,
  deduplication, or any other extra structure standardization.
- Uni-Mol v1 no-H molecular pretrained weights are selected through
  remove_hs=True in unimol_tools.
- Full end-to-end fine-tuning.
- Classification: unweighted BCEWithLogitsLoss, threshold=0.5,
  checkpoint selected by validation PR-AUC.
- Regression: MSE loss on log10_mic, checkpoint selected by validation RMSE.
- One shared per-dataset cache stores 11 deterministic no-H 3D conformers and
  is reused by classification, regression, all split types, and all training seeds.
- Model embeddings are recomputed in every run during end-to-end fine-tuning.
- Test set is evaluated only after the validation-selected checkpoint is loaded.
- NN and Activity Cliff diagnostics are computed only for test.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import logging
import math
import os
import platform
import pickle
import random
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import pearsonr, spearmanr
except Exception:
    pearsonr = None
    spearmanr = None

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

MODEL_NAME = "Uni-Mol"
MODEL_FAMILY = "3D_pretrained_transformer"
FEATURE_TYPE = "Uni-Mol_v1_noH_shared_11conformer_ETKDG"
DEFAULT_THRESHOLD = 0.5
DILUTION_1 = math.log10(2.0)
DILUTION_2 = math.log10(4.0)
AC_DELTA_LOG10 = math.log10(16.0)
NN_RADIUS = 2
NN_NBITS = 2048
NN_HIGH = 0.7
NN_MID = 0.4
AC_SIM_THRESHOLD = 0.8

IDENTITY_COLS = [
    "split", "dataset_name", "genus", "task", "split_type",
    "seed", "model", "model_family", "smiles"
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, set):
        return sorted(obj)
    return str(obj)


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=json_default)


def safe_float(value: Any) -> float:
    try:
        v = float(value)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def setup_logger(run_dir: Path, verbose: bool = True) -> logging.Logger:
    ensure_dir(run_dir)
    logger = logging.getLogger(f"unimol_benchmark_{hash(str(run_dir))}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(run_dir / "run.log", mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO if verbose else logging.WARNING)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def package_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def git_commit(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def infer_dataset_name(path: Path) -> str:
    # .../<dataset>/<task>/<split>/seed_x/train.csv
    return path.parents[3].name


def infer_split_type(path: Path) -> str:
    return path.parents[1].name


def infer_seed(path: Path) -> int:
    m = re.search(r"seed_(\d+)", path.parent.name)
    if not m:
        raise ValueError(f"Cannot infer seed from: {path}")
    return int(m.group(1))


def infer_genus(dataset_name: str) -> str:
    m = re.match(r"^\d+_([^_]+)_", dataset_name)
    return m.group(1) if m else dataset_name.split("_")[0]


def infer_threshold(dataset_name: str) -> float:
    m = re.search(r"_median_([0-9]+(?:[p.][0-9]+)?)_", dataset_name)
    if not m:
        return float("nan")
    return float(m.group(1).replace("p", "."))


def choose_column(df: pd.DataFrame, candidates: Sequence[str], required: bool = False) -> Optional[str]:
    lower_map = {str(c).lower(): str(c) for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    if required:
        raise ValueError(f"None of columns {list(candidates)} found. Available={list(df.columns)}")
    return None


def load_split_csv(path: Path, task: str) -> Tuple[pd.DataFrame, str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if len(df) == 0:
        raise ValueError(f"Empty CSV: {path}")

    smiles_col = choose_column(
        df, ["smiles", "SMILES", "smiles_std", "SMILES_std", "canonical_smiles"],
        required=True,
    )
    if task == "classification":
        target_col = choose_column(df, ["active", "label", "target", "Class"], required=True)
        y = pd.to_numeric(df[target_col], errors="raise")
        unique = set(y.dropna().astype(int).unique().tolist())
        if not unique.issubset({0, 1}):
            raise ValueError(f"Classification target must be 0/1, got {sorted(unique)} in {path}")
        df[target_col] = y.astype(int)
    else:
        target_col = choose_column(
            df, ["log10_mic", "log10_MIC", "log10mic", "target", "y"],
            required=True,
        )
        df[target_col] = pd.to_numeric(df[target_col], errors="raise").astype(float)

    if df[smiles_col].isna().any():
        raise ValueError(f"Missing SMILES in {path}")
    df[smiles_col] = df[smiles_col].astype(str)
    return df.reset_index(drop=True), smiles_col, target_col


def validate_smiles_exact(df: pd.DataFrame, smiles_col: str, path: Path) -> None:
    """
    Parse-only validation. It does not rewrite or standardize the SMILES.
    """
    try:
        from rdkit import Chem
    except Exception as e:
        raise RuntimeError("RDKit is required.") from e

    bad = []
    for i, smi in enumerate(df[smiles_col].tolist()):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            bad.append((i, smi))
            if len(bad) >= 20:
                break
    if bad:
        raise ValueError(
            f"Invalid SMILES found in {path}; first examples={bad}. "
            "The benchmark script does not silently remove rows."
        )


def match_columns_across_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_smiles: str,
    val_smiles: str,
    test_smiles: str,
    train_target: str,
    val_target: str,
    test_target: str,
    task: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Normalize only column names used by the model; molecular strings remain unchanged.
    canonical_target = "active" if task == "classification" else "log10_mic"

    def ren(df: pd.DataFrame, smi: str, target: str) -> pd.DataFrame:
        out = df.copy()
        rename = {}
        if smi != "smiles":
            rename[smi] = "smiles"
        if target != canonical_target:
            rename[target] = canonical_target
        out = out.rename(columns=rename)
        return out

    return (
        ren(train_df, train_smiles, train_target),
        ren(val_df, val_smiles, val_target),
        ren(test_df, test_smiles, test_target),
    )


def safe_roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    try:
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, p))
    except Exception:
        return float("nan")


def safe_pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    try:
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(average_precision_score(y, p))
    except Exception:
        return float("nan")


def safe_log_loss(y: np.ndarray, p: np.ndarray) -> float:
    try:
        p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
        return float(log_loss(y, p, labels=[0, 1]))
    except Exception:
        return float("nan")


def classification_metrics(
    y_true: Sequence[int],
    pred_prob: Sequence[float],
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=int).reshape(-1)
    p = np.asarray(pred_prob, dtype=float).reshape(-1)
    pred = (p >= threshold).astype(int)

    try:
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    except Exception:
        tn = fp = fn = tp = 0

    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")

    out = {
        "n": int(len(y)),
        "pos": int((y == 1).sum()),
        "neg": int((y == 0).sum()),
        "pos_ratio": float((y == 1).mean()) if len(y) else float("nan"),
        "roc_auc": safe_roc_auc(y, p),
        "pr_auc": safe_pr_auc(y, p),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)) if len(np.unique(pred)) > 1 or len(np.unique(y)) > 1 else 0.0,
        "specificity": float(specificity),
        "sensitivity": float(sensitivity),
        "log_loss": safe_log_loss(y, p),
        "brier": float(brier_score_loss(y, p)),
    }
    out.update(screening_metrics(y, p, higher_is_better=True))
    return out


def _top_n(n: int, frac: float) -> int:
    return max(1, int(math.ceil(n * frac)))


def screening_metrics(
    true_active: Sequence[int],
    score: Sequence[float],
    higher_is_better: bool = True,
) -> Dict[str, float]:
    y = np.asarray(true_active, dtype=int).reshape(-1)
    s = np.asarray(score, dtype=float).reshape(-1)
    n = len(y)
    if n == 0:
        return {}
    order = np.argsort(-s if higher_is_better else s, kind="mergesort")
    prevalence = float(y.mean())
    out: Dict[str, float] = {}

    for pct, label in [(0.01, "1pct"), (0.05, "5pct"), (0.10, "10pct")]:
        k = _top_n(n, pct)
        hits = int(y[order[:k]].sum())
        hit_rate = hits / k
        ef = hit_rate / prevalence if prevalence > 0 else float("nan")
        out[f"top_{label}_n"] = k
        out[f"top_{label}_hits"] = hits
        out[f"hit_rate_top_{label}"] = hit_rate
        out[f"ef_{label}"] = ef
        out[f"EF{int(pct*100)}%"] = ef
        out[f"Hit_rate@Top{int(pct*100)}%"] = hit_rate

    positives = int(y.sum())
    for k0 in (50, 100, 500):
        k = min(k0, n)
        hits = int(y[order[:k]].sum())
        out[f"Top{k0}_hits"] = hits
        out[f"Recall@Top{k0}"] = hits / positives if positives > 0 else float("nan")
    return out


def _safe_corr(fn, y: np.ndarray, p: np.ndarray) -> float:
    if fn is None or len(y) < 2 or np.std(y) == 0 or np.std(p) == 0:
        return float("nan")
    try:
        return float(fn(y, p)[0])
    except Exception:
        return float("nan")


def regression_metrics(
    y_true: Sequence[float],
    pred: Sequence[float],
    true_active_for_screening: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(pred, dtype=float).reshape(-1)
    err = p - y
    abs_err = np.abs(err)
    out = {
        "n": int(len(y)),
        "log10mic_mean": float(np.mean(y)) if len(y) else float("nan"),
        "log10mic_std": float(np.std(y, ddof=1)) if len(y) > 1 else float("nan"),
        "mae": float(mean_absolute_error(y, p)),
        "mse": float(mean_squared_error(y, p)),
        "rmse": float(math.sqrt(mean_squared_error(y, p))),
        "medae": float(median_absolute_error(y, p)),
        "r2": float(r2_score(y, p)) if len(y) > 1 else float("nan"),
        "pearson": _safe_corr(pearsonr, y, p),
        "spearman": _safe_corr(spearmanr, y, p),
        "within_1_dilution": float(np.mean(abs_err <= DILUTION_1)),
        "within_2_dilutions": float(np.mean(abs_err <= DILUTION_2)),
        "median_fold_error": float(np.median(np.power(10.0, abs_err))),
        "mean_fold_error": float(np.mean(np.power(10.0, abs_err))),
        "max_abs_error_log10": float(np.max(abs_err)) if len(abs_err) else float("nan"),
    }
    if true_active_for_screening is not None:
        out.update(
            screening_metrics(
                true_active_for_screening,
                -p,
                higher_is_better=True,
            )
        )
    return out


def best_f1_threshold(y_true: Sequence[int], pred_prob: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(pred_prob, dtype=float)
    candidates = np.unique(np.concatenate(([0.0], p, [1.0])))
    if len(candidates) > 2001:
        candidates = np.linspace(0, 1, 1001)
    best_t = 0.5
    best_f = -1.0
    for t in candidates:
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > best_f:
            best_f = f
            best_t = float(t)
    return best_t


def prefix_dict(prefix: str, d: Mapping[str, Any]) -> Dict[str, Any]:
    return {f"{prefix}_{k}": v for k, v in d.items()}


def extract_mic_numeric(df: pd.DataFrame, log10_values: Optional[np.ndarray] = None) -> np.ndarray:
    col = choose_column(
        df,
        ["mic_numeric", "MIC", "mic", "mic_value", "value", "median_mic"],
        required=False,
    )
    if col is not None:
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(values).any():
            return values
    log_col = choose_column(
        df,
        ["log10_mic", "log10_MIC", "log10mic", "log10_mic_for_diagnosis"],
        required=False,
    )
    if log_col is not None:
        vals = pd.to_numeric(df[log_col], errors="coerce").to_numpy(dtype=float)
        return np.power(10.0, vals)
    if log10_values is not None:
        return np.power(10.0, np.asarray(log10_values, dtype=float))
    return np.full(len(df), np.nan)


def extract_log10_mic_for_diagnosis(df: pd.DataFrame) -> np.ndarray:
    col = choose_column(
        df,
        ["log10_mic", "log10_MIC", "log10mic", "log10_mic_for_diagnosis"],
        required=False,
    )
    if col is not None:
        return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    mic = extract_mic_numeric(df)
    out = np.full(len(df), np.nan)
    ok = np.isfinite(mic) & (mic > 0)
    out[ok] = np.log10(mic[ok])
    return out


def mic_range_group(mic: np.ndarray, threshold: float) -> np.ndarray:
    if not math.isfinite(threshold):
        return np.array(["unknown"] * len(mic), dtype=object)
    result = np.full(len(mic), "unknown", dtype=object)
    ok = np.isfinite(mic)
    result[ok & (mic <= threshold / 2.0)] = "strong_MIC_le_T_over_2"
    result[ok & (mic > threshold / 2.0) & (mic <= 2.0 * threshold)] = "boundary_T_over_2_to_2T"
    result[ok & (mic > 2.0 * threshold)] = "weak_MIC_gt_2T"
    return result


def boundary_flag(mic: np.ndarray, threshold: float) -> np.ndarray:
    if not math.isfinite(threshold):
        return np.zeros(len(mic), dtype=int)
    return (
        np.isfinite(mic)
        & (mic > threshold / 2.0)
        & (mic <= 2.0 * threshold)
    ).astype(int)


def morgan_fingerprints(smiles: Sequence[str]):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    fps = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            raise ValueError(f"Invalid SMILES at row {i}: {smi}")
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=NN_RADIUS, nBits=NN_NBITS
        )
        fps.append(fp)
    return fps


def nearest_neighbor_diagnostics(
    train_smiles: Sequence[str],
    query_smiles: Sequence[str],
) -> Dict[str, Any]:
    from rdkit import DataStructs

    train_fps = morgan_fingerprints(train_smiles)
    query_fps = morgan_fingerprints(query_smiles)
    nn_idx = np.full(len(query_fps), -1, dtype=int)
    nn_sim = np.full(len(query_fps), np.nan, dtype=float)
    all_sims: List[np.ndarray] = []

    for i, fp in enumerate(query_fps):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, train_fps), dtype=float)
        all_sims.append(sims)
        if len(sims):
            j = int(np.argmax(sims))
            nn_idx[i] = j
            nn_sim[i] = float(sims[j])

    bins = np.where(
        nn_sim >= NN_HIGH,
        ">=0.7",
        np.where(nn_sim >= NN_MID, "0.4-0.7", "<0.4"),
    )
    return {
        "train_fps": train_fps,
        "query_fps": query_fps,
        "nn_idx": nn_idx,
        "nn_sim": nn_sim,
        "nn_bin": bins,
        "all_sims": all_sims,
    }


def add_common_identity(
    df: pd.DataFrame,
    split_name: str,
    dataset_name: str,
    genus: str,
    task: str,
    split_type: str,
    seed: int,
) -> pd.DataFrame:
    out = df.copy()
    out["split"] = split_name
    out["dataset_name"] = dataset_name
    out["genus"] = genus
    out["task"] = task
    out["split_type"] = split_type
    out["seed"] = int(seed)
    out["model"] = MODEL_NAME
    out["model_family"] = MODEL_FAMILY
    return out


def add_blank_test_diagnostic_columns(out: pd.DataFrame, task: str) -> pd.DataFrame:
    common = [
        "boundary_flag", "threshold_for_boundary", "mic_range_group",
        "nn_train_row", "nn_sim", "nn_bin", "nn_train_smiles",
        "ac_flag", "ac_neighbor_count", "ac_pair_count_for_molecule",
    ]
    cls = [
        "log2_mic_for_diagnosis", "log10_mic_for_diagnosis", "mic_numeric",
        "nn_train_active", "nn_train_score",
        "ac_max_abs_log2mic_diff",
        "ac_pair_directional_accuracy_for_molecule",
    ]
    reg = [
        "nn_train_log10mic", "nn_train_pred_log10mic",
        "local_train_log10mic_variance_k10", "local_train_log10mic_sd_k10",
        "ac_max_abs_log10mic_diff",
        "ac_pair_directional_accuracy_for_molecule",
    ]
    for c in common + (cls if task == "classification" else reg):
        if c not in out:
            out[c] = np.nan
    return out


def classification_prediction_frame(
    raw_df: pd.DataFrame,
    y_true: np.ndarray,
    pred_prob: np.ndarray,
    split_name: str,
    dataset_name: str,
    genus: str,
    split_type: str,
    seed: int,
    threshold: float,
    val_f1_threshold: float,
) -> pd.DataFrame:
    out = add_common_identity(
        raw_df, split_name, dataset_name, genus,
        "classification", split_type, seed
    )
    out["smiles"] = raw_df["smiles"].astype(str).to_numpy()
    out["active"] = y_true.astype(int)
    out["true_active"] = y_true.astype(int)
    out["pred_score"] = pred_prob
    out["pred_prob"] = pred_prob
    out["threshold_used"] = threshold
    out["pred_label"] = (pred_prob >= threshold).astype(int)
    out["pred_label_0p5"] = (pred_prob >= 0.5).astype(int)
    out["val_best_f1_threshold"] = val_f1_threshold
    out["error"] = (out["pred_label"].to_numpy() != y_true).astype(int)

    mic = extract_mic_numeric(raw_df)
    log10mic = extract_log10_mic_for_diagnosis(raw_df)
    threshold_mic = infer_threshold(dataset_name)
    out["mic_numeric"] = mic
    out["log10_mic_for_diagnosis"] = log10mic
    out["log2_mic_for_diagnosis"] = np.where(
        np.isfinite(mic) & (mic > 0), np.log2(mic), np.nan
    )
    out["threshold_for_boundary"] = threshold_mic
    out["boundary_flag"] = boundary_flag(mic, threshold_mic)
    out["mic_range_group"] = mic_range_group(mic, threshold_mic)
    return add_blank_test_diagnostic_columns(out, "classification")


def regression_prediction_frame(
    raw_df: pd.DataFrame,
    y_true: np.ndarray,
    pred: np.ndarray,
    split_name: str,
    dataset_name: str,
    genus: str,
    split_type: str,
    seed: int,
) -> pd.DataFrame:
    out = add_common_identity(
        raw_df, split_name, dataset_name, genus,
        "regression", split_type, seed
    )
    out["smiles"] = raw_df["smiles"].astype(str).to_numpy()
    out["log10_mic"] = y_true
    out["true_log10MIC"] = y_true
    out["pred_log10MIC"] = pred
    out["true_MIC"] = np.power(10.0, y_true)
    out["mic_numeric"] = out["true_MIC"]
    out["pred_MIC"] = np.power(10.0, pred)
    out["score_reg"] = -pred
    abs_err = np.abs(pred - y_true)
    out["abs_error_log10"] = abs_err
    out["sq_error_log10"] = np.square(pred - y_true)
    out["fold_error"] = np.power(10.0, abs_err)
    out["within_1_dilution"] = (abs_err <= DILUTION_1).astype(int)
    out["within_2_dilutions"] = (abs_err <= DILUTION_2).astype(int)

    threshold_mic = infer_threshold(dataset_name)
    out["threshold_for_boundary"] = threshold_mic
    out["true_active_for_screening"] = (
        out["true_MIC"].to_numpy() <= threshold_mic
    ).astype(int) if math.isfinite(threshold_mic) else np.nan
    out["boundary_flag"] = boundary_flag(out["true_MIC"].to_numpy(), threshold_mic)
    out["mic_range_group"] = mic_range_group(
        out["true_MIC"].to_numpy(), threshold_mic
    )
    return add_blank_test_diagnostic_columns(out, "regression")


def apply_classification_test_diagnostics(
    test_pred: pd.DataFrame,
    train_pred: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    diag = nearest_neighbor_diagnostics(
        train_pred["smiles"].astype(str).tolist(),
        test_pred["smiles"].astype(str).tolist(),
    )
    idx = diag["nn_idx"]
    test_pred = test_pred.copy()
    test_pred["nn_train_row"] = idx
    test_pred["nn_sim"] = diag["nn_sim"]
    test_pred["nn_bin"] = diag["nn_bin"]
    test_pred["nn_train_smiles"] = [
        train_pred.iloc[j]["smiles"] if j >= 0 else np.nan for j in idx
    ]
    test_pred["nn_train_active"] = [
        train_pred.iloc[j]["true_active"] if j >= 0 else np.nan for j in idx
    ]
    test_pred["nn_train_score"] = [
        train_pred.iloc[j]["pred_score"] if j >= 0 else np.nan for j in idx
    ]

    train_log10 = pd.to_numeric(
        train_pred["log10_mic_for_diagnosis"], errors="coerce"
    ).to_numpy(float)
    test_log10 = pd.to_numeric(
        test_pred["log10_mic_for_diagnosis"], errors="coerce"
    ).to_numpy(float)
    train_score = train_pred["pred_score"].to_numpy(float)
    test_score = test_pred["pred_score"].to_numpy(float)

    ac_flag = np.zeros(len(test_pred), dtype=int)
    ac_count = np.zeros(len(test_pred), dtype=int)
    ac_max_log2 = np.full(len(test_pred), np.nan)
    ac_dir_acc = np.full(len(test_pred), np.nan)

    for i, sims in enumerate(diag["all_sims"]):
        if not np.isfinite(test_log10[i]):
            continue
        cand = np.where(
            (sims >= AC_SIM_THRESHOLD)
            & np.isfinite(train_log10)
            & (np.abs(train_log10 - test_log10[i]) >= AC_DELTA_LOG10)
        )[0]
        if len(cand) == 0:
            continue
        ac_flag[i] = 1
        ac_count[i] = len(cand)
        diffs = np.abs(train_log10[cand] - test_log10[i])
        ac_max_log2[i] = float(np.max(diffs) / math.log10(2.0))
        # Higher score should correspond to lower MIC.
        truth_delta = test_log10[i] - train_log10[cand]
        score_delta = test_score[i] - train_score[cand]
        valid = np.isfinite(truth_delta) & np.isfinite(score_delta) & (truth_delta != 0)
        if valid.any():
            ac_dir_acc[i] = float(np.mean(score_delta[valid] * truth_delta[valid] < 0))

    test_pred["ac_flag"] = ac_flag
    test_pred["ac_neighbor_count"] = ac_count
    test_pred["ac_pair_count_for_molecule"] = ac_count
    test_pred["ac_max_abs_log2mic_diff"] = ac_max_log2
    test_pred["ac_pair_directional_accuracy_for_molecule"] = ac_dir_acc
    return test_pred, diag


def apply_regression_test_diagnostics(
    test_pred: pd.DataFrame,
    train_pred: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    diag = nearest_neighbor_diagnostics(
        train_pred["smiles"].astype(str).tolist(),
        test_pred["smiles"].astype(str).tolist(),
    )
    idx = diag["nn_idx"]
    test_pred = test_pred.copy()
    test_pred["nn_train_row"] = idx
    test_pred["nn_sim"] = diag["nn_sim"]
    test_pred["nn_bin"] = diag["nn_bin"]
    test_pred["nn_train_smiles"] = [
        train_pred.iloc[j]["smiles"] if j >= 0 else np.nan for j in idx
    ]
    test_pred["nn_train_log10mic"] = [
        train_pred.iloc[j]["true_log10MIC"] if j >= 0 else np.nan for j in idx
    ]
    test_pred["nn_train_pred_log10mic"] = [
        train_pred.iloc[j]["pred_log10MIC"] if j >= 0 else np.nan for j in idx
    ]

    train_true = train_pred["true_log10MIC"].to_numpy(float)
    train_pred_y = train_pred["pred_log10MIC"].to_numpy(float)
    test_true = test_pred["true_log10MIC"].to_numpy(float)
    test_pred_y = test_pred["pred_log10MIC"].to_numpy(float)

    local_var = np.full(len(test_pred), np.nan)
    local_sd = np.full(len(test_pred), np.nan)
    ac_flag = np.zeros(len(test_pred), dtype=int)
    ac_count = np.zeros(len(test_pred), dtype=int)
    ac_max = np.full(len(test_pred), np.nan)
    ac_dir_acc = np.full(len(test_pred), np.nan)

    for i, sims in enumerate(diag["all_sims"]):
        if len(sims):
            k = min(10, len(sims))
            top = np.argsort(-sims, kind="mergesort")[:k]
            vals = train_true[top]
            if np.isfinite(vals).sum() >= 2:
                local_var[i] = float(np.nanvar(vals, ddof=1))
                local_sd[i] = float(np.nanstd(vals, ddof=1))

        cand = np.where(
            (sims >= AC_SIM_THRESHOLD)
            & np.isfinite(train_true)
            & np.isfinite(test_true[i])
            & (np.abs(train_true - test_true[i]) >= AC_DELTA_LOG10)
        )[0]
        if len(cand) == 0:
            continue
        ac_flag[i] = 1
        ac_count[i] = len(cand)
        ac_max[i] = float(np.max(np.abs(train_true[cand] - test_true[i])))
        true_delta = test_true[i] - train_true[cand]
        pred_delta = test_pred_y[i] - train_pred_y[cand]
        valid = np.isfinite(true_delta) & np.isfinite(pred_delta) & (true_delta != 0)
        if valid.any():
            ac_dir_acc[i] = float(np.mean(true_delta[valid] * pred_delta[valid] > 0))

    test_pred["local_train_log10mic_variance_k10"] = local_var
    test_pred["local_train_log10mic_sd_k10"] = local_sd
    test_pred["ac_flag"] = ac_flag
    test_pred["ac_neighbor_count"] = ac_count
    test_pred["ac_pair_count_for_molecule"] = ac_count
    test_pred["ac_max_abs_log10mic_diff"] = ac_max
    test_pred["ac_pair_directional_accuracy_for_molecule"] = ac_dir_acc
    return test_pred, diag


def subgroup_classification_metrics(
    test_pred: pd.DataFrame,
    group_type: str,
    group_col: str,
) -> List[Dict[str, Any]]:
    rows = []
    for value, sub in test_pred.groupby(group_col, dropna=False):
        m = classification_metrics(
            sub["true_active"].to_numpy(int),
            sub["pred_prob"].to_numpy(float),
            threshold=DEFAULT_THRESHOLD,
        )
        row = {
            "group_type": group_type,
            "group_col": group_col,
            "group_value": value,
        }
        row.update(m)
        rows.append(row)
    return rows


def subgroup_regression_metrics(
    test_pred: pd.DataFrame,
    group_type: str,
    group_col: str,
) -> List[Dict[str, Any]]:
    rows = []
    for value, sub in test_pred.groupby(group_col, dropna=False):
        active = None
        if "true_active_for_screening" in sub and pd.to_numeric(
            sub["true_active_for_screening"], errors="coerce"
        ).notna().all():
            active = sub["true_active_for_screening"].to_numpy(int)
        m = regression_metrics(
            sub["true_log10MIC"].to_numpy(float),
            sub["pred_log10MIC"].to_numpy(float),
            active,
        )
        row = {
            "group_type": group_type,
            "group_col": group_col,
            "group_value": value,
        }
        row.update(m)
        rows.append(row)
    return rows


def diagnostic_outputs_classification(
    test_pred: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for gt, gc in [
        ("MIC_boundary", "boundary_flag"),
        ("NN_similarity", "nn_bin"),
        ("Activity_cliff", "ac_flag"),
        ("MIC_range", "mic_range_group"),
    ]:
        rows.extend(subgroup_classification_metrics(test_pred, gt, gc))
    long_df = pd.DataFrame(rows)

    wide: Dict[str, Any] = {}
    boundary = test_pred[test_pred["boundary_flag"] == 1]
    non_boundary = test_pred[test_pred["boundary_flag"] == 0]
    ac = test_pred[test_pred["ac_flag"] == 1]
    non_ac = test_pred[test_pred["ac_flag"] == 0]

    def err_rate(df: pd.DataFrame) -> float:
        return float(df["error"].mean()) if len(df) else float("nan")

    wide["boundary_n"] = len(boundary)
    wide["non_boundary_n"] = len(non_boundary)
    wide["boundary_error_rate"] = err_rate(boundary)
    wide["non_boundary_error_rate"] = err_rate(non_boundary)
    wide["boundary_error_enrichment"] = (
        wide["boundary_error_rate"] / wide["non_boundary_error_rate"]
        if safe_float(wide["non_boundary_error_rate"]) > 0 else float("nan")
    )

    for label, mask in [
        ("nn_ge_0p7", test_pred["nn_bin"] == ">=0.7"),
        ("nn_0p4_0p7", test_pred["nn_bin"] == "0.4-0.7"),
        ("nn_lt_0p4", test_pred["nn_bin"] == "<0.4"),
    ]:
        sub = test_pred[mask]
        met = classification_metrics(
            sub["true_active"].to_numpy(int),
            sub["pred_prob"].to_numpy(float),
        ) if len(sub) else {}
        wide[f"{label}_n"] = len(sub)
        wide[f"{label}_error_rate"] = err_rate(sub)
        for k in ("roc_auc", "pr_auc", "accuracy", "f1", "mcc", "EF1%"):
            wide[f"{label}_{k}"] = met.get(k, float("nan"))

    wide["ac_n"] = len(ac)
    wide["non_ac_n"] = len(non_ac)
    wide["ac_error_rate"] = err_rate(ac)
    wide["non_ac_error_rate"] = err_rate(non_ac)
    wide["ac_error_enrichment"] = (
        wide["ac_error_rate"] / wide["non_ac_error_rate"]
        if safe_float(wide["non_ac_error_rate"]) > 0 else float("nan")
    )
    wide["ac_subgroup_pr_auc"] = (
        safe_pr_auc(ac["true_active"], ac["pred_prob"]) if len(ac) else float("nan")
    )
    wide["ac_pair_directional_accuracy"] = float(
        pd.to_numeric(
            ac["ac_pair_directional_accuracy_for_molecule"], errors="coerce"
        ).mean()
    ) if len(ac) else float("nan")
    wide["ac_pair_total_count"] = int(
        pd.to_numeric(test_pred["ac_pair_count_for_molecule"], errors="coerce")
        .fillna(0).sum()
    )
    return pd.DataFrame([wide]), long_df


def diagnostic_outputs_regression(
    test_pred: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for gt, gc in [
        ("MIC_boundary", "boundary_flag"),
        ("NN_similarity", "nn_bin"),
        ("Activity_cliff", "ac_flag"),
        ("MIC_range", "mic_range_group"),
    ]:
        rows.extend(subgroup_regression_metrics(test_pred, gt, gc))

    median_local = pd.to_numeric(
        test_pred["local_train_log10mic_variance_k10"], errors="coerce"
    ).median()
    local_group = np.where(
        pd.to_numeric(
            test_pred["local_train_log10mic_variance_k10"], errors="coerce"
        ) >= median_local,
        "high_local_variance",
        "low_local_variance",
    )
    test_pred = test_pred.copy()
    test_pred["local_variance_group"] = local_group
    rows.extend(
        subgroup_regression_metrics(
            test_pred, "Local_variance", "local_variance_group"
        )
    )
    long_df = pd.DataFrame(rows)

    wide: Dict[str, Any] = {}
    boundary = test_pred[test_pred["boundary_flag"] == 1]
    non_boundary = test_pred[test_pred["boundary_flag"] == 0]
    ac = test_pred[test_pred["ac_flag"] == 1]
    non_ac = test_pred[test_pred["ac_flag"] == 0]

    def rmse(df: pd.DataFrame) -> float:
        if len(df) == 0:
            return float("nan")
        return float(math.sqrt(mean_squared_error(
            df["true_log10MIC"], df["pred_log10MIC"]
        )))

    def mae(df: pd.DataFrame) -> float:
        if len(df) == 0:
            return float("nan")
        return float(mean_absolute_error(
            df["true_log10MIC"], df["pred_log10MIC"]
        ))

    wide["boundary_n"] = len(boundary)
    wide["non_boundary_n"] = len(non_boundary)
    wide["boundary_mae"] = mae(boundary)
    wide["non_boundary_mae"] = mae(non_boundary)
    wide["boundary_rmse"] = rmse(boundary)
    wide["non_boundary_rmse"] = rmse(non_boundary)
    wide["boundary_regression_error_enrichment"] = (
        wide["boundary_mae"] / wide["non_boundary_mae"]
        if safe_float(wide["non_boundary_mae"]) > 0 else float("nan")
    )
    wide["boundary_within_1_dilution"] = (
        float(boundary["within_1_dilution"].mean()) if len(boundary) else float("nan")
    )
    wide["non_boundary_within_1_dilution"] = (
        float(non_boundary["within_1_dilution"].mean())
        if len(non_boundary) else float("nan")
    )

    for label, mask in [
        ("nn_ge_0p7", test_pred["nn_bin"] == ">=0.7"),
        ("nn_0p4_0p7", test_pred["nn_bin"] == "0.4-0.7"),
        ("nn_lt_0p4", test_pred["nn_bin"] == "<0.4"),
    ]:
        sub = test_pred[mask]
        active = sub["true_active_for_screening"].to_numpy(int) if (
            len(sub) and pd.to_numeric(
                sub["true_active_for_screening"], errors="coerce"
            ).notna().all()
        ) else None
        met = regression_metrics(
            sub["true_log10MIC"], sub["pred_log10MIC"], active
        ) if len(sub) else {}
        wide[f"{label}_n"] = len(sub)
        for k in (
            "mae", "rmse", "mse", "medae", "r2", "pearson",
            "spearman", "within_1_dilution", "within_2_dilutions",
            "EF1%",
        ):
            wide[f"{label}_{k}"] = met.get(k, float("nan"))

    wide["ac_n"] = len(ac)
    wide["non_ac_n"] = len(non_ac)
    wide["ac_mae"] = mae(ac)
    wide["non_ac_mae"] = mae(non_ac)
    wide["ac_rmse"] = rmse(ac)
    wide["non_ac_rmse"] = rmse(non_ac)
    wide["ac_error_enrichment"] = (
        wide["ac_mae"] / wide["non_ac_mae"]
        if safe_float(wide["non_ac_mae"]) > 0 else float("nan")
    )
    wide["ac_pair_directional_accuracy"] = float(
        pd.to_numeric(
            ac["ac_pair_directional_accuracy_for_molecule"], errors="coerce"
        ).mean()
    ) if len(ac) else float("nan")
    wide["ac_pair_total_count"] = int(
        pd.to_numeric(test_pred["ac_pair_count_for_molecule"], errors="coerce")
        .fillna(0).sum()
    )

    local = pd.to_numeric(
        test_pred["local_train_log10mic_variance_k10"], errors="coerce"
    )
    wide["mean_local_train_log10mic_variance_k10"] = float(local.mean())
    wide["median_local_train_log10mic_variance_k10"] = float(local.median())
    high = test_pred[test_pred["local_variance_group"] == "high_local_variance"]
    low = test_pred[test_pred["local_variance_group"] == "low_local_variance"]
    wide["high_local_variance_mae"] = mae(high)
    wide["low_local_variance_mae"] = mae(low)
    wide["local_variance_error_enrichment"] = (
        wide["high_local_variance_mae"] / wide["low_local_variance_mae"]
        if safe_float(wide["low_local_variance_mae"]) > 0 else float("nan")
    )
    return pd.DataFrame([wide]), long_df


def add_identity_to_diagnostics(
    df: pd.DataFrame,
    dataset_name: str,
    genus: str,
    task: str,
    split_type: str,
    seed: int,
) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "model", MODEL_NAME)
    out.insert(1, "model_family", MODEL_FAMILY)
    out.insert(2, "task", task)
    out.insert(3, "dataset_name", dataset_name)
    out.insert(4, "genus", genus)
    out.insert(5, "split_type", split_type)
    out.insert(6, "seed", int(seed))
    return out


def make_train_log_long(train_log: pd.DataFrame, task: str) -> pd.DataFrame:
    id_cols = [
        c for c in [
            "iteration", "epoch", "learning_rate",
            "best_epoch_so_far", "best_val_metric_so_far", "note"
        ] if c in train_log.columns
    ]
    metric_cols = [c for c in train_log.columns if c not in id_cols]
    rows = []
    for _, row in train_log.iterrows():
        for col in metric_cols:
            if col.startswith("train_"):
                split = "train"
                metric = col[len("train_"):]
            elif col.startswith("val_"):
                split = "validation"
                metric = col[len("val_"):]
            else:
                split = "general"
                metric = col
            rows.append({
                "iteration": row.get("iteration"),
                "epoch": row.get("epoch"),
                "split": split,
                "metric": metric,
                "value": row.get(col),
                "learning_rate": row.get("learning_rate"),
                "task": task,
            })
    return pd.DataFrame(rows)


def _autocast(enabled: bool):
    import torch
    if not enabled or not torch.cuda.is_available():
        return contextlib.nullcontext()
    try:
        return torch.amp.autocast(device_type="cuda", enabled=True)
    except Exception:
        return torch.cuda.amp.autocast(enabled=True)


def _grad_scaler(enabled: bool):
    import torch
    if not enabled or not torch.cuda.is_available():
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=True)


def _move_batch(batch, device, task: str):
    net_input, target = batch
    net_input = {k: v.to(device, non_blocking=True) for k, v in net_input.items()}
    target = target.to(device, non_blocking=True)
    if task == "classification":
        target = target.float().view(-1)
    else:
        target = target.float().view(-1)
    return net_input, target


class MultiConformerDataset:
    """
    One row represents one molecule and stores a fixed list of cached conformers.

    train mode:
        Select one cached conformer deterministically for each molecule and epoch.
        No RDKit conformer generation occurs during training.
    fixed mode:
        Use one fixed cached conformer index.
    """

    def __init__(
        self,
        features: Sequence[Sequence[Mapping[str, np.ndarray]]],
        targets: Sequence[Any],
        mode: str = "train",
        seed: int = 0,
        epoch: int = 0,
        fixed_conformer_index: int = 0,
    ):
        self.features = features
        self.targets = np.asarray(targets)
        self.mode = mode
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.fixed_conformer_index = int(fixed_conformer_index)

    def __len__(self) -> int:
        return len(self.features)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _select_index(self, idx: int) -> int:
        conformers = self.features[idx]
        n = len(conformers)
        if n < 1:
            raise RuntimeError(f"Molecule index {idx} has no cached conformers.")
        if self.mode == "train":
            # Deterministic cycling; the training seed changes the starting point,
            # but the cached coordinates themselves are identical across runs.
            return (idx + self.epoch + self.seed) % n
        return self.fixed_conformer_index % n

    def __getitem__(self, idx: int):
        conf_idx = self._select_index(idx)
        return self.features[idx][conf_idx], self.targets[idx]



class FlattenedConformerDataset:
    """Flatten molecule × conformer into one deterministic evaluation dataset."""

    def __init__(
        self,
        features: Sequence[Sequence[Mapping[str, np.ndarray]]],
        targets: Sequence[Any],
    ):
        self.features = features
        self.targets = np.asarray(targets)
        if len(features) == 0:
            self.n_confs = 0
        else:
            counts = {len(x) for x in features}
            if len(counts) != 1:
                raise RuntimeError(
                    f"All molecules must have equal conformer counts; got {counts}"
                )
            self.n_confs = next(iter(counts))

    def __len__(self) -> int:
        return len(self.features) * self.n_confs

    def __getitem__(self, flat_idx: int):
        mol_idx, conf_idx = divmod(flat_idx, self.n_confs)
        return self.features[mol_idx][conf_idx], self.targets[mol_idx]


class IdentityTargetScaler:
    def transform(self, x):
        return x

    def inverse_transform(self, x):
        return x


def build_dataloader(
    features,
    targets,
    model,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    mode: str = "train",
    seed: int = 0,
    epoch: int = 0,
    fixed_conformer_index: int = 0,
):
    import torch
    from torch.utils.data import DataLoader

    dataset = MultiConformerDataset(
        features=features,
        targets=targets,
        mode=mode,
        seed=seed,
        epoch=epoch,
        fixed_conformer_index=fixed_conformer_index,
    )
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": (num_workers > 0),
        "collate_fn": model.batch_collate_fn,
        "drop_last": False,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(**loader_kwargs)



def polynomial_warmup_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    if total_steps <= 0:
        return 1.0
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    remaining = max(0, total_steps - step)
    decay_steps = max(1, total_steps - warmup_steps)
    return max(0.0, float(remaining) / float(decay_steps))


def _evaluate_single_conformer(
    model,
    features,
    targets,
    task: str,
    batch_size: int,
    device,
    use_amp: bool,
    num_workers: int,
    conformer_index: int,
) -> Tuple[np.ndarray, float]:
    import torch
    import torch.nn.functional as F

    loader = build_dataloader(
        features=features,
        targets=targets,
        model=model,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        mode="fixed",
        fixed_conformer_index=conformer_index,
    )
    model.eval()
    preds = []
    losses = []
    with torch.no_grad():
        for batch in loader:
            net_input, target = _move_batch(batch, device, task)
            with _autocast(use_amp):
                logits = model(**net_input)
                if task == "classification":
                    logit = logits.view(-1)
                    loss = F.binary_cross_entropy_with_logits(logit, target)
                    pred = torch.sigmoid(logit)
                else:
                    pred = logits.view(-1)
                    loss = F.mse_loss(pred, target)
            preds.append(pred.detach().float().cpu().numpy())
            losses.append(float(loss.detach().float().cpu()))
    return (
        np.concatenate(preds) if preds else np.array([], dtype=float),
        float(np.mean(losses)) if losses else float("nan"),
    )


def evaluate_model(
    model,
    features,
    targets,
    task: str,
    batch_size: int,
    device,
    use_amp: bool,
    num_workers: int,
    conformer_mode: str = "mean_all",
    conformer_index: int = 0,
) -> Tuple[np.ndarray, float]:
    """
    Evaluate cached conformers.

    conformer_mode="single":
        Use one specified conformer. Used for the inexpensive per-epoch train
        diagnostics.
    conformer_mode="mean_all":
        Run all cached conformers and average per-molecule predictions. Used for
        validation checkpoint selection and final train/validation/test output.
    """
    import torch
    import torch.nn.functional as F

    y = np.asarray(targets)
    if len(features) != len(y):
        raise ValueError("features and targets have different lengths")
    if len(features) == 0:
        return np.array([], dtype=float), float("nan")

    if conformer_mode == "single":
        return _evaluate_single_conformer(
            model, features, y, task, batch_size, device,
            use_amp, num_workers, conformer_index,
        )
    if conformer_mode != "mean_all":
        raise ValueError(f"Unknown conformer_mode={conformer_mode}")

    from torch.utils.data import DataLoader

    flat_dataset = FlattenedConformerDataset(features, y)
    n_confs = flat_dataset.n_confs
    loader_kwargs = {
        "dataset": flat_dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": (num_workers > 0),
        "collate_fn": model.batch_collate_fn,
        "drop_last": False,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    flat_loader = DataLoader(**loader_kwargs)

    model.eval()
    flat_predictions = []
    with torch.no_grad():
        for batch in flat_loader:
            net_input, target = _move_batch(batch, device, task)
            with _autocast(use_amp):
                logits = model(**net_input)
                if task == "classification":
                    pred = torch.sigmoid(logits.view(-1))
                else:
                    pred = logits.view(-1)
            flat_predictions.append(
                pred.detach().float().cpu().numpy()
            )
    flat_pred = (
        np.concatenate(flat_predictions)
        if flat_predictions else np.array([], dtype=float)
    )
    expected = len(features) * n_confs
    if len(flat_pred) != expected:
        raise RuntimeError(
            f"Flattened conformer prediction length={len(flat_pred)}, "
            f"expected={expected}"
        )
    pred_stack = flat_pred.reshape(len(features), n_confs)
    mean_pred = pred_stack.mean(axis=1)

    y_t = torch.as_tensor(y, dtype=torch.float32)
    p_t = torch.as_tensor(mean_pred, dtype=torch.float32)
    if task == "classification":
        p_t = torch.clamp(p_t, 1e-7, 1 - 1e-7)
        aggregate_loss = float(F.binary_cross_entropy(p_t, y_t).item())
    else:
        aggregate_loss = float(F.mse_loss(p_t, y_t).item())
    return mean_pred, aggregate_loss



def train_custom_unimol(
    train_features,
    train_y,
    val_features,
    val_y,
    task: str,
    model,
    run_dir: Path,
    logger: logging.Logger,
    learning_rate: float,
    epochs: int,
    patience: int,
    batch_size: int,
    eval_batch_size: int,
    warmup_ratio: float,
    max_norm: float,
    use_amp: bool,
    num_workers: int,
    log_interval: int,
    training_seed: int,
) -> Tuple[Any, pd.DataFrame, int, float, float]:
    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal Uni-Mol benchmark run.")
    device = torch.device("cuda:0")
    model = model.to(device)

    for p in model.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.99),
        eps=1e-6,
        weight_decay=0.0,
    )
    train_loader = build_dataloader(
        features=train_features,
        targets=train_y,
        model=model,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        mode="train",
        seed=training_seed,
        epoch=0,
    )
    total_steps = max(1, epochs * len(train_loader))
    warmup_steps = int(round(total_steps * warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda s: polynomial_warmup_lambda(
            s, warmup_steps, total_steps
        ),
    )
    scaler = _grad_scaler(use_amp)

    maximize = task == "classification"
    best_metric = -float("inf") if maximize else float("inf")
    best_epoch = 0
    no_improve = 0
    rows = []
    start = time.time()
    conf_size = len(train_features[0])

    logger.info(
        "Training starts: task=%s, epochs=%d, batch=%d, eval_batch=%d, "
        "lr=%.3e, warmup_ratio=%.3f, total_steps=%d, warmup_steps=%d, "
        "patience=%d, AMP=%s, class_weight=disabled, cached_conformers=%d",
        task, epochs, batch_size, eval_batch_size, learning_rate,
        warmup_ratio, total_steps, warmup_steps, patience, use_amp, conf_size,
    )
    logger.info(
        "Conformer policy: train=one cached conformer per molecule per epoch "
        "(deterministic cycling); validation=mean prediction over all %d cached "
        "conformers; no conformer generation occurs inside a training run.",
        conf_size,
    )
    logger.info("Trainable parameters: %d", sum(
        p.numel() for p in model.parameters() if p.requires_grad
    ))

    global_step = 0
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loader.dataset.set_epoch(epoch)
        model.train()
        batch_losses = []
        for batch_idx, batch in enumerate(train_loader, start=1):
            net_input, target = _move_batch(batch, device, task)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(use_amp):
                logits = model(**net_input)
                if task == "classification":
                    loss = F.binary_cross_entropy_with_logits(
                        logits.view(-1), target
                    )
                else:
                    loss = F.mse_loss(logits.view(-1), target)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()
            scheduler.step()

            global_step += 1
            batch_losses.append(float(loss.detach().float().cpu()))
            if (
                batch_idx == 1
                or batch_idx % log_interval == 0
                or batch_idx == len(train_loader)
            ):
                logger.info(
                    "epoch=%04d batch=%05d/%05d global_step=%07d "
                    "batch_loss=%.6f running_train_loss=%.6f lr=%.6e",
                    epoch, batch_idx, len(train_loader), global_step,
                    batch_losses[-1], float(np.mean(batch_losses)),
                    optimizer.param_groups[0]["lr"],
                )

        # Per-epoch train metrics use one deterministic cached conformer to avoid
        # multiplying full-dataset train evaluation cost by 11 every epoch.
        train_eval_conf = (epoch + training_seed) % conf_size
        train_pred, train_eval_loss = evaluate_model(
            model, train_features, train_y, task, eval_batch_size,
            device, use_amp, num_workers,
            conformer_mode="single",
            conformer_index=train_eval_conf,
        )
        # Checkpoint selection always uses the mean prediction across all 11
        # cached conformers.
        val_pred, val_loss = evaluate_model(
            model, val_features, val_y, task, eval_batch_size,
            device, use_amp, num_workers,
            conformer_mode="mean_all",
        )

        if task == "classification":
            train_m = classification_metrics(train_y, train_pred)
            val_m = classification_metrics(val_y, val_pred)
            selection_metric = val_m["pr_auc"]
            improved = (
                math.isfinite(selection_metric)
                and selection_metric > best_metric + 1e-12
            )
            row = {
                "iteration": epoch,
                "epoch": epoch,
                "train_loss": float(np.mean(batch_losses)),
                "train_eval_loss": train_eval_loss,
                "val_loss": val_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_log_loss": train_m["log_loss"],
                "val_log_loss": val_m["log_loss"],
                "train_auc": train_m["roc_auc"],
                "val_auc": val_m["roc_auc"],
                "train_pr_auc": train_m["pr_auc"],
                "val_pr_auc": val_m["pr_auc"],
                "train_accuracy": train_m["accuracy"],
                "val_accuracy": val_m["accuracy"],
                "train_f1": train_m["f1"],
                "val_f1": val_m["f1"],
                "train_mcc": train_m["mcc"],
                "val_mcc": val_m["mcc"],
                "train_eval_conformer_index": train_eval_conf,
                "val_conformer_aggregation": f"mean_{conf_size}",
                "note": (
                    "unweighted_BCEWithLogitsLoss;"
                    "full_end_to_end_finetuning;"
                    "shared_static_3D_cache"
                ),
            }
        else:
            train_m = regression_metrics(train_y, train_pred)
            val_m = regression_metrics(val_y, val_pred)
            selection_metric = val_m["rmse"]
            improved = (
                math.isfinite(selection_metric)
                and selection_metric < best_metric - 1e-12
            )
            row = {
                "iteration": epoch,
                "epoch": epoch,
                "train_loss": float(np.mean(batch_losses)),
                "train_eval_loss": train_eval_loss,
                "val_loss": val_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_mse": train_m["mse"],
                "val_mse": val_m["mse"],
                "train_rmse": train_m["rmse"],
                "val_rmse": val_m["rmse"],
                "train_mae": train_m["mae"],
                "val_mae": val_m["mae"],
                "train_spearman": train_m["spearman"],
                "val_spearman": val_m["spearman"],
                "train_pearson": train_m["pearson"],
                "val_pearson": val_m["pearson"],
                "train_within_1_dilution": train_m["within_1_dilution"],
                "val_within_1_dilution": val_m["within_1_dilution"],
                "train_within_2_dilution": train_m["within_2_dilutions"],
                "val_within_2_dilution": val_m["within_2_dilutions"],
                "train_eval_conformer_index": train_eval_conf,
                "val_conformer_aggregation": f"mean_{conf_size}",
                "note": (
                    "mse_loss;full_end_to_end_finetuning;"
                    "shared_static_3D_cache"
                ),
            }

        if improved:
            best_metric = float(selection_metric)
            best_epoch = epoch
            no_improve = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_val_metric": best_metric,
                "task": task,
                "model": MODEL_NAME,
                "conformer_aggregation": f"mean_{conf_size}",
            }
            torch.save(checkpoint, run_dir / "best_model.pth")
            torch.save(checkpoint, run_dir / "model_0.pth")
        else:
            no_improve += 1

        row["best_epoch_so_far"] = best_epoch
        row["best_val_metric_so_far"] = best_metric
        row["epoch_seconds"] = time.time() - epoch_start
        rows.append(row)

        pd.DataFrame(rows).to_csv(run_dir / "train_log.csv", index=False)
        pd.DataFrame(rows).to_csv(run_dir / "training_curve.csv", index=False)
        make_train_log_long(pd.DataFrame(rows), task).to_csv(
            run_dir / "train_log_long.csv", index=False
        )

        if task == "classification":
            logger.info(
                "epoch=%04d train_loss=%.6f val_loss=%.6f "
                "train_PR_AUC=%.6f val_PR_AUC=%.6f "
                "train_ROC_AUC=%.6f val_ROC_AUC=%.6f "
                "train_eval_conf=%d val_agg=mean_%d "
                "best_epoch=%d best_val_PR_AUC=%.6f "
                "no_improve=%d epoch_seconds=%.1f",
                epoch, row["train_loss"], val_loss,
                train_m["pr_auc"], val_m["pr_auc"],
                train_m["roc_auc"], val_m["roc_auc"],
                train_eval_conf, conf_size,
                best_epoch, best_metric, no_improve, row["epoch_seconds"],
            )
        else:
            logger.info(
                "epoch=%04d train_loss=%.6f val_loss=%.6f "
                "train_RMSE=%.6f val_RMSE=%.6f "
                "train_Spearman=%.6f val_Spearman=%.6f "
                "train_eval_conf=%d val_agg=mean_%d "
                "best_epoch=%d best_val_RMSE=%.6f "
                "no_improve=%d epoch_seconds=%.1f",
                epoch, row["train_loss"], val_loss,
                train_m["rmse"], val_m["rmse"],
                train_m["spearman"], val_m["spearman"],
                train_eval_conf, conf_size,
                best_epoch, best_metric, no_improve, row["epoch_seconds"],
            )

        if no_improve >= patience:
            logger.info(
                "Early stopping: epoch=%d, best_epoch=%d, best_val_metric=%.6f",
                epoch, best_epoch, best_metric,
            )
            break

    if best_epoch == 0 or not (run_dir / "best_model.pth").is_file():
        raise RuntimeError("No valid checkpoint was selected.")

    checkpoint = torch.load(
        run_dir / "best_model.pth", map_location=device
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    train_seconds = time.time() - start
    return model, pd.DataFrame(rows), best_epoch, best_metric, train_seconds



def initialize_unimol_from_cache(
    task: str,
    run_dir: Path,
    training_seed: int,
    pooler_dropout: float,
    logger: logging.Logger,
):
    """
    Initialize the official Uni-Mol v1 no-H pretrained backbone without calling
    DataHub. All molecular inputs are loaded from the shared cache.
    """
    try:
        from unimol_tools.tasks import Trainer
        from unimol_tools.models import NNModel
    except Exception as e:
        raise RuntimeError(
            "Cannot import unimol_tools internals. Run preflight_unimol.py "
            "inside the installed Uni-Mol environment."
        ) from e

    params = {
        "task": task,
        "data_type": "molecule",
        "model_name": "unimolv1",
        "remove_hs": True,
        "target_normalize": "none",
        "split": "random",
        "split_method": "1fold_random",
        "kfold": 1,
        "split_seed": training_seed,
        "seed": training_seed,
        "metrics": "auprc" if task == "classification" else "mse",
        "pooler_dropout": pooler_dropout,
        "use_cuda": True,
        "use_amp": True,
        "use_ddp": False,
        "use_gpu": "0",
        "batch_size": 32,
        "epochs": 1,
        "patience": 1,
        "learning_rate": 1e-4,
        "warmup_ratio": 0.06,
        "max_norm": 1.0,
    }
    dummy_feature = {
        "src_tokens": np.array([0, 0], dtype=np.int64),
        "src_distance": np.zeros((2, 2), dtype=np.float32),
        "src_coord": np.zeros((2, 3), dtype=np.float32),
        "src_edge_type": np.zeros((2, 2), dtype=np.int64),
    }
    dummy_data = {
        "num_classes": 1,
        "target_scaler": IdentityTargetScaler(),
        "unimol_input": [dummy_feature],
        "target": np.zeros((1, 1), dtype=np.float32),
        "group": None,
        "kfold": 1,
        "split_nfolds": [(np.array([0]), np.array([0]))],
    }
    trainer = Trainer(save_path=str(run_dir), **params)
    bundle = NNModel(dummy_data, trainer, **params)
    model = bundle.model

    if task == "classification":
        try:
            from unimol_tools.models.unimol import LinearHead
            input_dim = int(model.args.encoder_embed_dim)
            model.classification_head = LinearHead(
                input_dim=input_dim,
                num_classes=1,
                pooler_dropout=float(pooler_dropout),
            )
            model.output_dim = 1
            logger.info(
                "Classification head reset to one output logit for "
                "unweighted BCEWithLogitsLoss."
            )
        except Exception as e:
            raise RuntimeError(
                "Failed to create the one-logit Uni-Mol classification head."
            ) from e

    changed = []
    if hasattr(model, "classification_head"):
        for name, module in model.classification_head.named_modules():
            if module.__class__.__name__ == "Dropout" and hasattr(module, "p"):
                module.p = float(pooler_dropout)
                changed.append(f"classification_head.{name or '<root>'}")
    logger.info(
        "Prediction-head dropout set to %.3f; changed_modules=%s",
        pooler_dropout, changed
    )
    return model, params, changed


def load_shared_conformer_cache(
    cache_dir: Path,
    expected_dataset_name: str,
    expected_conf_size: int,
    expected_conformer_seed: int,
    logger: logging.Logger,
) -> Tuple[Dict[str, Sequence[Mapping[str, np.ndarray]]], Dict[str, Any]]:
    cache_dir = Path(cache_dir).resolve()
    info_path = cache_dir / "cache_info.json"
    feature_path = cache_dir / "unimol_v1_noh_conformers.pkl"
    completed_path = cache_dir / "completed.flag"

    for path in (info_path, feature_path, completed_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Shared 3D cache is incomplete: missing {path}. "
                "Run submit_unimol_3d_cache_slurm.sh before model training."
            )

    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    checks = {
        "dataset_name": (info.get("dataset_name"), expected_dataset_name),
        "model_name": (info.get("model_name"), "unimolv1"),
        "remove_hs": (info.get("remove_hs"), True),
        "conf_size": (int(info.get("conf_size", -1)), expected_conf_size),
        "conformer_seed": (
            int(info.get("conformer_seed", -999999)),
            expected_conformer_seed,
        ),
    }
    bad = {
        k: {"cache": got, "expected": expected}
        for k, (got, expected) in checks.items()
        if got != expected
    }
    if bad:
        raise RuntimeError(
            f"Shared cache configuration mismatch in {cache_dir}: {bad}"
        )

    logger.info("Loading shared 3D cache: %s", feature_path.name)
    started = time.time()
    with feature_path.open("rb") as f:
        payload = pickle.load(f)
    smiles = payload["smiles"]
    conformers = payload["conformers"]
    if len(smiles) != len(conformers):
        raise RuntimeError("Shared cache smiles/conformers length mismatch")
    if len(set(smiles)) != len(smiles):
        raise RuntimeError("Shared cache contains duplicate SMILES keys")
    for i, confs in enumerate(conformers):
        if len(confs) != expected_conf_size:
            raise RuntimeError(
                f"Cache molecule index {i} has {len(confs)} conformers; "
                f"expected {expected_conf_size}"
            )
    mapping = dict(zip(smiles, conformers))
    logger.info(
        "Shared 3D cache loaded: molecules=%d conformers_per_molecule=%d "
        "seconds=%.1f",
        len(mapping), expected_conf_size, time.time() - started,
    )
    return mapping, info


def features_for_split(
    df: pd.DataFrame,
    cache_mapping: Mapping[str, Sequence[Mapping[str, np.ndarray]]],
    split_name: str,
) -> List[Sequence[Mapping[str, np.ndarray]]]:
    result = []
    missing = []
    for row_idx, smi in enumerate(df["smiles"].astype(str).tolist()):
        if smi not in cache_mapping:
            missing.append((row_idx, smi))
            if len(missing) >= 20:
                break
        else:
            result.append(cache_mapping[smi])
    if missing:
        raise KeyError(
            f"{split_name} contains SMILES absent from shared cache; "
            f"first examples={missing}"
        )
    if len(result) != len(df):
        raise RuntimeError(
            f"{split_name} cache mapping length mismatch: "
            f"{len(result)} vs {len(df)}"
        )
    return result






def run_benchmark(args: argparse.Namespace, task: str) -> None:
    import torch

    torch.set_num_threads(max(1, int(args.cpu_threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    run_dir = ensure_dir(Path(args.out_dir).resolve())
    logger = setup_logger(run_dir)
    started_at = time.time()
    status_path = run_dir / "status.json"

    train_path = Path(args.train_csv).resolve()
    val_path = Path(args.validation_csv).resolve()
    test_path = Path(args.test_csv).resolve()
    shared_cache_dir = Path(args.shared_cache_dir).resolve()

    dataset_name = args.dataset_name or infer_dataset_name(train_path)
    split_type = args.split_type or infer_split_type(train_path)
    seed = int(args.seed if args.seed is not None else infer_seed(train_path))
    genus = infer_genus(dataset_name)

    write_json(status_path, {
        "status": "running",
        "task": task,
        "dataset_name": dataset_name,
        "split_type": split_type,
        "seed": seed,
        "started_at_unix": started_at,
    })

    try:
        set_global_seed(seed)
        logger.info("=" * 100)
        logger.info(
            "RUN START model=%s task=%s dataset=%s genus=%s split=%s seed=%d",
            MODEL_NAME, task, dataset_name, genus, split_type, seed
        )
        logger.info("train_csv=%s", train_path.name)
        logger.info("validation_csv=%s", val_path.name)
        logger.info("test_csv=%s", test_path.name)
        logger.info("out_dir=%s", run_dir.name)
        logger.info("shared_cache_dir=%s", shared_cache_dir.name)
        logger.info(
            "CPU controls: SLURM_CPUS_PER_TASK=%s torch_cpu_threads=%d "
            "dataloader_num_workers=%d OMP_NUM_THREADS=%s",
            os.environ.get("SLURM_CPUS_PER_TASK"),
            args.cpu_threads,
            args.num_workers,
            os.environ.get("OMP_NUM_THREADS"),
        )
        logger.info(
            "python=%s torch=%s cuda_available=%s gpu=%s",
            sys.version.replace("\n", " "),
            torch.__version__, torch.cuda.is_available(),
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        )

        train_df, train_smi, train_target = load_split_csv(train_path, task)
        val_df, val_smi, val_target = load_split_csv(val_path, task)
        test_df, test_smi, test_target = load_split_csv(test_path, task)
        validate_smiles_exact(train_df, train_smi, train_path)
        validate_smiles_exact(val_df, val_smi, val_path)
        validate_smiles_exact(test_df, test_smi, test_path)

        train_df, val_df, test_df = match_columns_across_splits(
            train_df, val_df, test_df,
            train_smi, val_smi, test_smi,
            train_target, val_target, test_target, task,
        )
        target_col = "active" if task == "classification" else "log10_mic"
        y_train = train_df[target_col].to_numpy()
        y_val = val_df[target_col].to_numpy()
        y_test = test_df[target_col].to_numpy()

        logger.info(
            "Data sizes: train=%d validation=%d test=%d",
            len(train_df), len(val_df), len(test_df)
        )
        if task == "classification":
            logger.info(
                "Class ratios: train_pos_ratio=%.6f val_pos_ratio=%.6f test_pos_ratio=%.6f; "
                "class weighting is DISABLED",
                float(np.mean(y_train)), float(np.mean(y_val)), float(np.mean(y_test))
            )
        else:
            logger.info(
                "Target summaries: train_mean=%.6f val_mean=%.6f test_mean=%.6f",
                float(np.mean(y_train)), float(np.mean(y_val)), float(np.mean(y_test))
            )

        cache_mapping, cache_info = load_shared_conformer_cache(
            cache_dir=shared_cache_dir,
            expected_dataset_name=dataset_name,
            expected_conf_size=args.conf_size,
            expected_conformer_seed=args.conformer_seed,
            logger=logger,
        )
        train_features = features_for_split(
            train_df, cache_mapping, "train"
        )
        val_features = features_for_split(
            val_df, cache_mapping, "validation"
        )
        test_features = features_for_split(
            test_df, cache_mapping, "test"
        )
        # Free the large dictionary shell after split lists have been created.
        del cache_mapping

        model, model_init_params, changed_dropout_modules = initialize_unimol_from_cache(
            task=task,
            run_dir=run_dir,
            training_seed=seed,
            pooler_dropout=args.pooler_dropout,
            logger=logger,
        )

        model, train_log, best_epoch, best_val_metric, train_seconds = train_custom_unimol(
            train_features=train_features,
            train_y=y_train,
            val_features=val_features,
            val_y=y_val,
            task=task,
            model=model,
            run_dir=run_dir,
            logger=logger,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            warmup_ratio=args.warmup_ratio,
            max_norm=args.max_norm,
            use_amp=args.use_amp,
            num_workers=args.num_workers,
            log_interval=args.log_interval,
            training_seed=seed,
        )

        device = torch.device("cuda:0")
        logger.info(
            "Best checkpoint loaded. Computing final train/validation "
            "predictions by averaging all %d cached conformers.",
            args.conf_size,
        )
        train_pred_y, train_eval_loss = evaluate_model(
            model, train_features, y_train, task, args.eval_batch_size,
            device, args.use_amp, args.num_workers,
            conformer_mode="mean_all",
        )
        val_pred_y, val_eval_loss = evaluate_model(
            model, val_features, y_val, task, args.eval_batch_size,
            device, args.use_amp, args.num_workers,
            conformer_mode="mean_all",
        )
        logger.info(
            "The held-out test set has not been used for training or checkpoint "
            "selection. Starting the single final test evaluation using the "
            "same shared cache and mean prediction across all %d conformers.",
            args.conf_size,
        )
        test_pred_y, test_eval_loss = evaluate_model(
            model, test_features, y_test, task, args.eval_batch_size,
            device, args.use_amp, args.num_workers,
            conformer_mode="mean_all",
        )

        threshold_mic = infer_threshold(dataset_name)
        metrics_row: Dict[str, Any] = {
            "model": MODEL_NAME,
            "model_family": MODEL_FAMILY,
            "task": task,
            "dataset_name": dataset_name,
            "genus": genus,
            "split_type": split_type,
            "seed": seed,
            "feature_type": FEATURE_TYPE,
            "target_scale": "raw_binary" if task == "classification" else "log10_mic",
            "threshold_from_dataset_name": threshold_mic,
            "threshold_for_boundary": threshold_mic,
            "train_seconds": train_seconds,
            "best_epoch": best_epoch,
            "best_val_metric": best_val_metric,
            "shared_3d_cache_dir": shared_cache_dir.name,
            "conformer_seed": args.conformer_seed,
            "conformer_number": args.conf_size,
            "prediction_conformer_aggregation": f"mean_{args.conf_size}",
            "train_loss": float(
                train_log.loc[train_log["epoch"] == best_epoch, "train_loss"].iloc[0]
            ),
            "val_loss": float(
                train_log.loc[train_log["epoch"] == best_epoch, "val_loss"].iloc[0]
            ),
            "test_loss": test_eval_loss,
        }

        if task == "classification":
            val_f1_t = best_f1_threshold(y_val, val_pred_y)
            train_pred_df = classification_prediction_frame(
                train_df, y_train.astype(int), train_pred_y,
                "train", dataset_name, genus, split_type, seed,
                DEFAULT_THRESHOLD, val_f1_t,
            )
            val_pred_df = classification_prediction_frame(
                val_df, y_val.astype(int), val_pred_y,
                "validation", dataset_name, genus, split_type, seed,
                DEFAULT_THRESHOLD, val_f1_t,
            )
            test_pred_df = classification_prediction_frame(
                test_df, y_test.astype(int), test_pred_y,
                "test", dataset_name, genus, split_type, seed,
                DEFAULT_THRESHOLD, val_f1_t,
            )
            test_pred_df, _ = apply_classification_test_diagnostics(
                test_pred_df, train_pred_df
            )
            diag_wide, diag_long = diagnostic_outputs_classification(test_pred_df)

            metrics_row.update({
                "threshold_mode": "fixed_0p5",
                "threshold_used": DEFAULT_THRESHOLD,
                "val_best_f1_threshold": val_f1_t,
                "class_weight": "none",
                "use_pos_weight": False,
            })
            metrics_row.update(prefix_dict(
                "train", classification_metrics(y_train, train_pred_y)
            ))
            metrics_row.update(prefix_dict(
                "val", classification_metrics(y_val, val_pred_y)
            ))
            metrics_row.update(prefix_dict(
                "test", classification_metrics(y_test, test_pred_y)
            ))
        else:
            train_pred_df = regression_prediction_frame(
                train_df, y_train.astype(float), train_pred_y,
                "train", dataset_name, genus, split_type, seed
            )
            val_pred_df = regression_prediction_frame(
                val_df, y_val.astype(float), val_pred_y,
                "validation", dataset_name, genus, split_type, seed
            )
            test_pred_df = regression_prediction_frame(
                test_df, y_test.astype(float), test_pred_y,
                "test", dataset_name, genus, split_type, seed
            )
            test_pred_df, _ = apply_regression_test_diagnostics(
                test_pred_df, train_pred_df
            )
            diag_wide, diag_long = diagnostic_outputs_regression(test_pred_df)

            train_active = (
                np.power(10.0, y_train) <= threshold_mic
            ).astype(int) if math.isfinite(threshold_mic) else None
            val_active = (
                np.power(10.0, y_val) <= threshold_mic
            ).astype(int) if math.isfinite(threshold_mic) else None
            test_active = (
                np.power(10.0, y_test) <= threshold_mic
            ).astype(int) if math.isfinite(threshold_mic) else None
            metrics_row.update(prefix_dict(
                "train", regression_metrics(y_train, train_pred_y, train_active)
            ))
            metrics_row.update(prefix_dict(
                "val", regression_metrics(y_val, val_pred_y, val_active)
            ))
            metrics_row.update(prefix_dict(
                "test", regression_metrics(y_test, test_pred_y, test_active)
            ))

        diag_wide = add_identity_to_diagnostics(
            diag_wide, dataset_name, genus, task, split_type, seed
        )
        diag_long = add_identity_to_diagnostics(
            diag_long, dataset_name, genus, task, split_type, seed
        )
        for c in diag_wide.columns:
            if c not in metrics_row and c not in {
                "model", "model_family", "task", "dataset_name",
                "genus", "split_type", "seed"
            }:
                metrics_row[c] = diag_wide.iloc[0][c]

        train_pred_df.to_csv(run_dir / "predictions_train.csv", index=False)
        val_pred_df.to_csv(run_dir / "predictions_validation.csv", index=False)
        test_pred_df.to_csv(run_dir / "predictions_test.csv", index=False)
        diag_wide.to_csv(run_dir / "diagnostics_metrics.csv", index=False)
        diag_long.to_csv(run_dir / "diagnostic_subgroup_metrics.csv", index=False)
        pd.DataFrame([metrics_row]).to_csv(
            run_dir / "metrics_full.csv", index=False
        )

        feature_summary = {
            "model": MODEL_NAME,
            "model_version": "unimolv1",
            "model_family": MODEL_FAMILY,
            "feature_type": FEATURE_TYPE,
            "input_source": "shared_precomputed_3D_cache_keyed_by_exact_smiles",
            "shared_cache_dir": shared_cache_dir.name,
            "shared_cache_format_version": cache_info.get("format_version"),
            "shared_across_tasks_splits_and_training_seeds": True,
            "extra_structure_standardization": False,
            "smiles_parse_validation_only": True,
            "hydrogen_setting": "no-H",
            "remove_hs": True,
            "conformer_number": args.conf_size,
            "conformer_seed": args.conformer_seed,
            "conformer_seeds": cache_info.get("conformer_seeds"),
            "train_conformer_policy": "deterministic_cycle_one_cached_conformer_per_epoch",
            "validation_conformer_policy": f"mean_prediction_over_{args.conf_size}",
            "final_prediction_conformer_policy": f"mean_prediction_over_{args.conf_size}",
            "embedding_recomputed_each_run": True,
            "representation": [
                "atom_type_tokens",
                "3D_coordinates",
                "pairwise_distances",
                "edge_types",
                "CLS molecular representation",
            ],
            "nn_diagnostic": {
                "reference": "train",
                "fingerprint": "Morgan",
                "radius": NN_RADIUS,
                "n_bits": NN_NBITS,
                "similarity": "Tanimoto",
            },
            "activity_cliff": {
                "similarity_threshold": AC_SIM_THRESHOLD,
                "minimum_abs_log10MIC_difference": AC_DELTA_LOG10,
            },
        }
        write_json(run_dir / "feature_summary.json", feature_summary)

        model_params = {
            "model": MODEL_NAME,
            "model_name": "unimolv1",
            "pretrained_weights": "official molecular no-H checkpoint selected by unimol_tools",
            "remove_hs": True,
            "full_end_to_end_finetuning": True,
            "optimizer": "Adam",
            "adam_betas": [0.9, 0.99],
            "adam_epsilon": 1e-6,
            "learning_rate": args.learning_rate,
            "weight_decay": 0.0,
            "lr_scheduler": "polynomial_decay_linear_power1",
            "warmup_ratio": args.warmup_ratio,
            "batch_size": args.batch_size,
            "effective_batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "max_epochs": args.epochs,
            "patience": args.patience,
            "pooler_dropout": args.pooler_dropout,
            "pooler_dropout_modules_changed": changed_dropout_modules,
            "gradient_clip_norm": args.max_norm,
            "conformer_number": args.conf_size,
            "conformer_seed": args.conformer_seed,
            "shared_3d_cache_dir": shared_cache_dir.name,
            "shared_3d_cache": True,
            "train_conformer_policy": "deterministic_cycle_one_per_molecule_per_epoch",
            "validation_and_final_prediction": f"mean_over_{args.conf_size}_conformers",
            "mixed_precision": args.use_amp,
            "classification_loss": (
                "BCEWithLogitsLoss"
                if task == "classification" else None
            ),
            "class_weight": "none",
            "use_pos_weight": False,
            "regression_loss": "MSE" if task == "regression" else None,
            "checkpoint_selection": (
                "max_validation_PR_AUC"
                if task == "classification"
                else "min_validation_RMSE"
            ),
            "classification_threshold": (
                DEFAULT_THRESHOLD if task == "classification" else None
            ),
            "test_evaluation": "once_after_loading_validation_selected_checkpoint",
            "seed": seed,
            "cpu_threads": args.cpu_threads,
            "dataloader_num_workers": args.num_workers,
        }
        write_json(run_dir / "model_params.json", model_params)

        run_info = {
            "model": MODEL_NAME,
            "task": task,
            "dataset_name": dataset_name,
            "genus": genus,
            "split_type": split_type,
            "seed": seed,
            "train_csv": train_path.name,
            "validation_csv": val_path.name,
            "test_csv": test_path.name,
            "train_csv_sha256": sha256_file(train_path),
            "validation_csv_sha256": sha256_file(val_path),
            "test_csv_sha256": sha256_file(test_path),
            "out_dir": run_dir.name,
            "smiles_col": "smiles",
            "target_col": target_col,
            "threshold_used": DEFAULT_THRESHOLD if task == "classification" else None,
            "threshold_for_boundary": threshold_mic,
            "software": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": package_version("torch"),
                "unimol_tools": package_version("unimol_tools"),
                "rdkit": package_version("rdkit"),
                "numpy": package_version("numpy"),
                "pandas": package_version("pandas"),
                "scipy": package_version("scipy"),
                "scikit_learn": package_version("scikit-learn"),
            },
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "started_at_unix": started_at,
            "finished_at_unix": time.time(),
            "elapsed_seconds": time.time() - started_at,
            "script": Path(__file__).name,
            "unimol_repo_commit": git_commit(os.environ.get("UNIMOL_REPO")),
            "model_init_params": model_init_params,
            "shared_3d_cache_dir": shared_cache_dir.name,
            "shared_3d_cache_info": public_cache_info(cache_info),
            "cpu_controls": {
                "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
                "torch_cpu_threads": args.cpu_threads,
                "dataloader_num_workers": args.num_workers,
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            },
        }
        write_json(run_dir / "run_info.json", run_info)
        write_json(status_path, {
            "status": "completed",
            "task": task,
            "dataset_name": dataset_name,
            "split_type": split_type,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_val_metric": best_val_metric,
            "elapsed_seconds": time.time() - started_at,
        })
        logger.info(
            "RUN COMPLETED best_epoch=%d best_val_metric=%.6f elapsed_seconds=%.1f",
            best_epoch, best_val_metric, time.time() - started_at
        )
        logger.info("=" * 100)

    except Exception as e:
        logger.error("RUN FAILED: %s", type(e).__name__)
        write_json(status_path, {
            "status": "failed",
            "task": task,
            "dataset_name": dataset_name,
            "split_type": split_type,
            "seed": seed,
            "error": type(e).__name__,
            "traceback": None,
            "elapsed_seconds": time.time() - started_at,
        })
        raise SystemExit(1) from None


def public_cache_info(info: Mapping[str, Any]) -> Dict[str, Any]:
    """Remove local host and filesystem details from shareable run metadata."""
    result = dict(info)
    result.pop("host", None)
    if result.get("dataset_dir"):
        result["dataset_dir"] = Path(result["dataset_dir"]).name
    if "source_split_files" in result:
        paths = []
        for value in result["source_split_files"]:
            parts = Path(value).parts
            start = next((i for i, part in enumerate(parts)
                          if part in ("classification", "regression")), len(parts) - 1)
            paths.append(Path(*parts[start:]).as_posix())
        result["source_split_files"] = paths
    return result


def add_common_cli(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--validation-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--shared-cache-dir",
        required=True,
        help="Dataset-specific shared Uni-Mol v1 no-H 3D cache directory.",
    )
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--split-type", default=None)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--pooler-dropout", type=float, default=0.10)
    parser.add_argument("--max-norm", type=float, default=1.0)
    parser.add_argument("--conf-size", type=int, default=11)
    parser.add_argument("--conformer-seed", type=int, default=0)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="PyTorch DataLoader worker processes per training run.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=1,
        help="CPU threads used by the main PyTorch process.",
    )
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument(
        "--use-amp", action=argparse.BooleanOptionalAction, default=True
    )
