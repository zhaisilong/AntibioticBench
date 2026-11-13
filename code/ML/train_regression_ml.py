#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AntibioticBench regression trainer for traditional ML models.

This is the regression counterpart of train_ml.py.

Expected directory layout:
    DATA_ROOT/
      1_Staphylococcus_median_16_MolWt_50_1000_value_le_8192_multitask/
        regression/
          random/seed_0/train.csv validation.csv test.csv
          scaffold/seed_0/...
          repeated_stratified_perimeter/seed_0/...
          strict_perimeter/seed_0/...

Task:
    Predict continuous log10(MIC ug/mL).

Features:
    Morgan fingerprint radius=2 nBits=2048 + RDKit 2D descriptors.

Supported models:
    rf          -> RandomForestRegressor
    svm         -> SVR with RBF kernel
    xgb         -> XGBRegressor, objective=reg:squarederror
    lightgbm    -> LGBMRegressor, objective=regression

Main outputs per run:
    metrics_full.csv
    predictions_train.csv / predictions_validation.csv / predictions_test.csv
    diagnostics_metrics.csv
    diagnostic_subgroup_metrics.csv
    train_log.csv
    training_curve.csv
    train_log_long.csv
    feature_summary.json
    model_params.json
    run_info.json
    model.joblib

Aggregate outputs per model:
    all_run_metrics.csv
    aggregate_mean_sd.csv
    ood_change_random_vs_ood_splits.csv

Notes:
    - Lower log10MIC means stronger antibacterial activity.
    - Regression screening score is score_reg = -predicted_log10MIC.
    - within 1 dilution uses |pred - obs| <= log10(2).
    - within 2 dilutions uses |pred - obs| <= log10(4).
    - MIC-specific activity cliff: Tanimoto >= 0.7 and |log10MIC_i - log10MIC_j| >= log10(16).
"""

from privacy_utils import sanitize_metadata

import argparse
import json
import math
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


N_BITS = 2048
RADIUS = 2
HARD_CODED_SMILES_COL = "smiles"

DEFAULT_TARGET_COLS = [
    "log10_mic", "log10MIC", "log10mic", "log10_value", "log10_mic_value",
    "log10_MIC", "log10_median_value", "log10_label", "y_reg", "target_regression",
    "regression_target", "target", "label",
]
DEFAULT_VALUE_COLS = [
    "mic_numeric", "value", "MIC", "mic", "mic_value", "median_value",
    "value_ug_ml", "mic_ug_ml", "MIC_ug_ml", "mic_ug_per_ml",
]
DEFAULT_ACTIVE_COLS = ["active", "label_active", "is_active", "activity_label"]
DESC_FUNCS = Descriptors._descList
DILUTION1_LOG10 = math.log10(2.0)
DILUTION2_LOG10 = math.log10(4.0)
AC_LOG10_DIFF_CUTOFF = math.log10(16.0)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def numeric_sort_key(path: Path):
    m = re.match(r"^(\d+)_", path.name)
    return (int(m.group(1)) if m else 10**9, path.name)


def parse_threshold_from_dataset_name(name: str) -> Optional[float]:
    m = re.search(r"median_([0-9]+(?:\.[0-9]+)?)", name)
    return float(m.group(1)) if m else None


def parse_genus_from_dataset_name(name: str) -> str:
    parts = name.split("_")
    if len(parts) >= 2 and parts[0].isdigit():
        return parts[1]
    return name


def find_col(df: pd.DataFrame, preferred: Optional[str], candidates: List[str], required: bool = True) -> Optional[str]:
    if preferred and preferred in df.columns:
        return preferred
    # exact first
    for c in candidates:
        if c in df.columns:
            return c
    # case-insensitive fallback
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    if required:
        raise ValueError(f"Cannot find required column. Preferred={preferred}; candidates={candidates}; columns={list(df.columns)}")
    return None


def require_smiles_column(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> str:
    for split_name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        if HARD_CODED_SMILES_COL not in df.columns:
            raise ValueError(
                f"Missing required column 'smiles' in {split_name}. "
                f"Available columns: {list(df.columns)}"
            )
    return HARD_CODED_SMILES_COL


def safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def finite_array(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr[np.isfinite(arr)]


def safe_corr(y_true, y_pred, method: str = "pearson") -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if len(y) < 2 or len(np.unique(y)) < 2 or len(np.unique(p)) < 2:
        return np.nan
    if method == "spearman":
        y = pd.Series(y).rank(method="average").values
        p = pd.Series(p).rank(method="average").values
    try:
        return float(np.corrcoef(y, p)[0, 1])
    except Exception:
        return np.nan


# -----------------------------
# Chemistry features
# -----------------------------

def mol_from_smiles(smiles: str):
    try:
        return Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None


def morgan_fp_array(smiles: str, radius: int = RADIUS, n_bits: int = N_BITS) -> np.ndarray:
    mol = mol_from_smiles(smiles)
    arr = np.zeros((n_bits,), dtype=np.int8)
    if mol is None:
        return arr.astype(np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)


def morgan_fp_bitvect(smiles: str, radius: int = RADIUS, n_bits: int = N_BITS):
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def rdkit2d_desc(smiles: str) -> np.ndarray:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return np.full(len(DESC_FUNCS), np.nan, dtype=np.float32)
    values = []
    for _, func in DESC_FUNCS:
        try:
            v = func(mol)
            v = np.nan if v is None else float(v)
            if not np.isfinite(v) or abs(v) > 1e12:
                v = np.nan
        except Exception:
            v = np.nan
        values.append(v)
    return np.array(values, dtype=np.float32)


def build_features(smiles_list: Iterable[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    smiles_list = list(smiles_list)
    print(f"[{now()}] Computing Morgan fingerprints for {len(smiles_list)} molecules ...", flush=True)
    fps = np.vstack([morgan_fp_array(s) for s in smiles_list]).astype(np.float32)

    print(f"[{now()}] Computing RDKit 2D descriptors ...", flush=True)
    desc = np.vstack([rdkit2d_desc(s) for s in smiles_list]).astype(np.float32)
    desc[~np.isfinite(desc)] = np.nan
    desc[np.abs(desc) > 1e12] = np.nan

    X = np.hstack([fps, desc]).astype(np.float32)
    X[~np.isfinite(X)] = np.nan
    X[np.abs(X) > 1e12] = np.nan

    finite = X[np.isfinite(X)]
    summary = {
        "n_molecules": len(smiles_list),
        "morgan_bits": int(fps.shape[1]),
        "rdkit2d_descriptors": int(desc.shape[1]),
        "total_features": int(X.shape[1]),
        "nan_count": int(np.isnan(X).sum()),
        "inf_count": int(np.isinf(X).sum()),
        "finite_min": float(np.min(finite)) if finite.size else np.nan,
        "finite_max": float(np.max(finite)) if finite.size else np.nan,
    }
    print(f"[{now()}] Feature dimensions: Morgan={fps.shape[1]}, RDKit2D={desc.shape[1]}, total={X.shape[1]}", flush=True)
    print(f"[{now()}] Feature check: shape={X.shape}, nan={summary['nan_count']}, inf={summary['inf_count']}", flush=True)
    return X, fps, desc, summary


def build_bitvectors(smiles_list: Iterable[str]) -> List:
    return [morgan_fp_bitvect(s) for s in smiles_list]


def tanimoto_to_train(test_fp, train_fps: List) -> np.ndarray:
    if test_fp is None:
        return np.zeros(len(train_fps), dtype=np.float32)
    valid_indices = [i for i, fp in enumerate(train_fps) if fp is not None]
    sims = np.zeros(len(train_fps), dtype=np.float32)
    if not valid_indices:
        return sims
    valid_fps = [train_fps[i] for i in valid_indices]
    valid_sims = DataStructs.BulkTanimotoSimilarity(test_fp, valid_fps)
    sims[np.array(valid_indices, dtype=int)] = np.array(valid_sims, dtype=np.float32)
    return sims


# -----------------------------
# Labels and metrics
# -----------------------------

def infer_target_log10mic(df: pd.DataFrame, target_col: Optional[str], value_col: Optional[str]) -> Tuple[pd.Series, str, str]:
    col = find_col(df, target_col, DEFAULT_TARGET_COLS, required=False)
    vcol = find_col(df, value_col, DEFAULT_VALUE_COLS, required=False)

    # Avoid accidentally using a generic binary label column as the regression target.
    # If the detected generic column is only 0/1 and an MIC value column is available,
    # compute log10MIC from the MIC value instead.
    if col is not None:
        vals = pd.to_numeric(df[col], errors="coerce")
        finite_vals = vals[np.isfinite(vals)]
        unique_vals = set(np.unique(finite_vals)) if len(finite_vals) else set()
        is_generic_name = str(col).lower() in {"target", "label", "y", "regression_target", "target_regression"}
        looks_binary = bool(unique_vals) and unique_vals.issubset({0, 1, 0.0, 1.0})
        if not (is_generic_name and looks_binary and vcol is not None):
            return vals, col, "existing_log10_column"

    if vcol is not None:
        vals = pd.to_numeric(df[vcol], errors="coerce")
        return np.log10(vals.where(vals > 0)), vcol, "computed_log10_from_mic_value"

    raise ValueError(
        f"Cannot infer regression target. Expected one of log10 target columns {DEFAULT_TARGET_COLS} "
        f"or MIC value columns {DEFAULT_VALUE_COLS}. Columns={list(df.columns)}"
    )


def infer_value_col(df: pd.DataFrame, preferred: Optional[str]) -> Optional[str]:
    return find_col(df, preferred, DEFAULT_VALUE_COLS, required=False)


def infer_active_flags(df: pd.DataFrame, active_col: Optional[str], value_col: Optional[str], threshold: Optional[float]) -> pd.Series:
    col = find_col(df, active_col, DEFAULT_ACTIVE_COLS, required=False)
    if col is not None:
        return pd.to_numeric(df[col], errors="coerce")
    if value_col is not None and threshold is not None:
        vals = pd.to_numeric(df[value_col], errors="coerce")
        return (vals <= threshold).astype(float)
    return pd.Series(np.nan, index=df.index)


def infer_boundary_flags(df: pd.DataFrame, value_col: Optional[str], y_log10: pd.Series, threshold: Optional[float]) -> pd.Series:
    for c in ["boundary_flag", "mic_boundary_flag", "mic_boundary", "MIC_boundary", "is_boundary"]:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    if threshold is not None:
        if value_col is not None:
            vals = pd.to_numeric(df[value_col], errors="coerce")
        else:
            vals = 10 ** pd.to_numeric(y_log10, errors="coerce")
        return ((vals >= threshold / 2.0) & (vals <= threshold * 2.0)).astype(float)
    return pd.Series(np.nan, index=df.index)


def mic_range_group(y_log10: pd.Series, threshold: Optional[float]) -> pd.Series:
    if threshold is None:
        return pd.Series("unknown", index=y_log10.index)
    vals = 10 ** pd.to_numeric(y_log10, errors="coerce")
    out = pd.Series("unknown", index=y_log10.index, dtype=object)
    out[vals <= threshold / 2.0] = "strong_MIC_le_T_over_2"
    out[(vals > threshold / 2.0) & (vals <= threshold * 2.0)] = "boundary_T_over_2_to_2T"
    out[vals > threshold * 2.0] = "weak_MIC_gt_2T"
    return out


def regression_metrics(y_true, y_pred, prefix: str = "") -> Dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    out = {}
    pr = prefix
    out[pr + "n"] = int(len(y))
    if len(y) == 0:
        for k in ["mae", "rmse", "mse", "medae", "r2", "pearson", "spearman", "within_1_dilution", "within_2_dilutions", "median_fold_error", "mean_fold_error", "max_abs_error_log10"]:
            out[pr + k] = np.nan
        return out
    abs_err = np.abs(p - y)
    sq_err = (p - y) ** 2
    out[pr + "mae"] = float(mean_absolute_error(y, p))
    out[pr + "mse"] = float(mean_squared_error(y, p))
    out[pr + "rmse"] = float(np.sqrt(out[pr + "mse"]))
    out[pr + "medae"] = float(median_absolute_error(y, p))
    out[pr + "r2"] = float(r2_score(y, p)) if len(y) >= 2 else np.nan
    out[pr + "pearson"] = safe_corr(y, p, method="pearson")
    out[pr + "spearman"] = safe_corr(y, p, method="spearman")
    out[pr + "within_1_dilution"] = float(np.mean(abs_err <= DILUTION1_LOG10))
    out[pr + "within_2_dilutions"] = float(np.mean(abs_err <= DILUTION2_LOG10))
    out[pr + "median_fold_error"] = float(np.median(10 ** abs_err))
    out[pr + "mean_fold_error"] = float(np.mean(10 ** abs_err))
    out[pr + "max_abs_error_log10"] = float(np.max(abs_err))
    return out


def screening_metrics_from_regression(active_true, pred_log10mic, prefix: str = "") -> Dict[str, float]:
    y = np.asarray(active_true, dtype=float)
    pred = np.asarray(pred_log10mic, dtype=float)
    mask = np.isfinite(y) & np.isfinite(pred)
    y = y[mask].astype(int)
    pred = pred[mask]
    n = len(y)
    out = {}
    pr = prefix
    out[pr + "screening_n"] = int(n)
    out[pr + "screening_pos"] = int(np.sum(y == 1)) if n else 0
    if n == 0 or len(np.unique(y)) < 2:
        for name in ["EF1%", "EF5%", "EF10%", "Hit_rate@Top1%", "Hit_rate@Top5%", "Hit_rate@Top10%", "Recall@Top50", "Recall@Top100", "Recall@Top500"]:
            out[pr + name] = np.nan
        return out

    # Lower predicted log10MIC means stronger activity, so score = -predicted log10MIC.
    score = -pred
    order = np.argsort(-score)
    base_rate = float(np.mean(y == 1))
    for pct, label in [(0.01, "1pct"), (0.05, "5pct"), (0.10, "10pct")]:
        k = max(1, int(math.ceil(n * pct)))
        top = order[:k]
        hits = int(np.sum(y[top] == 1))
        hit_rate = float(hits / k)
        ef = float(hit_rate / base_rate) if base_rate > 0 else np.nan
        out[pr + f"top_{label}_n"] = int(k)
        out[pr + f"top_{label}_hits"] = int(hits)
        out[pr + f"hit_rate_top_{label}"] = hit_rate
        out[pr + f"ef_{label}"] = ef
    out[pr + "EF1%"] = out[pr + "ef_1pct"]
    out[pr + "EF5%"] = out[pr + "ef_5pct"]
    out[pr + "EF10%"] = out[pr + "ef_10pct"]
    out[pr + "Hit_rate@Top1%"] = out[pr + "hit_rate_top_1pct"]
    out[pr + "Hit_rate@Top5%"] = out[pr + "hit_rate_top_5pct"]
    out[pr + "Hit_rate@Top10%"] = out[pr + "hit_rate_top_10pct"]
    total_pos = max(int(np.sum(y == 1)), 1)
    for k in [50, 100, 500]:
        kk = min(k, n)
        hits = int(np.sum(y[order[:kk]] == 1)) if kk > 0 else 0
        out[pr + f"Recall@Top{k}"] = float(hits / total_pos) if n else np.nan
    return out


# -----------------------------
# Models and preprocessing
# -----------------------------

def build_model(model_name: str, random_state: int) -> object:
    model_name = model_name.lower()
    if model_name == "rf":
        return RandomForestRegressor(
            n_estimators=500,
            max_features="sqrt",
            n_jobs=-1,
            random_state=random_state,
        )
    if model_name == "svm":
        return SVR(C=1.0, kernel="rbf", gamma="scale")
    if model_name == "xgb":
        try:
            from xgboost import XGBRegressor
        except Exception as e:
            raise ImportError("xgboost is required for --model xgb") from e
        return XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            eval_metric=["rmse", "mae"],
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except Exception as e:
            raise ImportError("lightgbm is required for --model lightgbm") from e
        return LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="regression",
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def fit_preprocess(X_train, X_val, X_test, X_all_for_train_pred, model_name: str):
    imputer = SimpleImputer(strategy="median")
    X_train_i = imputer.fit_transform(X_train)
    X_val_i = imputer.transform(X_val)
    X_test_i = imputer.transform(X_test)
    X_all_i = imputer.transform(X_all_for_train_pred)

    scaler = None
    if model_name.lower() == "svm":
        scaler = StandardScaler()
        X_train_i = scaler.fit_transform(X_train_i)
        X_val_i = scaler.transform(X_val_i)
        X_test_i = scaler.transform(X_test_i)
        X_all_i = scaler.transform(X_all_i)
    return imputer, scaler, X_train_i, X_val_i, X_test_i, X_all_i


def get_predictions(model, X) -> np.ndarray:
    return np.asarray(model.predict(X), dtype=float)


def save_training_log_regression(model_name: str, model, X_train, y_train, X_val, y_val, outdir: Path):
    """Save train/validation regression loss curves.

    Always writes train_log.csv, training_curve.csv and train_log_long.csv.
    For XGB/LightGBM: one row per boosting iteration when eval history is available.
    For RF/SVR: one final row with train/validation MSE as train_loss/val_loss.
    """
    mname = model_name.lower()
    long_rows = []
    wide_by_iter = {}

    def split_alias(raw_split: str, split_index: int = None) -> str:
        s = str(raw_split).lower()
        if s in {"train", "training", "train_0"}:
            return "train"
        if s in {"valid", "validation", "val", "valid_0", "valid_1"}:
            return "validation"
        if s == "validation_0":
            return "train"
        if s == "validation_1":
            return "validation"
        if split_index == 0:
            return "train"
        if split_index == 1:
            return "validation"
        if "train" in s:
            return "train"
        if "valid" in s or "val" in s:
            return "validation"
        return str(raw_split)

    def metric_alias(raw_metric: str) -> str:
        s = str(raw_metric).lower()
        if s in {"l2", "mse", "mean_squared_error", "regression", "loss"}:
            return "mse"
        if s in {"rmse", "root_mean_squared_error"}:
            return "rmse"
        if s in {"l1", "mae", "mean_absolute_error"}:
            return "mae"
        return s

    def add_metric(iteration: int, raw_split: str, raw_metric: str, value, split_index: int = None):
        try:
            value = float(value)
        except Exception:
            value = np.nan
        split = split_alias(raw_split, split_index)
        metric = metric_alias(raw_metric)
        long_rows.append({
            "iteration": int(iteration),
            "raw_split": str(raw_split),
            "split": split,
            "raw_metric": str(raw_metric),
            "metric": metric,
            "value": value,
        })
        row = wide_by_iter.setdefault(int(iteration), {"iteration": int(iteration)})
        prefix = "train" if split == "train" else "val" if split == "validation" else split
        if prefix in {"train", "val"}:
            if metric == "mse":
                row[f"{prefix}_loss"] = value
                row[f"{prefix}_mse"] = value
                row.setdefault(f"{prefix}_rmse", float(np.sqrt(value)) if value >= 0 else np.nan)
            elif metric == "rmse":
                row[f"{prefix}_rmse"] = value
                row.setdefault(f"{prefix}_loss", value ** 2 if np.isfinite(value) else np.nan)
                row.setdefault(f"{prefix}_mse", value ** 2 if np.isfinite(value) else np.nan)
            elif metric == "mae":
                row[f"{prefix}_mae"] = value

    if mname == "xgb" and hasattr(model, "evals_result"):
        try:
            res = model.evals_result()
            for split_index, (split_key, metrics) in enumerate(res.items()):
                for metric_name, vals in metrics.items():
                    for i, v in enumerate(vals):
                        add_metric(i + 1, split_key, metric_name, v, split_index=split_index)
        except Exception as e:
            long_rows.append({"iteration": 1, "raw_split": "NA", "split": "NA", "raw_metric": "evals_result_error", "metric": "error", "value": np.nan, "note": type(e).__name__})

    elif mname == "lightgbm" and hasattr(model, "evals_result_"):
        try:
            res = model.evals_result_
            for split_index, (split_key, metrics) in enumerate(res.items()):
                for metric_name, vals in metrics.items():
                    for i, v in enumerate(vals):
                        add_metric(i + 1, split_key, metric_name, v, split_index=split_index)
        except Exception as e:
            long_rows.append({"iteration": 1, "raw_split": "NA", "split": "NA", "raw_metric": "evals_result_error", "metric": "error", "value": np.nan, "note": type(e).__name__})

    if not wide_by_iter:
        train_pred = get_predictions(model, X_train)
        val_pred = get_predictions(model, X_val)
        train_m = regression_metrics(y_train, train_pred, prefix="")
        val_m = regression_metrics(y_val, val_pred, prefix="")
        wide_by_iter[1] = {
            "iteration": 1,
            "train_loss": train_m.get("mse", np.nan),
            "val_loss": val_m.get("mse", np.nan),
            "train_mse": train_m.get("mse", np.nan),
            "val_mse": val_m.get("mse", np.nan),
            "train_rmse": train_m.get("rmse", np.nan),
            "val_rmse": val_m.get("rmse", np.nan),
            "train_mae": train_m.get("mae", np.nan),
            "val_mae": val_m.get("mae", np.nan),
            "train_spearman": train_m.get("spearman", np.nan),
            "val_spearman": val_m.get("spearman", np.nan),
            "train_within_1_dilution": train_m.get("within_1_dilution", np.nan),
            "val_within_1_dilution": val_m.get("within_1_dilution", np.nan),
            "train_within_2_dilutions": train_m.get("within_2_dilutions", np.nan),
            "val_within_2_dilutions": val_m.get("within_2_dilutions", np.nan),
            "note": "final_only_no_epoch_loss_for_this_model",
        }
        for split_name, metrics in [("train", train_m), ("validation", val_m)]:
            for metric_name, value in metrics.items():
                try:
                    value = float(value)
                except Exception:
                    value = np.nan
                long_rows.append({
                    "iteration": 1,
                    "raw_split": split_name,
                    "split": split_name,
                    "raw_metric": metric_name,
                    "metric": metric_alias(metric_name),
                    "value": value,
                    "note": "final_only_no_epoch_loss_for_this_model",
                })

    curve = pd.DataFrame([wide_by_iter[k] for k in sorted(wide_by_iter)])
    preferred_cols = [
        "iteration",
        "train_loss", "val_loss",
        "train_mse", "val_mse",
        "train_rmse", "val_rmse",
        "train_mae", "val_mae",
        "train_spearman", "val_spearman",
        "train_within_1_dilution", "val_within_1_dilution",
        "train_within_2_dilutions", "val_within_2_dilutions",
        "note",
    ]
    ordered_cols = [c for c in preferred_cols if c in curve.columns] + [c for c in curve.columns if c not in preferred_cols]
    curve = curve[ordered_cols]
    curve.to_csv(outdir / "train_log.csv", index=False)
    curve.to_csv(outdir / "training_curve.csv", index=False)
    pd.DataFrame(long_rows).to_csv(outdir / "train_log_long.csv", index=False)


# -----------------------------
# Diagnostics
# -----------------------------

def nn_similarity_and_activity_cliff_regression(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    train_pred: np.ndarray,
    eval_pred: np.ndarray,
    smiles_col: str,
    y_train_log10: pd.Series,
    y_eval_log10: pd.Series,
    k_local: int = 10,
) -> pd.DataFrame:
    train_smiles = train_df[smiles_col].astype(str).tolist()
    eval_smiles = eval_df[smiles_col].astype(str).tolist()
    train_fps = build_bitvectors(train_smiles)
    eval_fps = build_bitvectors(eval_smiles)
    train_y = pd.to_numeric(y_train_log10, errors="coerce").values.astype(float)
    eval_y = pd.to_numeric(y_eval_log10, errors="coerce").values.astype(float)
    train_pred = np.asarray(train_pred, dtype=float)
    eval_pred = np.asarray(eval_pred, dtype=float)

    rows = []
    for i, fp in enumerate(eval_fps):
        sims = tanimoto_to_train(fp, train_fps)
        if len(sims):
            nn_idx = int(np.argmax(sims))
            nn_sim = float(sims[nn_idx])
        else:
            nn_idx = -1
            nn_sim = np.nan

        local_var = np.nan
        if len(sims):
            order = np.argsort(-sims)
            top_idx = order[:min(k_local, len(order))]
            vals = train_y[top_idx]
            vals = vals[np.isfinite(vals)]
            if len(vals) >= 2:
                local_var = float(np.var(vals, ddof=1))

        ac_mask = np.zeros(len(train_df), dtype=bool)
        ac_max_abs_diff = np.nan
        ac_pair_directional_correct = []
        if np.isfinite(eval_y[i]) and len(sims):
            diff = np.abs(train_y - eval_y[i])
            ac_mask = (sims >= 0.7) & np.isfinite(diff) & (diff >= AC_LOG10_DIFF_CUTOFF)
            if np.any(ac_mask):
                ac_max_abs_diff = float(np.nanmax(diff[ac_mask]))
                ac_indices = np.where(ac_mask)[0]
                for j in ac_indices:
                    # Lower log10MIC means stronger activity.
                    true_test_stronger = eval_y[i] < train_y[j]
                    pred_test_stronger = eval_pred[i] < train_pred[j]
                    ac_pair_directional_correct.append(int(true_test_stronger == pred_test_stronger))

        rows.append({
            "nn_train_row": nn_idx,
            "nn_sim": nn_sim,
            "nn_bin": ">=0.7" if nn_sim >= 0.7 else ("0.4-0.7" if nn_sim >= 0.4 else "<0.4"),
            "nn_train_smiles": train_smiles[nn_idx] if nn_idx >= 0 else None,
            "nn_train_log10mic": float(train_y[nn_idx]) if nn_idx >= 0 and np.isfinite(train_y[nn_idx]) else np.nan,
            "nn_train_pred_log10mic": float(train_pred[nn_idx]) if nn_idx >= 0 and np.isfinite(train_pred[nn_idx]) else np.nan,
            "local_train_log10mic_variance_k10": local_var,
            "ac_flag": int(np.any(ac_mask)),
            "ac_neighbor_count": int(np.sum(ac_mask)),
            "ac_max_abs_log10mic_diff": ac_max_abs_diff,
            "ac_pair_directional_accuracy_for_molecule": float(np.mean(ac_pair_directional_correct)) if ac_pair_directional_correct else np.nan,
            "ac_pair_count_for_molecule": int(len(ac_pair_directional_correct)),
        })
    return pd.DataFrame(rows)


def subgroup_regression_metrics(df: pd.DataFrame, y_col="true_log10MIC", pred_col="pred_log10MIC", active_col="true_active_for_screening") -> Dict[str, float]:
    out = regression_metrics(df[y_col].values, df[pred_col].values, prefix="")
    if active_col in df.columns:
        out.update(screening_metrics_from_regression(df[active_col].values, df[pred_col].values, prefix=""))
    return out


def diagnostic_summary(pred_df: pd.DataFrame) -> Dict[str, float]:
    out = {}
    df = pred_df.copy()
    df["abs_error_log10"] = np.abs(df["pred_log10MIC"].astype(float) - df["true_log10MIC"].astype(float))
    df["sq_error_log10"] = df["abs_error_log10"] ** 2

    # 12.1 MIC-boundary regression error analysis
    if "boundary_flag" in df.columns and df["boundary_flag"].notna().any():
        b = df[df["boundary_flag"] == 1]
        nb = df[df["boundary_flag"] == 0]
        out["boundary_n"] = int(len(b))
        out["non_boundary_n"] = int(len(nb))
        out["boundary_mae"] = float(b["abs_error_log10"].mean()) if len(b) else np.nan
        out["non_boundary_mae"] = float(nb["abs_error_log10"].mean()) if len(nb) else np.nan
        out["boundary_rmse"] = float(np.sqrt(b["sq_error_log10"].mean())) if len(b) else np.nan
        out["non_boundary_rmse"] = float(np.sqrt(nb["sq_error_log10"].mean())) if len(nb) else np.nan
        out["boundary_regression_error_enrichment"] = out["boundary_mae"] / out["non_boundary_mae"] if out["non_boundary_mae"] and out["non_boundary_mae"] > 0 else np.nan
        out["boundary_within_1_dilution"] = float(np.mean(b["abs_error_log10"] <= DILUTION1_LOG10)) if len(b) else np.nan
        out["non_boundary_within_1_dilution"] = float(np.mean(nb["abs_error_log10"] <= DILUTION1_LOG10)) if len(nb) else np.nan

    # 12.2 NN subgroup analysis
    if "nn_bin" in df.columns:
        for bin_name in [">=0.7", "0.4-0.7", "<0.4"]:
            g = df[df["nn_bin"] == bin_name]
            key = bin_name.replace(">=", "ge_").replace("<", "lt_").replace("-", "_").replace(".", "p")
            prefix = f"nn_{key}_"
            m = subgroup_regression_metrics(g)
            for k, v in m.items():
                out[prefix + k] = v

    # 12.3 Activity cliff subgroup analysis
    if "ac_flag" in df.columns:
        ac = df[df["ac_flag"] == 1]
        non_ac = df[df["ac_flag"] == 0]
        out["ac_n"] = int(len(ac))
        out["non_ac_n"] = int(len(non_ac))
        out["ac_mae"] = float(ac["abs_error_log10"].mean()) if len(ac) else np.nan
        out["non_ac_mae"] = float(non_ac["abs_error_log10"].mean()) if len(non_ac) else np.nan
        out["ac_rmse"] = float(np.sqrt(ac["sq_error_log10"].mean())) if len(ac) else np.nan
        out["non_ac_rmse"] = float(np.sqrt(non_ac["sq_error_log10"].mean())) if len(non_ac) else np.nan
        out["ac_error_enrichment"] = out["ac_mae"] / out["non_ac_mae"] if out["non_ac_mae"] and out["non_ac_mae"] > 0 else np.nan
        if "ac_pair_directional_accuracy_for_molecule" in df.columns:
            out["ac_pair_directional_accuracy"] = float(df["ac_pair_directional_accuracy_for_molecule"].dropna().mean()) if df["ac_pair_directional_accuracy_for_molecule"].notna().any() else np.nan
            out["ac_pair_total_count"] = int(df.get("ac_pair_count_for_molecule", pd.Series(dtype=float)).fillna(0).sum())

    # 12.5 local landscape roughness proxy
    if "local_train_log10mic_variance_k10" in df.columns and df["local_train_log10mic_variance_k10"].notna().any():
        out["mean_local_train_log10mic_variance_k10"] = float(df["local_train_log10mic_variance_k10"].mean())
        med = float(df["local_train_log10mic_variance_k10"].median())
        high = df[df["local_train_log10mic_variance_k10"] > med]
        low = df[df["local_train_log10mic_variance_k10"] <= med]
        out["high_local_variance_n"] = int(len(high))
        out["low_local_variance_n"] = int(len(low))
        out["high_local_variance_mae"] = float(high["abs_error_log10"].mean()) if len(high) else np.nan
        out["low_local_variance_mae"] = float(low["abs_error_log10"].mean()) if len(low) else np.nan
        out["local_variance_error_enrichment"] = out["high_local_variance_mae"] / out["low_local_variance_mae"] if out["low_local_variance_mae"] and out["low_local_variance_mae"] > 0 else np.nan

    # 12.6 MIC range-specific error analysis
    if "mic_range_group" in df.columns:
        for group in ["strong_MIC_le_T_over_2", "boundary_T_over_2_to_2T", "weak_MIC_gt_2T"]:
            g = df[df["mic_range_group"] == group]
            prefix = f"range_{group}_"
            m = subgroup_regression_metrics(g)
            for k, v in m.items():
                out[prefix + k] = v

    return out


# -----------------------------
# Run discovery and aggregation
# -----------------------------

def discover_runs(data_root: Path, split_types: List[str], seeds: List[int], datasets: Optional[List[str]] = None) -> List[Dict]:
    dataset_dirs = [p for p in data_root.iterdir() if p.is_dir() and (p / "regression").exists()]
    dataset_dirs = sorted(dataset_dirs, key=numeric_sort_key)
    if datasets:
        keep = set(datasets)
        dataset_dirs = [p for p in dataset_dirs if p.name in keep or parse_genus_from_dataset_name(p.name) in keep]

    runs = []
    for ds in dataset_dirs:
        for split_type in split_types:
            split_dir = ds / "regression" / split_type
            if not split_dir.exists():
                print(f"[{now()}] Skip missing regression split dir: {split_dir.name}", flush=True)
                continue
            for seed in seeds:
                seed_dir = split_dir / f"seed_{seed}"
                train_csv = seed_dir / "train.csv"
                val_csv = seed_dir / "validation.csv"
                test_csv = seed_dir / "test.csv"
                if train_csv.exists() and val_csv.exists() and test_csv.exists():
                    runs.append({
                        "dataset_dir": ds,
                        "dataset_name": ds.name,
                        "genus": parse_genus_from_dataset_name(ds.name),
                        "threshold": parse_threshold_from_dataset_name(ds.name),
                        "split_type": split_type,
                        "seed": seed,
                        "seed_dir": seed_dir,
                        "train_csv": train_csv,
                        "val_csv": val_csv,
                        "test_csv": test_csv,
                    })
                else:
                    print(f"[{now()}] Skip incomplete regression seed dir: {seed_dir.name}", flush=True)
    return runs


def aggregate_results(out_root: Path, model_name: str) -> None:
    model_root = out_root / model_name
    metric_files = list(model_root.glob("**/metrics_full.csv"))
    if not metric_files:
        return
    dfs = []
    for f in metric_files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception:
            pass
    if not dfs:
        return
    all_df = pd.concat(dfs, ignore_index=True)
    all_df.to_csv(model_root / "all_run_metrics.csv", index=False)

    group_cols = ["model", "dataset_name", "genus", "split_type"]
    metric_cols = [
        c for c in all_df.columns
        if (
            c.startswith("test_")
            or c.startswith("val_")
            or c.startswith("boundary_")
            or c.startswith("non_boundary_")
            or c.startswith("nn_")
            or c.startswith("ac_")
            or c.startswith("range_")
            or c.startswith("mean_local_")
            or c.startswith("high_local_")
            or c.startswith("low_local_")
            or c.startswith("local_variance_")
        )
    ]
    numeric_cols = [c for c in metric_cols if pd.api.types.is_numeric_dtype(all_df[c])]
    agg = all_df.groupby(group_cols)[numeric_cols].agg(["mean", "std", "count"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(model_root / "aggregate_mean_sd.csv", index=False)

    mean_df = all_df.groupby(["model", "dataset_name", "genus", "split_type"])[numeric_cols].mean().reset_index()
    rows = []
    lower_better = ["test_mae", "test_rmse", "test_mse", "test_medae", "test_median_fold_error", "test_mean_fold_error"]
    higher_better = ["test_spearman", "test_pearson", "test_r2", "test_within_1_dilution", "test_within_2_dilutions", "test_EF1%", "test_EF5%", "test_EF10%", "test_Hit_rate@Top1%", "test_Hit_rate@Top5%"]
    for (model, dataset_name, genus), sub in mean_df.groupby(["model", "dataset_name", "genus"]):
        random_row = sub[sub["split_type"] == "random"]
        if random_row.empty:
            continue
        random_row = random_row.iloc[0]
        for target_split in ["scaffold", "repeated_stratified_perimeter", "strict_perimeter", "perimeter", "max_distance"]:
            target_row = sub[sub["split_type"] == target_split]
            if target_row.empty:
                continue
            target_row = target_row.iloc[0]
            row = {"model": model, "dataset_name": dataset_name, "genus": genus, "reference_split": "random", "target_split": target_split}
            for m in lower_better:
                if m in random_row.index and m in target_row.index:
                    row[f"ood_increase_{m}"] = target_row[m] - random_row[m]
            for m in higher_better:
                if m in random_row.index and m in target_row.index:
                    row[f"ood_drop_{m}"] = random_row[m] - target_row[m]
            rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(model_root / "ood_change_random_vs_ood_splits.csv", index=False)


# -----------------------------
# Main single run
# -----------------------------

def run_one(run: Dict, args) -> Optional[Dict]:
    model_name = args.model.lower()
    dataset_name = run["dataset_name"]
    genus = run["genus"]
    split_type = run["split_type"]
    seed = run["seed"]
    threshold_from_name = run["threshold"]

    outdir = Path(args.outdir) / model_name / dataset_name / "regression" / split_type / f"seed_{seed}"
    ensure_dir(outdir)

    print("\n" + "=" * 100, flush=True)
    print(f"[{now()}] RUN: REGRESSION | model={model_name} | dataset={dataset_name} | split={split_type} | seed={seed}", flush=True)
    print("=" * 100, flush=True)

    train_df = pd.read_csv(run["train_csv"])
    val_df = pd.read_csv(run["val_csv"])
    test_df = pd.read_csv(run["test_csv"])

    smiles_col = require_smiles_column(train_df, val_df, test_df)
    value_col = find_col(train_df, args.value_col, DEFAULT_VALUE_COLS, required=False)

    y_train_s, target_source_col, target_source_type = infer_target_log10mic(train_df, args.target_col, args.value_col)
    y_val_s, val_target_source_col, _ = infer_target_log10mic(val_df, target_source_col if target_source_type == "existing_log10_column" else None, args.value_col)
    y_test_s, test_target_source_col, _ = infer_target_log10mic(test_df, target_source_col if target_source_type == "existing_log10_column" else None, args.value_col)

    # Drop rows with invalid target before training/evaluation.
    def valid_target_filter(df, y):
        mask = pd.to_numeric(y, errors="coerce").notna() & np.isfinite(pd.to_numeric(y, errors="coerce"))
        return df.loc[mask].reset_index(drop=True), pd.to_numeric(y.loc[mask], errors="coerce").reset_index(drop=True)

    train_df, y_train_s = valid_target_filter(train_df, y_train_s)
    val_df, y_val_s = valid_target_filter(val_df, y_val_s)
    test_df, y_test_s = valid_target_filter(test_df, y_test_s)

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise RuntimeError(f"Empty train/validation/test after filtering invalid regression targets: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    train_df.to_csv(outdir / "train_input.csv", index=False)
    val_df.to_csv(outdir / "validation_input.csv", index=False)
    test_df.to_csv(outdir / "test_input.csv", index=False)

    y_train = y_train_s.values.astype(float)
    y_val = y_val_s.values.astype(float)
    y_test = y_test_s.values.astype(float)

    print(f"[{now()}] Columns: smiles_col={smiles_col}, target_source_col={target_source_col}, target_source_type={target_source_type}, value_col={value_col}", flush=True)
    for name, y in [("train", y_train), ("validation", y_val), ("test", y_test)]:
        print(f"[{now()}] {name}: n={len(y)}, log10MIC mean={float(np.mean(y)):.4f}, sd={float(np.std(y)):.4f}, min={float(np.min(y)):.4f}, max={float(np.max(y)):.4f}", flush=True)

    all_smiles = pd.concat([train_df[smiles_col], val_df[smiles_col], test_df[smiles_col]], ignore_index=True).astype(str).tolist()
    X_all, _, _, feature_summary = build_features(all_smiles)
    with open(outdir / "feature_summary.json", "w", encoding="utf-8") as f:
        json.dump(feature_summary, f, indent=2, ensure_ascii=False, default=str)

    n_tr, n_val = len(train_df), len(val_df)
    X_train_raw = X_all[:n_tr]
    X_val_raw = X_all[n_tr:n_tr + n_val]
    X_test_raw = X_all[n_tr + n_val:]

    imputer, scaler, X_train, X_val, X_test, X_train_for_pred = fit_preprocess(
        X_train_raw, X_val_raw, X_test_raw, X_train_raw, model_name
    )

    random_state = seed if args.use_split_seed_as_model_seed else args.random_state
    model = build_model(model_name, random_state=random_state)
    model_params = model.get_params() if hasattr(model, "get_params") else {}
    with open(outdir / "model_params.json", "w", encoding="utf-8") as f:
        json.dump(model_params, f, indent=2, ensure_ascii=False, default=str)

    print(f"[{now()}] Training {model_name} regressor with fixed hyperparameters ...", flush=True)
    t0 = time.time()
    if model_name == "xgb":
        model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
    elif model_name == "lightgbm":
        try:
            model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], eval_names=["train", "validation"], eval_metric=["l2", "l1"])
        except TypeError:
            model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], eval_metric=["l2", "l1"])
    else:
        model.fit(X_train, y_train)
    train_seconds = time.time() - t0
    print(f"[{now()}] Finished training. seconds={train_seconds:.2f}", flush=True)

    train_pred = get_predictions(model, X_train)
    val_pred = get_predictions(model, X_val)
    test_pred = get_predictions(model, X_test)

    save_training_log_regression(model_name, model, X_train, y_train, X_val, y_val, outdir)

    threshold_for_boundary = args.threshold if args.threshold is not None else threshold_from_name
    train_value_col = infer_value_col(train_df, args.value_col)
    val_value_col = infer_value_col(val_df, args.value_col)
    test_value_col = infer_value_col(test_df, args.value_col)

    train_boundary = infer_boundary_flags(train_df, train_value_col, y_train_s, threshold_for_boundary)
    val_boundary = infer_boundary_flags(val_df, val_value_col, y_val_s, threshold_for_boundary)
    test_boundary = infer_boundary_flags(test_df, test_value_col, y_test_s, threshold_for_boundary)

    train_active = infer_active_flags(train_df, args.active_col, train_value_col, threshold_for_boundary)
    val_active = infer_active_flags(val_df, args.active_col, val_value_col, threshold_for_boundary)
    test_active = infer_active_flags(test_df, args.active_col, test_value_col, threshold_for_boundary)

    print(f"[{now()}] Computing nearest-neighbor, activity-cliff and local roughness diagnostics for validation/test ...", flush=True)
    val_diag = nn_similarity_and_activity_cliff_regression(train_df, val_df, train_pred, val_pred, smiles_col, y_train_s, y_val_s, k_local=args.local_k)
    test_diag = nn_similarity_and_activity_cliff_regression(train_df, test_df, train_pred, test_pred, smiles_col, y_train_s, y_test_s, k_local=args.local_k)

    def make_pred_df(df, y_true, y_pred, split_name, boundary, active, diag=None):
        pred = df.copy()
        pred.insert(0, "split", split_name)
        pred["dataset_name"] = dataset_name
        pred["genus"] = genus
        pred["task"] = "regression"
        pred["split_type"] = split_type
        pred["seed"] = seed
        pred["model"] = model_name
        pred["true_log10MIC"] = np.asarray(y_true, dtype=float)
        pred["pred_log10MIC"] = np.asarray(y_pred, dtype=float)
        pred["true_MIC"] = 10 ** pred["true_log10MIC"]
        pred["pred_MIC"] = 10 ** pred["pred_log10MIC"]
        pred["score_reg"] = -pred["pred_log10MIC"]
        pred["abs_error_log10"] = np.abs(pred["pred_log10MIC"] - pred["true_log10MIC"])
        pred["sq_error_log10"] = pred["abs_error_log10"] ** 2
        pred["fold_error"] = 10 ** pred["abs_error_log10"]
        pred["within_1_dilution"] = (pred["abs_error_log10"] <= DILUTION1_LOG10).astype(int)
        pred["within_2_dilutions"] = (pred["abs_error_log10"] <= DILUTION2_LOG10).astype(int)
        pred["boundary_flag"] = boundary.values
        pred["true_active_for_screening"] = active.values
        pred["mic_range_group"] = mic_range_group(pd.Series(y_true), threshold_for_boundary).values
        pred["threshold_for_boundary"] = threshold_for_boundary
        pred["regression_target_col"] = target_source_col
        pred["target_source_type"] = target_source_type
        if diag is not None:
            pred = pd.concat([pred.reset_index(drop=True), diag.reset_index(drop=True)], axis=1)
        return pred

    train_pred_df = make_pred_df(train_df, y_train, train_pred, "train", train_boundary, train_active, diag=None)
    val_pred_df = make_pred_df(val_df, y_val, val_pred, "validation", val_boundary, val_active, diag=val_diag)
    test_pred_df = make_pred_df(test_df, y_test, test_pred, "test", test_boundary, test_active, diag=test_diag)

    train_pred_df.to_csv(outdir / "predictions_train.csv", index=False)
    val_pred_df.to_csv(outdir / "predictions_validation.csv", index=False)
    test_pred_df.to_csv(outdir / "predictions_test.csv", index=False)

    row = {
        "model": model_name,
        "dataset_name": dataset_name,
        "genus": genus,
        "task": "regression",
        "split_type": split_type,
        "seed": seed,
        "threshold_from_dataset_name": threshold_from_name,
        "threshold_for_boundary": threshold_for_boundary,
        "train_seconds": train_seconds,
        "feature_type": "Morgan_r2_2048_plus_RDKit2D",
        "target_scale": "log10MIC",
        "regression_target_col": target_source_col,
        "target_source_type": target_source_type,
        "random_state": random_state,
    }
    for split_name, y, pred, active in [
        ("train", y_train, train_pred, train_active),
        ("val", y_val, val_pred, val_active),
        ("test", y_test, test_pred, test_active),
    ]:
        row.update(regression_metrics(y, pred, prefix=f"{split_name}_"))
        row.update(screening_metrics_from_regression(active.values, pred, prefix=f"{split_name}_"))

    diag_summary = diagnostic_summary(test_pred_df)
    row.update(diag_summary)

    metrics_df = pd.DataFrame([row])
    metrics_df.to_csv(outdir / "metrics_full.csv", index=False)
    pd.DataFrame([diag_summary]).to_csv(outdir / "diagnostics_metrics.csv", index=False)

    subgroup_rows = []
    for group_col, group_type in [
        ("boundary_flag", "MIC_boundary"),
        ("nn_bin", "NN_similarity"),
        ("ac_flag", "Activity_cliff"),
        ("mic_range_group", "MIC_range"),
    ]:
        if group_col in test_pred_df.columns:
            for group_value, g in test_pred_df.groupby(group_col, dropna=False):
                r = {"group_type": group_type, "group_col": group_col, "group_value": group_value, "n": int(len(g))}
                r.update(subgroup_regression_metrics(g))
                subgroup_rows.append(r)
    if "local_train_log10mic_variance_k10" in test_pred_df.columns and test_pred_df["local_train_log10mic_variance_k10"].notna().any():
        med = test_pred_df["local_train_log10mic_variance_k10"].median()
        for label, g in [("low_or_equal_median", test_pred_df[test_pred_df["local_train_log10mic_variance_k10"] <= med]), ("high_median", test_pred_df[test_pred_df["local_train_log10mic_variance_k10"] > med])]:
            r = {"group_type": "local_log10MIC_variance", "group_col": "local_train_log10mic_variance_k10", "group_value": label, "n": int(len(g))}
            r.update(subgroup_regression_metrics(g))
            subgroup_rows.append(r)
    if subgroup_rows:
        pd.DataFrame(subgroup_rows).to_csv(outdir / "diagnostic_subgroup_metrics.csv", index=False)

    run_info = {
        "script": Path(__file__).name,
        "finish_time": now(),
        "data_root": str(args.data_root),
        "dataset_name": dataset_name,
        "genus": genus,
        "regression_seed_dir": str(run["seed_dir"]),
        "train_csv": str(run["train_csv"]),
        "validation_csv": str(run["val_csv"]),
        "test_csv": str(run["test_csv"]),
        "model": model_name,
        "feature_type": "Morgan fingerprint radius=2 nBits=2048 + RDKit 2D descriptors",
        "smiles_col": smiles_col,
        "regression_target_col": target_source_col,
        "target_source_type": target_source_type,
        "target_scale": "log10MIC",
        "value_col_train": train_value_col,
        "threshold_for_boundary": threshold_for_boundary,
        "within_1_dilution_cutoff_log10": DILUTION1_LOG10,
        "within_2_dilutions_cutoff_log10": DILUTION2_LOG10,
        "activity_cliff_sim_cutoff": 0.7,
        "activity_cliff_log10diff_cutoff": AC_LOG10_DIFF_CUTOFF,
        "random_state": random_state,
        "train_seconds": train_seconds,
        "python": sys.version,
    }
    run_info = sanitize_metadata(run_info)
    with open(outdir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False, default=str)

    joblib.dump({"imputer": imputer, "scaler": scaler, "model": model, "run_info": run_info}, outdir / "model.joblib")

    print(
        f"[{now()}] TEST regression metrics: RMSE={row.get('test_rmse'):.4f}, MAE={row.get('test_mae'):.4f}, "
        f"Spearman={row.get('test_spearman'):.4f}, within1={row.get('test_within_1_dilution'):.4f}, "
        f"within2={row.get('test_within_2_dilutions'):.4f}",
        flush=True,
    )
    print(f"[{now()}] Saved outputs to: {outdir.name}", flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate/diagnose RF, SVR, XGBRegressor and LightGBMRegressor for AntibioticBench log10MIC regression splits.")
    parser.add_argument("--data-root", required=True, help="Root directory containing dataset folders with regression splits")
    parser.add_argument("--outdir", default=str(Path(__file__).resolve().parents[2] / "outputs" / "ML" / "regression"), help="Output root directory")
    parser.add_argument("--model", required=True, choices=["xgb", "lightgbm", "rf", "svm"], help="Regressor to train. svm means SVR.")
    parser.add_argument("--split-types", nargs="+", default=["random", "scaffold", "repeated_stratified_perimeter", "strict_perimeter"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--datasets", nargs="+", default=None, help="Optional dataset folder names or genus names to include")
    parser.add_argument("--smiles-col", default=None, help="Deprecated/ignored. Molecular input column is hard-coded as 'smiles'.")
    parser.add_argument("--target-col", default=None, help="Optional log10MIC target column name. Auto-detected if omitted.")
    parser.add_argument("--value-col", default=None, help="Optional MIC value column. Used to compute log10MIC if no log10 target column exists.")
    parser.add_argument("--active-col", default=None, help="Optional active label column for EF/Hit/Recall. If absent, inferred from MIC <= threshold when possible.")
    parser.add_argument("--threshold", type=float, default=None, help="Override MIC threshold for boundary/screening active inference")
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--use-split-seed-as-model-seed", action="store_true")
    parser.add_argument("--local-k", type=int, default=10, help="k nearest train molecules used for local log10MIC variance diagnostic")
    parser.add_argument("--max-runs", type=int, default=None, help="Debug option: maximum number of runs")
    args = parser.parse_args()

    args.data_root = Path(args.data_root).resolve()
    args.outdir = Path(args.outdir).resolve()
    ensure_dir(args.outdir)

    runs = discover_runs(args.data_root, args.split_types, args.seeds, args.datasets)
    if args.max_runs is not None:
        runs = runs[:args.max_runs]

    print(f"[{now()}] Discovered {len(runs)} regression runs.", flush=True)
    if not runs:
        raise RuntimeError("No regression runs found. Please check --data-root and DATA_ROOT/dataset/regression/split/seed layout.")

    all_rows = []
    failed = []
    for run in runs:
        try:
            row = run_one(run, args)
            if row is not None:
                all_rows.append(row)
        except Exception as e:
            print(f"[{now()}] ERROR in run: {type(e).__name__}", flush=True)
            failed.append(sanitize_metadata({**{k: str(v) for k, v in run.items()}, "error": type(e).__name__}))
            fail_dir = Path(args.outdir) / args.model / "_failed_runs"
            ensure_dir(fail_dir)
            pd.DataFrame(failed).to_csv(fail_dir / "failed_runs.csv", index=False)
            if args.max_runs:
                raise SystemExit(1) from None
            continue

    if all_rows:
        model_root = Path(args.outdir) / args.model
        ensure_dir(model_root)
        pd.DataFrame(all_rows).to_csv(model_root / "all_run_metrics_from_current_invocation.csv", index=False)
        aggregate_results(Path(args.outdir), args.model)

    if failed:
        fail_dir = Path(args.outdir) / args.model / "_failed_runs"
        ensure_dir(fail_dir)
        pd.DataFrame(failed).to_csv(fail_dir / "failed_runs.csv", index=False)
        print(f"[{now()}] Finished with {len(failed)} failed runs. See failed_runs.csv", flush=True)
    else:
        print(f"[{now()}] Finished all regression runs successfully.", flush=True)


if __name__ == "__main__":
    main()
