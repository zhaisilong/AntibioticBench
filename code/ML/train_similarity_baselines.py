#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AntibioticBench similarity baselines: 1-NN and 5-NN using Morgan fingerprints
and Tanimoto similarity.

Classification expected layout:
    DATA_ROOT/dataset_name/classification/split_type/seed_0/train.csv
    DATA_ROOT/dataset_name/classification/split_type/seed_0/validation.csv
    DATA_ROOT/dataset_name/classification/split_type/seed_0/test.csv

Regression expected layout:
    DATA_ROOT/dataset_name/regression/split_type/seed_0/train.csv
    DATA_ROOT/dataset_name/regression/split_type/seed_0/validation.csv
    DATA_ROOT/dataset_name/regression/split_type/seed_0/test.csv

Core design:
    - Molecule input: smiles
    - Similarity representation: Morgan fingerprint, radius=2, nBits=2048
    - Similarity: Tanimoto
    - Reference library for validation/test: train split only
    - Training-set predictions: leave-one-out within train split, so train metrics
      are not trivially perfect.

Models:
    1nn: k=1
    5nn: k=5

Classification:
    predicted probability = weighted mean of neighbor active labels
    predicted label = prob >= 0.5

Regression:
    predicted log10MIC = weighted mean of neighbor log10MIC values
    screening score = -predicted_log10MIC

By default, 5-NN uses similarity weighting. Set --weighting uniform for simple
majority vote / arithmetic mean.
"""

from privacy_utils import sanitize_metadata

import argparse
import json
import math
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
from rdkit.Chem import AllChem

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
    recall_score,
    r2_score,
    roc_auc_score,
)

N_BITS_DEFAULT = 2048
RADIUS_DEFAULT = 2
HARD_CODED_SMILES_COL = "smiles"

DEFAULT_CLASSIFICATION_TARGET_COLS = [
    "active", "label_active", "is_active", "activity_label", "target", "label", "y",
]
DEFAULT_REGRESSION_TARGET_COLS = [
    "log10_mic", "log10MIC", "log10mic", "log10_value", "log10_mic_value",
    "log10_MIC", "log10_median_value", "log10_label", "y_reg", "target_regression",
    "regression_target", "target", "label",
]
DEFAULT_VALUE_COLS = [
    "mic_numeric", "value", "MIC", "mic", "mic_value", "median_value",
    "value_ug_ml", "mic_ug_ml", "MIC_ug_ml", "mic_ug_per_ml",
]
DEFAULT_ACTIVE_COLS = ["active", "label_active", "is_active", "activity_label"]
DILUTION1_LOG10 = math.log10(2.0)
DILUTION2_LOG10 = math.log10(4.0)
EPS = 1e-12


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def numeric_sort_key(path: Path):
    m = re.match(r"^(\d+)_", path.name)
    return (int(m.group(1)) if m else 10**9, path.name)


def parse_genus_from_dataset_name(name: str) -> str:
    parts = name.split("_")
    if len(parts) >= 2 and parts[0].isdigit():
        return parts[1]
    return name


def parse_threshold_from_dataset_name(name: str) -> Optional[float]:
    m = re.search(r"median_([0-9]+(?:\.[0-9]+)?)", name)
    return float(m.group(1)) if m else None


def find_col(df: pd.DataFrame, preferred: Optional[str], candidates: List[str], required: bool = True) -> Optional[str]:
    if preferred and preferred in df.columns:
        return preferred
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    if required:
        raise ValueError(
            f"Cannot find required column. Preferred={preferred}; candidates={candidates}; columns={list(df.columns)}"
        )
    return None


def require_smiles_column(*dfs: pd.DataFrame) -> str:
    for i, df in enumerate(dfs):
        if HARD_CODED_SMILES_COL not in df.columns:
            raise ValueError(f"Missing required column 'smiles' in dataframe #{i}. Columns={list(df.columns)}")
    return HARD_CODED_SMILES_COL


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


def finite_array(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr[np.isfinite(arr)]


def safe_mean(x) -> float:
    arr = finite_array(x)
    return float(np.mean(arr)) if arr.size else np.nan


def safe_std(x) -> float:
    arr = finite_array(x)
    return float(np.std(arr, ddof=1)) if arr.size >= 2 else np.nan


# -----------------------------
# Chemistry helpers
# -----------------------------


def mol_from_smiles(smiles: str):
    try:
        return Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None


def morgan_fp_bitvect(smiles: str, radius: int, n_bits: int):
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def build_bitvectors(smiles_list: Iterable[str], radius: int, n_bits: int) -> List:
    smiles_list = list(smiles_list)
    print(f"[{now()}] Computing Morgan bit vectors: n={len(smiles_list)}, radius={radius}, nBits={n_bits}", flush=True)
    return [morgan_fp_bitvect(s, radius, n_bits) for s in smiles_list]


def tanimoto_to_reference(query_fp, ref_fps: List) -> np.ndarray:
    sims = np.zeros(len(ref_fps), dtype=np.float32)
    if query_fp is None or len(ref_fps) == 0:
        return sims
    valid_idx = [i for i, fp in enumerate(ref_fps) if fp is not None]
    if not valid_idx:
        return sims
    vals = DataStructs.BulkTanimotoSimilarity(query_fp, [ref_fps[i] for i in valid_idx])
    sims[np.array(valid_idx, dtype=int)] = np.asarray(vals, dtype=np.float32)
    return sims


# -----------------------------
# Labels and inferred flags
# -----------------------------


def infer_classification_target(df: pd.DataFrame, target_col: Optional[str]) -> Tuple[pd.Series, str]:
    col = find_col(df, target_col, DEFAULT_CLASSIFICATION_TARGET_COLS, required=True)
    y = pd.to_numeric(df[col], errors="coerce")
    return y, col


def infer_regression_target(df: pd.DataFrame, target_col: Optional[str], value_col: Optional[str]) -> Tuple[pd.Series, str, str]:
    col = find_col(df, target_col, DEFAULT_REGRESSION_TARGET_COLS, required=False)
    vcol = find_col(df, value_col, DEFAULT_VALUE_COLS, required=False)
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
        f"Cannot infer regression target. Expected one of {DEFAULT_REGRESSION_TARGET_COLS} "
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


def infer_boundary_flags(df: pd.DataFrame, value_col: Optional[str], y_log10: Optional[pd.Series], threshold: Optional[float]) -> pd.Series:
    for c in ["boundary_flag", "mic_boundary_flag", "mic_boundary", "MIC_boundary", "is_boundary"]:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    if threshold is not None:
        if value_col is not None:
            vals = pd.to_numeric(df[value_col], errors="coerce")
        elif y_log10 is not None:
            vals = 10 ** pd.to_numeric(y_log10, errors="coerce")
        else:
            return pd.Series(np.nan, index=df.index)
        return ((vals >= threshold / 2.0) & (vals <= threshold * 2.0)).astype(float)
    return pd.Series(np.nan, index=df.index)


def mic_range_group_from_values(df: pd.DataFrame, value_col: Optional[str], y_log10: Optional[pd.Series], threshold: Optional[float]) -> pd.Series:
    if threshold is None:
        return pd.Series("unknown", index=df.index, dtype=object)
    if value_col is not None:
        vals = pd.to_numeric(df[value_col], errors="coerce")
    elif y_log10 is not None:
        vals = 10 ** pd.to_numeric(y_log10, errors="coerce")
    else:
        return pd.Series("unknown", index=df.index, dtype=object)
    out = pd.Series("unknown", index=df.index, dtype=object)
    out[vals <= threshold / 2.0] = "strong_MIC_le_T_over_2"
    out[(vals > threshold / 2.0) & (vals <= threshold * 2.0)] = "boundary_T_over_2_to_2T"
    out[vals > threshold * 2.0] = "weak_MIC_gt_2T"
    return out




# -----------------------------
# Consistency helpers matching traditional ML outputs
# -----------------------------


def infer_log2mic_for_classification(df: pd.DataFrame, value_col: Optional[str]) -> pd.Series:
    """Match train_ml.py diagnostic convention: log2MIC is inferred from MIC value column when available."""
    if value_col is not None and value_col in df.columns:
        vals = pd.to_numeric(df[value_col], errors="coerce")
        return np.log2(vals.where(vals > 0))
    return pd.Series(np.nan, index=df.index)


def merge_diagnostics_with_topk(primary_diag: Optional[pd.DataFrame], topk_diag: pd.DataFrame) -> pd.DataFrame:
    """Keep standard NN/AC diagnostic columns and append top-k audit columns without duplicate names."""
    if primary_diag is None or primary_diag.empty:
        return topk_diag.reset_index(drop=True)
    drop_cols = [
        "neighbor_count_used", "nn_train_row", "nn_sim", "nn_bin",
    ]
    aux = topk_diag.drop(columns=[c for c in drop_cols if c in topk_diag.columns], errors="ignore")
    return pd.concat([primary_diag.reset_index(drop=True), aux.reset_index(drop=True)], axis=1)


def classification_diagnostics_to_train(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    train_fps: List,
    eval_fps: List,
    train_scores: np.ndarray,
    eval_scores: np.ndarray,
    train_y: np.ndarray,
    value_col_train: Optional[str],
    value_col_eval: Optional[str],
    smiles_col: str,
) -> pd.DataFrame:
    """Create the same key diagnostic columns as train_ml.py for classification outputs."""
    train_smiles = train_df[smiles_col].astype(str).tolist()
    train_y = np.asarray(train_y).astype(int)
    train_scores = np.asarray(train_scores, dtype=float)
    eval_scores = np.asarray(eval_scores, dtype=float)
    train_logmic_arr = pd.to_numeric(infer_log2mic_for_classification(train_df, value_col_train), errors="coerce").values.astype(float)
    eval_logmic_arr = pd.to_numeric(infer_log2mic_for_classification(eval_df, value_col_eval), errors="coerce").values.astype(float)

    rows = []
    for i, fp in enumerate(eval_fps):
        sims = tanimoto_to_reference(fp, train_fps)
        if len(sims):
            nn_idx = int(np.argmax(sims))
            nn_sim = float(sims[nn_idx])
        else:
            nn_idx = -1
            nn_sim = np.nan

        ac_mask = np.zeros(len(train_df), dtype=bool)
        ac_max_abs_diff = np.nan
        ac_pair_directional_correct = []
        if i < len(eval_logmic_arr) and np.isfinite(eval_logmic_arr[i]) and len(sims):
            sim_mask = sims >= 0.7
            diff = np.abs(train_logmic_arr - eval_logmic_arr[i])
            ac_mask = sim_mask & np.isfinite(diff) & (diff >= 4.0)
            if np.any(ac_mask):
                ac_max_abs_diff = float(np.nanmax(diff[ac_mask]))
                ac_indices = np.where(ac_mask)[0]
                for j in ac_indices:
                    true_test_stronger = eval_logmic_arr[i] < train_logmic_arr[j]
                    pred_test_stronger = eval_scores[i] > train_scores[j]
                    ac_pair_directional_correct.append(int(true_test_stronger == pred_test_stronger))

        rows.append({
            "nn_train_row": nn_idx,
            "nn_sim": nn_sim,
            "nn_bin": ">=0.7" if nn_sim >= 0.7 else ("0.4-0.7" if nn_sim >= 0.4 else "<0.4"),
            "nn_train_smiles": train_smiles[nn_idx] if nn_idx >= 0 else None,
            "nn_train_active": int(train_y[nn_idx]) if nn_idx >= 0 and nn_idx < len(train_y) else np.nan,
            "nn_train_score": float(train_scores[nn_idx]) if nn_idx >= 0 and nn_idx < len(train_scores) else np.nan,
            "nn_train_log2mic": float(train_logmic_arr[nn_idx]) if nn_idx >= 0 and nn_idx < len(train_logmic_arr) and np.isfinite(train_logmic_arr[nn_idx]) else np.nan,
            "ac_flag": int(np.any(ac_mask)),
            "ac_neighbor_count": int(np.sum(ac_mask)),
            "ac_max_abs_log2mic_diff": ac_max_abs_diff,
            "ac_pair_directional_accuracy_for_molecule": float(np.mean(ac_pair_directional_correct)) if ac_pair_directional_correct else np.nan,
            "ac_pair_count_for_molecule": int(len(ac_pair_directional_correct)),
        })
    return pd.DataFrame(rows)


def regression_diagnostics_to_train(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    train_fps: List,
    eval_fps: List,
    train_pred: np.ndarray,
    eval_pred: np.ndarray,
    train_y_log10: np.ndarray,
    eval_y_log10: np.ndarray,
    smiles_col: str,
    k_local: int = 10,
) -> pd.DataFrame:
    """Create the same key diagnostic columns as train_regression_ml.py for regression outputs."""
    train_smiles = train_df[smiles_col].astype(str).tolist()
    train_y = np.asarray(train_y_log10, dtype=float)
    eval_y = np.asarray(eval_y_log10, dtype=float)
    train_pred = np.asarray(train_pred, dtype=float)
    eval_pred = np.asarray(eval_pred, dtype=float)

    rows = []
    for i, fp in enumerate(eval_fps):
        sims = tanimoto_to_reference(fp, train_fps)
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
        if i < len(eval_y) and np.isfinite(eval_y[i]) and len(sims):
            diff = np.abs(train_y - eval_y[i])
            ac_mask = (sims >= 0.7) & np.isfinite(diff) & (diff >= math.log10(16.0))
            if np.any(ac_mask):
                ac_max_abs_diff = float(np.nanmax(diff[ac_mask]))
                ac_indices = np.where(ac_mask)[0]
                for j in ac_indices:
                    true_test_stronger = eval_y[i] < train_y[j]
                    pred_test_stronger = eval_pred[i] < train_pred[j]
                    ac_pair_directional_correct.append(int(true_test_stronger == pred_test_stronger))

        rows.append({
            "nn_train_row": nn_idx,
            "nn_sim": nn_sim,
            "nn_bin": ">=0.7" if nn_sim >= 0.7 else ("0.4-0.7" if nn_sim >= 0.4 else "<0.4"),
            "nn_train_smiles": train_smiles[nn_idx] if nn_idx >= 0 else None,
            "nn_train_log10mic": float(train_y[nn_idx]) if nn_idx >= 0 and nn_idx < len(train_y) and np.isfinite(train_y[nn_idx]) else np.nan,
            "nn_train_pred_log10mic": float(train_pred[nn_idx]) if nn_idx >= 0 and nn_idx < len(train_pred) and np.isfinite(train_pred[nn_idx]) else np.nan,
            "local_train_log10mic_variance_k10": local_var,
            "ac_flag": int(np.any(ac_mask)),
            "ac_neighbor_count": int(np.sum(ac_mask)),
            "ac_max_abs_log10mic_diff": ac_max_abs_diff,
            "ac_pair_directional_accuracy_for_molecule": float(np.mean(ac_pair_directional_correct)) if ac_pair_directional_correct else np.nan,
            "ac_pair_count_for_molecule": int(len(ac_pair_directional_correct)),
        })
    return pd.DataFrame(rows)

# -----------------------------
# kNN prediction
# -----------------------------


def parse_knn_model(model_name: str) -> int:
    m = model_name.lower().replace("-", "")
    if m in {"1nn", "nn1"}:
        return 1
    if m in {"5nn", "nn5"}:
        return 5
    raise ValueError(f"Unsupported similarity baseline: {model_name}. Use 1nn or 5nn.")


def weighted_average(values: np.ndarray, sims: np.ndarray, weighting: str) -> float:
    values = np.asarray(values, dtype=float)
    sims = np.asarray(sims, dtype=float)
    mask = np.isfinite(values)
    values = values[mask]
    sims = sims[mask]
    if values.size == 0:
        return np.nan
    if weighting == "similarity" and np.sum(np.maximum(sims, 0.0)) > EPS:
        w = np.maximum(sims, 0.0)
        return float(np.sum(w * values) / np.sum(w))
    return float(np.mean(values))


def predict_knn(
    ref_fps: List,
    query_fps: List,
    ref_y: np.ndarray,
    k: int,
    weighting: str,
    exclude_self: bool = False,
) -> Tuple[np.ndarray, pd.DataFrame]:
    ref_y = np.asarray(ref_y, dtype=float)
    preds = np.full(len(query_fps), np.nan, dtype=float)
    rows = []
    finite_ref = np.isfinite(ref_y)
    finite_count = int(np.sum(finite_ref))
    if finite_count == 0:
        raise ValueError("No finite reference targets available for kNN prediction.")

    for i, fp in enumerate(query_fps):
        sims = tanimoto_to_reference(fp, ref_fps)
        if exclude_self and i < len(sims):
            sims[i] = -1.0
        score_for_order = sims.copy()
        score_for_order[~finite_ref] = -1.0
        k_eff = min(k, finite_count)
        order = np.argsort(-score_for_order)[:k_eff]
        order = np.asarray([j for j in order if score_for_order[j] >= 0 and np.isfinite(ref_y[j])], dtype=int)
        if order.size == 0:
            # Fall back to finite targets if all fingerprints are invalid.
            order = np.where(finite_ref)[0][:k_eff]
        top_sims = sims[order]
        top_y = ref_y[order]
        pred = weighted_average(top_y, top_sims, weighting=weighting)
        preds[i] = pred
        top_sims_nonneg = np.maximum(top_sims, 0.0)
        rows.append({
            "neighbor_count_used": int(order.size),
            "nn_train_row": int(order[0]) if order.size else -1,
            "nn_sim": float(top_sims[0]) if order.size else np.nan,
            "nn_bin": ">=0.7" if order.size and top_sims[0] >= 0.7 else ("0.4-0.7" if order.size and top_sims[0] >= 0.4 else "<0.4"),
            "topk_train_rows": ";".join(map(str, order.tolist())),
            "topk_sims": ";".join(f"{x:.6g}" for x in top_sims.tolist()),
            "topk_targets": ";".join(f"{x:.6g}" for x in top_y.tolist()),
            "topk_sim_mean": float(np.mean(top_sims_nonneg)) if order.size else np.nan,
            "topk_sim_min": float(np.min(top_sims_nonneg)) if order.size else np.nan,
            "topk_target_mean": safe_mean(top_y),
            "topk_target_sd": safe_std(top_y),
        })
    return preds, pd.DataFrame(rows)


# -----------------------------
# Metrics
# -----------------------------


def classification_metrics(y_true, y_prob, threshold: float = 0.5, prefix: str = "") -> Dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask].astype(int)
    p = np.clip(p[mask], EPS, 1.0 - EPS)
    pred = (p >= threshold).astype(int)
    out: Dict[str, float] = {prefix + "n": int(len(y)), prefix + "pos": int(np.sum(y == 1)) if len(y) else 0}
    out[prefix + "neg"] = int(np.sum(y == 0)) if len(y) else 0
    out[prefix + "pos_ratio"] = float(np.mean(y == 1)) if len(y) else np.nan
    out[prefix + "threshold"] = float(threshold)
    if len(y) == 0:
        for name in ["roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "precision", "recall", "f1", "mcc", "log_loss", "brier", "specificity", "sensitivity"]:
            out[prefix + name] = np.nan
        return out
    if len(np.unique(y)) >= 2:
        out[prefix + "roc_auc"] = float(roc_auc_score(y, p))
        out[prefix + "pr_auc"] = float(average_precision_score(y, p))
        out[prefix + "log_loss"] = float(log_loss(y, p, labels=[0, 1]))
    else:
        out[prefix + "roc_auc"] = np.nan
        out[prefix + "pr_auc"] = np.nan
        out[prefix + "log_loss"] = np.nan
    out[prefix + "roc"] = out[prefix + "roc_auc"]
    out[prefix + "pr"] = out[prefix + "pr_auc"]
    out[prefix + "accuracy"] = float(accuracy_score(y, pred))
    out[prefix + "balanced_accuracy"] = float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) >= 2 else np.nan
    out[prefix + "precision"] = float(precision_score(y, pred, zero_division=0))
    out[prefix + "recall"] = float(recall_score(y, pred, zero_division=0))
    out[prefix + "f1"] = float(f1_score(y, pred, zero_division=0))
    out[prefix + "mcc"] = float(matthews_corrcoef(y, pred)) if len(np.unique(pred)) >= 2 and len(np.unique(y)) >= 2 else 0.0
    out[prefix + "brier"] = float(brier_score_loss(y, p))
    try:
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        out[prefix + "specificity"] = float(tn / (tn + fp)) if (tn + fp) else np.nan
        out[prefix + "sensitivity"] = float(tp / (tp + fn)) if (tp + fn) else np.nan
    except Exception:
        out[prefix + "specificity"] = np.nan
        out[prefix + "sensitivity"] = np.nan
    return out


def screening_metrics(active_true, score, prefix: str = "") -> Dict[str, float]:
    y = np.asarray(active_true, dtype=float)
    s = np.asarray(score, dtype=float)
    mask = np.isfinite(y) & np.isfinite(s)
    y = y[mask].astype(int)
    s = s[mask]
    n = len(y)
    out: Dict[str, float] = {prefix + "screening_n": int(n), prefix + "screening_pos": int(np.sum(y == 1)) if n else 0}
    if n == 0 or len(np.unique(y)) < 2:
        for name in ["EF1%", "EF5%", "EF10%", "Hit_rate@Top1%", "Hit_rate@Top5%", "Hit_rate@Top10%", "Recall@Top50", "Recall@Top100", "Recall@Top500"]:
            out[prefix + name] = np.nan
        return out
    order = np.argsort(-s)
    base_rate = float(np.mean(y == 1))
    for pct, label in [(0.01, "1pct"), (0.05, "5pct"), (0.10, "10pct")]:
        kk = max(1, int(math.ceil(n * pct)))
        top = order[:kk]
        hits = int(np.sum(y[top] == 1))
        hit_rate = float(hits / kk)
        out[prefix + f"top_{label}_n"] = int(kk)
        out[prefix + f"top_{label}_hits"] = int(hits)
        out[prefix + f"hit_rate_top_{label}"] = hit_rate
        out[prefix + f"ef_{label}"] = float(hit_rate / base_rate) if base_rate > 0 else np.nan
    out[prefix + "EF1%"] = out[prefix + "ef_1pct"]
    out[prefix + "EF5%"] = out[prefix + "ef_5pct"]
    out[prefix + "EF10%"] = out[prefix + "ef_10pct"]
    out[prefix + "Hit_rate@Top1%"] = out[prefix + "hit_rate_top_1pct"]
    out[prefix + "Hit_rate@Top5%"] = out[prefix + "hit_rate_top_5pct"]
    out[prefix + "Hit_rate@Top10%"] = out[prefix + "hit_rate_top_10pct"]
    total_pos = max(int(np.sum(y == 1)), 1)
    for kk in [50, 100, 500]:
        kkk = min(kk, n)
        hits = int(np.sum(y[order[:kkk]] == 1)) if kkk > 0 else 0
        out[prefix + f"Recall@Top{kk}"] = float(hits / total_pos) if n else np.nan
    return out


def regression_metrics(y_true, y_pred, prefix: str = "") -> Dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    out: Dict[str, float] = {prefix + "n": int(len(y))}
    if len(y) == 0:
        for k in ["mae", "rmse", "mse", "medae", "r2", "pearson", "spearman", "within_1_dilution", "within_2_dilutions", "median_fold_error", "mean_fold_error", "max_abs_error_log10"]:
            out[prefix + k] = np.nan
        return out
    abs_err = np.abs(p - y)
    mse = mean_squared_error(y, p)
    out[prefix + "mae"] = float(mean_absolute_error(y, p))
    out[prefix + "mse"] = float(mse)
    out[prefix + "rmse"] = float(np.sqrt(mse))
    out[prefix + "medae"] = float(median_absolute_error(y, p))
    out[prefix + "r2"] = float(r2_score(y, p)) if len(y) >= 2 else np.nan
    out[prefix + "pearson"] = safe_corr(y, p, method="pearson")
    out[prefix + "spearman"] = safe_corr(y, p, method="spearman")
    out[prefix + "within_1_dilution"] = float(np.mean(abs_err <= DILUTION1_LOG10))
    out[prefix + "within_2_dilutions"] = float(np.mean(abs_err <= DILUTION2_LOG10))
    out[prefix + "median_fold_error"] = float(np.median(10 ** abs_err))
    out[prefix + "mean_fold_error"] = float(np.mean(10 ** abs_err))
    out[prefix + "max_abs_error_log10"] = float(np.max(abs_err))
    return out


# -----------------------------
# Diagnostics and output helpers
# -----------------------------


def subgroup_classification_metrics(df: pd.DataFrame) -> Dict[str, float]:
    score_col = "pred_score" if "pred_score" in df.columns else "pred_prob"
    out = classification_metrics(df["true_active"].values, df[score_col].values, prefix="")
    out.update(screening_metrics(df["true_active"].values, df[score_col].values, prefix=""))
    return out


def subgroup_regression_metrics(df: pd.DataFrame) -> Dict[str, float]:
    out = regression_metrics(df["true_log10MIC"].values, df["pred_log10MIC"].values, prefix="")
    if "true_active_for_screening" in df.columns:
        out.update(screening_metrics(df["true_active_for_screening"].values, -df["pred_log10MIC"].values, prefix=""))
    return out


def diagnostic_summary_classification(pred_df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    df = pred_df.copy()
    score_col = "pred_score" if "pred_score" in df.columns else "pred_prob"
    df["error"] = (df["true_active"].astype(int) != df["pred_label"].astype(int)).astype(int)
    if "boundary_flag" in df.columns and df["boundary_flag"].notna().any():
        b = df[df["boundary_flag"] == 1]
        nb = df[df["boundary_flag"] == 0]
        out["boundary_n"] = int(len(b))
        out["non_boundary_n"] = int(len(nb))
        out["boundary_error_rate"] = float(b["error"].mean()) if len(b) else np.nan
        out["non_boundary_error_rate"] = float(nb["error"].mean()) if len(nb) else np.nan
        out["boundary_error_enrichment"] = out["boundary_error_rate"] / out["non_boundary_error_rate"] if out.get("non_boundary_error_rate") and out["non_boundary_error_rate"] > 0 else np.nan
    if "nn_bin" in df.columns:
        for bin_name in [">=0.7", "0.4-0.7", "<0.4"]:
            g = df[df["nn_bin"] == bin_name]
            key = bin_name.replace(">=", "ge_").replace("<", "lt_").replace("-", "_").replace(".", "p")
            prefix = f"nn_{key}_"
            m = subgroup_classification_metrics(g)
            for k, v in m.items():
                out[prefix + k] = v
    if "ac_flag" in df.columns:
        ac = df[df["ac_flag"] == 1]
        non_ac = df[df["ac_flag"] == 0]
        out["ac_n"] = int(len(ac))
        out["non_ac_n"] = int(len(non_ac))
        out["ac_error_rate"] = float(ac["error"].mean()) if len(ac) else np.nan
        out["non_ac_error_rate"] = float(non_ac["error"].mean()) if len(non_ac) else np.nan
        out["ac_error_enrichment"] = (
            out["ac_error_rate"] / out["non_ac_error_rate"]
            if out.get("non_ac_error_rate") and out["non_ac_error_rate"] > 0 else np.nan
        )
        if len(ac) and len(np.unique(ac["true_active"].astype(int))) >= 2:
            out["ac_subgroup_pr_auc"] = float(average_precision_score(ac["true_active"].astype(int), ac[score_col].astype(float)))
        else:
            out["ac_subgroup_pr_auc"] = np.nan
        if "ac_pair_directional_accuracy_for_molecule" in df.columns:
            out["ac_pair_directional_accuracy"] = float(df["ac_pair_directional_accuracy_for_molecule"].dropna().mean()) if df["ac_pair_directional_accuracy_for_molecule"].notna().any() else np.nan
            out["ac_pair_total_count"] = int(df.get("ac_pair_count_for_molecule", pd.Series(dtype=float)).fillna(0).sum())

    if "mic_range_group" in df.columns:
        for group in ["strong_MIC_le_T_over_2", "boundary_T_over_2_to_2T", "weak_MIC_gt_2T"]:
            g = df[df["mic_range_group"] == group]
            prefix = f"range_{group}_"
            m = subgroup_classification_metrics(g)
            for k, v in m.items():
                out[prefix + k] = v
    return out


def diagnostic_summary_regression(pred_df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    df = pred_df.copy()
    df["abs_error_log10"] = np.abs(df["pred_log10MIC"].astype(float) - df["true_log10MIC"].astype(float))
    df["sq_error_log10"] = df["abs_error_log10"] ** 2
    if "boundary_flag" in df.columns and df["boundary_flag"].notna().any():
        b = df[df["boundary_flag"] == 1]
        nb = df[df["boundary_flag"] == 0]
        out["boundary_n"] = int(len(b))
        out["non_boundary_n"] = int(len(nb))
        out["boundary_mae"] = float(b["abs_error_log10"].mean()) if len(b) else np.nan
        out["non_boundary_mae"] = float(nb["abs_error_log10"].mean()) if len(nb) else np.nan
        out["boundary_rmse"] = float(np.sqrt(b["sq_error_log10"].mean())) if len(b) else np.nan
        out["non_boundary_rmse"] = float(np.sqrt(nb["sq_error_log10"].mean())) if len(nb) else np.nan
        out["boundary_regression_error_enrichment"] = out["boundary_mae"] / out["non_boundary_mae"] if out.get("non_boundary_mae") and out["non_boundary_mae"] > 0 else np.nan
        out["boundary_within_1_dilution"] = float(np.mean(b["abs_error_log10"] <= DILUTION1_LOG10)) if len(b) else np.nan
        out["non_boundary_within_1_dilution"] = float(np.mean(nb["abs_error_log10"] <= DILUTION1_LOG10)) if len(nb) else np.nan
    if "nn_bin" in df.columns:
        for bin_name in [">=0.7", "0.4-0.7", "<0.4"]:
            g = df[df["nn_bin"] == bin_name]
            key = bin_name.replace(">=", "ge_").replace("<", "lt_").replace("-", "_").replace(".", "p")
            prefix = f"nn_{key}_"
            m = subgroup_regression_metrics(g)
            for k, v in m.items():
                out[prefix + k] = v
    if "topk_target_sd" in df.columns and df["topk_target_sd"].notna().any():
        out["mean_local_train_log10mic_sd_k"] = float(df["topk_target_sd"].mean())
        med = float(df["topk_target_sd"].median())
        high = df[df["topk_target_sd"] > med]
        low = df[df["topk_target_sd"] <= med]
        out["high_local_variance_n"] = int(len(high))
        out["low_local_variance_n"] = int(len(low))
        out["high_local_variance_mae"] = float(high["abs_error_log10"].mean()) if len(high) else np.nan
        out["low_local_variance_mae"] = float(low["abs_error_log10"].mean()) if len(low) else np.nan
        out["local_variance_error_enrichment"] = out["high_local_variance_mae"] / out["low_local_variance_mae"] if out.get("low_local_variance_mae") and out["low_local_variance_mae"] > 0 else np.nan
    if "mic_range_group" in df.columns:
        for group in ["strong_MIC_le_T_over_2", "boundary_T_over_2_to_2T", "weak_MIC_gt_2T"]:
            g = df[df["mic_range_group"] == group]
            prefix = f"range_{group}_"
            m = subgroup_regression_metrics(g)
            for k, v in m.items():
                out[prefix + k] = v
    return out


def save_train_log(task: str, train_metrics: Dict[str, float], val_metrics: Dict[str, float], outdir: Path) -> None:
    if task == "classification":
        row = {
            "iteration": 1,
            "train_loss": train_metrics.get("train_log_loss", np.nan),
            "val_loss": val_metrics.get("val_log_loss", np.nan),
            "train_auc": train_metrics.get("train_roc_auc", np.nan),
            "val_auc": val_metrics.get("val_roc_auc", np.nan),
            "train_pr_auc": train_metrics.get("train_pr_auc", np.nan),
            "val_pr_auc": val_metrics.get("val_pr_auc", np.nan),
            "train_accuracy": train_metrics.get("train_accuracy", np.nan),
            "val_accuracy": val_metrics.get("val_accuracy", np.nan),
            "train_f1": train_metrics.get("train_f1", np.nan),
            "val_f1": val_metrics.get("val_f1", np.nan),
            "train_mcc": train_metrics.get("train_mcc", np.nan),
            "val_mcc": val_metrics.get("val_mcc", np.nan),
            "note": "final_only_no_epoch_loss_for_this_model",
        }
    else:
        row = {
            "iteration": 1,
            "train_loss": train_metrics.get("train_mse", np.nan),
            "val_loss": val_metrics.get("val_mse", np.nan),
            "train_mse": train_metrics.get("train_mse", np.nan),
            "val_mse": val_metrics.get("val_mse", np.nan),
            "train_rmse": train_metrics.get("train_rmse", np.nan),
            "val_rmse": val_metrics.get("val_rmse", np.nan),
            "train_mae": train_metrics.get("train_mae", np.nan),
            "val_mae": val_metrics.get("val_mae", np.nan),
            "train_spearman": train_metrics.get("train_spearman", np.nan),
            "val_spearman": val_metrics.get("val_spearman", np.nan),
            "train_within_1_dilution": train_metrics.get("train_within_1_dilution", np.nan),
            "val_within_1_dilution": val_metrics.get("val_within_1_dilution", np.nan),
            "train_within_2_dilutions": train_metrics.get("train_within_2_dilutions", np.nan),
            "val_within_2_dilutions": val_metrics.get("val_within_2_dilutions", np.nan),
            "note": "final_only_no_epoch_loss_for_this_model",
        }
    curve = pd.DataFrame([row])
    curve.to_csv(outdir / "train_log.csv", index=False)
    curve.to_csv(outdir / "training_curve.csv", index=False)
    long_rows = []
    for split_prefix, metrics in [("train", train_metrics), ("validation", val_metrics)]:
        for metric_name, value in metrics.items():
            if not metric_name.startswith(split_prefix[:3]) and not metric_name.startswith(split_prefix):
                continue
            long_rows.append({
                "iteration": 1,
                "split": split_prefix,
                "metric": metric_name,
                "value": value,
                "note": "final_only_no_epoch_loss_for_this_model",
            })
    pd.DataFrame(long_rows).to_csv(outdir / "train_log_long.csv", index=False)


# -----------------------------
# Run discovery
# -----------------------------


def discover_runs(data_root: Path, task: str, split_types: List[str], seeds: List[int], datasets: Optional[List[str]] = None) -> List[Dict]:
    dataset_dirs = [p for p in data_root.iterdir() if p.is_dir() and (p / task).exists()]
    dataset_dirs = sorted(dataset_dirs, key=numeric_sort_key)
    if datasets:
        keep = set(datasets)
        dataset_dirs = [p for p in dataset_dirs if p.name in keep or parse_genus_from_dataset_name(p.name) in keep]
    runs = []
    for ds in dataset_dirs:
        for split_type in split_types:
            split_dir = ds / task / split_type
            if not split_dir.exists():
                print(f"[{now()}] Skip missing {task} split dir: {split_dir.name}", flush=True)
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
                        "task": task,
                        "split_type": split_type,
                        "seed": seed,
                        "seed_dir": seed_dir,
                        "train_csv": train_csv,
                        "val_csv": val_csv,
                        "test_csv": test_csv,
                    })
                else:
                    print(f"[{now()}] Skip incomplete {task} seed dir: {seed_dir.name}", flush=True)
    return runs


# -----------------------------
# Main run logic
# -----------------------------


def valid_target_filter(df: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    y_num = pd.to_numeric(y, errors="coerce")
    mask = y_num.notna() & np.isfinite(y_num)
    return df.loc[mask].reset_index(drop=True), y_num.loc[mask].reset_index(drop=True)


def run_one(run: Dict, model_name: str, args) -> Optional[Dict]:
    task = run["task"]
    k = parse_knn_model(model_name)
    model_name = model_name.lower().replace("-", "")
    dataset_name = run["dataset_name"]
    genus = run["genus"]
    split_type = run["split_type"]
    seed = run["seed"]
    threshold_from_name = run["threshold"]
    threshold_for_boundary = args.threshold if args.threshold is not None else threshold_from_name

    outdir = Path(args.outdir) / model_name / dataset_name / task / split_type / f"seed_{seed}"
    ensure_dir(outdir)

    print("\n" + "=" * 100, flush=True)
    print(f"[{now()}] RUN: {task.upper()} | model={model_name} | k={k} | dataset={dataset_name} | split={split_type} | seed={seed}", flush=True)
    print("=" * 100, flush=True)

    train_df = pd.read_csv(run["train_csv"])
    val_df = pd.read_csv(run["val_csv"])
    test_df = pd.read_csv(run["test_csv"])
    smiles_col = require_smiles_column(train_df, val_df, test_df)

    value_col = infer_value_col(train_df, args.value_col)
    val_value_col = infer_value_col(val_df, args.value_col)
    test_value_col = infer_value_col(test_df, args.value_col)

    if task == "classification":
        y_train_s, target_col = infer_classification_target(train_df, args.classification_target_col)
        y_val_s, _ = infer_classification_target(val_df, target_col)
        y_test_s, _ = infer_classification_target(test_df, target_col)
        target_source_type = "classification_label"
    else:
        y_train_s, target_col, target_source_type = infer_regression_target(train_df, args.regression_target_col, args.value_col)
        y_val_s, _, _ = infer_regression_target(val_df, target_col if target_source_type == "existing_log10_column" else None, args.value_col)
        y_test_s, _, _ = infer_regression_target(test_df, target_col if target_source_type == "existing_log10_column" else None, args.value_col)

    train_df, y_train_s = valid_target_filter(train_df, y_train_s)
    val_df, y_val_s = valid_target_filter(val_df, y_val_s)
    test_df, y_test_s = valid_target_filter(test_df, y_test_s)
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise RuntimeError(f"Empty train/validation/test after filtering invalid targets: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    train_df.to_csv(outdir / "train_input.csv", index=False)
    val_df.to_csv(outdir / "validation_input.csv", index=False)
    test_df.to_csv(outdir / "test_input.csv", index=False)

    y_train = y_train_s.values.astype(float)
    y_val = y_val_s.values.astype(float)
    y_test = y_test_s.values.astype(float)

    print(f"[{now()}] Columns: smiles_col={smiles_col}, target_col={target_col}, target_source_type={target_source_type}, value_col={value_col}", flush=True)
    print(f"[{now()}] Sizes: train={len(train_df)}, validation={len(val_df)}, test={len(test_df)}", flush=True)

    t0 = time.time()
    train_fps = build_bitvectors(train_df[smiles_col].astype(str).tolist(), args.radius, args.n_bits)
    val_fps = build_bitvectors(val_df[smiles_col].astype(str).tolist(), args.radius, args.n_bits)
    test_fps = build_bitvectors(test_df[smiles_col].astype(str).tolist(), args.radius, args.n_bits)

    print(f"[{now()}] Predicting train by leave-one-out, validation/test by train-reference kNN ...", flush=True)
    train_pred_raw, train_nn = predict_knn(train_fps, train_fps, y_train, k=k, weighting=args.weighting, exclude_self=True)
    val_pred_raw, val_nn = predict_knn(train_fps, val_fps, y_train, k=k, weighting=args.weighting, exclude_self=False)
    test_pred_raw, test_nn = predict_knn(train_fps, test_fps, y_train, k=k, weighting=args.weighting, exclude_self=False)
    run_seconds = time.time() - t0
    train_seconds = run_seconds

    feature_summary = {
        "n_train": int(len(train_df)),
        "n_validation": int(len(val_df)),
        "n_test": int(len(test_df)),
        "morgan_bits": int(args.n_bits),
        "radius": int(args.radius),
        "rdkit2d_descriptors": 0,
        "total_features": int(args.n_bits),
        "similarity_metric": "Tanimoto",
        "feature_type": f"Morgan_r{args.radius}_{args.n_bits}_Tanimoto_only",
        "train_invalid_smiles_count": int(sum(fp is None for fp in train_fps)),
        "validation_invalid_smiles_count": int(sum(fp is None for fp in val_fps)),
        "test_invalid_smiles_count": int(sum(fp is None for fp in test_fps)),
    }
    with open(outdir / "feature_summary.json", "w", encoding="utf-8") as f:
        json.dump(feature_summary, f, indent=2, ensure_ascii=False, default=str)

    model_params = {
        "model": model_name,
        "k": int(k),
        "weighting": args.weighting,
        "radius": int(args.radius),
        "n_bits": int(args.n_bits),
        "similarity_metric": "Tanimoto",
        "classification_threshold": float(args.classification_threshold),
        "learned_parameters": False,
    }
    with open(outdir / "model_params.json", "w", encoding="utf-8") as f:
        json.dump(model_params, f, indent=2, ensure_ascii=False, default=str)

    train_boundary = infer_boundary_flags(train_df, value_col, y_train_s if task == "regression" else None, threshold_for_boundary)
    val_boundary = infer_boundary_flags(val_df, val_value_col, y_val_s if task == "regression" else None, threshold_for_boundary)
    test_boundary = infer_boundary_flags(test_df, test_value_col, y_test_s if task == "regression" else None, threshold_for_boundary)

    if task == "classification":
        train_prob = np.clip(train_pred_raw, 0.0, 1.0)
        val_prob = np.clip(val_pred_raw, 0.0, 1.0)
        test_prob = np.clip(test_pred_raw, 0.0, 1.0)
        train_label = (train_prob >= args.classification_threshold).astype(int)
        val_label = (val_prob >= args.classification_threshold).astype(int)
        test_label = (test_prob >= args.classification_threshold).astype(int)
        fixed_threshold = args.classification_threshold
        val_best_f1_threshold = fixed_threshold

        val_diag_core = classification_diagnostics_to_train(
            train_df, val_df, train_fps, val_fps, train_prob, val_prob, y_train, value_col, val_value_col, smiles_col
        )
        test_diag_core = classification_diagnostics_to_train(
            train_df, test_df, train_fps, test_fps, train_prob, test_prob, y_train, value_col, test_value_col, smiles_col
        )
        val_diag = merge_diagnostics_with_topk(val_diag_core, val_nn)
        test_diag = merge_diagnostics_with_topk(test_diag_core, test_nn)

        def make_pred_df(df, y_true, prob, label, split_name, boundary, nn_df=None):
            pred = df.copy()
            pred.insert(0, "split", split_name)
            pred["dataset_name"] = dataset_name
            pred["genus"] = genus
            pred["task"] = task
            pred["split_type"] = split_type
            pred["seed"] = seed
            pred["model"] = model_name
            pred["k"] = k
            pred["weighting"] = args.weighting
            pred["true_active"] = np.asarray(y_true, dtype=int)
            pred["pred_score"] = np.asarray(prob, dtype=float)
            pred["pred_prob"] = pred["pred_score"]
            pred["score_cls"] = pred["pred_score"]
            pred["pred_label"] = np.asarray(label, dtype=int)
            pred["pred_label_0p5"] = (pred["pred_score"].values >= 0.5).astype(int)
            pred["threshold_used"] = fixed_threshold
            pred["val_best_f1_threshold"] = val_best_f1_threshold
            pred["classification_threshold"] = fixed_threshold
            pred["error"] = (pred["true_active"].astype(int) != pred["pred_label"].astype(int)).astype(int)
            pred["boundary_flag"] = boundary.values
            pred["log2_mic_for_diagnosis"] = infer_log2mic_for_classification(df, infer_value_col(df, args.value_col)).values
            pred["mic_range_group"] = mic_range_group_from_values(df, infer_value_col(df, args.value_col), None, threshold_for_boundary).values
            pred["threshold_for_boundary"] = threshold_for_boundary
            if nn_df is not None:
                pred = pd.concat([pred.reset_index(drop=True), nn_df.reset_index(drop=True)], axis=1)
            return pred

        train_pred_df = make_pred_df(train_df, y_train, train_prob, train_label, "train", train_boundary, None)
        val_pred_df = make_pred_df(val_df, y_val, val_prob, val_label, "validation", val_boundary, val_diag)
        test_pred_df = make_pred_df(test_df, y_test, test_prob, test_label, "test", test_boundary, test_diag)

        train_m = classification_metrics(y_train, train_prob, threshold=args.classification_threshold, prefix="train_")
        train_m.update(screening_metrics(y_train, train_prob, prefix="train_"))
        val_m = classification_metrics(y_val, val_prob, threshold=args.classification_threshold, prefix="val_")
        val_m.update(screening_metrics(y_val, val_prob, prefix="val_"))
        test_m = classification_metrics(y_test, test_prob, threshold=args.classification_threshold, prefix="test_")
        test_m.update(screening_metrics(y_test, test_prob, prefix="test_"))
        diag_summary = diagnostic_summary_classification(test_pred_df)
        feature_type = f"Morgan_r{args.radius}_{args.n_bits}_Tanimoto"
        target_scale = "binary_active"
    else:
        train_active = infer_active_flags(train_df, args.active_col, value_col, threshold_for_boundary)
        val_active = infer_active_flags(val_df, args.active_col, val_value_col, threshold_for_boundary)
        test_active = infer_active_flags(test_df, args.active_col, test_value_col, threshold_for_boundary)

        val_diag_core = regression_diagnostics_to_train(
            train_df, val_df, train_fps, val_fps, train_pred_raw, val_pred_raw, y_train, y_val, smiles_col, k_local=10
        )
        test_diag_core = regression_diagnostics_to_train(
            train_df, test_df, train_fps, test_fps, train_pred_raw, test_pred_raw, y_train, y_test, smiles_col, k_local=10
        )
        val_diag = merge_diagnostics_with_topk(val_diag_core, val_nn)
        test_diag = merge_diagnostics_with_topk(test_diag_core, test_nn)

        def make_pred_df(df, y_true, pred_log10, split_name, boundary, active, nn_df=None):
            pred = df.copy()
            pred.insert(0, "split", split_name)
            pred["dataset_name"] = dataset_name
            pred["genus"] = genus
            pred["task"] = task
            pred["split_type"] = split_type
            pred["seed"] = seed
            pred["model"] = model_name
            pred["k"] = k
            pred["weighting"] = args.weighting
            pred["true_log10MIC"] = np.asarray(y_true, dtype=float)
            pred["pred_log10MIC"] = np.asarray(pred_log10, dtype=float)
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
            pred["mic_range_group"] = mic_range_group_from_values(df, infer_value_col(df, args.value_col), pd.Series(y_true), threshold_for_boundary).values
            pred["threshold_for_boundary"] = threshold_for_boundary
            pred["regression_target_col"] = target_col
            pred["target_source_type"] = target_source_type
            if nn_df is not None:
                pred = pd.concat([pred.reset_index(drop=True), nn_df.reset_index(drop=True)], axis=1)
            return pred

        train_pred_df = make_pred_df(train_df, y_train, train_pred_raw, "train", train_boundary, train_active, None)
        val_pred_df = make_pred_df(val_df, y_val, val_pred_raw, "validation", val_boundary, val_active, val_diag)
        test_pred_df = make_pred_df(test_df, y_test, test_pred_raw, "test", test_boundary, test_active, test_diag)

        train_m = regression_metrics(y_train, train_pred_raw, prefix="train_")
        train_m.update(screening_metrics(train_active.values, -train_pred_raw, prefix="train_"))
        val_m = regression_metrics(y_val, val_pred_raw, prefix="val_")
        val_m.update(screening_metrics(val_active.values, -val_pred_raw, prefix="val_"))
        test_m = regression_metrics(y_test, test_pred_raw, prefix="test_")
        test_m.update(screening_metrics(test_active.values, -test_pred_raw, prefix="test_"))
        diag_summary = diagnostic_summary_regression(test_pred_df)
        feature_type = f"Morgan_r{args.radius}_{args.n_bits}_Tanimoto"
        target_scale = "log10MIC"

    train_pred_df.to_csv(outdir / "predictions_train.csv", index=False)
    val_pred_df.to_csv(outdir / "predictions_validation.csv", index=False)
    test_pred_df.to_csv(outdir / "predictions_test.csv", index=False)

    row = {
        "model": model_name,
        "dataset_name": dataset_name,
        "genus": genus,
        "task": task,
        "split_type": split_type,
        "seed": seed,
        "threshold_from_dataset_name": threshold_from_name,
        "threshold_for_boundary": threshold_for_boundary,
        "threshold_mode": "fixed_0.5" if task == "classification" else None,
        "threshold_used": args.classification_threshold if task == "classification" else np.nan,
        "val_best_f1_threshold": args.classification_threshold if task == "classification" else np.nan,
        "train_seconds": train_seconds,
        "feature_type": feature_type,
        "scale_pos_weight": (float(np.sum(y_train == 0) / max(np.sum(y_train == 1), 1)) if task == "classification" else np.nan),
        "target_scale": target_scale,
        "smiles_col": smiles_col,
        "target_col": target_col,
        "target_source_type": target_source_type,
        "k": k,
        "weighting": args.weighting,
        "radius": args.radius,
        "n_bits": args.n_bits,
        "train_invalid_smiles_count": int(sum(fp is None for fp in train_fps)),
        "validation_invalid_smiles_count": int(sum(fp is None for fp in val_fps)),
        "test_invalid_smiles_count": int(sum(fp is None for fp in test_fps)),
    }
    row.update(train_m)
    row.update(val_m)
    row.update(test_m)
    row.update(diag_summary)

    pd.DataFrame([row]).to_csv(outdir / "metrics_full.csv", index=False)
    pd.DataFrame([diag_summary]).to_csv(outdir / "diagnostics_metrics.csv", index=False)

    subgroup_rows = []
    if task == "classification":
        group_specs = [("boundary_flag", "MIC_boundary"), ("nn_bin", "NN_similarity"), ("mic_range_group", "MIC_range")]
        for group_col, group_type in group_specs:
            if group_col in test_pred_df.columns:
                for group_value, g in test_pred_df.groupby(group_col, dropna=False):
                    r = {"group_type": group_type, "group_col": group_col, "group_value": group_value, "n": int(len(g))}
                    r.update(subgroup_classification_metrics(g))
                    subgroup_rows.append(r)
    else:
        group_specs = [("boundary_flag", "MIC_boundary"), ("nn_bin", "NN_similarity"), ("mic_range_group", "MIC_range")]
        for group_col, group_type in group_specs:
            if group_col in test_pred_df.columns:
                for group_value, g in test_pred_df.groupby(group_col, dropna=False):
                    r = {"group_type": group_type, "group_col": group_col, "group_value": group_value, "n": int(len(g))}
                    r.update(subgroup_regression_metrics(g))
                    subgroup_rows.append(r)
        if "topk_target_sd" in test_pred_df.columns and test_pred_df["topk_target_sd"].notna().any():
            med = test_pred_df["topk_target_sd"].median()
            for label, g in [("low_or_equal_median", test_pred_df[test_pred_df["topk_target_sd"] <= med]), ("high_median", test_pred_df[test_pred_df["topk_target_sd"] > med])]:
                r = {"group_type": "local_log10MIC_sd", "group_col": "topk_target_sd", "group_value": label, "n": int(len(g))}
                r.update(subgroup_regression_metrics(g))
                subgroup_rows.append(r)
    if subgroup_rows:
        pd.DataFrame(subgroup_rows).to_csv(outdir / "diagnostic_subgroup_metrics.csv", index=False)
    else:
        pd.DataFrame().to_csv(outdir / "diagnostic_subgroup_metrics.csv", index=False)

    save_train_log(task, train_m, val_m, outdir)

    run_info = {
        "script": Path(__file__).name,
        "finish_time": now(),
        "data_root": str(args.data_root),
        "dataset_name": dataset_name,
        "genus": genus,
        "task": task,
        "seed_dir": str(run["seed_dir"]),
        "train_csv": str(run["train_csv"]),
        "validation_csv": str(run["val_csv"]),
        "test_csv": str(run["test_csv"]),
        "model": model_name,
        "k": k,
        "weighting": args.weighting,
        "feature_type": "Morgan fingerprint + Tanimoto similarity only; no RDKit descriptors",
        "radius": args.radius,
        "n_bits": args.n_bits,
        "smiles_col": smiles_col,
        "target_col": target_col,
        "target_source_type": target_source_type,
        "target_scale": target_scale,
        "threshold_for_boundary": threshold_for_boundary,
        "classification_threshold": args.classification_threshold,
        "train_seconds": train_seconds,
        "python": sys.version,
    }
    run_info = sanitize_metadata(run_info)
    with open(outdir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False, default=str)

    # This is not a learned model. Save reference targets and metadata for reproducibility.
    joblib.dump({
        "model": model_name,
        "k": k,
        "weighting": args.weighting,
        "radius": args.radius,
        "n_bits": args.n_bits,
        "train_smiles": train_df[smiles_col].astype(str).tolist(),
        "train_target": y_train.tolist(),
        "run_info": run_info,
    }, outdir / "model.joblib")

    if task == "classification":
        print(
            f"[{now()}] TEST metrics: ROC-AUC={row.get('test_roc_auc', np.nan):.4f}, "
            f"PR-AUC={row.get('test_pr_auc', np.nan):.4f}, F1={row.get('test_f1', np.nan):.4f}, "
            f"MCC={row.get('test_mcc', np.nan):.4f}",
            flush=True,
        )
    else:
        print(
            f"[{now()}] TEST metrics: RMSE={row.get('test_rmse', np.nan):.4f}, "
            f"MAE={row.get('test_mae', np.nan):.4f}, Spearman={row.get('test_spearman', np.nan):.4f}, "
            f"within1={row.get('test_within_1_dilution', np.nan):.4f}",
            flush=True,
        )
    print(f"[{now()}] Saved outputs to: {outdir.name}", flush=True)
    return row


# -----------------------------
# Aggregation
# -----------------------------


def metric_direction(metric: str) -> str:
    lower_terms = ["loss", "mae", "rmse", "mse", "medae", "brier", "fold_error", "error_rate", "max_abs_error"]
    if any(t in metric.lower() for t in lower_terms):
        return "lower"
    return "higher"


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

    group_cols = ["model", "task", "dataset_name", "genus", "split_type"]
    numeric_cols = [c for c in all_df.columns if c not in ["seed"] and pd.api.types.is_numeric_dtype(all_df[c])]
    if numeric_cols:
        agg = all_df.groupby(group_cols)[numeric_cols].agg(["mean", "std", "count"]).reset_index()
        agg.columns = ["_".join([str(x) for x in col if str(x)]) for col in agg.columns]
        for m in ["test_roc_auc", "test_pr_auc", "test_mcc", "test_f1", "test_rmse", "test_mae", "test_spearman", "test_within_1_dilution", "test_EF1%"]:
            mean_col, std_col = f"{m}_mean", f"{m}_std"
            if mean_col in agg.columns and std_col in agg.columns:
                agg[f"{m}_mean±sd"] = [
                    "" if pd.isna(a) else (f"{a:.4f}" if pd.isna(b) else f"{a:.4f} ± {b:.4f}")
                    for a, b in zip(agg[mean_col], agg[std_col])
                ]
        agg.to_csv(model_root / "aggregate_mean_sd.csv", index=False)

        mean_df = all_df.groupby(group_cols)[numeric_cols].mean().reset_index()
        rows = []
        for (model, task, dataset_name, genus), sub in mean_df.groupby(["model", "task", "dataset_name", "genus"]):
            random_row = sub[sub["split_type"] == "random"]
            if random_row.empty:
                continue
            random_row = random_row.iloc[0]
            for target_split in ["scaffold", "repeated_stratified_perimeter", "strict_perimeter", "perimeter", "max_distance"]:
                target_df = sub[sub["split_type"] == target_split]
                if target_df.empty:
                    continue
                target_row = target_df.iloc[0]
                row = {"model": model, "task": task, "dataset_name": dataset_name, "genus": genus, "reference_split": "random", "target_split": target_split}
                for m in numeric_cols:
                    if m in random_row.index and m in target_row.index:
                        if metric_direction(m) == "lower":
                            row[f"ood_increase_{m}"] = target_row[m] - random_row[m]
                        else:
                            row[f"ood_drop_{m}"] = random_row[m] - target_row[m]
                rows.append(row)
        if rows:
            pd.DataFrame(rows).to_csv(model_root / "ood_change_random_vs_ood_splits.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Run AntibioticBench 1-NN/5-NN similarity baselines for classification and/or regression.")
    parser.add_argument("--data-root", required=True, help="Root directory containing dataset folders")
    parser.add_argument("--outdir", default=str(Path(__file__).resolve().parents[2] / "outputs" / "ML" / "similarity"), help="Output root directory")
    parser.add_argument("--task", choices=["classification", "regression", "both"], default="both")
    parser.add_argument("--models", nargs="+", default=["1nn", "5nn"], help="Similarity baselines to run: 1nn 5nn")
    parser.add_argument("--split-types", nargs="+", default=["random", "scaffold", "repeated_stratified_perimeter", "strict_perimeter"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--datasets", nargs="+", default=None, help="Optional dataset folder names or genus names to include")
    parser.add_argument("--classification-target-col", default=None, help="Default auto-detects active")
    parser.add_argument("--regression-target-col", default=None, help="Default auto-detects log10_mic")
    parser.add_argument("--value-col", default=None, help="Optional MIC value column for boundary/screening/log10MIC fallback")
    parser.add_argument("--active-col", default=None, help="Optional active column for regression screening metrics")
    parser.add_argument("--threshold", type=float, default=None, help="Override MIC threshold for boundary/screening active inference")
    parser.add_argument("--classification-threshold", type=float, default=0.5)
    parser.add_argument("--weighting", choices=["similarity", "uniform"], default="similarity")
    parser.add_argument("--radius", type=int, default=RADIUS_DEFAULT)
    parser.add_argument("--n-bits", type=int, default=N_BITS_DEFAULT)
    parser.add_argument("--max-runs", type=int, default=None, help="Debug option: max number of split runs per task")
    args = parser.parse_args()

    args.data_root = Path(args.data_root).resolve()
    args.outdir = Path(args.outdir).resolve()
    ensure_dir(args.outdir)

    tasks = ["classification", "regression"] if args.task == "both" else [args.task]
    all_rows = []
    failed = []
    for task in tasks:
        runs = discover_runs(args.data_root, task, args.split_types, args.seeds, args.datasets)
        if args.max_runs is not None:
            runs = runs[:args.max_runs]
        print(f"[{now()}] Discovered {len(runs)} {task} split runs.", flush=True)
        if not runs:
            print(f"[{now()}] WARNING: No {task} runs found.", flush=True)
            continue
        for model_name in args.models:
            for run in runs:
                try:
                    row = run_one(run, model_name, args)
                    if row is not None:
                        all_rows.append(row)
                except Exception as e:
                    print(f"[{now()}] ERROR in {task} {model_name} run: {type(e).__name__}", flush=True)
                    failed.append(sanitize_metadata({**{k: str(v) for k, v in run.items()}, "model": model_name, "error": type(e).__name__}))
                    fail_dir = args.outdir / model_name.lower().replace("-", "") / "_failed_runs"
                    ensure_dir(fail_dir)
                    pd.DataFrame(failed).to_csv(fail_dir / "failed_runs.csv", index=False)
                    if args.max_runs:
                        raise SystemExit(1) from None
                    continue
            aggregate_results(args.outdir, model_name.lower().replace("-", ""))

    if all_rows:
        pd.DataFrame(all_rows).to_csv(args.outdir / "all_run_metrics_from_current_invocation.csv", index=False)
    if failed:
        pd.DataFrame(failed).to_csv(args.outdir / "failed_runs.csv", index=False)
        print(f"[{now()}] Finished with {len(failed)} failed runs. See failed_runs.csv", flush=True)
    else:
        print(f"[{now()}] Finished all similarity baseline runs successfully.", flush=True)


if __name__ == "__main__":
    main()
