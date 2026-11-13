#!/usr/bin/env python3
"""Shared metrics, diagnostics, run enumeration, and output helpers for MoLFormer AntibioticBench."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
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

SPLIT_TYPES: Tuple[str, ...] = (
    "random",
    "scaffold",
    "repeated_stratified_perimeter",
    "strict_perimeter",
)
STANDARD_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
STRICT_PERIMETER_SEEDS: Tuple[int, ...] = (0,)

SMILES_CANDIDATES = ("smiles", "SMILES", "canonical_smiles", "smiles_std")
CLASS_TARGET_CANDIDATES = ("active", "label", "target", "y")
REG_TARGET_CANDIDATES = (
    "log10_mic",
    "log10MIC",
    "log10_mic_value",
    "log10_mic_numeric",
    "log10_value",
)
MIC_CANDIDATES = (
    "mic_numeric",
    "MIC",
    "mic",
    "value",
    "mic_value",
    "MIC_numeric",
)
LOG10_MIC_CANDIDATES = (
    "log10_mic",
    "log10MIC",
    "log10_mic_for_diagnosis",
    "true_log10MIC",
)
LOG2_MIC_CANDIDATES = ("log2_mic", "log2MIC", "log2_mic_for_diagnosis")


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if not math.isfinite(value) else value
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=json_default)
    tmp.replace(path)


def sanitize_name(value: str, max_len: int = 100) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.-")
    if len(value) <= max_len:
        return value
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    return f"{value[:max_len-11]}_{digest}"


def sha256_files(paths: Sequence[Path], extra: str = "") -> str:
    h = hashlib.sha256()
    for path in paths:
        h.update(str(path.resolve()).encode("utf-8"))
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    h.update(extra.encode("utf-8"))
    return h.hexdigest()


def find_column(df: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    direct = {str(c): c for c in df.columns}
    lower = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in direct:
            return str(direct[candidate])
        if candidate.lower() in lower:
            return str(lower[candidate.lower()])
    if required:
        raise ValueError(
            f"None of the expected columns {list(candidates)} were found. Available columns: {list(df.columns)}"
        )
    return None


def coerce_binary(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        out = series.astype(float).to_numpy()
    else:
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() == len(series):
            out = numeric.to_numpy(dtype=float)
        else:
            mapping = {
                "active": 1.0,
                "inactive": 0.0,
                "positive": 1.0,
                "negative": 0.0,
                "true": 1.0,
                "false": 0.0,
                "yes": 1.0,
                "no": 0.0,
                "1": 1.0,
                "0": 0.0,
            }
            out = series.astype(str).str.strip().str.lower().map(mapping).to_numpy(dtype=float)
    finite = out[np.isfinite(out)]
    unique = set(np.unique(finite).tolist())
    if not unique.issubset({0.0, 1.0}):
        raise ValueError(f"Classification target must be binary 0/1; observed finite values: {sorted(unique)[:20]}")
    return out


def safe_float(value: Any) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else float("nan")
    except Exception:
        return float("nan")


def safe_div(num: float, den: float) -> float:
    if den is None or not math.isfinite(float(den)) or float(den) == 0:
        return float("nan")
    return float(num) / float(den)


def safe_corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x2, y2 = x[mask], y[mask]
    if len(x2) < 2 or np.nanstd(x2) == 0 or np.nanstd(y2) == 0:
        return float("nan")
    try:
        if method == "pearson":
            return float(np.corrcoef(x2, y2)[0, 1])
        if method == "spearman":
            from scipy.stats import spearmanr

            return float(spearmanr(x2, y2).statistic)
    except Exception:
        return float("nan")
    raise ValueError(method)


def screening_metrics(
    true_active: np.ndarray,
    scores: np.ndarray,
    top_percents: Sequence[int] = (1, 5, 10),
    top_ks: Sequence[int] = (50, 100, 500),
) -> Dict[str, float]:
    y = np.asarray(true_active, dtype=float)
    s = np.asarray(scores, dtype=float)
    mask = np.isfinite(y) & np.isfinite(s)
    y, s = y[mask], s[mask]
    n = len(y)
    out: Dict[str, float] = {}
    if n == 0:
        for pct in top_percents:
            out.update(
                {
                    f"top_{pct}pct_n": 0,
                    f"top_{pct}pct_hits": 0,
                    f"Hit_rate@Top{pct}%": float("nan"),
                    f"EF{pct}%": float("nan"),
                }
            )
        for k in top_ks:
            out[f"Top{k}_hits"] = 0
            out[f"Recall@Top{k}"] = float("nan")
        return out

    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    base_rate = float(np.mean(y))
    total_pos = float(np.sum(y == 1))
    for pct in top_percents:
        n_top = max(1, int(math.ceil(n * pct / 100.0)))
        hits = int(np.sum(y_sorted[:n_top] == 1))
        hit_rate = hits / n_top
        out[f"top_{pct}pct_n"] = int(n_top)
        out[f"top_{pct}pct_hits"] = hits
        out[f"Hit_rate@Top{pct}%"] = hit_rate
        out[f"EF{pct}%"] = safe_div(hit_rate, base_rate)
    for k in top_ks:
        kk = min(int(k), n)
        hits = int(np.sum(y_sorted[:kk] == 1))
        out[f"Top{k}_hits"] = hits
        out[f"Recall@Top{k}"] = safe_div(hits, total_pos)
    return out


def classification_metrics(y_true: np.ndarray, pred_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(pred_prob, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask].astype(int), p[mask]
    pred = (p >= threshold).astype(int)
    out: Dict[str, float] = {
        "n": int(len(y)),
        "pos": int(np.sum(y == 1)),
        "neg": int(np.sum(y == 0)),
        "pos_ratio": float(np.mean(y)) if len(y) else float("nan"),
    }
    if len(y) == 0:
        metric_names = (
            "roc_auc",
            "pr_auc",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "mcc",
            "specificity",
            "sensitivity",
            "log_loss",
            "brier",
        )
        out.update({name: float("nan") for name in metric_names})
        out.update(screening_metrics(y, p))
        return out

    out["accuracy"] = float(accuracy_score(y, pred))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) == 2 else float("nan")
    out["precision"] = float(precision_score(y, pred, zero_division=0))
    out["recall"] = float(recall_score(y, pred, zero_division=0))
    out["f1"] = float(f1_score(y, pred, zero_division=0))
    out["mcc"] = float(matthews_corrcoef(y, pred)) if len(np.unique(y)) == 2 else float("nan")
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    tp = int(np.sum((y == 1) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    out["specificity"] = safe_div(tn, tn + fp)
    out["sensitivity"] = safe_div(tp, tp + fn)
    if len(np.unique(y)) == 2:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    out["log_loss"] = float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]))
    out["brier"] = float(brier_score_loss(y, p))
    out.update(screening_metrics(y, p))
    return out


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold_for_screening: Optional[float],
) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    out: Dict[str, float] = {"n": int(len(y))}
    if len(y) == 0:
        for name in (
            "mae",
            "mse",
            "rmse",
            "medae",
            "r2",
            "pearson",
            "spearman",
            "within_1_dilution",
            "within_2_dilutions",
            "median_fold_error",
            "mean_fold_error",
            "max_abs_error_log10",
        ):
            out[name] = float("nan")
        out.update(screening_metrics(np.array([]), np.array([])))
        return out

    err = p - y
    abs_err = np.abs(err)
    sq_err = err**2
    out["mae"] = float(np.mean(abs_err))
    out["mse"] = float(np.mean(sq_err))
    out["rmse"] = float(np.sqrt(np.mean(sq_err)))
    out["medae"] = float(np.median(abs_err))
    out["r2"] = float(r2_score(y, p)) if len(y) >= 2 and np.nanstd(y) > 0 else float("nan")
    out["pearson"] = safe_corr(y, p, "pearson")
    out["spearman"] = safe_corr(y, p, "spearman")
    out["within_1_dilution"] = float(np.mean(abs_err <= math.log10(2)))
    out["within_2_dilutions"] = float(np.mean(abs_err <= math.log10(4)))
    fold_error = np.power(10.0, abs_err)
    out["median_fold_error"] = float(np.median(fold_error))
    out["mean_fold_error"] = float(np.mean(fold_error))
    out["max_abs_error_log10"] = float(np.max(abs_err))
    if threshold_for_screening is not None and math.isfinite(float(threshold_for_screening)):
        true_active = (np.power(10.0, y) <= float(threshold_for_screening)).astype(int)
        out.update(screening_metrics(true_active, -p))
    else:
        out.update(screening_metrics(np.full(len(y), np.nan), -p))
    return out


def prefix_metrics(metrics: Mapping[str, Any], prefix: str) -> Dict[str, Any]:
    return {f"{prefix}_{k}": v for k, v in metrics.items()}


def best_f1_threshold(y_true: np.ndarray, pred_prob: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(pred_prob, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask].astype(int), p[mask]
    if len(y) == 0:
        return 0.5
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], p)))
    best_t, best = 0.5, -1.0
    for t in candidates:
        score = f1_score(y, (p >= t).astype(int), zero_division=0)
        if score > best + 1e-12 or (abs(score - best) <= 1e-12 and abs(t - 0.5) < abs(best_t - 0.5)):
            best, best_t = score, float(t)
    return best_t


def extract_mic_arrays(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(df)
    mic = np.full(n, np.nan, dtype=float)
    log10mic = np.full(n, np.nan, dtype=float)
    log2mic = np.full(n, np.nan, dtype=float)

    col = find_column(df, MIC_CANDIDATES, required=False)
    if col is not None:
        mic = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        mic[mic <= 0] = np.nan
    col10 = find_column(df, LOG10_MIC_CANDIDATES, required=False)
    if col10 is not None:
        log10mic = pd.to_numeric(df[col10], errors="coerce").to_numpy(dtype=float)
    col2 = find_column(df, LOG2_MIC_CANDIDATES, required=False)
    if col2 is not None:
        log2mic = pd.to_numeric(df[col2], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(mic) & ~np.isfinite(log10mic)
    log10mic[mask] = np.log10(mic[mask])
    mask = np.isfinite(mic) & ~np.isfinite(log2mic)
    log2mic[mask] = np.log2(mic[mask])
    mask = np.isfinite(log10mic) & ~np.isfinite(mic)
    mic[mask] = np.power(10.0, log10mic[mask])
    mask = np.isfinite(log10mic) & ~np.isfinite(log2mic)
    log2mic[mask] = log10mic[mask] / math.log10(2.0)
    mask = np.isfinite(log2mic) & ~np.isfinite(log10mic)
    log10mic[mask] = log2mic[mask] * math.log10(2.0)
    mask = np.isfinite(log2mic) & ~np.isfinite(mic)
    mic[mask] = np.power(2.0, log2mic[mask])
    return mic, log10mic, log2mic


def mic_range_fields(mic: np.ndarray, threshold: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
    n = len(mic)
    boundary = np.full(n, np.nan, dtype=float)
    group = np.full(n, None, dtype=object)
    if threshold is None or not math.isfinite(float(threshold)) or float(threshold) <= 0:
        return boundary, group
    t = float(threshold)
    valid = np.isfinite(mic) & (mic > 0)
    boundary[valid] = ((mic[valid] > t / 2.0) & (mic[valid] <= 2.0 * t)).astype(float)
    group[valid & (mic <= t / 2.0)] = "strong_MIC_le_T_over_2"
    group[valid & (mic > t / 2.0) & (mic <= 2.0 * t)] = "boundary_T_over_2_to_2T"
    group[valid & (mic > 2.0 * t)] = "weak_MIC_gt_2T"
    return boundary, group


def make_fingerprints(smiles: Sequence[str], radius: int = 2, n_bits: int = 2048):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    fps = []
    valid_rows = []
    invalid_rows = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            invalid_rows.append(i)
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        fps.append(fp)
        valid_rows.append(i)
    return fps, np.asarray(valid_rows, dtype=int), invalid_rows


def add_test_nn_and_activity_cliff(
    task: str,
    train_pred: pd.DataFrame,
    test_pred: pd.DataFrame,
    smiles_col: str = "smiles",
    fp_radius: int = 2,
    fp_bits: int = 2048,
    nn_hi: float = 0.7,
    nn_mid: float = 0.4,
    ac_similarity_threshold: float = 0.7,
    ac_log10_diff_threshold: float = math.log10(16.0),
    local_k: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Populate NN and test-vs-train activity-cliff fields; train is the sole reference library."""
    from rdkit import DataStructs

    result = test_pred.copy()
    train_smiles = train_pred[smiles_col].astype(str).tolist()
    test_smiles = result[smiles_col].astype(str).tolist()
    train_fps, train_valid_rows, train_invalid = make_fingerprints(train_smiles, fp_radius, fp_bits)
    test_fps, test_valid_rows, test_invalid = make_fingerprints(test_smiles, fp_radius, fp_bits)

    n = len(result)
    nn_train_row = np.full(n, np.nan)
    nn_sim = np.full(n, np.nan)
    nn_bin = np.full(n, None, dtype=object)
    nn_train_smiles = np.full(n, None, dtype=object)
    ac_flag = np.zeros(n, dtype=int)
    ac_count = np.zeros(n, dtype=int)
    ac_pair_count = np.zeros(n, dtype=int)
    ac_direction_acc = np.full(n, np.nan)

    if task == "classification":
        train_true = pd.to_numeric(train_pred["true_active"], errors="coerce").to_numpy(dtype=float)
        test_true = pd.to_numeric(result["true_active"], errors="coerce").to_numpy(dtype=float)
        train_model_value = pd.to_numeric(train_pred["pred_score"], errors="coerce").to_numpy(dtype=float)
        nn_train_value_name = "nn_train_active"
        nn_train_pred_name = "nn_train_score"
        train_mic, train_log10mic, _ = extract_mic_arrays(train_pred)
        test_mic, test_log10mic, _ = extract_mic_arrays(result)
        use_mic_ac = bool(np.isfinite(train_log10mic).any() and np.isfinite(test_log10mic).any())
        ac_max_diff = np.full(n, np.nan)
    else:
        train_true = pd.to_numeric(train_pred["true_log10MIC"], errors="coerce").to_numpy(dtype=float)
        test_true = pd.to_numeric(result["true_log10MIC"], errors="coerce").to_numpy(dtype=float)
        train_model_value = pd.to_numeric(train_pred["pred_log10MIC"], errors="coerce").to_numpy(dtype=float)
        nn_train_value_name = "nn_train_log10mic"
        nn_train_pred_name = "nn_train_pred_log10mic"
        train_log10mic = train_true.copy()
        test_log10mic = test_true.copy()
        use_mic_ac = True
        ac_max_diff = np.full(n, np.nan)
        local_var = np.full(n, np.nan)
        local_sd = np.full(n, np.nan)

    nn_train_true = np.full(n, np.nan)
    nn_train_pred = np.full(n, np.nan)

    test_fp_by_row = {int(row): fp for row, fp in zip(test_valid_rows, test_fps)}
    for test_i, fp in test_fp_by_row.items():
        if not train_fps:
            continue
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, train_fps), dtype=float)
        best_valid_pos = int(np.argmax(sims))
        best_train_i = int(train_valid_rows[best_valid_pos])
        best_sim = float(sims[best_valid_pos])
        nn_train_row[test_i] = best_train_i
        nn_sim[test_i] = best_sim
        nn_train_smiles[test_i] = train_smiles[best_train_i]
        nn_train_true[test_i] = train_true[best_train_i]
        nn_train_pred[test_i] = train_model_value[best_train_i]
        if best_sim >= nn_hi:
            nn_bin[test_i] = ">=0.7"
        elif best_sim >= nn_mid:
            nn_bin[test_i] = "0.4-0.7"
        else:
            nn_bin[test_i] = "<0.4"

        if task == "regression":
            valid_true_positions = np.array(
                [j for j, row in enumerate(train_valid_rows) if np.isfinite(train_true[int(row)])], dtype=int
            )
            if len(valid_true_positions):
                order = valid_true_positions[np.argsort(-sims[valid_true_positions])]
                chosen = order[: min(local_k, len(order))]
                values = np.array([train_true[int(train_valid_rows[j])] for j in chosen], dtype=float)
                if len(values):
                    local_var[test_i] = float(np.var(values, ddof=0))
                    local_sd[test_i] = float(np.std(values, ddof=0))

        candidate_positions = np.where(sims >= ac_similarity_threshold)[0]
        corrects: List[float] = []
        diffs: List[float] = []
        for pos in candidate_positions:
            tr_i = int(train_valid_rows[int(pos)])
            if use_mic_ac:
                if not (np.isfinite(test_log10mic[test_i]) and np.isfinite(train_log10mic[tr_i])):
                    continue
                diff = float(test_log10mic[test_i] - train_log10mic[tr_i])
                if abs(diff) < ac_log10_diff_threshold:
                    continue
                pred_diff = float(
                    (result.iloc[test_i]["pred_log10MIC"] if task == "regression" else result.iloc[test_i]["pred_score"])
                    - train_model_value[tr_i]
                )
                if task == "classification":
                    # Larger score means more active/lower MIC, so the expected direction is opposite to logMIC.
                    correct = float(np.sign(pred_diff) == -np.sign(diff)) if pred_diff != 0 else 0.0
                else:
                    correct = float(np.sign(pred_diff) == np.sign(diff)) if pred_diff != 0 else 0.0
                diffs.append(abs(diff))
                corrects.append(correct)
            else:
                if not (np.isfinite(test_true[test_i]) and np.isfinite(train_true[tr_i])):
                    continue
                if test_true[test_i] == train_true[tr_i]:
                    continue
                true_diff = float(test_true[test_i] - train_true[tr_i])
                pred_diff = float(result.iloc[test_i]["pred_score"] - train_model_value[tr_i])
                corrects.append(float(np.sign(pred_diff) == np.sign(true_diff)) if pred_diff != 0 else 0.0)
                diffs.append(float(abs(true_diff)))
        if corrects:
            ac_flag[test_i] = 1
            ac_count[test_i] = len(corrects)
            ac_pair_count[test_i] = len(corrects)
            ac_direction_acc[test_i] = float(np.mean(corrects))
            ac_max_diff[test_i] = float(np.max(diffs))

    result["nn_train_row"] = nn_train_row
    result["nn_sim"] = nn_sim
    result["nn_bin"] = nn_bin
    result["nn_train_smiles"] = nn_train_smiles
    result[nn_train_value_name] = nn_train_true
    result[nn_train_pred_name] = nn_train_pred
    result["ac_flag"] = ac_flag
    result["ac_neighbor_count"] = ac_count
    result["ac_pair_directional_accuracy_for_molecule"] = ac_direction_acc
    result["ac_pair_count_for_molecule"] = ac_pair_count
    if task == "classification":
        result["ac_max_abs_log2mic_diff"] = ac_max_diff / math.log10(2.0) if use_mic_ac else np.nan
    else:
        result["ac_max_abs_log10mic_diff"] = ac_max_diff
        result["local_train_log10mic_variance_k10"] = local_var
        result["local_train_log10mic_sd_k10"] = local_sd

    meta = {
        "nn_reference": "train",
        "fingerprint": f"Morgan radius={fp_radius}, nBits={fp_bits}",
        "nn_bins": [">=0.7", "0.4-0.7", "<0.4"],
        "activity_cliff_similarity_threshold": ac_similarity_threshold,
        "activity_cliff_log10mic_difference_threshold": ac_log10_diff_threshold,
        "activity_cliff_fold_difference_threshold": float(10**ac_log10_diff_threshold),
        "activity_cliff_method": "MIC_difference" if use_mic_ac else "opposite_binary_label_fallback",
        "invalid_train_smiles_count": len(train_invalid),
        "invalid_test_smiles_count": len(test_invalid),
    }
    return result, meta


def _classification_subgroup_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    return classification_metrics(
        pd.to_numeric(df["true_active"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(df["pred_score"], errors="coerce").to_numpy(dtype=float),
        threshold=0.5,
    )


def _regression_subgroup_metrics(df: pd.DataFrame, threshold: Optional[float]) -> Dict[str, Any]:
    return regression_metrics(
        pd.to_numeric(df["true_log10MIC"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(df["pred_log10MIC"], errors="coerce").to_numpy(dtype=float),
        threshold_for_screening=threshold,
    )


def build_test_diagnostics(
    task: str,
    test_df: pd.DataFrame,
    identity: Mapping[str, Any],
    threshold_for_boundary: Optional[float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create one-row wide diagnostics and a long subgroup table from test predictions only."""
    wide: Dict[str, Any] = dict(identity)
    subgroup_rows: List[Dict[str, Any]] = []

    def add_group(group_type: str, group_col: str, group_value: Any, subset: pd.DataFrame) -> Dict[str, Any]:
        if task == "classification":
            m = _classification_subgroup_metrics(subset)
        else:
            m = _regression_subgroup_metrics(subset, threshold_for_boundary)
        row = dict(identity)
        row.update({"group_type": group_type, "group_col": group_col, "group_value": group_value})
        row.update(m)
        subgroup_rows.append(row)
        return m

    # MIC boundary
    if "boundary_flag" in test_df:
        b = test_df[pd.to_numeric(test_df["boundary_flag"], errors="coerce") == 1]
        nb = test_df[pd.to_numeric(test_df["boundary_flag"], errors="coerce") == 0]
        bm, nbm = add_group("MIC_boundary", "boundary_flag", 1, b), add_group(
            "MIC_boundary", "boundary_flag", 0, nb
        )
        if task == "classification":
            wide.update(
                {
                    "boundary_n": bm.get("n"),
                    "non_boundary_n": nbm.get("n"),
                    "boundary_error_rate": 1 - bm.get("accuracy", np.nan),
                    "non_boundary_error_rate": 1 - nbm.get("accuracy", np.nan),
                }
            )
            wide["boundary_error_enrichment"] = safe_div(
                wide["boundary_error_rate"], wide["non_boundary_error_rate"]
            )
        else:
            wide.update(
                {
                    "boundary_n": bm.get("n"),
                    "non_boundary_n": nbm.get("n"),
                    "boundary_mae": bm.get("mae"),
                    "non_boundary_mae": nbm.get("mae"),
                    "boundary_rmse": bm.get("rmse"),
                    "non_boundary_rmse": nbm.get("rmse"),
                    "boundary_within_1_dilution": bm.get("within_1_dilution"),
                    "non_boundary_within_1_dilution": nbm.get("within_1_dilution"),
                }
            )
            wide["boundary_regression_error_enrichment"] = safe_div(
                wide["boundary_mae"], wide["non_boundary_mae"]
            )

    # NN bins
    nn_prefixes = {">=0.7": "nn_ge_0p7", "0.4-0.7": "nn_0p4_0p7", "<0.4": "nn_lt_0p4"}
    if "nn_bin" in test_df:
        for value, prefix in nn_prefixes.items():
            sub = test_df[test_df["nn_bin"] == value]
            m = add_group("NN_similarity", "nn_bin", value, sub)
            for k, v in m.items():
                wide[f"{prefix}_{k}"] = v
            if task == "classification":
                wide[f"{prefix}_error_rate"] = 1 - m.get("accuracy", np.nan)

    # Activity cliff
    if "ac_flag" in test_df:
        ac = test_df[pd.to_numeric(test_df["ac_flag"], errors="coerce") == 1]
        nonac = test_df[pd.to_numeric(test_df["ac_flag"], errors="coerce") == 0]
        am, nam = add_group("Activity_cliff", "ac_flag", 1, ac), add_group(
            "Activity_cliff", "ac_flag", 0, nonac
        )
        pair_total = int(pd.to_numeric(test_df.get("ac_pair_count_for_molecule", 0), errors="coerce").fillna(0).sum())
        pair_acc = pd.to_numeric(
            test_df.get("ac_pair_directional_accuracy_for_molecule", np.nan), errors="coerce"
        ).mean()
        if task == "classification":
            wide.update(
                {
                    "ac_n": am.get("n"),
                    "non_ac_n": nam.get("n"),
                    "ac_error_rate": 1 - am.get("accuracy", np.nan),
                    "non_ac_error_rate": 1 - nam.get("accuracy", np.nan),
                    "ac_subgroup_pr_auc": am.get("pr_auc"),
                    "ac_pair_directional_accuracy": pair_acc,
                    "ac_pair_total_count": pair_total,
                }
            )
            wide["ac_error_enrichment"] = safe_div(wide["ac_error_rate"], wide["non_ac_error_rate"])
        else:
            wide.update(
                {
                    "ac_n": am.get("n"),
                    "non_ac_n": nam.get("n"),
                    "ac_mae": am.get("mae"),
                    "non_ac_mae": nam.get("mae"),
                    "ac_rmse": am.get("rmse"),
                    "non_ac_rmse": nam.get("rmse"),
                    "ac_pair_directional_accuracy": pair_acc,
                    "ac_pair_total_count": pair_total,
                }
            )
            wide["ac_error_enrichment"] = safe_div(wide["ac_mae"], wide["non_ac_mae"])

    # MIC range
    if "mic_range_group" in test_df:
        for value in (
            "strong_MIC_le_T_over_2",
            "boundary_T_over_2_to_2T",
            "weak_MIC_gt_2T",
        ):
            sub = test_df[test_df["mic_range_group"] == value]
            m = add_group("MIC_range", "mic_range_group", value, sub)
            prefix = "range_" + value
            for k, v in m.items():
                wide[f"{prefix}_{k}"] = v

    # Regression local variance
    if task == "regression" and "local_train_log10mic_variance_k10" in test_df:
        values = pd.to_numeric(test_df["local_train_log10mic_variance_k10"], errors="coerce")
        median = float(values.median()) if values.notna().any() else float("nan")
        local_group = np.where(values >= median, "high_local_variance", "low_local_variance")
        local_group[values.isna().to_numpy()] = "missing"
        work = test_df.copy()
        work["local_variance_group"] = local_group
        high = work[work["local_variance_group"] == "high_local_variance"]
        low = work[work["local_variance_group"] == "low_local_variance"]
        hm = add_group("Local_variance", "local_variance_group", "high_local_variance", high)
        lm = add_group("Local_variance", "local_variance_group", "low_local_variance", low)
        wide.update(
            {
                "mean_local_train_log10mic_variance_k10": float(values.mean()) if values.notna().any() else np.nan,
                "median_local_train_log10mic_variance_k10": median,
                "high_local_variance_mae": hm.get("mae"),
                "low_local_variance_mae": lm.get("mae"),
                "local_variance_error_enrichment": safe_div(hm.get("mae", np.nan), lm.get("mae", np.nan)),
            }
        )

    return pd.DataFrame([wide]), pd.DataFrame(subgroup_rows)


def package_versions() -> Dict[str, str]:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "pandas", "sklearn", "torch", "rdkit", "transformers"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception as e:
            versions[name] = f"unavailable: {e}"
    return versions


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    except Exception:
        pass


def load_dataset_config(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"dataset_name", "genus", "threshold_for_boundary"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"dataset config missing columns: {sorted(missing)}")
    if df["dataset_name"].duplicated().any():
        dup = df.loc[df["dataset_name"].duplicated(), "dataset_name"].tolist()
        raise ValueError(f"duplicate dataset_name values in config: {dup}")
    return df


def enumerate_runs(config_df: pd.DataFrame, task: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    index = 0
    for _, ds in config_df.iterrows():
        for split_type in SPLIT_TYPES:
            seeds = STRICT_PERIMETER_SEEDS if split_type == "strict_perimeter" else STANDARD_SEEDS
            for seed in seeds:
                rows.append(
                    {
                        "run_index": index,
                        "dataset_name": str(ds["dataset_name"]),
                        "genus": str(ds["genus"]),
                        "threshold_for_boundary": float(ds["threshold_for_boundary"]),
                        "task": task,
                        "split_type": split_type,
                        "seed": int(seed),
                    }
                )
                index += 1
    return rows


def validate_input_run(data_root: Path, run: Mapping[str, Any]) -> Dict[str, Path]:
    base = data_root / str(run["dataset_name"]) / str(run["task"]) / str(run["split_type"]) / f"seed_{run['seed']}"
    paths = {name: base / f"{name}.csv" for name in ("train", "validation", "test")}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing input files for run {run}: {missing}")
    return paths


def reorder_prediction_columns(df: pd.DataFrame, task: str) -> pd.DataFrame:
    if task == "classification":
        first = [
            "split",
            "dataset_name",
            "genus",
            "task",
            "split_type",
            "seed",
            "model",
            "model_family",
            "smiles",
            "active",
            "true_active",
            "pred_score",
            "pred_prob",
            "pred_label",
            "pred_label_0p5",
            "threshold_used",
            "val_best_f1_threshold",
            "error",
            "boundary_flag",
            "threshold_for_boundary",
            "mic_range_group",
            "log2_mic_for_diagnosis",
            "log10_mic_for_diagnosis",
            "mic_numeric",
            "nn_train_row",
            "nn_sim",
            "nn_bin",
            "nn_train_smiles",
            "nn_train_active",
            "nn_train_score",
            "ac_flag",
            "ac_neighbor_count",
            "ac_max_abs_log2mic_diff",
            "ac_pair_directional_accuracy_for_molecule",
            "ac_pair_count_for_molecule",
        ]
    else:
        first = [
            "split",
            "dataset_name",
            "genus",
            "task",
            "split_type",
            "seed",
            "model",
            "model_family",
            "smiles",
            "log10_mic",
            "true_log10MIC",
            "true_MIC",
            "mic_numeric",
            "pred_log10MIC",
            "pred_MIC",
            "score_reg",
            "abs_error_log10",
            "sq_error_log10",
            "fold_error",
            "within_1_dilution",
            "within_2_dilutions",
            "true_active_for_screening",
            "threshold_for_boundary",
            "boundary_flag",
            "mic_range_group",
            "nn_train_row",
            "nn_sim",
            "nn_bin",
            "nn_train_smiles",
            "nn_train_log10mic",
            "nn_train_pred_log10mic",
            "local_train_log10mic_variance_k10",
            "local_train_log10mic_sd_k10",
            "ac_flag",
            "ac_neighbor_count",
            "ac_max_abs_log10mic_diff",
            "ac_pair_directional_accuracy_for_molecule",
            "ac_pair_count_for_molecule",
        ]
    existing = [c for c in first if c in df.columns]
    remaining = [c for c in df.columns if c not in existing]
    return df[existing + remaining]


def write_train_log_files(run_dir: Path, history: pd.DataFrame) -> None:
    history.to_csv(run_dir / "train_log.csv", index=False)
    history.to_csv(run_dir / "training_curve.csv", index=False)
    id_cols = [c for c in ("iteration", "epoch", "note") if c in history.columns]
    value_cols = [c for c in history.columns if c not in id_cols]
    long_df = history.melt(id_vars=id_cols, value_vars=value_cols, var_name="metric_full", value_name="value")
    long_df["split"] = np.select(
        [long_df["metric_full"].str.startswith("train_"), long_df["metric_full"].str.startswith("val_")],
        ["train", "validation"],
        default="global",
    )
    long_df["metric"] = long_df["metric_full"].str.replace(r"^(train_|val_)", "", regex=True)
    ordered = [c for c in ("iteration", "epoch", "split", "metric", "value", "metric_full", "note") if c in long_df]
    long_df[ordered].to_csv(run_dir / "train_log_long.csv", index=False)


def save_split_metrics(run_dir: Path, split_metrics: Mapping[str, Mapping[str, Any]]) -> None:
    for split, metrics in split_metrics.items():
        pd.DataFrame([metrics]).to_csv(run_dir / f"metrics_{split}.csv", index=False)


def make_empty_diagnostic_columns(df: pd.DataFrame, task: str) -> pd.DataFrame:
    out = df.copy()
    if task == "classification":
        cols = {
            "nn_train_row": np.nan,
            "nn_sim": np.nan,
            "nn_bin": None,
            "nn_train_smiles": None,
            "nn_train_active": np.nan,
            "nn_train_score": np.nan,
            "ac_flag": np.nan,
            "ac_neighbor_count": np.nan,
            "ac_max_abs_log2mic_diff": np.nan,
            "ac_pair_directional_accuracy_for_molecule": np.nan,
            "ac_pair_count_for_molecule": np.nan,
        }
    else:
        cols = {
            "nn_train_row": np.nan,
            "nn_sim": np.nan,
            "nn_bin": None,
            "nn_train_smiles": None,
            "nn_train_log10mic": np.nan,
            "nn_train_pred_log10mic": np.nan,
            "local_train_log10mic_variance_k10": np.nan,
            "local_train_log10mic_sd_k10": np.nan,
            "ac_flag": np.nan,
            "ac_neighbor_count": np.nan,
            "ac_max_abs_log10mic_diff": np.nan,
            "ac_pair_directional_accuracy_for_molecule": np.nan,
            "ac_pair_count_for_molecule": np.nan,
        }
    for col, value in cols.items():
        if col not in out:
            out[col] = value
    return out
