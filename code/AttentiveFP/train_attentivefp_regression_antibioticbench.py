#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AntibioticBench PyG AttentiveFP training script for regression.

Key outputs per run:
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
  best_model.pth
  run.log

This implementation uses the official ``torch_geometric.nn.models.AttentiveFP`` model.
RDKit generates the same 43-dimensional atom features and 14-dimensional bond features used by the GINE baseline.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import glob
import json
import logging
import math
import os
import platform
import random
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except Exception as e:  # pragma: no cover
    raise RuntimeError("PyTorch is required. Please install torch in the training environment.") from e

try:
    from sklearn.metrics import (
        accuracy_score,
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
        precision_score,
        r2_score,
        recall_score,
        roc_auc_score,
    )
except Exception as e:  # pragma: no cover
    raise RuntimeError("scikit-learn is required. Please install scikit-learn.") from e

try:
    from scipy.stats import pearsonr, spearmanr
except Exception as e:  # pragma: no cover
    raise RuntimeError("scipy is required. Please install scipy.") from e

try:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
except Exception as e:  # pragma: no cover
    raise RuntimeError("RDKit is required. Recommended: conda install -c conda-forge rdkit") from e

try:
    import torch_geometric
    from torch_geometric.nn.models import AttentiveFP as PyGAttentiveFP
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "PyTorch Geometric with torch_geometric.nn.models.AttentiveFP is required. "
        "Activate the installed AttentiveFP/PyG environment before running this script."
    ) from e

RDLogger.DisableLog("rdApp.*")


# ----------------------------- utilities -----------------------------


def now_str() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_float(x: Any, default: float = np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(out_dir: Path) -> logging.Logger:
    ensure_dir(out_dir)
    logger = logging.getLogger("antibioticbench_attentivefp")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(out_dir / "run.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def save_json(obj: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def parse_list_arg(x: Optional[str], default: Optional[List[str]] = None) -> List[str]:
    if x is None or str(x).strip() == "":
        return default or []
    return [i.strip() for i in re.split(r"[,\s]+", str(x)) if i.strip()]


def first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in cols:
            return cols[c.lower()]
    return None


def canonical_split_name(x: str) -> str:
    y = str(x).strip().lower()
    if y in {"valid", "validation", "val", "dev"}:
        return "validation"
    if y in {"train", "training"}:
        return "train"
    if y in {"test", "testing"}:
        return "test"
    return y


def infer_dataset_name_from_path(path: str | Path) -> str:
    stem = Path(path).stem
    for token in ["_classification", "_regression", "_random", "_scaffold", "_train", "_valid", "_validation", "_test"]:
        stem = stem.replace(token, "")
    return stem


# ----------------------------- RDKit graph and fingerprints -----------------------------

ATOM_TYPES = [
    1, 5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 19, 20, 26, 30, 35, 53
]
DEGREES = [0, 1, 2, 3, 4, 5, 6]
FORMAL_CHARGES = [-2, -1, 0, 1, 2]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BOND_STEREO = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
]


def one_hot_unknown(value: Any, allowable: Sequence[Any]) -> List[float]:
    return [1.0 if value == v else 0.0 for v in allowable] + [0.0 if value in allowable else 1.0]


def atom_features(atom: Chem.Atom) -> List[float]:
    feat: List[float] = []
    feat += one_hot_unknown(atom.GetAtomicNum(), ATOM_TYPES)
    feat += one_hot_unknown(atom.GetTotalDegree(), DEGREES)
    feat += one_hot_unknown(atom.GetFormalCharge(), FORMAL_CHARGES)
    feat += one_hot_unknown(atom.GetHybridization(), HYBRIDIZATIONS)
    feat += [
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        float(atom.GetTotalNumHs()),
        float(atom.GetMass() * 0.01),
    ]
    return feat


def bond_features(bond: Chem.Bond) -> List[float]:
    """Bond features supplied to the PyG AttentiveFP edge-aware attention layer.

    The feature vector is intentionally compact and fully RDKit-derived:
      - bond type: single/double/triple/aromatic/unknown
      - conjugation flag
      - ring-bond flag
      - stereo category with unknown bucket
    """
    feat: List[float] = []
    feat += one_hot_unknown(bond.GetBondType(), BOND_TYPES)
    feat += [
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]
    feat += one_hot_unknown(bond.GetStereo(), BOND_STEREO)
    return feat


BOND_FEATURE_DIM = len(BOND_TYPES) + 1 + 2 + len(BOND_STEREO) + 1


def mol_to_graph(smiles: str) -> Optional[Dict[str, Any]]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    x = np.array([atom_features(a) for a in mol.GetAtoms()], dtype=np.float32)
    edges: List[Tuple[int, int]] = []
    edge_attrs: List[List[float]] = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = bond_features(b)
        edges.append((i, j))
        edge_attrs.append(bf)
        edges.append((j, i))
        edge_attrs.append(bf)
    if len(edges) == 0:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, BOND_FEATURE_DIM), dtype=np.float32)
    else:
        edge_index = np.array(edges, dtype=np.int64).T
        edge_attr = np.array(edge_attrs, dtype=np.float32)
    return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr, "num_nodes": x.shape[0]}


def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    try:
        return Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None


def smiles_to_fp(smiles: str, radius: int = 2, n_bits: int = 2048):
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    except Exception:
        return None


def bulk_tanimoto_array(fp, train_fps: Sequence[Any]) -> np.ndarray:
    """Fast Tanimoto similarities to all train fingerprints. Invalid train fps are NaN."""
    sims = np.full(len(train_fps), np.nan, dtype=float)
    if fp is None or len(train_fps) == 0:
        return sims
    valid_idx = [i for i, tfp in enumerate(train_fps) if tfp is not None]
    if not valid_idx:
        return sims
    valid_fps = [train_fps[i] for i in valid_idx]
    try:
        vals = DataStructs.BulkTanimotoSimilarity(fp, valid_fps)
        sims[np.asarray(valid_idx, dtype=int)] = np.asarray(vals, dtype=float)
    except Exception:
        # Conservative fallback. This branch should rarely be used.
        for i in valid_idx:
            try:
                sims[i] = float(DataStructs.TanimotoSimilarity(fp, train_fps[i]))
            except Exception:
                sims[i] = np.nan
    return sims


def tanimoto_to_train(fp, train_fps: Sequence[Any]) -> Tuple[int, float]:
    sims = bulk_tanimoto_array(fp, train_fps)
    if not np.isfinite(sims).any():
        return -1, np.nan
    j = int(np.nanargmax(sims))
    return j, float(sims[j])


def topk_train_neighbors(fp, train_fps: Sequence[Any], k: int = 10) -> List[Tuple[int, float]]:
    sims = bulk_tanimoto_array(fp, train_fps)
    valid = np.where(np.isfinite(sims))[0]
    if len(valid) == 0:
        return []
    k2 = min(int(k), len(valid))
    if k2 <= 0:
        return []
    vals = sims[valid]
    if k2 < len(valid):
        part = np.argpartition(-vals, k2 - 1)[:k2]
        idx = valid[part]
    else:
        idx = valid
    idx = idx[np.argsort(-sims[idx])]
    return [(int(i), float(sims[i])) for i in idx[:k2]]


def nn_bin(sim: float) -> str:
    if pd.isna(sim):
        return "NA"
    if sim >= 0.7:
        return ">=0.7"
    if sim >= 0.4:
        return "0.4-0.7"
    return "<0.4"


# ----------------------------- dataset -----------------------------

@dataclass
class RunSpec:
    dataset_name: str
    task: str
    split_type: str
    seed: int
    threshold_for_boundary: float
    train_path: Optional[str] = None
    val_path: Optional[str] = None
    test_path: Optional[str] = None
    data_path: Optional[str] = None


def public_run_spec(spec: RunSpec) -> Dict[str, Any]:
    """Keep run-plan fields while excluding local directory names."""
    return {k: Path(v).name if k.endswith("_path") and v else v for k, v in asdict(spec).items()}


def public_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {k: Path(v).name if any(x in k for x in ("root", "path", "config")) and isinstance(v, str) and v else v for k, v in vars(args).items()}


class MolGraphDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        task: str,
        smiles_col: str,
        target_col: str,
        logger: Optional[logging.Logger] = None,
    ):
        self.df_original = df.reset_index(drop=True).copy()
        self.task = task
        self.smiles_col = smiles_col
        self.target_col = target_col
        self.items: List[Dict[str, Any]] = []
        bad_rows: List[int] = []
        for idx, row in self.df_original.iterrows():
            smi = row[smiles_col]
            g = mol_to_graph(smi)
            y = safe_float(row[target_col])
            if g is None or pd.isna(y):
                bad_rows.append(idx)
                continue
            self.items.append({
                "row_idx": idx,
                "smiles": str(smi),
                "x": g["x"],
                "edge_index": g["edge_index"],
                "edge_attr": g["edge_attr"],
                "num_nodes": g["num_nodes"],
                "y": float(y),
            })
        self.df = self.df_original.iloc[[it["row_idx"] for it in self.items]].reset_index(drop=True).copy()
        if logger is not None:
            logger.info(f"Dataset built: task={task}, rows_in={len(df)}, usable={len(self.items)}, dropped_invalid={len(bad_rows)}")
            if bad_rows[:10]:
                logger.info(f"First invalid/dropped row indices: {bad_rows[:10]}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


def collate_graphs(batch: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor | List[str]]:
    xs = []
    edge_indices = []
    edge_attrs = []
    batch_vec = []
    ys = []
    smiles = []
    row_idx = []
    offset = 0
    for gi, item in enumerate(batch):
        x = torch.tensor(item["x"], dtype=torch.float32)
        ei = torch.tensor(item["edge_index"], dtype=torch.long)
        ea = torch.tensor(item["edge_attr"], dtype=torch.float32)
        xs.append(x)
        if ei.numel() > 0:
            edge_indices.append(ei + offset)
            edge_attrs.append(ea)
        batch_vec.append(torch.full((x.shape[0],), gi, dtype=torch.long))
        ys.append(float(item["y"]))
        smiles.append(item["smiles"])
        row_idx.append(int(item["row_idx"]))
        offset += x.shape[0]
    edge_index = torch.cat(edge_indices, dim=1) if edge_indices else torch.empty((2, 0), dtype=torch.long)
    edge_attr_dim = int(batch[0]["edge_attr"].shape[1]) if len(batch) else BOND_FEATURE_DIM
    edge_attr = torch.cat(edge_attrs, dim=0) if edge_attrs else torch.empty((0, edge_attr_dim), dtype=torch.float32)
    return {
        "x": torch.cat(xs, dim=0),
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "batch": torch.cat(batch_vec, dim=0),
        "y": torch.tensor(ys, dtype=torch.float32),
        "smiles": smiles,
        "row_idx": torch.tensor(row_idx, dtype=torch.long),
        "num_graphs": torch.tensor(len(batch), dtype=torch.long),
    }


# ----------------------------- model -----------------------------

EXPECTED_TRAINABLE_PARAMS = 897_201


class AttentiveFPModel(nn.Module):
    """Thin wrapper around the official PyG AttentiveFP implementation."""

    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        hidden_dim: int = 200,
        num_layers: int = 2,
        num_timesteps: int = 2,
        dropout: float = 0.20,
    ):
        super().__init__()
        self.net = PyGAttentiveFP(
            in_channels=int(in_dim),
            hidden_channels=int(hidden_dim),
            out_channels=1,
            edge_dim=int(edge_dim),
            num_layers=int(num_layers),
            num_timesteps=int(num_timesteps),
            dropout=float(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(x, edge_index, edge_attr, batch).view(-1)


# ----------------------------- metrics -----------------------------


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        if len(np.unique(y_true[~pd.isna(y_true)])) < 2:
            return np.nan
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return np.nan


def safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        if len(np.unique(y_true[~pd.isna(y_true)])) < 2:
            return np.nan
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return np.nan


def safe_corr(y_true: np.ndarray, y_pred: np.ndarray, method: str = "spearman") -> float:
    try:
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if mask.sum() < 3:
            return np.nan
        if np.nanstd(y_true[mask]) == 0 or np.nanstd(y_pred[mask]) == 0:
            return np.nan
        if method == "spearman":
            return float(spearmanr(y_true[mask], y_pred[mask]).correlation)
        return float(pearsonr(y_true[mask], y_pred[mask])[0])
    except Exception:
        return np.nan


def classification_metrics(y_true: Sequence[float], y_score: Sequence[float], threshold: float = 0.5) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(y_score, dtype=float)
    mask = np.isfinite(y) & np.isfinite(s)
    y = y[mask].astype(int)
    s = np.clip(s[mask], 1e-7, 1 - 1e-7)
    if len(y) == 0:
        return {k: np.nan for k in [
            "roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "precision", "recall", "f1", "mcc",
            "specificity", "sensitivity", "log_loss", "brier"
        ]}
    pred = (s >= threshold).astype(int)
    out: Dict[str, float] = {}
    out["roc_auc"] = safe_auc(y, s)
    out["pr_auc"] = safe_pr_auc(y, s)
    out["accuracy"] = float(accuracy_score(y, pred))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) > 1 else np.nan
    out["precision"] = float(precision_score(y, pred, zero_division=0))
    out["recall"] = float(recall_score(y, pred, zero_division=0))
    out["f1"] = float(f1_score(y, pred, zero_division=0))
    out["mcc"] = float(matthews_corrcoef(y, pred)) if len(np.unique(y)) > 1 and len(np.unique(pred)) > 1 else 0.0
    try:
        cm = confusion_matrix(y, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        out["specificity"] = float(tn / (tn + fp)) if (tn + fp) else np.nan
        out["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) else np.nan
    except Exception:
        out["specificity"] = np.nan
        out["sensitivity"] = np.nan
    try:
        out["log_loss"] = float(log_loss(y, s, labels=[0, 1]))
    except Exception:
        out["log_loss"] = np.nan
    try:
        out["brier"] = float(brier_score_loss(y, s))
    except Exception:
        out["brier"] = np.nan
    return out


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if len(y) == 0:
        return {k: np.nan for k in [
            "mae", "mse", "rmse", "medae", "r2", "pearson", "spearman",
            "within_1_dilution", "within_2_dilutions", "median_fold_error", "mean_fold_error", "max_abs_error_log10"
        ]}
    abs_err = np.abs(p - y)
    mse = mean_squared_error(y, p)
    out: Dict[str, float] = {
        "mae": float(mean_absolute_error(y, p)),
        "mse": float(mse),
        "rmse": float(math.sqrt(mse)),
        "medae": float(median_absolute_error(y, p)),
        "r2": float(r2_score(y, p)) if len(y) > 1 else np.nan,
        "pearson": safe_corr(y, p, "pearson"),
        "spearman": safe_corr(y, p, "spearman"),
        "within_1_dilution": float(np.mean(abs_err <= math.log10(2))),
        "within_2_dilutions": float(np.mean(abs_err <= math.log10(4))),
        "median_fold_error": float(np.nanmedian(np.power(10.0, abs_err))),
        "mean_fold_error": float(np.nanmean(np.power(10.0, abs_err))),
        "max_abs_error_log10": float(np.nanmax(abs_err)),
    }
    return out


def virtual_screen_metrics(
    y_active: Sequence[float],
    score: Sequence[float],
    fractions: Sequence[float] = (0.01, 0.05, 0.10),
    top_ns: Sequence[int] = (50, 100, 500),
) -> Dict[str, float]:
    y = np.asarray(y_active, dtype=float)
    s = np.asarray(score, dtype=float)
    mask = np.isfinite(y) & np.isfinite(s)
    y = y[mask].astype(int)
    s = s[mask]
    out: Dict[str, float] = {}
    n = len(y)
    total_hits = int(np.sum(y == 1))
    base_rate = float(total_hits / n) if n else np.nan
    if n == 0:
        for frac in fractions:
            pct = int(frac * 100)
            out[f"top_{pct}pct_n"] = 0
            out[f"top_{pct}pct_hits"] = 0
            out[f"hit_rate_top_{pct}pct"] = np.nan
            out[f"ef_{pct}pct"] = np.nan
            out[f"EF{pct}%"] = np.nan
            out[f"Hit_rate@Top{pct}%"] = np.nan
        for topn in top_ns:
            out[f"Top{topn}_hits"] = 0
            out[f"Recall@Top{topn}"] = np.nan
        return out
    order = np.argsort(-s)
    for frac in fractions:
        pct = int(frac * 100)
        k = max(1, int(math.ceil(n * frac)))
        idx = order[:k]
        hits = int(np.sum(y[idx] == 1))
        hit_rate = float(hits / k) if k else np.nan
        ef = float(hit_rate / base_rate) if base_rate and np.isfinite(base_rate) and base_rate > 0 else np.nan
        out[f"top_{pct}pct_n"] = int(k)
        out[f"top_{pct}pct_hits"] = hits
        out[f"hit_rate_top_{pct}pct"] = hit_rate
        out[f"ef_{pct}pct"] = ef
        out[f"EF{pct}%"] = ef
        out[f"Hit_rate@Top{pct}%"] = hit_rate
    for topn in top_ns:
        k = min(int(topn), n)
        idx = order[:k]
        hits = int(np.sum(y[idx] == 1))
        recall = float(hits / total_hits) if total_hits else np.nan
        out[f"Top{topn}_hits"] = hits
        out[f"Recall@Top{topn}"] = recall
    return out


def best_f1_threshold(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    mask = np.isfinite(y) & np.isfinite(s)
    y = y[mask]
    s = s[mask]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return 0.5
    thresholds = np.linspace(0.01, 0.99, 99)
    vals = [f1_score(y, (s >= t).astype(int), zero_division=0) for t in thresholds]
    return float(thresholds[int(np.argmax(vals))])


# ----------------------------- data loading -----------------------------


def read_csv_smart(path: str | Path) -> pd.DataFrame:
    path = str(path)
    # sep=None lets pandas sniff csv/tsv; python engine is slower but convenient for benchmark-sized data.
    return pd.read_csv(path, sep=None, engine="python")


def possible_dataset_dirs(data_root: Path, dataset: str) -> List[Path]:
    """Return possible directories for a dataset.

    Supports both exact dataset folder names and the AntibioticBench folder style, e.g.
    dataset_name="Staphylococcus" can match
    1_Staphylococcus_median_16_MolWt_50_1000_value_le_8192_multitask.
    Exact matches are kept first.
    """
    dataset = str(dataset).strip()
    exact = data_root / dataset
    dirs: List[Path] = [exact]
    if data_root.exists() and data_root.is_dir():
        ds_l = dataset.lower()
        for child in sorted(data_root.iterdir(), key=lambda p: p.name):
            if not child.is_dir():
                continue
            name_l = child.name.lower()
            if child == exact:
                continue
            if name_l == ds_l or name_l.startswith(ds_l + "_") or f"_{ds_l}_" in name_l or ds_l in name_l:
                dirs.append(child)
    # de-duplicate preserving order
    seen = set()
    out: List[Path] = []
    for d in dirs:
        s = str(d)
        if s not in seen:
            seen.add(s)
            out.append(d)
    return out


def candidate_split_paths(data_root: Path, dataset: str, task: str, split_type: str, seed: int, split_name: str) -> List[Path]:
    seed_dir = f"seed_{seed}"
    names = {
        "train": ["train.csv", "training.csv", f"{dataset}_train.csv", f"{dataset}_{split_type}_seed{seed}_train.csv"],
        "validation": ["validation.csv", "valid.csv", "val.csv", f"{dataset}_validation.csv", f"{dataset}_valid.csv", f"{dataset}_{split_type}_seed{seed}_valid.csv"],
        "test": ["test.csv", f"{dataset}_test.csv", f"{dataset}_{split_type}_seed{seed}_test.csv"],
    }[split_name]
    dirs: List[Path] = []
    for ds_dir in possible_dataset_dirs(data_root, dataset):
        dirs.extend([
            ds_dir / task / split_type / seed_dir,
            ds_dir / split_type / seed_dir / task,
            ds_dir / split_type / seed_dir,
            ds_dir / "splits" / task / split_type / seed_dir,
            ds_dir / "splits" / split_type / seed_dir,
            ds_dir,
        ])
    dirs.extend([
        data_root / task / dataset / split_type / seed_dir,
        data_root / task / split_type / seed_dir / dataset,
        data_root,
    ])
    paths: List[Path] = []
    for d in dirs:
        for n in names:
            paths.append(d / n)
    # Also support flat or nested filenames.
    suffix = "valid" if split_name == "validation" else split_name
    flat_patterns = [
        f"{dataset}*{task}*{split_type}*seed*{seed}*{suffix}*.csv",
        f"{dataset}*{split_type}*seed*{seed}*{suffix}*.csv",
        f"{dataset}*{suffix}*.csv",
        f"**/{dataset}*{task}*{split_type}*seed*{seed}*{suffix}*.csv",
        f"**/{dataset}*{split_type}*seed*{seed}*{suffix}*.csv",
    ]
    for pat in flat_patterns:
        paths.extend(Path(p) for p in glob.glob(str(data_root / pat), recursive=True))
    # de-duplicate preserving order
    seen = set()
    out = []
    for p in paths:
        ps = str(p)
        if ps not in seen:
            seen.add(ps)
            out.append(p)
    return out


def find_existing(candidates: Sequence[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


def find_monolithic_path(data_root: Path, dataset: str, task: str, config_path: Optional[str] = None) -> Optional[Path]:
    candidates: List[Path] = []
    if config_path:
        p = Path(config_path)
        if p.exists() and p.is_file():
            return p
    for ds_dir in possible_dataset_dirs(data_root, dataset):
        candidates.extend([
            ds_dir / f"{dataset}_{task}.csv",
            ds_dir / f"{dataset}.csv",
            ds_dir / f"{ds_dir.name}_{task}.csv",
            ds_dir / f"{ds_dir.name}.csv",
        ])
    candidates.extend([
        data_root / task / f"{dataset}.csv",
        data_root / f"{dataset}_{task}.csv",
        data_root / f"{dataset}.csv",
    ])
    for pat in [f"**/{dataset}*{task}*.csv", f"**/{dataset}*.csv"]:
        candidates.extend(Path(p) for p in glob.glob(str(data_root / pat), recursive=True))
    return find_existing(candidates)


def load_split_data(
    spec: RunSpec,
    data_root: Path,
    smiles_col_arg: str,
    active_col_arg: str,
    log10_col_arg: str,
    mic_col_arg: str,
    split_col_arg: str,
    split_type_col_arg: str,
    seed_col_arg: str,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    info: Dict[str, Any] = {"loading_mode": None, "paths": {}}

    explicit_paths = [spec.train_path, spec.val_path, spec.test_path]
    if all(explicit_paths):
        train_p, val_p, test_p = map(Path, explicit_paths)  # type: ignore[arg-type]
    else:
        train_p = find_existing(candidate_split_paths(data_root, spec.dataset_name, spec.task, spec.split_type, spec.seed, "train"))
        val_p = find_existing(candidate_split_paths(data_root, spec.dataset_name, spec.task, spec.split_type, spec.seed, "validation"))
        test_p = find_existing(candidate_split_paths(data_root, spec.dataset_name, spec.task, spec.split_type, spec.seed, "test"))

    if train_p and val_p and test_p:
        info["loading_mode"] = "three_split_files"
        info["paths"] = {"train": str(train_p), "validation": str(val_p), "test": str(test_p)}
        train_df = read_csv_smart(train_p)
        val_df = read_csv_smart(val_p)
        test_df = read_csv_smart(test_p)
    else:
        mono = find_monolithic_path(data_root, spec.dataset_name, spec.task, spec.data_path)
        if mono is None:
            raise FileNotFoundError(
                f"Cannot find split files or monolithic CSV for dataset={spec.dataset_name}, task={spec.task}, "
                f"split_type={spec.split_type}, seed={spec.seed}. Please check --data-root or dataset_config.csv."
            )
        info["loading_mode"] = "monolithic_csv_filtered_by_split_columns"
        info["paths"] = {"data": str(mono)}
        df = read_csv_smart(mono)
        split_col = first_existing_col(df, [split_col_arg, "split", "set", "subset"])
        if split_col is None:
            raise ValueError(f"Monolithic CSV {mono.name} has no split column. Expected one of split/set/subset or --split-col.")
        df = df.copy()
        df["__split_norm"] = df[split_col].map(canonical_split_name)
        split_type_col = first_existing_col(df, [split_type_col_arg, "split_type", "split_strategy"])
        seed_col = first_existing_col(df, [seed_col_arg, "seed", "split_seed"])
        if split_type_col is not None:
            df = df[df[split_type_col].astype(str).str.lower() == str(spec.split_type).lower()].copy()
        if seed_col is not None:
            df = df[df[seed_col].astype(str).astype(float).astype(int) == int(spec.seed)].copy()
        train_df = df[df["__split_norm"] == "train"].drop(columns=["__split_norm"], errors="ignore")
        val_df = df[df["__split_norm"] == "validation"].drop(columns=["__split_norm"], errors="ignore")
        test_df = df[df["__split_norm"] == "test"].drop(columns=["__split_norm"], errors="ignore")
        if train_df.empty or val_df.empty or test_df.empty:
            raise ValueError(f"Filtered monolithic CSV {mono.name} produced empty train/val/test for split_type={spec.split_type}, seed={spec.seed}.")

    for name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        df["split"] = name
        df["dataset_name"] = spec.dataset_name
        df["task"] = spec.task
        df["split_type"] = spec.split_type
        df["seed"] = spec.seed
        if "genus" not in df.columns:
            df["genus"] = spec.dataset_name

    logger.info(f"Loaded splits: train={len(train_df)}, validation={len(val_df)}, test={len(test_df)}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True), info


def prepare_target_columns(
    df: pd.DataFrame,
    task: str,
    active_col_arg: str,
    log10_col_arg: str,
    mic_col_arg: str,
    threshold: float,
) -> Tuple[pd.DataFrame, str, Optional[str], Optional[str]]:
    df = df.copy()
    mic_col = first_existing_col(df, [mic_col_arg, "mic_numeric", "MIC", "mic", "MIC_ug_ml", "mic_ug_ml", "value"])
    log10_col = first_existing_col(df, [log10_col_arg, "log10_mic", "log10MIC", "log10_MIC", "log_mic"])
    active_col = first_existing_col(df, [active_col_arg, "active", "label", "y", "activity"])

    if log10_col is None and mic_col is not None:
        vals = pd.to_numeric(df[mic_col], errors="coerce")
        vals = vals.where(vals > 0)
        df["log10_mic"] = np.log10(vals)
        log10_col = "log10_mic"
    if mic_col is None and log10_col is not None:
        df["mic_numeric"] = np.power(10.0, pd.to_numeric(df[log10_col], errors="coerce"))
        mic_col = "mic_numeric"
    if active_col is None and mic_col is not None:
        df["active"] = (pd.to_numeric(df[mic_col], errors="coerce") <= float(threshold)).astype(float)
        active_col = "active"
    if task == "classification":
        if active_col is None:
            raise ValueError("Classification task requires active/label column, or MIC column plus threshold to infer active.")
        df[active_col] = pd.to_numeric(df[active_col], errors="coerce")
        # enforce binary 0/1 when possible
        df[active_col] = (df[active_col] > 0).astype(float)
        return df, active_col, log10_col, mic_col
    else:
        if log10_col is None:
            raise ValueError("Regression task requires log10_mic column, or positive MIC column to compute log10_mic.")
        df[log10_col] = pd.to_numeric(df[log10_col], errors="coerce")
        return df, log10_col, log10_col, mic_col


# ----------------------------- predictions and diagnostics -----------------------------

@torch.no_grad()
def predict_model(
    model: nn.Module,
    loader: DataLoader,
    task: str,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    model.eval()
    ys: List[float] = []
    outs: List[float] = []
    rows: List[int] = []
    for batch in loader:
        x = batch["x"].to(device)
        edge_index = batch["edge_index"].to(device)
        edge_attr = batch["edge_attr"].to(device)
        bvec = batch["batch"].to(device)
        y = batch["y"].detach().cpu().numpy().astype(float)
        out = model(x, edge_index, edge_attr, bvec).detach().cpu().numpy().astype(float)
        if task == "classification":
            out = 1.0 / (1.0 + np.exp(-out))
        ys.extend(y.tolist())
        outs.extend(out.tolist())
        rows.extend(batch["row_idx"].detach().cpu().numpy().astype(int).tolist())
    return np.asarray(ys), np.asarray(outs), rows


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    try:
        return float(optimizer.param_groups[0]["lr"])
    except Exception:
        return np.nan


def make_prediction_df(
    base_df: pd.DataFrame,
    split_name: str,
    task: str,
    target_col: str,
    pred_values: Sequence[float],
    spec: RunSpec,
    model_name: str,
    threshold_used: float,
    val_best_f1_threshold: float,
    mic_col: Optional[str],
    log10_col: Optional[str],
    train_ref_df: pd.DataFrame,
    train_ref_smiles_col: str,
    train_ref_target_col: str,
    train_ref_pred_col: str,
    smiles_col: str,
    logger: Optional[logging.Logger] = None,
    include_diagnostics: bool = False,
) -> pd.DataFrame:
    """Build per-molecule prediction output.

    Output policy:
      - train / validation: predictions and fields needed for overall metrics only.
      - test: predictions plus train-reference NN and activity-cliff diagnostics.

    This deliberately avoids computing NN/AC diagnostics for train and validation,
    which reduces repeated Morgan-similarity work and keeps the split roles clear.
    """
    df = base_df.reset_index(drop=True).copy()
    if len(df) != len(pred_values):
        raise ValueError(f"Prediction length mismatch for {split_name}: df={len(df)}, preds={len(pred_values)}")

    out = pd.DataFrame(index=np.arange(len(df)))
    out["split"] = split_name
    out["dataset_name"] = spec.dataset_name
    out["genus"] = df["genus"] if "genus" in df.columns else spec.dataset_name
    out["task"] = spec.task
    out["split_type"] = spec.split_type
    out["seed"] = spec.seed
    out["model"] = model_name
    out["model_family"] = "AttentiveFP"
    out["smiles"] = df[smiles_col].astype(str).values

    if mic_col and mic_col in df.columns:
        mic_numeric = pd.to_numeric(df[mic_col], errors="coerce").values.astype(float)
    elif log10_col and log10_col in df.columns:
        mic_numeric = np.power(10.0, pd.to_numeric(df[log10_col], errors="coerce").values.astype(float))
    else:
        mic_numeric = np.full(len(df), np.nan)

    if log10_col and log10_col in df.columns:
        log10_mic_diag = pd.to_numeric(df[log10_col], errors="coerce").values.astype(float)
    else:
        log10_mic_diag = np.log10(np.where(mic_numeric > 0, mic_numeric, np.nan))

    pred_values_np = np.asarray(pred_values, dtype=float)
    if task == "classification":
        true_active = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int).values
        pred_prob = np.clip(pred_values_np, 1e-7, 1 - 1e-7)
        pred_label_0p5 = (pred_prob >= 0.5).astype(int)
        pred_label = (pred_prob >= threshold_used).astype(int)
        out["active"] = true_active
        out["true_active"] = true_active
        out["pred_score"] = pred_prob
        out["pred_prob"] = pred_prob
        out["pred_label"] = pred_label
        out["pred_label_0p5"] = pred_label_0p5
        out["threshold_used"] = threshold_used
        out["val_best_f1_threshold"] = val_best_f1_threshold
        out["error"] = (pred_label != true_active).astype(int)
        if include_diagnostics:
            # MIC values are retained only on test because activity-cliff diagnosis needs them.
            out["mic_numeric"] = mic_numeric
            out["log2_mic_for_diagnosis"] = np.log2(np.where(mic_numeric > 0, mic_numeric, np.nan))
            out["log10_mic_for_diagnosis"] = log10_mic_diag
    else:
        true_log10 = pd.to_numeric(df[target_col], errors="coerce").values.astype(float)
        pred_log10 = pred_values_np.astype(float)
        true_mic = np.power(10.0, true_log10)
        pred_mic = np.power(10.0, pred_log10)
        abs_err = np.abs(pred_log10 - true_log10)
        out["log10_mic"] = true_log10
        out["true_log10MIC"] = true_log10
        out["true_MIC"] = true_mic
        out["mic_numeric"] = mic_numeric if np.isfinite(mic_numeric).any() else true_mic
        out["pred_log10MIC"] = pred_log10
        out["pred_MIC"] = pred_mic
        out["score_reg"] = -pred_log10
        out["abs_error_log10"] = abs_err
        out["sq_error_log10"] = (pred_log10 - true_log10) ** 2
        out["fold_error"] = np.power(10.0, abs_err)
        out["within_1_dilution"] = (abs_err <= math.log10(2)).astype(int)
        out["within_2_dilutions"] = (abs_err <= math.log10(4)).astype(int)
        # Needed for train/validation/test virtual-screening metrics, not a diagnostic.
        out["true_active_for_screening"] = (true_mic <= spec.threshold_for_boundary).astype(int)
        out["threshold_for_screening"] = spec.threshold_for_boundary

    if not include_diagnostics:
        return out

    # Test-only nearest-neighbor diagnostics, always using train as reference library.
    train_smiles = train_ref_df[train_ref_smiles_col].astype(str).tolist()
    train_fps = [smiles_to_fp(s) for s in train_smiles]
    this_fps = [smiles_to_fp(s) for s in out["smiles"].astype(str).tolist()]
    nn_rows: List[int] = []
    nn_sims: List[float] = []
    nn_smiles: List[Any] = []
    for fp in this_fps:
        neighbors = topk_train_neighbors(fp, train_fps, k=1)
        if neighbors:
            nn_i, nn_s = neighbors[0]
        else:
            nn_i, nn_s = -1, np.nan
        nn_rows.append(int(nn_i))
        nn_sims.append(float(nn_s) if np.isfinite(nn_s) else np.nan)
        nn_smiles.append(train_smiles[nn_i] if 0 <= nn_i < len(train_smiles) else np.nan)

    out["nn_train_row"] = nn_rows
    out["nn_sim"] = nn_sims
    out["nn_bin"] = [nn_bin(s) for s in nn_sims]
    out["nn_train_smiles"] = nn_smiles

    if task == "classification":
        train_active = pd.to_numeric(train_ref_df[train_ref_target_col], errors="coerce").fillna(0).astype(int).values
        train_score = (
            pd.to_numeric(train_ref_df[train_ref_pred_col], errors="coerce").values.astype(float)
            if train_ref_pred_col in train_ref_df.columns
            else np.full(len(train_ref_df), np.nan)
        )
        out["nn_train_active"] = [train_active[i] if 0 <= i < len(train_active) else np.nan for i in nn_rows]
        out["nn_train_score"] = [train_score[i] if 0 <= i < len(train_score) else np.nan for i in nn_rows]
    else:
        train_log10_true = pd.to_numeric(train_ref_df[train_ref_target_col], errors="coerce").values.astype(float)
        train_log10_pred = (
            pd.to_numeric(train_ref_df[train_ref_pred_col], errors="coerce").values.astype(float)
            if train_ref_pred_col in train_ref_df.columns
            else np.full(len(train_ref_df), np.nan)
        )
        out["nn_train_log10mic"] = [train_log10_true[i] if 0 <= i < len(train_log10_true) else np.nan for i in nn_rows]
        out["nn_train_pred_log10mic"] = [train_log10_pred[i] if 0 <= i < len(train_log10_pred) else np.nan for i in nn_rows]

    # Test-only activity-cliff diagnosis against the train reference library.
    ac_flag, ac_count, ac_maxdiff, ac_dir_acc, ac_pair_count = activity_cliff_fields(
        out_df=out,
        task=task,
        train_ref_df=train_ref_df,
        train_fps=train_fps,
        log10_col=log10_col,
        mic_col=mic_col,
        smiles_col=train_ref_smiles_col,
        threshold=spec.threshold_for_boundary,
    )
    out["ac_flag"] = ac_flag
    out["ac_neighbor_count"] = ac_count
    if task == "classification":
        out["ac_max_abs_log2mic_diff"] = ac_maxdiff
    else:
        out["ac_max_abs_log10mic_diff"] = ac_maxdiff
    out["ac_pair_directional_accuracy_for_molecule"] = ac_dir_acc
    out["ac_pair_count_for_molecule"] = ac_pair_count
    return out

def activity_cliff_fields(
    out_df: pd.DataFrame,
    task: str,
    train_ref_df: pd.DataFrame,
    train_fps: Sequence[Any],
    log10_col: Optional[str],
    mic_col: Optional[str],
    smiles_col: str,
    threshold: float,
) -> Tuple[List[int], List[int], List[float], List[float], List[int]]:
    # Activity cliff definition: Tanimoto >= 0.7 and MIC differs >= 16-fold.
    if log10_col and log10_col in train_ref_df.columns:
        train_log10 = pd.to_numeric(train_ref_df[log10_col], errors="coerce").values.astype(float)
    elif mic_col and mic_col in train_ref_df.columns:
        train_mic = pd.to_numeric(train_ref_df[mic_col], errors="coerce").values.astype(float)
        train_log10 = np.log10(np.where(train_mic > 0, train_mic, np.nan))
    else:
        train_log10 = np.full(len(train_ref_df), np.nan)

    if task == "regression":
        this_log10 = out_df["true_log10MIC"].values.astype(float) if "true_log10MIC" in out_df.columns else np.full(len(out_df), np.nan)
        this_pred = out_df["pred_log10MIC"].values.astype(float) if "pred_log10MIC" in out_df.columns else np.full(len(out_df), np.nan)
        # Use the prediction of the actual train-side activity-cliff partner, not merely
        # the prediction of the nearest neighbour of the current molecule.
        if "__train_pred_log10mic" in train_ref_df.columns:
            train_pred_all = pd.to_numeric(train_ref_df["__train_pred_log10mic"], errors="coerce").values.astype(float)
        else:
            train_pred_all = np.full(len(train_ref_df), np.nan)
        diff_cut = math.log10(16.0)
        scale = "log10"
    else:
        if "log10_mic_for_diagnosis" in out_df.columns:
            this_log10 = out_df["log10_mic_for_diagnosis"].values.astype(float)
        else:
            mic = out_df["mic_numeric"].values.astype(float) if "mic_numeric" in out_df.columns else np.full(len(out_df), np.nan)
            this_log10 = np.log10(np.where(mic > 0, mic, np.nan))
        this_pred = out_df["pred_score"].values.astype(float) if "pred_score" in out_df.columns else np.full(len(out_df), np.nan)
        if "__train_pred_score" in train_ref_df.columns:
            train_pred_all = pd.to_numeric(train_ref_df["__train_pred_score"], errors="coerce").values.astype(float)
        else:
            train_pred_all = np.full(len(train_ref_df), np.nan)
        diff_cut = math.log10(16.0)
        scale = "log2"

    flags: List[int] = []
    counts: List[int] = []
    maxdiffs: List[float] = []
    dir_accs: List[float] = []
    pair_counts: List[int] = []

    for idx, row in out_df.iterrows():
        fp = smiles_to_fp(row["smiles"])
        sims = bulk_tanimoto_array(fp, train_fps) if fp is not None else np.full(len(train_fps), np.nan)
        neighbor_idx = np.where(np.isfinite(sims) & (sims >= 0.7))[0]
        this_true = this_log10[idx]
        diffs = []
        dir_hits = []
        if np.isfinite(this_true) and len(neighbor_idx):
            for ti in neighbor_idx:
                tv = train_log10[ti] if ti < len(train_log10) else np.nan
                if not np.isfinite(tv):
                    continue
                d = abs(this_true - tv)
                if d >= diff_cut:
                    diffs.append(d)
                    # directional accuracy only if we have both predictions and true direction.
                    try:
                        if task == "regression":
                            # Correct if predicted lower MIC direction agrees with true lower MIC direction.
                            p_this = this_pred[idx]
                            p_train = train_pred_all[ti] if ti < len(train_pred_all) and np.isfinite(train_pred_all[ti]) else np.nan
                            if np.isfinite(p_this) and np.isfinite(p_train):
                                dir_hits.append(int(np.sign(tv - this_true) == np.sign(p_train - p_this)))
                        else:
                            # For classification, active means low MIC; pred_score higher = more active.
                            p_this = this_pred[idx]
                            p_train = train_pred_all[ti] if ti < len(train_pred_all) and np.isfinite(train_pred_all[ti]) else np.nan
                            if np.isfinite(p_this) and np.isfinite(p_train):
                                true_dir = np.sign(tv - this_true)  # positive: this molecule lower MIC/more active
                                pred_dir = np.sign(p_this - p_train)
                                dir_hits.append(int(true_dir == pred_dir))
                    except Exception:
                        pass
        flags.append(1 if diffs else 0)
        counts.append(int(len(diffs)))
        if not diffs:
            maxdiffs.append(np.nan)
        else:
            md = float(max(diffs))
            maxdiffs.append(md / math.log10(2.0) if scale == "log2" else md)
        pair_counts.append(int(len(dir_hits)))
        dir_accs.append(float(np.mean(dir_hits)) if dir_hits else np.nan)
    return flags, counts, maxdiffs, dir_accs, pair_counts


def prefix_metrics(prefix: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {f"{prefix}_{k}": v for k, v in metrics.items()}


def split_metrics_from_predictions(pred_df: pd.DataFrame, task: str, threshold: float = 0.5) -> Dict[str, Any]:
    if task == "classification":
        m = classification_metrics(pred_df["true_active"].values, pred_df["pred_score"].values, threshold=threshold)
        vs = virtual_screen_metrics(pred_df["true_active"].values, pred_df["pred_score"].values)
        return {**m, **vs}
    else:
        m = regression_metrics(pred_df["true_log10MIC"].values, pred_df["pred_log10MIC"].values)
        y_active = pred_df["true_active_for_screening"].values if "true_active_for_screening" in pred_df.columns else np.full(len(pred_df), np.nan)
        vs = virtual_screen_metrics(y_active, pred_df["score_reg"].values)
        return {**m, **vs}


def make_diagnostics_metrics(test_pred: pd.DataFrame, task: str) -> Dict[str, Any]:
    """Compute test-only NN and activity-cliff diagnostics."""
    out: Dict[str, Any] = {}
    n = len(test_pred)
    if n == 0:
        return out

    nn_groups = [
        ("nn_ge_0p7", ">=0.7"),
        ("nn_0p4_0p7", "0.4-0.7"),
        ("nn_lt_0p4", "<0.4"),
    ]

    if task == "classification":
        err = pd.to_numeric(test_pred["error"], errors="coerce").values.astype(float)
        for label, bin_value in nn_groups:
            sub = test_pred[test_pred["nn_bin"] == bin_value]
            out[f"{label}_n"] = int(len(sub))
            out[f"{label}_error_rate"] = float(sub["error"].mean()) if len(sub) else np.nan
            mets = split_metrics_from_predictions(sub, task, threshold=0.5) if len(sub) else {}
            for k in [
                "roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "precision",
                "recall", "f1", "mcc", "specificity", "sensitivity", "log_loss",
                "brier", "EF1%", "EF5%", "EF10%", "Hit_rate@Top1%",
                "Hit_rate@Top5%", "Hit_rate@Top10%", "Recall@Top50",
                "Recall@Top100", "Recall@Top500",
            ]:
                out[f"{label}_{k}"] = mets.get(k, np.nan)

        ac = pd.to_numeric(test_pred.get("ac_flag", pd.Series(np.zeros(n))), errors="coerce").fillna(0).values.astype(int)
        ac_mask = ac == 1
        non_ac_mask = ac == 0
        out["ac_n"] = int(ac_mask.sum())
        out["non_ac_n"] = int(non_ac_mask.sum())
        out["ac_error_rate"] = float(np.nanmean(err[ac_mask])) if ac_mask.any() else np.nan
        out["non_ac_error_rate"] = float(np.nanmean(err[non_ac_mask])) if non_ac_mask.any() else np.nan
        out["ac_error_enrichment"] = safe_ratio(out["ac_error_rate"], out["non_ac_error_rate"])
        ac_mets = split_metrics_from_predictions(test_pred.loc[ac_mask], task) if ac_mask.any() else {}
        for k in ["roc_auc", "pr_auc", "accuracy", "f1", "mcc", "EF1%", "Hit_rate@Top1%", "Recall@Top100"]:
            out[f"ac_subgroup_{k}"] = ac_mets.get(k, np.nan)
    else:
        abs_err = pd.to_numeric(test_pred["abs_error_log10"], errors="coerce").values.astype(float)
        for label, bin_value in nn_groups:
            sub = test_pred[test_pred["nn_bin"] == bin_value]
            out[f"{label}_n"] = int(len(sub))
            mets = split_metrics_from_predictions(sub, task) if len(sub) else {}
            for k in [
                "mae", "rmse", "mse", "medae", "r2", "pearson", "spearman",
                "within_1_dilution", "within_2_dilutions", "median_fold_error",
                "mean_fold_error", "max_abs_error_log10", "EF1%", "EF5%",
                "EF10%", "Hit_rate@Top1%", "Hit_rate@Top5%", "Hit_rate@Top10%",
                "Recall@Top50", "Recall@Top100", "Recall@Top500",
            ]:
                out[f"{label}_{k}"] = mets.get(k, np.nan)

        ac = pd.to_numeric(test_pred.get("ac_flag", pd.Series(np.zeros(n))), errors="coerce").fillna(0).values.astype(int)
        ac_mask = ac == 1
        non_ac_mask = ac == 0
        out["ac_n"] = int(ac_mask.sum())
        out["non_ac_n"] = int(non_ac_mask.sum())
        out["ac_mae"] = float(np.nanmean(abs_err[ac_mask])) if ac_mask.any() else np.nan
        out["non_ac_mae"] = float(np.nanmean(abs_err[non_ac_mask])) if non_ac_mask.any() else np.nan
        out["ac_rmse"] = rmse_from_error(abs_err[ac_mask]) if ac_mask.any() else np.nan
        out["non_ac_rmse"] = rmse_from_error(abs_err[non_ac_mask]) if non_ac_mask.any() else np.nan
        out["ac_error_enrichment"] = safe_ratio(out["ac_mae"], out["non_ac_mae"])
        ac_mets = split_metrics_from_predictions(test_pred.loc[ac_mask], task) if ac_mask.any() else {}
        for k in ["pearson", "spearman", "within_1_dilution", "within_2_dilutions", "EF1%", "Hit_rate@Top1%", "Recall@Top100"]:
            out[f"ac_subgroup_{k}"] = ac_mets.get(k, np.nan)

    if "ac_pair_directional_accuracy_for_molecule" in test_pred.columns:
        ac_dir = pd.to_numeric(test_pred["ac_pair_directional_accuracy_for_molecule"], errors="coerce").values.astype(float)
        out["ac_pair_directional_accuracy"] = float(np.nanmean(ac_dir)) if np.isfinite(ac_dir).any() else np.nan
    else:
        out["ac_pair_directional_accuracy"] = np.nan
    out["ac_pair_total_count"] = (
        int(np.nansum(pd.to_numeric(test_pred["ac_pair_count_for_molecule"], errors="coerce")))
        if "ac_pair_count_for_molecule" in test_pred.columns else 0
    )
    return out

def safe_ratio(a: Any, b: Any) -> float:
    try:
        aa = float(a)
        bb = float(b)
        if not np.isfinite(aa) or not np.isfinite(bb) or bb == 0:
            return np.nan
        return aa / bb
    except Exception:
        return np.nan


def rmse_from_error(abs_err: Sequence[float]) -> float:
    e = np.asarray(abs_err, dtype=float)
    e = e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e ** 2))) if len(e) else np.nan


def make_subgroup_metrics(test_pred: pd.DataFrame, task: str) -> pd.DataFrame:
    """Long-form test diagnostics restricted to NN similarity and activity cliffs."""
    rows: List[Dict[str, Any]] = []
    groups = [
        ("NN_similarity", "nn_bin"),
        ("Activity_cliff", "ac_flag"),
    ]
    for group_type, group_col in groups:
        if group_col not in test_pred.columns:
            continue
        for gv, sub in test_pred.groupby(group_col, dropna=False):
            row: Dict[str, Any] = {
                "group_type": group_type,
                "group_col": group_col,
                "group_value": gv,
                "n": int(len(sub)),
            }
            mets = split_metrics_from_predictions(sub, task) if len(sub) else {}
            if task == "classification":
                row["error_rate"] = float(sub["error"].mean()) if len(sub) else np.nan
                for k in [
                    "pr_auc", "roc_auc", "accuracy", "balanced_accuracy", "precision",
                    "recall", "f1", "mcc", "specificity", "sensitivity", "log_loss",
                    "brier", "EF1%", "EF5%", "EF10%", "Hit_rate@Top1%",
                    "Hit_rate@Top5%", "Hit_rate@Top10%", "Recall@Top50",
                    "Recall@Top100", "Recall@Top500",
                ]:
                    row[k] = mets.get(k, np.nan)
            else:
                for k in [
                    "mae", "rmse", "mse", "medae", "r2", "pearson", "spearman",
                    "within_1_dilution", "within_2_dilutions", "median_fold_error",
                    "mean_fold_error", "max_abs_error_log10", "EF1%", "EF5%",
                    "EF10%", "Hit_rate@Top1%", "Hit_rate@Top5%", "Hit_rate@Top10%",
                    "Recall@Top50", "Recall@Top100", "Recall@Top500",
                ]:
                    row[k] = mets.get(k, np.nan)
            rows.append(row)
    return pd.DataFrame(rows)

# ----------------------------- training -----------------------------


def train_one_run(args: argparse.Namespace, spec: RunSpec) -> Path:
    out_dir = Path(args.out_root) / spec.dataset_name / spec.task / spec.split_type / f"seed_{spec.seed}"
    ensure_dir(out_dir)
    logger = setup_logger(out_dir)
    start_time = time.time()
    set_seed(int(spec.seed))

    logger.info("=" * 100)
    logger.info(f"Starting AttentiveFP run: dataset={spec.dataset_name}, task={spec.task}, split={spec.split_type}, seed={spec.seed}")
    logger.info("Output directory configured")
    logger.info(f"Threshold for boundary/screening: {spec.threshold_for_boundary}")

    train_df, val_df, test_df, loading_info = load_split_data(
        spec=spec,
        data_root=Path(args.data_root),
        smiles_col_arg=args.smiles_col,
        active_col_arg=args.active_col,
        log10_col_arg=args.log10_mic_col,
        mic_col_arg=args.mic_col,
        split_col_arg=args.split_col,
        split_type_col_arg=args.split_type_col,
        seed_col_arg=args.seed_col,
        logger=logger,
    )

    all_df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
    smiles_col = first_existing_col(all_df, [args.smiles_col, "smiles", "SMILES", "canonical_smiles", "SMILES_std", "smiles_std"])
    if smiles_col is None:
        raise ValueError("Cannot find SMILES column. Please pass --smiles-col.")
    train_df, target_col, log10_col_train, mic_col_train = prepare_target_columns(
        train_df, spec.task, args.active_col, args.log10_mic_col, args.mic_col, spec.threshold_for_boundary
    )
    val_df, target_col_val, log10_col_val, mic_col_val = prepare_target_columns(
        val_df, spec.task, args.active_col, args.log10_mic_col, args.mic_col, spec.threshold_for_boundary
    )
    test_df, target_col_test, log10_col_test, mic_col_test = prepare_target_columns(
        test_df, spec.task, args.active_col, args.log10_mic_col, args.mic_col, spec.threshold_for_boundary
    )
    # Prefer resolved target col from train. For consistent naming, add missing train target col name to val/test if needed.
    target_col = target_col
    if target_col_val != target_col and target_col_val in val_df.columns:
        val_df[target_col] = val_df[target_col_val]
    if target_col_test != target_col and target_col_test in test_df.columns:
        test_df[target_col] = test_df[target_col_test]
    log10_col = log10_col_train or log10_col_val or log10_col_test
    mic_col = mic_col_train or mic_col_val or mic_col_test

    train_set = MolGraphDataset(train_df, spec.task, smiles_col, target_col, logger=logger)
    val_set = MolGraphDataset(val_df, spec.task, smiles_col, target_col, logger=logger)
    test_set = MolGraphDataset(test_df, spec.task, smiles_col, target_col, logger=logger)
    # Align dataframes to retained valid molecule rows.
    train_df = train_set.df.copy()
    val_df = val_set.df.copy()
    test_df = test_set.df.copy()

    if len(train_set) == 0 or len(val_set) == 0 or len(test_set) == 0:
        raise ValueError(f"Empty usable dataset after RDKit graph conversion: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

    in_dim = int(train_set[0]["x"].shape[1])
    edge_dim = int(train_set[0]["edge_attr"].shape[1])
    logger.info(f"Node feature dimension: {in_dim}")
    logger.info(f"Bond/edge feature dimension: {edge_dim}")

    num_workers = int(args.num_workers)
    train_loader = DataLoader(train_set, batch_size=int(args.batch_size), shuffle=True, collate_fn=collate_graphs, num_workers=num_workers)
    eval_loader_train = DataLoader(train_set, batch_size=int(args.eval_batch_size), shuffle=False, collate_fn=collate_graphs, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=int(args.eval_batch_size), shuffle=False, collate_fn=collate_graphs, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=int(args.eval_batch_size), shuffle=False, collate_fn=collate_graphs, num_workers=num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"CUDA device count={torch.cuda.device_count()}, current={torch.cuda.current_device()}, name={torch.cuda.get_device_name(torch.cuda.current_device())}")

    if in_dim != 43:
        raise ValueError(f"AttentiveFP benchmark expects 43 atom features, but featurizer produced {in_dim}.")
    if edge_dim != 14:
        raise ValueError(f"AttentiveFP benchmark expects 14 bond features, but featurizer produced {edge_dim}.")

    model = AttentiveFPModel(
        in_dim=in_dim,
        edge_dim=edge_dim,
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        num_timesteps=int(args.num_timesteps),
        dropout=float(args.dropout),
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: total={total_params:,}, trainable={trainable_params:,}")
    if trainable_params != EXPECTED_TRAINABLE_PARAMS:
        logger.warning(
            f"Expected {EXPECTED_TRAINABLE_PARAMS:,} trainable parameters for the fixed PyG AttentiveFP preset, "
            f"but found {trainable_params:,}. Check PyG version or changed hyperparameters."
        )

    if spec.task == "classification":
        # Fixed benchmark setting: unweighted binary cross-entropy.
        ytrain = pd.to_numeric(train_df[target_col], errors="coerce").fillna(0).astype(int).values
        pos = int(np.sum(ytrain == 1))
        neg = int(np.sum(ytrain == 0))
        criterion = nn.BCEWithLogitsLoss()
        logger.info(
            f"Classification target counts: pos={pos}, neg={neg}; "
            "class_weighting=none; pos_weight=disabled"
        )
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max" if spec.task == "classification" else "min",
        factor=float(args.lr_reduce_factor),
        patience=int(args.lr_patience),
        min_lr=float(args.min_lr),
    )

    best_epoch = -1
    best_val_metric = -np.inf if spec.task == "classification" else np.inf
    best_ckpt_path = out_dir / "best_model.pth"
    train_log_rows: List[Dict[str, Any]] = []
    no_improve = 0

    for epoch in range(1, int(args.epochs) + 1):
        epoch_start = time.time()
        model.train()
        batch_losses = []
        for batch_idx, batch in enumerate(train_loader, start=1):
            x = batch["x"].to(device)
            edge_index = batch["edge_index"].to(device)
            edge_attr = batch["edge_attr"].to(device)
            bvec = batch["batch"].to(device)
            y = batch["y"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits_or_pred = model(x, edge_index, edge_attr, bvec)
            if spec.task == "classification":
                loss = criterion(logits_or_pred, y)
            else:
                loss = criterion(logits_or_pred, y)
            loss.backward()
            if float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))
            if args.log_every_batches > 0 and (batch_idx % int(args.log_every_batches) == 0):
                logger.info(
                    f"epoch={epoch:04d} batch={batch_idx:04d}/{len(train_loader):04d} "
                    f"batch_loss={batch_losses[-1]:.6f} running_train_loss={np.mean(batch_losses):.6f} lr={get_lr(optimizer):.3e}"
                )

        # Full train/val prediction every epoch to save detailed curves.
        train_y, train_pred, _ = predict_model(model, eval_loader_train, spec.task, device)
        val_y, val_pred, _ = predict_model(model, val_loader, spec.task, device)
        if spec.task == "classification":
            train_loss_eval = float(log_loss(train_y.astype(int), np.clip(train_pred, 1e-7, 1 - 1e-7), labels=[0, 1])) if len(np.unique(train_y)) > 1 else float(np.mean(batch_losses))
            val_loss_eval = float(log_loss(val_y.astype(int), np.clip(val_pred, 1e-7, 1 - 1e-7), labels=[0, 1])) if len(np.unique(val_y)) > 1 else np.nan
            train_m = classification_metrics(train_y, train_pred, threshold=0.5)
            val_m = classification_metrics(val_y, val_pred, threshold=0.5)
            # best checkpoint metric: val_pr_auc by default; fallback to val_auc, then -val_loss.
            if args.select_metric == "val_roc_auc":
                cur_val_metric = val_m.get("roc_auc", np.nan)
            elif args.select_metric == "val_loss":
                cur_val_metric = -val_loss_eval
            else:
                cur_val_metric = val_m.get("pr_auc", np.nan)
            if not np.isfinite(cur_val_metric):
                cur_val_metric = -val_loss_eval if np.isfinite(val_loss_eval) else -np.inf
            better = (cur_val_metric > best_val_metric) or (best_epoch < 0)
            scheduler.step(cur_val_metric)
            row = {
                "iteration": epoch,
                "epoch": epoch,
                "train_loss": train_loss_eval,
                "val_loss": val_loss_eval,
                "learning_rate": get_lr(optimizer),
                "best_epoch_so_far": best_epoch if best_epoch > 0 else epoch,
                "best_val_metric_so_far": best_val_metric if np.isfinite(best_val_metric) else np.nan,
                "note": "epoch_training_attentivefp_bce_loss",
                "train_log_loss": train_loss_eval,
                "val_log_loss": val_loss_eval,
                "train_auc": train_m.get("roc_auc", np.nan),
                "val_auc": val_m.get("roc_auc", np.nan),
                "train_pr_auc": train_m.get("pr_auc", np.nan),
                "val_pr_auc": val_m.get("pr_auc", np.nan),
                "train_accuracy": train_m.get("accuracy", np.nan),
                "val_accuracy": val_m.get("accuracy", np.nan),
                "train_f1": train_m.get("f1", np.nan),
                "val_f1": val_m.get("f1", np.nan),
                "train_mcc": train_m.get("mcc", np.nan),
                "val_mcc": val_m.get("mcc", np.nan),
                "epoch_seconds": time.time() - epoch_start,
            }
            log_line = (
                f"epoch={epoch:04d} train_loss={train_loss_eval:.6f} val_loss={val_loss_eval:.6f} "
                f"train_auc={row['train_auc']:.4f} val_auc={row['val_auc']:.4f} "
                f"train_pr_auc={row['train_pr_auc']:.4f} val_pr_auc={row['val_pr_auc']:.4f} "
                f"train_f1={row['train_f1']:.4f} val_f1={row['val_f1']:.4f} "
                f"lr={row['learning_rate']:.3e}"
            )
        else:
            train_loss_eval = float(mean_squared_error(train_y, train_pred))
            val_loss_eval = float(mean_squared_error(val_y, val_pred))
            train_m = regression_metrics(train_y, train_pred)
            val_m = regression_metrics(val_y, val_pred)
            if args.select_metric == "val_spearman":
                cur_val_metric = val_m.get("spearman", np.nan)
                better = (cur_val_metric > (best_val_metric if np.isfinite(best_val_metric) else -np.inf)) or (best_epoch < 0)
                scheduler.step(cur_val_metric if np.isfinite(cur_val_metric) else -np.inf)
            else:
                cur_val_metric = val_m.get("rmse", np.nan)
                better = (cur_val_metric < best_val_metric) or (best_epoch < 0)
                scheduler.step(cur_val_metric if np.isfinite(cur_val_metric) else np.inf)
            row = {
                "iteration": epoch,
                "epoch": epoch,
                "train_loss": train_loss_eval,
                "val_loss": val_loss_eval,
                "learning_rate": get_lr(optimizer),
                "best_epoch_so_far": best_epoch if best_epoch > 0 else epoch,
                "best_val_metric_so_far": best_val_metric if np.isfinite(best_val_metric) else np.nan,
                "note": "epoch_training_attentivefp_mse_loss",
                "train_mse": train_m.get("mse", np.nan),
                "val_mse": val_m.get("mse", np.nan),
                "train_rmse": train_m.get("rmse", np.nan),
                "val_rmse": val_m.get("rmse", np.nan),
                "train_mae": train_m.get("mae", np.nan),
                "val_mae": val_m.get("mae", np.nan),
                "train_spearman": train_m.get("spearman", np.nan),
                "val_spearman": val_m.get("spearman", np.nan),
                "train_pearson": train_m.get("pearson", np.nan),
                "val_pearson": val_m.get("pearson", np.nan),
                "train_within_1_dilution": train_m.get("within_1_dilution", np.nan),
                "val_within_1_dilution": val_m.get("within_1_dilution", np.nan),
                "train_within_2_dilution": train_m.get("within_2_dilutions", np.nan),
                "val_within_2_dilution": val_m.get("within_2_dilutions", np.nan),
                "epoch_seconds": time.time() - epoch_start,
            }
            log_line = (
                f"epoch={epoch:04d} train_loss={train_loss_eval:.6f} val_loss={val_loss_eval:.6f} "
                f"train_rmse={row['train_rmse']:.4f} val_rmse={row['val_rmse']:.4f} "
                f"train_mae={row['train_mae']:.4f} val_mae={row['val_mae']:.4f} "
                f"train_spearman={row['train_spearman']:.4f} val_spearman={row['val_spearman']:.4f} "
                f"lr={row['learning_rate']:.3e}"
            )

        if better:
            best_epoch = epoch
            best_val_metric = float(cur_val_metric)
            no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "args": public_args(args),
                "run_spec": public_run_spec(spec),
                "epoch": epoch,
                "best_val_metric": best_val_metric,
                "input_dim": in_dim,
                "edge_dim": edge_dim,
                "model_config": {
                    "in_channels": in_dim,
                    "edge_dim": edge_dim,
                    "hidden_channels": int(args.hidden_dim),
                    "out_channels": 1,
                    "num_layers": int(args.num_layers),
                    "num_timesteps": int(args.num_timesteps),
                    "dropout": float(args.dropout),
                    "readout": "global_add_pool_initialization_plus_attention_GRU_refinement",
                    "predictor": f"Linear({int(args.hidden_dim)} -> 1)",
                },
            }, best_ckpt_path)
            row["checkpoint_saved"] = 1
            row["best_epoch_so_far"] = best_epoch
            row["best_val_metric_so_far"] = best_val_metric
            logger.info(log_line + f"  <-- best checkpoint saved ({args.select_metric}={best_val_metric:.6f})")
        else:
            no_improve += 1
            row["checkpoint_saved"] = 0
            logger.info(log_line + f"  no_improve={no_improve}/{args.patience}")

        train_log_rows.append(row)
        # Write live every epoch so interrupted jobs still retain curves.
        train_log_df_live = pd.DataFrame(train_log_rows)
        train_log_df_live.to_csv(out_dir / "train_log.csv", index=False)
        train_log_df_live.to_csv(out_dir / "training_curve.csv", index=False)
        make_train_long(train_log_df_live).to_csv(out_dir / "train_log_long.csv", index=False)

        if int(args.patience) > 0 and no_improve >= int(args.patience):
            logger.info(f"Early stopping at epoch={epoch}; best_epoch={best_epoch}; best_val_metric={best_val_metric:.6f}")
            break

    # Load best checkpoint for final predictions.
    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded best checkpoint: epoch={ckpt.get('epoch')}, best_val_metric={ckpt.get('best_val_metric')}")
    else:
        logger.warning("No best checkpoint found; using final model weights.")
        torch.save({"model_state_dict": model.state_dict(), "args": public_args(args), "run_spec": public_run_spec(spec)}, best_ckpt_path)

    train_y, train_pred_values, _ = predict_model(model, eval_loader_train, spec.task, device)
    val_y, val_pred_values, _ = predict_model(model, val_loader, spec.task, device)
    test_y, test_pred_values, _ = predict_model(model, test_loader, spec.task, device)

    threshold_used = 0.5
    val_best_thr = 0.5
    if spec.task == "classification":
        val_best_thr = best_f1_threshold(val_y.astype(int), val_pred_values)
        if args.threshold_mode == "val_best_f1":
            threshold_used = val_best_thr
        else:
            threshold_used = 0.5
    logger.info(f"threshold_used={threshold_used:.4f}; val_best_f1_threshold={val_best_thr:.4f}")

    # Build prediction dfs. For train reference predicted values, first build minimal train ref with predictions.
    train_ref_df = train_df.reset_index(drop=True).copy()
    if spec.task == "classification":
        train_ref_df["__train_pred_score"] = train_pred_values
        train_ref_pred_col = "__train_pred_score"
    else:
        train_ref_df["__train_pred_log10mic"] = train_pred_values
        train_ref_pred_col = "__train_pred_log10mic"

    pred_train = make_prediction_df(
        train_df, "train", spec.task, target_col, train_pred_values, spec, args.model_name,
        threshold_used, val_best_thr, mic_col, log10_col, train_ref_df, smiles_col, target_col, train_ref_pred_col, smiles_col, logger,
        include_diagnostics=False
    )
    pred_val = make_prediction_df(
        val_df, "validation", spec.task, target_col, val_pred_values, spec, args.model_name,
        threshold_used, val_best_thr, mic_col, log10_col, train_ref_df, smiles_col, target_col, train_ref_pred_col, smiles_col, logger,
        include_diagnostics=False
    )
    pred_test = make_prediction_df(
        test_df, "test", spec.task, target_col, test_pred_values, spec, args.model_name,
        threshold_used, val_best_thr, mic_col, log10_col, train_ref_df, smiles_col, target_col, train_ref_pred_col, smiles_col, logger,
        include_diagnostics=True
    )

    pred_train.to_csv(out_dir / "predictions_train.csv", index=False)
    pred_val.to_csv(out_dir / "predictions_validation.csv", index=False)
    pred_test.to_csv(out_dir / "predictions_test.csv", index=False)
    logger.info("Saved prediction files.")

    train_metrics = split_metrics_from_predictions(pred_train, spec.task, threshold=threshold_used)
    val_metrics = split_metrics_from_predictions(pred_val, spec.task, threshold=threshold_used)
    test_metrics = split_metrics_from_predictions(pred_test, spec.task, threshold=threshold_used)
    diag_metrics = make_diagnostics_metrics(pred_test, spec.task)

    last_train_loss = train_log_rows[-1]["train_loss"] if train_log_rows else np.nan
    last_val_loss = train_log_rows[-1]["val_loss"] if train_log_rows else np.nan
    best_log_row = next((r for r in train_log_rows if int(r.get("epoch", -1)) == int(best_epoch)), train_log_rows[-1] if train_log_rows else {})
    selected_train_loss = best_log_row.get("train_loss", np.nan)
    selected_val_loss = best_log_row.get("val_loss", np.nan)
    metrics_row: Dict[str, Any] = {
        "model": args.model_name,
        "model_family": "AttentiveFP",
        "hyperparameter_preset": "AttentiveFP_2x200_T2_fixed_v1",
        "task": spec.task,
        "dataset_name": spec.dataset_name,
        "genus": spec.dataset_name,
        "split_type": spec.split_type,
        "seed": spec.seed,
        "feature_type": "RDKit_molecular_graph_AttentiveFP_atom_and_bond_features",
        "target_scale": "binary_active" if spec.task == "classification" else "log10_mic",
        "threshold_from_dataset_name": spec.threshold_for_boundary,
        "threshold_for_boundary": spec.threshold_for_boundary,
        "threshold_mode": args.threshold_mode if spec.task == "classification" else "NA",
        "threshold_used": threshold_used if spec.task == "classification" else np.nan,
        "train_seconds": time.time() - start_time,
        "best_epoch": best_epoch,
        "best_val_metric": best_val_metric,
        "train_loss": selected_train_loss,
        "val_loss": selected_val_loss,
        "last_epoch": int(train_log_rows[-1]["epoch"]) if train_log_rows else np.nan,
        "last_epoch_train_loss": last_train_loss,
        "last_epoch_val_loss": last_val_loss,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "train_n": len(pred_train),
        "val_n": len(pred_val),
        "test_n": len(pred_test),
    }
    if spec.task == "classification":
        for split_name, pred in [("train", pred_train), ("val", pred_val), ("test", pred_test)]:
            y = pred["true_active"].astype(int).values
            metrics_row[f"{split_name}_pos"] = int(np.sum(y == 1))
            metrics_row[f"{split_name}_neg"] = int(np.sum(y == 0))
            metrics_row[f"{split_name}_pos_ratio"] = float(np.mean(y == 1)) if len(y) else np.nan
        metrics_row.update(prefix_metrics("train", train_metrics))
        metrics_row.update(prefix_metrics("val", val_metrics))
        metrics_row.update(prefix_metrics("test", test_metrics))
        # Minimum field aliases used by existing ML classification summaries.
        for pct in [1, 5, 10]:
            metrics_row[f"test_EF{pct}%"] = metrics_row.get(f"test_EF{pct}%", metrics_row.get(f"test_ef_{pct}pct", np.nan))
            metrics_row[f"test_Hit_rate@Top{pct}%"] = metrics_row.get(f"test_Hit_rate@Top{pct}%", metrics_row.get(f"test_hit_rate_top_{pct}pct", np.nan))
    else:
        for split_name, pred in [("train", pred_train), ("val", pred_val), ("test", pred_test)]:
            vals = pred["true_log10MIC"].astype(float).values
            metrics_row[f"{split_name}_log10mic_mean"] = float(np.nanmean(vals)) if len(vals) else np.nan
            metrics_row[f"{split_name}_log10mic_std"] = float(np.nanstd(vals)) if len(vals) else np.nan
        metrics_row.update(prefix_metrics("train", train_metrics))
        metrics_row.update(prefix_metrics("val", val_metrics))
        metrics_row.update(prefix_metrics("test", test_metrics))
        for pct in [1, 5, 10]:
            metrics_row[f"test_EF{pct}%"] = metrics_row.get(f"test_EF{pct}%", metrics_row.get(f"test_ef_{pct}pct", np.nan))
            metrics_row[f"test_Hit_rate@Top{pct}%"] = metrics_row.get(f"test_Hit_rate@Top{pct}%", metrics_row.get(f"test_hit_rate_top_{pct}pct", np.nan))
        metrics_row["test_within_2_dilutions"] = metrics_row.get("test_within_2_dilutions", np.nan)

    metrics_row.update(diag_metrics)
    pd.DataFrame([metrics_row]).to_csv(out_dir / "metrics_full.csv", index=False)
    pd.DataFrame([diag_metrics]).to_csv(out_dir / "diagnostics_metrics.csv", index=False)
    make_subgroup_metrics(pred_test, spec.task).to_csv(out_dir / "diagnostic_subgroup_metrics.csv", index=False)
    logger.info("Saved metrics and diagnostics.")

    save_json({
        "model": args.model_name,
        "model_family": "AttentiveFP",
        "hyperparameter_preset": "AttentiveFP_2x200_T2_fixed_v1",
        "feature_type": "RDKit_molecular_graph_AttentiveFP_atom_and_bond_features",
        "atom_feature_dim": in_dim,
        "atom_features": {
            "atom_types": ATOM_TYPES,
            "degrees": DEGREES,
            "formal_charges": FORMAL_CHARGES,
            "hybridizations": [str(x) for x in HYBRIDIZATIONS],
            "extra": ["is_aromatic", "is_in_ring", "total_num_hs", "mass_x0.01"],
        },
        "edge_features_used": True,
        "graph_pooling": "global_add_pool_initialization_plus_attention_GRU_refinement",
        "pyg_model_class": "torch_geometric.nn.models.AttentiveFP",
        "nn_diagnostic_fingerprint": {"type": "Morgan", "radius": 2, "nBits": 2048, "similarity": "Tanimoto"},
        "diagnostic_scope": {"train": "none", "validation": "none", "test": ["nearest_neighbor", "activity_cliff"]},
        "bond_features": {"bond_types": [str(x) for x in BOND_TYPES], "stereo": [str(x) for x in BOND_STEREO], "extra": ["is_conjugated", "is_in_ring"], "bond_feature_dim": edge_dim},
    }, out_dir / "feature_summary.json")

    save_json({
        "model_name": args.model_name,
        "hyperparameter_preset": "AttentiveFP_2x200_T2_fixed_v1",
        "in_channels": in_dim,
        "edge_dim": edge_dim,
        "hidden_channels": int(args.hidden_dim),
        "out_channels": 1,
        "num_layers": int(args.num_layers),
        "num_timesteps": int(args.num_timesteps),
        "dropout": float(args.dropout),
        "independent_prediction_head_hidden_layer": False,
        "final_output_layer": f"Linear({int(args.hidden_dim)} -> 1)",
        "graph_readout": "global_add_pool_initialization_plus_attention_GRU_refinement",
        "expected_trainable_params": EXPECTED_TRAINABLE_PARAMS,
        "actual_total_params": total_params,
        "actual_trainable_params": trainable_params,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "eval_batch_size": int(args.eval_batch_size),
        "lr": float(args.lr),
        "min_lr": float(args.min_lr),
        "weight_decay": float(args.weight_decay),
        "patience": int(args.patience),
        "grad_clip": float(args.grad_clip),
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau",
        "scheduler_mode": "max" if spec.task == "classification" else "min",
        "scheduler_factor": float(args.lr_reduce_factor),
        "scheduler_patience": int(args.lr_patience),
        "use_pos_weight": False,
        "class_weighting": "none",
        "select_metric": args.select_metric,
    }, out_dir / "model_params.json")

    save_json({
        "start_time": now_str(),
        "script": Path(sys.argv[0]).name,
        "run_spec": public_run_spec(spec),
        "hyperparameter_preset": "AttentiveFP_2x200_T2_fixed_v1",
        "diagnostic_scope": {"train": "none", "validation": "none", "test": ["nearest_neighbor", "activity_cliff"]},
        "loading_info": {"loading_mode": loading_info.get("loading_mode"), "files": {k: Path(v).name for k, v in loading_info.get("paths", {}).items()}},
        "columns": {
            "smiles_col": smiles_col,
            "target_col": target_col,
            "mic_col": mic_col,
            "log10_col": log10_col,
        },
        "versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_geometric": getattr(torch_geometric, "__version__", "unknown"),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "elapsed_seconds": time.time() - start_time,
    }, out_dir / "run_info.json")

    logger.info(f"Completed run in {time.time() - start_time:.1f} seconds. Best epoch={best_epoch}, best_val_metric={best_val_metric}")
    logger.info("=" * 100)
    return out_dir


def make_train_long(train_log_df: pd.DataFrame) -> pd.DataFrame:
    id_cols = [c for c in ["iteration", "epoch"] if c in train_log_df.columns]
    metric_cols = [c for c in train_log_df.columns if c not in set(id_cols + ["note", "checkpoint_saved", "epoch_seconds"])]
    rows: List[Dict[str, Any]] = []
    for _, r in train_log_df.iterrows():
        for c in metric_cols:
            val = r[c]
            if pd.isna(val):
                continue
            split = "train" if c.startswith("train_") or c == "train_loss" else "validation" if c.startswith("val_") or c == "val_loss" else "global"
            metric = c.replace("train_", "", 1).replace("val_", "", 1)
            rows.append({
                "iteration": r.get("iteration", np.nan),
                "epoch": r.get("epoch", np.nan),
                "split": split,
                "metric": metric,
                "value": val,
            })
    return pd.DataFrame(rows)


# ----------------------------- config and cli -----------------------------


def load_dataset_config(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"dataset_config not found: {p}")
    if p.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(p)
    else:
        df = pd.read_csv(p)
    records = []
    for _, r in df.iterrows():
        d = {k: (None if pd.isna(v) else v) for k, v in r.to_dict().items()}
        records.append(d)
    return records


def build_run_specs(args: argparse.Namespace) -> List[RunSpec]:
    tasks = parse_list_arg(args.tasks, ["classification"])
    split_types = parse_list_arg(args.split_types, ["random", "scaffold", "repeated_stratified_perimeter", "strict_perimeter"])
    seeds = [int(x) for x in parse_list_arg(args.seeds, ["0"])]
    configs = load_dataset_config(args.dataset_config)
    specs: List[RunSpec] = []

    def seeds_for_split(split_type: str) -> List[int]:
        """Return valid split seeds for each split type.

        In the current AntibioticBench data release, strict_perimeter has only
        one official split: seed_0.  Even if --seeds is set to 0,1,2,3,4 for
        the other split types, strict_perimeter is intentionally restricted to
        seed_0 to avoid submitting jobs for non-existent folders.
        """
        if str(split_type).strip().lower() == "strict_perimeter":
            return [0]
        return seeds

    if configs:
        for cfg in configs:
            dataset = str(cfg.get("dataset_name") or cfg.get("dataset") or cfg.get("genus") or "").strip()
            if not dataset:
                continue
            cfg_tasks = parse_list_arg(str(cfg.get("task") or cfg.get("tasks") or ",".join(tasks)), tasks)
            cfg_tasks = tasks if any(t.lower() in {"both", "all"} for t in cfg_tasks) else cfg_tasks
            threshold = safe_float(cfg.get("threshold_for_boundary") or cfg.get("threshold") or cfg.get("active_threshold"), args.threshold_for_boundary)
            data_path = cfg.get("data_path") or cfg.get("csv_path")
            class_path = cfg.get("classification_csv") or cfg.get("classification_path")
            reg_path = cfg.get("regression_csv") or cfg.get("regression_path")
            for task in cfg_tasks:
                if task not in tasks:
                    continue
                for split_type in split_types:
                    for seed in seeds_for_split(split_type):
                        specs.append(RunSpec(
                            dataset_name=dataset,
                            task=task,
                            split_type=split_type,
                            seed=seed,
                            threshold_for_boundary=threshold,
                            data_path=str(class_path if task == "classification" and class_path else reg_path if task == "regression" and reg_path else data_path) if (data_path or class_path or reg_path) else None,
                            train_path=str(cfg.get(f"{task}_{split_type}_seed{seed}_train") or cfg.get("train_path") or "") or None,
                            val_path=str(cfg.get(f"{task}_{split_type}_seed{seed}_validation") or cfg.get(f"{task}_{split_type}_seed{seed}_valid") or cfg.get("val_path") or cfg.get("validation_path") or "") or None,
                            test_path=str(cfg.get(f"{task}_{split_type}_seed{seed}_test") or cfg.get("test_path") or "") or None,
                        ))
    else:
        datasets = parse_list_arg(args.datasets, [])
        if not datasets:
            raise ValueError("Please provide --datasets or --dataset-config.")
        for dataset in datasets:
            for task in tasks:
                for split_type in split_types:
                    for seed in seeds_for_split(split_type):
                        specs.append(RunSpec(
                            dataset_name=dataset,
                            task=task,
                            split_type=split_type,
                            seed=seed,
                            threshold_for_boundary=float(args.threshold_for_boundary),
                        ))
    return specs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train AntibioticBench PyG AttentiveFP and save unified classification/regression outputs.")
    p.add_argument("--data-root", required=True, help="Root directory containing datasets/splits.")
    p.add_argument("--out-root", required=True, help="Output root directory.")
    p.add_argument("--dataset-config", default=None, help="CSV config with dataset_name, task, threshold/data_path columns. Recommended for 12 datasets.")
    p.add_argument("--datasets", default=None, help="Comma/space-separated dataset names if not using config.")
    p.add_argument("--tasks", default="regression", help="This entry script is for regression. Keep as regression unless debugging.")
    p.add_argument("--split-types", default="random,scaffold,repeated_stratified_perimeter,strict_perimeter", help="Comma-separated split types. Default includes random, scaffold, repeated_stratified_perimeter, and strict_perimeter.")
    p.add_argument("--seeds", default="0,1,2,3,4", help="Comma-separated split seeds for non-strict splits. strict_perimeter is always restricted to seed_0 because this split has only one official seed.")
    p.add_argument("--threshold-for-boundary", type=float, default=32.0, help="Default MIC threshold for boundary/screening if config does not specify it.")

    p.add_argument("--smiles-col", default="smiles")
    p.add_argument("--active-col", default="active")
    p.add_argument("--log10-mic-col", default="log10_mic")
    p.add_argument("--mic-col", default="mic_numeric")
    p.add_argument("--split-col", default="split")
    p.add_argument("--split-type-col", default="split_type")
    p.add_argument("--seed-col", default="split_seed")

    p.add_argument("--model-name", default="AttentiveFP")
    p.add_argument("--hidden-dim", type=int, default=200, help="PyG AttentiveFP hidden_channels.")
    p.add_argument("--num-layers", type=int, default=2, help="Number of atom-level GNN/message-passing layers.")
    p.add_argument("--num-timesteps", type=int, default=2, help="Number of global molecular readout refinement steps.")
    p.add_argument("--dropout", type=float, default=0.20)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--lr-reduce-factor", type=float, default=0.5)
    p.add_argument("--lr-patience", type=int, default=10)
    p.add_argument("--select-metric", default="val_rmse", choices=["val_pr_auc", "val_roc_auc", "val_loss", "val_rmse", "val_spearman"], help="Best checkpoint metric. Regression benchmark default: val_rmse.")
    p.add_argument("--threshold-mode", default="fixed_0p5", choices=["fixed_0p5", "val_best_f1"], help="Classification final threshold mode. Main benchmark should use fixed_0p5.")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--log-every-batches", type=int, default=0, help="Print batch-level loss every N batches; 0 disables batch logging.")
    p.add_argument("--torch-num-threads", type=int, default=1, help="CPU torch threads. GPU runs are usually unaffected; 1 avoids CPU oversubscription.")
    p.add_argument("--single-run-index", type=int, default=None, help="Run only one indexed spec. Useful for Slurm arrays.")
    p.add_argument("--print-run-plan", action="store_true", help="Print expanded run list and exit.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.tasks = "regression"
    if args.select_metric in {"val_pr_auc", "val_roc_auc"}:
        args.select_metric = "val_rmse"
    if int(args.torch_num_threads) > 0:
        torch.set_num_threads(int(args.torch_num_threads))
        try:
            torch.set_num_interop_threads(max(1, int(args.torch_num_threads)))
        except Exception:
            pass
    # Safer metric default for regression if user runs only regression and did not set select_metric explicitly is hard to detect;
    # we keep CLI explicit in launcher. Internally each run falls back sensibly if metric is unavailable.
    specs = build_run_specs(args)
    if args.print_run_plan:
        print(json.dumps([public_run_spec(s) for s in specs], ensure_ascii=False, indent=2))
        return
    if args.single_run_index is not None:
        idx = int(args.single_run_index)
        if idx < 0 or idx >= len(specs):
            raise IndexError(f"--single-run-index {idx} out of range 0..{len(specs)-1}")
        specs = [specs[idx]]
    print(f"[{now_str()}] Expanded {len(specs)} run(s).")
    for i, spec in enumerate(specs):
        # Override select metric per task when user did not pass a compatible metric.
        if spec.task == "regression" and args.select_metric in {"val_pr_auc", "val_roc_auc"}:
            args.select_metric = "val_rmse"
        elif spec.task == "classification" and args.select_metric in {"val_rmse", "val_spearman"}:
            args.select_metric = "val_pr_auc"
        print(f"[{now_str()}] Running {i+1}/{len(specs)}: dataset={spec.dataset_name}, task={spec.task}, split={spec.split_type}, seed={spec.seed}")
        train_one_run(args, spec)


if __name__ == "__main__":
    main()
