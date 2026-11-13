#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute molecular landscape roughness diagnostics for AntibioticBench splits.

This script is model-independent. It reads train/validation/test.csv from
classification split folders and computes:

Hard-coded input columns:
  - SMILES column: smiles
  - classification label column: active
  - MODI: mean kNN label consistency for classification labels.
  - 1 - MODI: roughness proxy for classification landscape.
  - mean local log2(MIC) variance among k nearest neighbors.
  - activity cliff density: fraction of high-similarity pairs with >=16-fold MIC difference.
  - split-level active ratio, boundary ratio, MIC/logMIC summaries.

Expected layout:
  DATA_ROOT/dataset_name/classification/{random,scaffold,repeated_stratified_perimeter,strict_perimeter}/seed_0/train.csv ...

Outputs:
  OUTDIR/landscape_roughness_by_split_seed.csv
  OUTDIR/landscape_roughness_mean_sd.csv
"""

import argparse
import math
import re
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

N_BITS = 2048
RADIUS = 2
# Hard-coded modeling/diagnostic columns.
# The molecule column is always df["smiles"], and the classification label column is df["active"].
HARD_CODED_SMILES_COL = "smiles"
HARD_CODED_TARGET_COL = "active"
DEFAULT_SMILES_COLS = [HARD_CODED_SMILES_COL]
DEFAULT_VALUE_COLS = ["mic_numeric", "value", "MIC", "mic", "mic_value", "median_value", "value_ug_ml", "mic_ug_ml"]
DEFAULT_LOGMIC_COLS = ["log2_value", "logMIC", "logmic", "log2_MIC", "log2_mic", "log2_mic_value"]


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


def find_col(df: pd.DataFrame, preferred: Optional[str], candidates: List[str], required=True) -> Optional[str]:
    if preferred and preferred in df.columns:
        return preferred
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Cannot find required column from candidates={candidates}. Columns={list(df.columns)}")
    return None


def require_hardcoded_columns(df: pd.DataFrame) -> tuple:
    smiles_col = HARD_CODED_SMILES_COL
    target_col = HARD_CODED_TARGET_COL
    missing = [c for c in [smiles_col, target_col] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required hard-coded column(s) {missing}. "
            f"This script requires molecule column 'smiles' and label column 'active'. "
            f"Available columns: {list(df.columns)}"
        )
    return smiles_col, target_col


def mol_from_smiles(smiles: str):
    try:
        return Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None


def morgan_fp_bitvect(smiles: str):
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=N_BITS)


def tanimoto_to_all(fp, fps: List) -> np.ndarray:
    sims = np.zeros(len(fps), dtype=np.float32)
    if fp is None:
        return sims
    valid_idx = [i for i, x in enumerate(fps) if x is not None]
    if not valid_idx:
        return sims
    vals = DataStructs.BulkTanimotoSimilarity(fp, [fps[i] for i in valid_idx])
    sims[np.array(valid_idx, dtype=int)] = np.array(vals, dtype=np.float32)
    return sims


def infer_log2mic(df: pd.DataFrame, logmic_col: Optional[str], value_col: Optional[str]) -> pd.Series:
    if logmic_col is not None:
        return pd.to_numeric(df[logmic_col], errors="coerce")
    if value_col is not None:
        v = pd.to_numeric(df[value_col], errors="coerce")
        return np.log2(v.where(v > 0))
    return pd.Series(np.nan, index=df.index)


def infer_boundary(df: pd.DataFrame, value_col: Optional[str], threshold: Optional[float]) -> pd.Series:
    for c in ["boundary_flag", "mic_boundary_flag", "mic_boundary", "MIC_boundary", "is_boundary"]:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    if value_col is not None and threshold is not None:
        v = pd.to_numeric(df[value_col], errors="coerce")
        return ((v >= threshold / 2.0) & (v <= threshold * 2.0)).astype(int)
    return pd.Series(np.nan, index=df.index)


def compute_roughness(df: pd.DataFrame, smiles_col: str, target_col: str, log2mic: pd.Series, k: int, ac_sim_cutoff: float, ac_log2diff_cutoff: float) -> dict:
    smiles = df[smiles_col].astype(str).tolist()
    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int).values
    log2_arr = pd.to_numeric(log2mic, errors="coerce").values.astype(float)
    fps = [morgan_fp_bitvect(s) for s in smiles]
    n = len(df)
    if n <= 1:
        return {"MODI": np.nan, "landscape_roughness_1_minus_MODI": np.nan, "mean_knn_log2mic_variance": np.nan, "activity_cliff_density": np.nan}

    modi_vals = []
    local_vars = []
    high_sim_pairs = 0
    ac_pairs = 0

    for i, fp in enumerate(fps):
        sims = tanimoto_to_all(fp, fps)
        sims[i] = -1.0
        valid_order = np.argsort(-sims)
        nn_idx = valid_order[:min(k, n - 1)]
        if len(nn_idx):
            modi_vals.append(float(np.mean(y[nn_idx] == y[i])))
            vals = log2_arr[nn_idx]
            vals = vals[np.isfinite(vals)]
            if len(vals) >= 2:
                local_vars.append(float(np.var(vals, ddof=1)))

        # Count activity cliff pairs once, using j > i.
        js = np.where(sims >= ac_sim_cutoff)[0]
        js = js[js > i]
        if len(js) and np.isfinite(log2_arr[i]):
            diffs = np.abs(log2_arr[js] - log2_arr[i])
            valid = np.isfinite(diffs)
            high_sim_pairs += int(np.sum(valid))
            ac_pairs += int(np.sum(valid & (diffs >= ac_log2diff_cutoff)))

    modi = float(np.mean(modi_vals)) if modi_vals else np.nan
    return {
        "MODI_k": int(k),
        "MODI": modi,
        "landscape_roughness_1_minus_MODI": float(1 - modi) if np.isfinite(modi) else np.nan,
        "mean_knn_log2mic_variance": float(np.mean(local_vars)) if local_vars else np.nan,
        "activity_cliff_sim_cutoff": float(ac_sim_cutoff),
        "activity_cliff_log2diff_cutoff": float(ac_log2diff_cutoff),
        "high_similarity_pair_count": int(high_sim_pairs),
        "activity_cliff_pair_count": int(ac_pairs),
        "activity_cliff_density": float(ac_pairs / high_sim_pairs) if high_sim_pairs > 0 else np.nan,
    }


def split_distribution(df: pd.DataFrame, target_col: str, value_col: Optional[str], log2mic: pd.Series, boundary: pd.Series, prefix: str) -> dict:
    y = pd.to_numeric(df[target_col], errors="coerce")
    log2_arr = pd.to_numeric(log2mic, errors="coerce")
    out = {
        f"{prefix}_n": int(len(df)),
        f"{prefix}_active": int(np.nansum(y == 1)),
        f"{prefix}_inactive": int(np.nansum(y == 0)),
        f"{prefix}_active_ratio": float(np.nanmean(y == 1)) if len(df) else np.nan,
        f"{prefix}_boundary_ratio": float(np.nanmean(boundary == 1)) if len(df) and boundary.notna().any() else np.nan,
        f"{prefix}_log2mic_mean": float(log2_arr.mean()) if log2_arr.notna().any() else np.nan,
        f"{prefix}_log2mic_std": float(log2_arr.std()) if log2_arr.notna().sum() >= 2 else np.nan,
        f"{prefix}_log2mic_median": float(log2_arr.median()) if log2_arr.notna().any() else np.nan,
        f"{prefix}_log2mic_q25": float(log2_arr.quantile(0.25)) if log2_arr.notna().any() else np.nan,
        f"{prefix}_log2mic_q75": float(log2_arr.quantile(0.75)) if log2_arr.notna().any() else np.nan,
    }
    if value_col is not None:
        v = pd.to_numeric(df[value_col], errors="coerce")
        out[f"{prefix}_mic_median"] = float(v.median()) if v.notna().any() else np.nan
    return out


def discover_runs(data_root: Path, split_types: List[str], seeds: List[int], datasets=None):
    dataset_dirs = [p for p in data_root.iterdir() if p.is_dir() and (p / "classification").exists()]
    dataset_dirs = sorted(dataset_dirs, key=numeric_sort_key)
    if datasets:
        keep = set(datasets)
        dataset_dirs = [p for p in dataset_dirs if p.name in keep or parse_genus_from_dataset_name(p.name) in keep]
    runs = []
    for ds in dataset_dirs:
        for sp in split_types:
            for seed in seeds:
                sd = ds / "classification" / sp / f"seed_{seed}"
                if (sd / "train.csv").exists() and (sd / "validation.csv").exists() and (sd / "test.csv").exists():
                    runs.append((ds, sp, seed, sd))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parents[2] / "outputs" / "ML" / "landscape_roughness"))
    ap.add_argument("--split-types", nargs="+", default=["random", "scaffold", "repeated_stratified_perimeter", "strict_perimeter"], help="Split folders to evaluate. strict_perimeter may only contain seed_0; missing seed folders are skipped automatically.")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--smiles-col", default=None, help="Deprecated/ignored. Molecule column is hard-coded as 'smiles'.")
    ap.add_argument("--target-col", default="active", help="Deprecated/ignored. Classification label column is hard-coded as 'active'.")
    ap.add_argument("--value-col", default=None)
    ap.add_argument("--logmic-col", default=None)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--ac-sim-cutoff", type=float, default=0.7)
    ap.add_argument("--ac-log2diff-cutoff", type=float, default=4.0)
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds, split_type, seed, seed_dir in discover_runs(data_root, args.split_types, args.seeds, args.datasets):
        dataset_name = ds.name
        genus = parse_genus_from_dataset_name(dataset_name)
        threshold = parse_threshold_from_dataset_name(dataset_name)
        print(f"Computing landscape: {dataset_name} | {split_type} | seed_{seed}", flush=True)
        train = pd.read_csv(seed_dir / "train.csv")
        val = pd.read_csv(seed_dir / "validation.csv")
        test = pd.read_csv(seed_dir / "test.csv")
        full = pd.concat([
            train.assign(split="train"), val.assign(split="validation"), test.assign(split="test")
        ], ignore_index=True)
        # Hard-coded input columns for consistency with model training.
        # Command-line --smiles-col and --target-col are intentionally ignored.
        smiles_col, target_col = require_hardcoded_columns(full)
        value_col = find_col(full, args.value_col, DEFAULT_VALUE_COLS, required=False)
        logmic_col = find_col(full, args.logmic_col, DEFAULT_LOGMIC_COLS, required=False)
        log2mic = infer_log2mic(full, logmic_col, value_col)
        boundary = infer_boundary(full, value_col, threshold)
        row = {
            "dataset_name": dataset_name,
            "genus": genus,
            "split_type": split_type,
            "seed": seed,
            "smiles_col": smiles_col,
            "target_col": target_col,
            "value_col": value_col,
            "logmic_col": logmic_col,
            "threshold": threshold,
        }
        row.update(compute_roughness(full, smiles_col, target_col, log2mic, args.k, args.ac_sim_cutoff, args.ac_log2diff_cutoff))
        for split_name, df_split in [("train", train), ("validation", val), ("test", test)]:
            idx = full["split"] == split_name
            row.update(split_distribution(full[idx], target_col, value_col, log2mic[idx], boundary[idx], split_name))
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "landscape_roughness_by_split_seed.csv", index=False)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in ["seed"]]
    if not df.empty:
        agg = df.groupby(["dataset_name", "genus", "split_type"])[numeric_cols].agg(["mean", "std", "count"]).reset_index()
        agg.columns = ["_".join([str(x) for x in col if str(x)]) for col in agg.columns]
        agg.to_csv(outdir / "landscape_roughness_mean_sd.csv", index=False)
    print(f"Saved outputs to {outdir.name}")


if __name__ == "__main__":
    main()
