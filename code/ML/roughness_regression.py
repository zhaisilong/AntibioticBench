#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute model-independent molecular landscape roughness diagnostics for
AntibioticBench log10MIC regression splits.

Expected input layout:
    DATA_ROOT/dataset_name/regression/{random,scaffold,repeated_stratified_perimeter,strict_perimeter}/seed_0/train.csv
    DATA_ROOT/dataset_name/regression/{...}/seed_0/validation.csv
    DATA_ROOT/dataset_name/regression/{...}/seed_0/test.csv

Hard-coded columns:
    smiles      : molecular input column
    log10_mic   : regression target, log10(MIC ug/mL)

Outputs:
    regression_landscape_roughness_by_split_seed.csv
    regression_landscape_roughness_mean_sd.csv
    regression_landscape_roughness_molecule_level.csv  (optional)

Main diagnostics:
    - local kNN log10MIC variance and absolute difference
    - nearest-neighbor log10MIC continuity
    - MIC-specific activity cliff density:
        Tanimoto >= 0.7 and |log10_mic_i - log10_mic_j| >= log10(16)
    - test-to-train NN similarity and log10MIC discontinuity
    - boundary and MIC-range distribution based on dataset threshold
"""

import argparse
from privacy_utils import sanitize_metadata
import math
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


N_BITS = 2048
RADIUS = 2
HARD_CODED_SMILES_COL = "smiles"
HARD_CODED_TARGET_COL = "log10_mic"
DEFAULT_VALUE_COLS = [
    "mic_numeric", "value", "MIC", "mic", "mic_value", "median_value",
    "value_ug_ml", "mic_ug_ml", "MIC_ug_ml", "mic_ug_per_ml",
]
DEFAULT_ACTIVE_COLS = ["active", "label_active", "is_active", "activity_label"]
DILUTION1_LOG10 = math.log10(2.0)
DILUTION2_LOG10 = math.log10(4.0)
DEFAULT_AC_SIM_CUTOFF = 0.7
DEFAULT_AC_LOG10_DIFF_CUTOFF = math.log10(16.0)


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


def find_col(df: pd.DataFrame, preferred: Optional[str], candidates: List[str], required: bool = False) -> Optional[str]:
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
        raise ValueError(f"Cannot find required column from candidates={candidates}. Columns={list(df.columns)}")
    return None


def require_hardcoded_columns(df: pd.DataFrame, split_name: str = "data") -> Tuple[str, str]:
    missing = [c for c in [HARD_CODED_SMILES_COL, HARD_CODED_TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required hard-coded column(s) {missing} in {split_name}. "
            f"Regression roughness requires '{HARD_CODED_SMILES_COL}' and '{HARD_CODED_TARGET_COL}'. "
            f"Available columns: {list(df.columns)}"
        )
    return HARD_CODED_SMILES_COL, HARD_CODED_TARGET_COL


def finite(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr[np.isfinite(arr)]


def safe_mean(x):
    arr = finite(x)
    return float(np.mean(arr)) if arr.size else np.nan


def safe_median(x):
    arr = finite(x)
    return float(np.median(arr)) if arr.size else np.nan


def safe_std(x):
    arr = finite(x)
    return float(np.std(arr, ddof=1)) if arr.size >= 2 else np.nan


def safe_quantile(x, q):
    arr = finite(x)
    return float(np.quantile(arr, q)) if arr.size else np.nan


def safe_ratio(num, den):
    if den is None or pd.isna(den) or den == 0:
        return np.nan
    if num is None or pd.isna(num):
        return np.nan
    return float(num / den)


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


def build_bitvectors(smiles_list: Iterable[str]) -> List:
    return [morgan_fp_bitvect(s) for s in smiles_list]


def tanimoto_to_all(fp, fps: List) -> np.ndarray:
    sims = np.zeros(len(fps), dtype=np.float32)
    if fp is None:
        return sims
    valid_idx = [i for i, x in enumerate(fps) if x is not None]
    if not valid_idx:
        return sims
    vals = DataStructs.BulkTanimotoSimilarity(fp, [fps[i] for i in valid_idx])
    sims[np.array(valid_idx, dtype=int)] = np.asarray(vals, dtype=np.float32)
    return sims


def infer_mic_from_log10(log10_values: pd.Series) -> pd.Series:
    y = pd.to_numeric(log10_values, errors="coerce")
    return 10.0 ** y


def infer_boundary_from_log10(log10_values: pd.Series, threshold: Optional[float]) -> pd.Series:
    if threshold is None or threshold <= 0:
        return pd.Series(np.nan, index=log10_values.index)
    y = pd.to_numeric(log10_values, errors="coerce")
    lo = math.log10(threshold / 2.0)
    hi = math.log10(threshold * 2.0)
    return ((y >= lo) & (y <= hi)).astype(int)


def infer_active_from_log10(log10_values: pd.Series, threshold: Optional[float]) -> pd.Series:
    if threshold is None or threshold <= 0:
        return pd.Series(np.nan, index=log10_values.index)
    y = pd.to_numeric(log10_values, errors="coerce")
    return (y <= math.log10(threshold)).astype(int)


def mic_range_group_from_log10(log10_values: pd.Series, threshold: Optional[float]) -> pd.Series:
    if threshold is None or threshold <= 0:
        return pd.Series("unknown", index=log10_values.index)
    y = pd.to_numeric(log10_values, errors="coerce")
    lo = math.log10(threshold / 2.0)
    hi = math.log10(threshold * 2.0)
    out = pd.Series("unknown", index=log10_values.index, dtype=object)
    out[y <= lo] = "strong_MIC_le_T_over_2"
    out[(y > lo) & (y <= hi)] = "boundary_T_over_2_to_2T"
    out[y > hi] = "weak_MIC_gt_2T"
    return out


def split_distribution(df: pd.DataFrame, target_col: str, threshold: Optional[float], prefix: str) -> Dict:
    y = pd.to_numeric(df[target_col], errors="coerce")
    mic = infer_mic_from_log10(y)
    boundary = infer_boundary_from_log10(y, threshold)
    active = infer_active_from_log10(y, threshold)
    return {
        f"{prefix}_n": int(len(df)),
        f"{prefix}_log10mic_mean": safe_mean(y),
        f"{prefix}_log10mic_std": safe_std(y),
        f"{prefix}_log10mic_median": safe_median(y),
        f"{prefix}_log10mic_q25": safe_quantile(y, 0.25),
        f"{prefix}_log10mic_q75": safe_quantile(y, 0.75),
        f"{prefix}_log10mic_min": float(np.nanmin(y)) if np.isfinite(y).any() else np.nan,
        f"{prefix}_log10mic_max": float(np.nanmax(y)) if np.isfinite(y).any() else np.nan,
        f"{prefix}_log10mic_range": (float(np.nanmax(y) - np.nanmin(y)) if np.isfinite(y).any() else np.nan),
        f"{prefix}_mic_median": safe_median(mic),
        f"{prefix}_active_ratio_inferred_by_threshold": safe_mean(active),
        f"{prefix}_boundary_ratio": safe_mean(boundary),
        f"{prefix}_strong_ratio_MIC_le_T_over_2": safe_mean(mic_range_group_from_log10(y, threshold) == "strong_MIC_le_T_over_2"),
        f"{prefix}_boundary_range_ratio_T_over_2_to_2T": safe_mean(mic_range_group_from_log10(y, threshold) == "boundary_T_over_2_to_2T"),
        f"{prefix}_weak_ratio_MIC_gt_2T": safe_mean(mic_range_group_from_log10(y, threshold) == "weak_MIC_gt_2T"),
    }


def compute_full_landscape_metrics(
    df: pd.DataFrame,
    smiles_col: str,
    target_col: str,
    k: int,
    ac_sim_cutoff: float,
    ac_log10diff_cutoff: float,
    save_molecule_level: bool = False,
) -> Tuple[Dict, pd.DataFrame]:
    smiles = df[smiles_col].astype(str).tolist()
    y = pd.to_numeric(df[target_col], errors="coerce").values.astype(float)
    fps = build_bitvectors(smiles)
    n = len(df)

    if n <= 1:
        base = {
            "full_n": int(n),
            "valid_smiles_count": int(sum(fp is not None for fp in fps)),
            "invalid_smiles_count": int(sum(fp is None for fp in fps)),
            "mean_knn_log10mic_variance": np.nan,
            "activity_cliff_density": np.nan,
        }
        return base, pd.DataFrame()

    local_var = np.full(n, np.nan, dtype=float)
    local_absdiff_mean = np.full(n, np.nan, dtype=float)
    local_absdiff_median = np.full(n, np.nan, dtype=float)
    nn1_sim = np.full(n, np.nan, dtype=float)
    nn1_absdiff = np.full(n, np.nan, dtype=float)
    knn_mean_sim = np.full(n, np.nan, dtype=float)
    ac_neighbor_count = np.zeros(n, dtype=int)
    rough_neighbor_count = np.zeros(n, dtype=int)

    high_sim_pairs = 0
    ac_pairs = 0
    rough_pairs_8fold = 0
    rough_pairs_4fold = 0
    all_pair_absdiffs_highsim = []

    for i, fp in enumerate(fps):
        sims = tanimoto_to_all(fp, fps)
        if n:
            sims[i] = -1.0
        valid_order = np.argsort(-sims)
        nn_idx = valid_order[:min(k, n - 1)]
        nn_idx = [j for j in nn_idx if sims[j] >= 0 and np.isfinite(y[j])]
        if nn_idx and np.isfinite(y[i]):
            vals = y[nn_idx]
            diffs = np.abs(vals - y[i])
            local_absdiff_mean[i] = float(np.mean(diffs))
            local_absdiff_median[i] = float(np.median(diffs))
            knn_mean_sim[i] = float(np.mean(sims[nn_idx]))
            if len(vals) >= 2:
                local_var[i] = float(np.var(vals, ddof=1))
            if len(nn_idx) >= 1:
                nn1 = nn_idx[0]
                nn1_sim[i] = float(sims[nn1])
                nn1_absdiff[i] = float(abs(y[nn1] - y[i])) if np.isfinite(y[nn1]) else np.nan

        js = np.where(sims >= ac_sim_cutoff)[0]
        js = js[js > i]
        if len(js) and np.isfinite(y[i]):
            diffs = np.abs(y[js] - y[i])
            valid = np.isfinite(diffs)
            diffs = diffs[valid]
            high_sim_pairs += int(len(diffs))
            ac_here = diffs >= ac_log10diff_cutoff
            ac_pairs += int(np.sum(ac_here))
            rough_pairs_8fold += int(np.sum(diffs >= math.log10(8.0)))
            rough_pairs_4fold += int(np.sum(diffs >= math.log10(4.0)))
            if diffs.size:
                all_pair_absdiffs_highsim.extend(diffs.tolist())
            # Per-molecule counts: count all high-sim neighbors, not just j > i.
        js_all = np.where(sims >= ac_sim_cutoff)[0]
        js_all = js_all[js_all != i]
        if len(js_all) and np.isfinite(y[i]):
            diffs_all = np.abs(y[js_all] - y[i])
            valid_all = np.isfinite(diffs_all)
            ac_neighbor_count[i] = int(np.sum(valid_all & (diffs_all >= ac_log10diff_cutoff)))
            rough_neighbor_count[i] = int(np.sum(valid_all & (diffs_all >= math.log10(4.0))))

    highsim_absdiff = np.asarray(all_pair_absdiffs_highsim, dtype=float)
    metrics = {
        "full_n": int(n),
        "valid_smiles_count": int(sum(fp is not None for fp in fps)),
        "invalid_smiles_count": int(sum(fp is None for fp in fps)),
        "MODI": np.nan,  # classification-only metric, kept as NA for schema clarity
        "landscape_roughness_1_minus_MODI": np.nan,
        "knn_k": int(k),
        "mean_knn_log10mic_variance": safe_mean(local_var),
        "median_knn_log10mic_variance": safe_median(local_var),
        "mean_knn_log10mic_absdiff": safe_mean(local_absdiff_mean),
        "median_knn_log10mic_absdiff": safe_median(local_absdiff_median),
        "mean_nearest_neighbor_log10mic_absdiff": safe_mean(nn1_absdiff),
        "median_nearest_neighbor_log10mic_absdiff": safe_median(nn1_absdiff),
        "mean_nearest_neighbor_similarity": safe_mean(nn1_sim),
        "median_nearest_neighbor_similarity": safe_median(nn1_sim),
        "mean_knn_similarity": safe_mean(knn_mean_sim),
        "activity_cliff_sim_cutoff": float(ac_sim_cutoff),
        "activity_cliff_log10diff_cutoff": float(ac_log10diff_cutoff),
        "activity_cliff_fold_change_cutoff": float(10 ** ac_log10diff_cutoff),
        "high_similarity_pair_count": int(high_sim_pairs),
        "activity_cliff_pair_count": int(ac_pairs),
        "activity_cliff_density": safe_ratio(ac_pairs, high_sim_pairs),
        "rough_pair_4fold_count": int(rough_pairs_4fold),
        "rough_pair_4fold_density": safe_ratio(rough_pairs_4fold, high_sim_pairs),
        "rough_pair_8fold_count": int(rough_pairs_8fold),
        "rough_pair_8fold_density": safe_ratio(rough_pairs_8fold, high_sim_pairs),
        "highsim_pair_absdiff_log10_mean": safe_mean(highsim_absdiff),
        "highsim_pair_absdiff_log10_median": safe_median(highsim_absdiff),
    }

    mol_df = pd.DataFrame()
    if save_molecule_level:
        mol_df = df[[smiles_col, target_col]].copy()
        mol_df["local_knn_log10mic_variance"] = local_var
        mol_df["local_knn_log10mic_absdiff_mean"] = local_absdiff_mean
        mol_df["local_knn_log10mic_absdiff_median"] = local_absdiff_median
        mol_df["nearest_neighbor_similarity"] = nn1_sim
        mol_df["nearest_neighbor_log10mic_absdiff"] = nn1_absdiff
        mol_df["mean_knn_similarity"] = knn_mean_sim
        mol_df["ac_neighbor_count"] = ac_neighbor_count
        mol_df["rough_neighbor_4fold_count"] = rough_neighbor_count
        mol_df["local_ac_flag"] = (ac_neighbor_count > 0).astype(int)

    return metrics, mol_df


def compute_test_to_train_metrics(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    smiles_col: str,
    target_col: str,
    split_prefix: str,
    ac_sim_cutoff: float,
    ac_log10diff_cutoff: float,
) -> Tuple[Dict, pd.DataFrame]:
    train_smiles = train_df[smiles_col].astype(str).tolist()
    eval_smiles = eval_df[smiles_col].astype(str).tolist()
    y_train = pd.to_numeric(train_df[target_col], errors="coerce").values.astype(float)
    y_eval = pd.to_numeric(eval_df[target_col], errors="coerce").values.astype(float)
    train_fps = build_bitvectors(train_smiles)
    eval_fps = build_bitvectors(eval_smiles)

    rows = []
    nn_sims, nn_absdiffs, ac_flags, ac_counts = [], [], [], []
    for i, fp in enumerate(eval_fps):
        sims = tanimoto_to_all(fp, train_fps)
        if len(sims):
            nn_idx = int(np.argmax(sims))
            nn_sim = float(sims[nn_idx])
        else:
            nn_idx, nn_sim = -1, np.nan
        nn_absdiff = np.nan
        if nn_idx >= 0 and np.isfinite(y_eval[i]) and np.isfinite(y_train[nn_idx]):
            nn_absdiff = float(abs(y_eval[i] - y_train[nn_idx]))
        ac_mask = np.zeros(len(train_df), dtype=bool)
        if np.isfinite(y_eval[i]):
            diffs = np.abs(y_train - y_eval[i])
            ac_mask = (sims >= ac_sim_cutoff) & np.isfinite(diffs) & (diffs >= ac_log10diff_cutoff)
        ac_count = int(np.sum(ac_mask))
        ac_flag = int(ac_count > 0)
        nn_bin = ">=0.7" if nn_sim >= 0.7 else ("0.4-0.7" if nn_sim >= 0.4 else "<0.4")
        rows.append({
            "eval_row": i,
            "nn_train_row": nn_idx,
            "nn_sim": nn_sim,
            "nn_bin": nn_bin,
            "nn_train_log10_mic": float(y_train[nn_idx]) if nn_idx >= 0 and np.isfinite(y_train[nn_idx]) else np.nan,
            "nn_log10mic_absdiff": nn_absdiff,
            "ac_relative_to_train_flag": ac_flag,
            "ac_relative_to_train_neighbor_count": ac_count,
        })
        nn_sims.append(nn_sim)
        nn_absdiffs.append(nn_absdiff)
        ac_flags.append(ac_flag)
        ac_counts.append(ac_count)

    diag = pd.DataFrame(rows)
    out = {
        f"{split_prefix}_to_train_nn_sim_mean": safe_mean(nn_sims),
        f"{split_prefix}_to_train_nn_sim_median": safe_median(nn_sims),
        f"{split_prefix}_to_train_nn_sim_q25": safe_quantile(nn_sims, 0.25),
        f"{split_prefix}_to_train_nn_sim_q75": safe_quantile(nn_sims, 0.75),
        f"{split_prefix}_to_train_nn_lt_0p4_ratio": safe_mean(np.asarray(nn_sims) < 0.4),
        f"{split_prefix}_to_train_nn_lt_0p7_ratio": safe_mean(np.asarray(nn_sims) < 0.7),
        f"{split_prefix}_to_train_nn_log10mic_absdiff_mean": safe_mean(nn_absdiffs),
        f"{split_prefix}_to_train_nn_log10mic_absdiff_median": safe_median(nn_absdiffs),
        f"{split_prefix}_to_train_ac_relative_flag_ratio": safe_mean(ac_flags),
        f"{split_prefix}_to_train_ac_relative_molecule_count": int(np.nansum(ac_flags)),
        f"{split_prefix}_to_train_ac_relative_neighbor_count_mean": safe_mean(ac_counts),
    }

    if not diag.empty:
        for bin_name in [">=0.7", "0.4-0.7", "<0.4"]:
            g = diag[diag["nn_bin"] == bin_name]
            key = bin_name.replace(">=", "ge_").replace("<", "lt_").replace("-", "_").replace(".", "p")
            out[f"{split_prefix}_nn_{key}_n"] = int(len(g))
            out[f"{split_prefix}_nn_{key}_log10mic_absdiff_mean"] = safe_mean(g["nn_log10mic_absdiff"]) if len(g) else np.nan
            out[f"{split_prefix}_nn_{key}_ac_relative_flag_ratio"] = safe_mean(g["ac_relative_to_train_flag"]) if len(g) else np.nan
    return out, diag


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


def main():
    ap = argparse.ArgumentParser(description="Compute log10MIC regression landscape roughness for AntibioticBench.")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parents[2] / "outputs" / "ML" / "regression_landscape_roughness"))
    ap.add_argument("--split-types", nargs="+", default=["random", "scaffold", "repeated_stratified_perimeter", "strict_perimeter"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--ac-sim-cutoff", type=float, default=DEFAULT_AC_SIM_CUTOFF)
    ap.add_argument("--ac-log10diff-cutoff", type=float, default=DEFAULT_AC_LOG10_DIFF_CUTOFF)
    ap.add_argument("--save-molecule-level", action="store_true", help="Save per-molecule local roughness diagnostics. Can be large.")
    ap.add_argument("--max-runs", type=int, default=None)
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    outdir = Path(args.outdir).resolve()
    ensure_dir(outdir)

    runs = discover_runs(data_root, args.split_types, args.seeds, args.datasets)
    if args.max_runs is not None:
        runs = runs[:args.max_runs]
    if not runs:
        raise RuntimeError(f"No regression runs found under {data_root}. Expected DATA_ROOT/dataset/regression/split/seed_x/*.csv")

    print(f"[{now()}] Discovered {len(runs)} regression split runs.", flush=True)
    rows = []
    mol_rows = []
    failed = []

    for run in runs:
        try:
            dataset_name = run["dataset_name"]
            genus = run["genus"]
            split_type = run["split_type"]
            seed = run["seed"]
            threshold = run["threshold"]
            print("\n" + "=" * 100, flush=True)
            print(f"[{now()}] Roughness regression: {dataset_name} | {split_type} | seed_{seed}", flush=True)
            print("=" * 100, flush=True)

            train = pd.read_csv(run["train_csv"])
            val = pd.read_csv(run["val_csv"])
            test = pd.read_csv(run["test_csv"])
            for split_name, df in [("train", train), ("validation", val), ("test", test)]:
                require_hardcoded_columns(df, split_name)

            full = pd.concat([
                train.assign(split="train"),
                val.assign(split="validation"),
                test.assign(split="test"),
            ], ignore_index=True)
            smiles_col, target_col = HARD_CODED_SMILES_COL, HARD_CODED_TARGET_COL

            row = {
                "dataset_name": dataset_name,
                "genus": genus,
                "task": "regression",
                "split_type": split_type,
                "seed": seed,
                "threshold_from_dataset_name": threshold,
                "smiles_col": smiles_col,
                "target_col": target_col,
                "k": args.k,
                "ac_sim_cutoff": args.ac_sim_cutoff,
                "ac_log10diff_cutoff": args.ac_log10diff_cutoff,
                "ac_fold_change_cutoff": 10 ** args.ac_log10diff_cutoff,
            }
            for split_name, df_split in [("train", train), ("validation", val), ("test", test), ("full", full)]:
                row.update(split_distribution(df_split, target_col, threshold, split_name))

            full_metrics, mol_df = compute_full_landscape_metrics(
                full, smiles_col, target_col, k=args.k,
                ac_sim_cutoff=args.ac_sim_cutoff,
                ac_log10diff_cutoff=args.ac_log10diff_cutoff,
                save_molecule_level=args.save_molecule_level,
            )
            row.update(full_metrics)

            val_to_train, val_diag = compute_test_to_train_metrics(
                train, val, smiles_col, target_col, "validation",
                ac_sim_cutoff=args.ac_sim_cutoff,
                ac_log10diff_cutoff=args.ac_log10diff_cutoff,
            )
            test_to_train, test_diag = compute_test_to_train_metrics(
                train, test, smiles_col, target_col, "test",
                ac_sim_cutoff=args.ac_sim_cutoff,
                ac_log10diff_cutoff=args.ac_log10diff_cutoff,
            )
            row.update(val_to_train)
            row.update(test_to_train)

            rows.append(row)

            if args.save_molecule_level and not mol_df.empty:
                mol_df.insert(0, "dataset_name", dataset_name)
                mol_df.insert(1, "genus", genus)
                mol_df.insert(2, "split_type", split_type)
                mol_df.insert(3, "seed", seed)
                mol_df.insert(4, "analysis_scope", "full_train_validation_test")
                mol_rows.append(mol_df)

            # Save per-run test-to-train diagnostics for debugging.
            per_run_dir = outdir / "per_run" / dataset_name / split_type / f"seed_{seed}"
            ensure_dir(per_run_dir)
            val_diag.to_csv(per_run_dir / "validation_to_train_nn_roughness.csv", index=False)
            test_diag.to_csv(per_run_dir / "test_to_train_nn_roughness.csv", index=False)
            pd.DataFrame([row]).to_csv(per_run_dir / "regression_landscape_roughness_metrics.csv", index=False)

        except Exception as e:
            failed.append(sanitize_metadata({**{k: str(v) for k, v in run.items()}, "error": type(e).__name__}))
            print(f"[{now()}] ERROR: {type(e).__name__}", flush=True)
            continue

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "regression_landscape_roughness_by_split_seed.csv", index=False)

    if not df.empty:
        id_cols = ["dataset_name", "genus", "task", "split_type"]
        numeric_cols = [c for c in df.columns if c not in ["seed"] and pd.api.types.is_numeric_dtype(df[c])]
        agg = df.groupby(id_cols)[numeric_cols].agg(["mean", "std", "count"]).reset_index()
        agg.columns = ["_".join([str(x) for x in col if str(x)]) for col in agg.columns]
        # Add human-readable mean±sd columns for selected key metrics.
        key_metrics = [
            "mean_knn_log10mic_variance", "mean_knn_log10mic_absdiff",
            "mean_nearest_neighbor_log10mic_absdiff", "activity_cliff_density",
            "test_to_train_nn_sim_mean", "test_to_train_nn_log10mic_absdiff_mean",
            "test_to_train_ac_relative_flag_ratio", "test_to_train_nn_lt_0p4_ratio",
        ]
        for m in key_metrics:
            mean_col, std_col = f"{m}_mean", f"{m}_std"
            if mean_col in agg.columns and std_col in agg.columns:
                agg[f"{m}_mean±sd"] = [
                    "" if pd.isna(a) else (f"{a:.4f}" if pd.isna(b) else f"{a:.4f} ± {b:.4f}")
                    for a, b in zip(agg[mean_col], agg[std_col])
                ]
        agg.to_csv(outdir / "regression_landscape_roughness_mean_sd.csv", index=False)

    if args.save_molecule_level and mol_rows:
        pd.concat(mol_rows, ignore_index=True).to_csv(outdir / "regression_landscape_roughness_molecule_level.csv", index=False)

    if failed:
        pd.DataFrame(failed).to_csv(outdir / "failed_runs.csv", index=False)
        print(f"[{now()}] Finished with {len(failed)} failed runs. See failed_runs.csv", flush=True)
    else:
        print(f"[{now()}] Finished all regression roughness runs successfully.", flush=True)
    print(f"Saved outputs to: {outdir.name}", flush=True)


if __name__ == "__main__":
    main()
