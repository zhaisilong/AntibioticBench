#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict activity-cliff pair analysis for AntibioticBench GCN outputs.

This script is intentionally separated from GCN training.
It computes activity-cliff (AC) pairs after predictions are available, using:
  - all_internal: all molecules in train + validation + test within one run
  - test_test: only molecule pairs inside test set

Definition:
  Tanimoto similarity >= sim_cutoff and |log10MIC_i - log10MIC_j| >= log10(fold_change)

Outputs per run:
  <run_dir>/activity_cliff_pair_analysis/
    ac_pairs_all_internal.csv
    ac_molecule_level_all_internal.csv
    ac_pairs_test_test.csv
    ac_molecule_level_test_test.csv
    ac_pair_summary.csv

Combined outputs:
  OUT_ROOT/model_name/_combined_summary/activity_cliff_pair_analysis/
    ac_pair_summary_all_runs.csv
    ac_molecule_level_all_runs.csv
"""
from __future__ import annotations

if __name__ == "__main__":
    from privacy_utils import install_privacy_excepthook
    install_privacy_excepthook()

import argparse
import glob
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
except Exception as e:  # pragma: no cover
    raise RuntimeError("RDKit is required. Recommended: conda install -c conda-forge rdkit") from e

RDLogger.DisableLog("rdApp.*")


def ensure_dir(p: str | Path) -> Path:
    q = Path(p)
    q.mkdir(parents=True, exist_ok=True)
    return q


def smiles_to_fp(smiles: str):
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    except Exception:
        return None


def bulk_tanimoto(fp, fps: Sequence[Any]) -> np.ndarray:
    if fp is None or not fps:
        return np.full(len(fps), np.nan)
    valid = [x is not None for x in fps]
    valid_fps = [x for x in fps if x is not None]
    out = np.full(len(fps), np.nan, dtype=float)
    if not valid_fps:
        return out
    sims = DataStructs.BulkTanimotoSimilarity(fp, valid_fps)
    k = 0
    for i, ok in enumerate(valid):
        if ok:
            out[i] = float(sims[k])
            k += 1
    return out


def find_run_dirs(model_root: Path) -> List[Path]:
    files = [Path(f) for f in glob.glob(str(model_root / "**" / "predictions_test.csv"), recursive=True)]
    files = [f for f in files if "_combined_summary" not in str(f)]
    return sorted({f.parent for f in files})


def read_run_predictions(run_dir: Path) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for split_name, fn in [("train", "predictions_train.csv"), ("validation", "predictions_validation.csv"), ("test", "predictions_test.csv")]:
        p = run_dir / fn
        if p.exists():
            df = pd.read_csv(p)
            if "split" not in df.columns:
                df["split"] = split_name
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["__row_in_run"] = np.arange(len(out), dtype=int)
    return out


def get_true_log10(df: pd.DataFrame) -> np.ndarray:
    for c in ["true_log10MIC", "log10_mic", "log10_mic_for_diagnosis"]:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").values.astype(float)
    if "mic_numeric" in df.columns:
        mic = pd.to_numeric(df["mic_numeric"], errors="coerce").values.astype(float)
        return np.log10(np.where(mic > 0, mic, np.nan))
    return np.full(len(df), np.nan)


def get_mic_numeric(df: pd.DataFrame, true_log10: np.ndarray) -> np.ndarray:
    for c in ["mic_numeric", "true_MIC"]:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").values.astype(float)
    return np.power(10.0, true_log10)


def infer_task(df: pd.DataFrame) -> str:
    if "task" in df.columns and df["task"].notna().any():
        return str(df["task"].dropna().iloc[0])
    if "true_active" in df.columns:
        return "classification"
    if "true_log10MIC" in df.columns:
        return "regression"
    return "unknown"


def get_prediction_value(df: pd.DataFrame, task: str) -> Tuple[np.ndarray, str]:
    if task == "classification":
        for c in ["pred_score", "pred_prob"]:
            if c in df.columns:
                return pd.to_numeric(df[c], errors="coerce").values.astype(float), c
    else:
        for c in ["pred_log10MIC"]:
            if c in df.columns:
                return pd.to_numeric(df[c], errors="coerce").values.astype(float), c
    return np.full(len(df), np.nan), "NA"


def sign_or_nan(x: float) -> float:
    if not np.isfinite(x) or x == 0:
        return np.nan
    return float(np.sign(x))


def pair_direction_correct(task: str, log10_i: float, log10_j: float, pred_i: float, pred_j: float) -> float:
    if not all(np.isfinite(v) for v in [log10_i, log10_j, pred_i, pred_j]):
        return np.nan
    # true_dir > 0 means molecule i has lower MIC and is more active than molecule j.
    true_dir = sign_or_nan(log10_j - log10_i)
    if not np.isfinite(true_dir):
        return np.nan
    if task == "classification":
        # Higher score means more active. Correct if i gets higher score when i has lower MIC.
        pred_dir = sign_or_nan(pred_i - pred_j)
    else:
        # Lower predicted log10MIC means more active. Correct if predicted MIC ordering agrees with true MIC ordering.
        pred_dir = sign_or_nan(pred_j - pred_i)
    if not np.isfinite(pred_dir):
        return np.nan
    return float(pred_dir == true_dir)


def base_meta(df: pd.DataFrame) -> Dict[str, Any]:
    cols = ["model", "model_family", "task", "dataset_name", "genus", "split_type", "seed"]
    meta: Dict[str, Any] = {}
    for c in cols:
        meta[c] = df[c].dropna().iloc[0] if c in df.columns and df[c].notna().any() else np.nan
    return meta


def compute_scope_pairs(
    df: pd.DataFrame,
    scope: str,
    sim_cutoff: float,
    diff_cut: float,
    fold_change: float,
    max_pairs_per_run: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if scope == "test_test":
        work = df[df["split"].astype(str).str.lower() == "test"].reset_index(drop=True).copy()
    elif scope == "all_internal":
        work = df.reset_index(drop=True).copy()
    else:
        raise ValueError(f"Unknown scope: {scope}")

    meta = base_meta(df)
    task = str(meta.get("task")) if str(meta.get("task")) != "nan" else infer_task(work)
    true_log10 = get_true_log10(work)
    mic = get_mic_numeric(work, true_log10)
    pred_value, pred_col = get_prediction_value(work, task)
    smiles = work["smiles"].astype(str).tolist() if "smiles" in work.columns else [""] * len(work)
    fps = [smiles_to_fp(s) for s in smiles]

    pair_rows: List[Dict[str, Any]] = []
    mol_counts = np.zeros(len(work), dtype=int)
    mol_maxdiff = np.full(len(work), np.nan, dtype=float)
    mol_dir_hits: List[List[float]] = [[] for _ in range(len(work))]
    possible_pairs = int(len(work) * (len(work) - 1) // 2)
    sim_ge_pairs = 0
    ac_pairs = 0
    truncated = False

    for j in range(len(work)):
        if j == 0:
            continue
        sims = bulk_tanimoto(fps[j], fps[:j])
        if len(sims) == 0:
            continue
        cand_idx = np.where(np.isfinite(sims) & (sims >= sim_cutoff))[0]
        if len(cand_idx):
            sim_ge_pairs += int(len(cand_idx))
        for i in cand_idx:
            if not np.isfinite(true_log10[i]) or not np.isfinite(true_log10[j]):
                continue
            d = float(abs(true_log10[i] - true_log10[j]))
            if d < diff_cut:
                continue
            ac_pairs += 1
            corr = pair_direction_correct(task, true_log10[i], true_log10[j], pred_value[i], pred_value[j])
            mol_counts[i] += 1
            mol_counts[j] += 1
            mol_maxdiff[i] = np.nanmax([mol_maxdiff[i], d]) if np.isfinite(mol_maxdiff[i]) else d
            mol_maxdiff[j] = np.nanmax([mol_maxdiff[j], d]) if np.isfinite(mol_maxdiff[j]) else d
            if np.isfinite(corr):
                mol_dir_hits[i].append(float(corr))
                mol_dir_hits[j].append(float(corr))
            if max_pairs_per_run and max_pairs_per_run > 0 and len(pair_rows) >= max_pairs_per_run:
                truncated = True
                continue
            pair_rows.append({
                **meta,
                "scope": scope,
                "pair_i_row_in_scope": int(i),
                "pair_j_row_in_scope": int(j),
                "pair_i_original_row": work["__row_in_run"].iloc[i] if "__row_in_run" in work.columns else int(i),
                "pair_j_original_row": work["__row_in_run"].iloc[j] if "__row_in_run" in work.columns else int(j),
                "split_i": work["split"].iloc[i] if "split" in work.columns else np.nan,
                "split_j": work["split"].iloc[j] if "split" in work.columns else np.nan,
                "smiles_i": smiles[i],
                "smiles_j": smiles[j],
                "tanimoto_similarity": float(sims[i]),
                "log10MIC_i": float(true_log10[i]),
                "log10MIC_j": float(true_log10[j]),
                "MIC_i": float(mic[i]) if np.isfinite(mic[i]) else np.nan,
                "MIC_j": float(mic[j]) if np.isfinite(mic[j]) else np.nan,
                "abs_log10MIC_diff": d,
                "fold_diff": float(10.0 ** d),
                "pred_col": pred_col,
                "pred_i": float(pred_value[i]) if np.isfinite(pred_value[i]) else np.nan,
                "pred_j": float(pred_value[j]) if np.isfinite(pred_value[j]) else np.nan,
                "pair_direction_correct": corr,
            })

    pair_columns = [
        "model", "model_family", "task", "dataset_name", "genus", "split_type", "seed", "scope",
        "pair_i_row_in_scope", "pair_j_row_in_scope", "pair_i_original_row", "pair_j_original_row",
        "split_i", "split_j", "smiles_i", "smiles_j", "tanimoto_similarity",
        "log10MIC_i", "log10MIC_j", "MIC_i", "MIC_j", "abs_log10MIC_diff", "fold_diff",
        "pred_col", "pred_i", "pred_j", "pair_direction_correct",
    ]
    pair_df = pd.DataFrame(pair_rows, columns=pair_columns)
    mol_rows: List[Dict[str, Any]] = []
    for i in range(len(work)):
        acc = float(np.mean(mol_dir_hits[i])) if mol_dir_hits[i] else np.nan
        mol_rows.append({
            **meta,
            "scope": scope,
            "row_in_scope": int(i),
            "original_row": work["__row_in_run"].iloc[i] if "__row_in_run" in work.columns else int(i),
            "split": work["split"].iloc[i] if "split" in work.columns else np.nan,
            "smiles": smiles[i],
            "true_log10MIC": float(true_log10[i]) if np.isfinite(true_log10[i]) else np.nan,
            "mic_numeric": float(mic[i]) if np.isfinite(mic[i]) else np.nan,
            "pred_col": pred_col,
            "pred_value": float(pred_value[i]) if np.isfinite(pred_value[i]) else np.nan,
            "ac_flag_strict_pairwise": int(mol_counts[i] > 0),
            "ac_pair_count_strict_pairwise": int(mol_counts[i]),
            "ac_max_abs_log10MIC_diff_strict_pairwise": float(mol_maxdiff[i]) if np.isfinite(mol_maxdiff[i]) else np.nan,
            "ac_pair_directional_accuracy_strict_pairwise": acc,
        })
    mol_df = pd.DataFrame(mol_rows)
    summary: Dict[str, Any] = {
        **meta,
        "scope": scope,
        "n_molecules": int(len(work)),
        "possible_pairs": possible_pairs,
        "similarity_cutoff": float(sim_cutoff),
        "fold_change_cutoff": float(fold_change),
        "log10_diff_cutoff": float(diff_cut),
        "similarity_ge_cutoff_pairs": int(sim_ge_pairs),
        "ac_pairs": int(ac_pairs),
        "ac_pair_rate_among_all_pairs": float(ac_pairs / possible_pairs) if possible_pairs else np.nan,
        "ac_pair_rate_among_similar_pairs": float(ac_pairs / sim_ge_pairs) if sim_ge_pairs else np.nan,
        "ac_molecules": int((mol_counts > 0).sum()),
        "ac_molecule_rate": float((mol_counts > 0).mean()) if len(work) else np.nan,
        "pair_directional_accuracy": float(pair_df["pair_direction_correct"].mean()) if (not pair_df.empty and "pair_direction_correct" in pair_df.columns) else np.nan,
        "pair_file_truncated": bool(truncated),
        "max_pairs_per_run": int(max_pairs_per_run),
    }
    return pair_df, mol_df, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute strict all-internal and test-test activity-cliff pair analysis from GCN prediction files.")
    ap.add_argument("--out-root", required=True, help="Output directory used by train_gcn_antibioticbench.py; AC summaries are written directly under this folder")
    ap.add_argument("--model-name", default="GCN", help="Model name kept for compatibility; output root is used directly and no extra model-name subdirectory is added")
    ap.add_argument("--scopes", default="all_internal,test_test", help="Comma-separated scopes: all_internal,test_test")
    ap.add_argument("--sim-cutoff", type=float, default=0.7)
    ap.add_argument("--fold-change", type=float, default=16.0)
    ap.add_argument("--max-pairs-per-run", type=int, default=0, help="0 means write all AC pairs; positive value truncates pair CSV rows but summary still counts all AC pairs.")
    ap.add_argument("--skip-pair-files", action="store_true", help="Only write molecule-level and summary files, not pair-level CSV files.")
    args = ap.parse_args()

    model_root = Path(args.out_root)
    run_dirs = find_run_dirs(model_root)
    if not run_dirs:
        raise RuntimeError("No predictions_test.csv found under the configured output root")
    scopes = [s.strip() for s in str(args.scopes).replace(" ", ",").split(",") if s.strip()]
    diff_cut = math.log10(float(args.fold_change))

    all_summary: List[Dict[str, Any]] = []
    all_mol: List[pd.DataFrame] = []
    for k, run_dir in enumerate(run_dirs, start=1):
        print(f"[{k}/{len(run_dirs)}] {run_dir.relative_to(model_root)}")
        df = read_run_predictions(run_dir)
        if df.empty or "smiles" not in df.columns:
            print(f"  [WARN] missing predictions or smiles, skipped")
            continue
        ac_dir = ensure_dir(run_dir / "activity_cliff_pair_analysis")
        run_summaries: List[Dict[str, Any]] = []
        for scope in scopes:
            pair_df, mol_df, summary = compute_scope_pairs(
                df=df,
                scope=scope,
                sim_cutoff=float(args.sim_cutoff),
                diff_cut=diff_cut,
                fold_change=float(args.fold_change),
                max_pairs_per_run=int(args.max_pairs_per_run),
            )
            if not args.skip_pair_files:
                pair_df.to_csv(ac_dir / f"ac_pairs_{scope}.csv", index=False)
            mol_df.to_csv(ac_dir / f"ac_molecule_level_{scope}.csv", index=False)
            run_summaries.append(summary)
            all_summary.append({**summary, "run_dir": str(run_dir.relative_to(model_root))})
            all_mol.append(mol_df.assign(run_dir=str(run_dir.relative_to(model_root))))
        pd.DataFrame(run_summaries).to_csv(ac_dir / "ac_pair_summary.csv", index=False)

    combined_dir = ensure_dir(model_root / "_combined_summary" / "activity_cliff_pair_analysis")
    summary_df = pd.DataFrame(all_summary)
    summary_df.to_csv(combined_dir / "ac_pair_summary_all_runs.csv", index=False)
    if all_mol:
        pd.concat(all_mol, ignore_index=True, sort=False).to_csv(combined_dir / "ac_molecule_level_all_runs.csv", index=False)
    print("Saved combined AC pair analysis.")


if __name__ == "__main__":
    main()
