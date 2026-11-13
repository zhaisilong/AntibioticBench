#!/usr/bin/env python3
"""Model-independent molecular landscape roughness for AntibioticBench.

This is intentionally separate from KPGT training so the same roughness values can be
reused by RF/XGB/GCN/GIN/Chemprop/KPGT/Uni-Mol instead of recomputing per model.
"""
from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from kpgt_benchmark_utils import (
    CLASS_TARGET_CANDIDATES,
    REG_TARGET_CANDIDATES,
    SMILES_CANDIDATES,
    coerce_binary,
    enumerate_runs,
    extract_mic_arrays,
    find_column,
    load_dataset_config,
    mic_range_fields,
    validate_input_run,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["classification", "regression"], required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--dataset-config", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--n-bits", type=int, default=2048)
    p.add_argument("--ac-sim-cutoff", type=float, default=0.7)
    p.add_argument("--ac-log10diff-cutoff", type=float, default=math.log10(16.0))
    return p.parse_args()


def fp_for_smiles(smiles: str, radius: int, n_bits: int):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def build_canonical(frames: Mapping[str, pd.DataFrame], task: str) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], str, str]:
    first = frames["train"]
    smiles_col = find_column(first, SMILES_CANDIDATES, required=True)
    target_col = find_column(first, CLASS_TARGET_CANDIDATES if task == "classification" else REG_TARGET_CANDIDATES, required=True)
    parts = []
    for split_name in ("train", "validation", "test"):
        df = frames[split_name]
        target = coerce_binary(df[target_col]) if task == "classification" else pd.to_numeric(df[target_col], errors="coerce").to_numpy(float)
        part = df.copy().reset_index(drop=True)
        part["__smiles"] = part[smiles_col].astype(str)
        part["__target"] = target
        part["__source_split"] = split_name
        part["__source_row"] = np.arange(len(part), dtype=int)
        parts.append(part)
    full = pd.concat(parts, ignore_index=True)
    full["__dup_occ"] = full.groupby(["__smiles", "__target"], sort=False, dropna=False).cumcount()
    canonical = full.sort_values(["__smiles", "__target", "__dup_occ"], kind="mergesort").reset_index(drop=True)
    canonical["canonical_row"] = np.arange(len(canonical), dtype=int)
    key_map = {
        (str(r["__smiles"]), float(r["__target"]), int(r["__dup_occ"])): int(r["canonical_row"])
        for _, r in canonical.iterrows()
    }
    split_rows = {}
    for split_name in ("train", "validation", "test"):
        sub = full[full["__source_split"] == split_name].sort_values("__source_row")
        split_rows[split_name] = np.asarray([
            key_map[(str(r["__smiles"]), float(r["__target"]), int(r["__dup_occ"]))]
            for _, r in sub.iterrows()
        ], dtype=int)
    return canonical, split_rows, smiles_col, target_col


def compute_molecule_roughness(
    canonical: pd.DataFrame,
    task: str,
    k: int,
    radius: int,
    n_bits: int,
    ac_sim_cutoff: float,
    ac_log10diff_cutoff: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    smiles = canonical["__smiles"].astype(str).tolist()
    target = canonical["__target"].to_numpy(float)
    _, log10mic, _ = extract_mic_arrays(canonical)
    if task == "regression":
        log10mic = target.copy()
    fps = [fp_for_smiles(s, radius, n_bits) for s in smiles]
    valid_positions = [i for i, fp in enumerate(fps) if fp is not None]
    valid_fps = [fps[i] for i in valid_positions]
    pos_to_valid = {row: pos for pos, row in enumerate(valid_positions)}
    n = len(canonical)
    rows: List[Dict[str, Any]] = []
    high_pair_count = 0
    cliff_pair_count = 0

    for i in range(n):
        row: Dict[str, Any] = {
            "canonical_row": i,
            "smiles": smiles[i],
            "target": target[i],
            "valid_smiles": int(fps[i] is not None),
        }
        if fps[i] is None or not valid_fps:
            row.update({
                "nearest_neighbor_row": np.nan,
                "nearest_neighbor_similarity": np.nan,
                "nearest_neighbor_abs_log10mic_diff": np.nan,
                "local_log10mic_variance_k10": np.nan,
                "local_log10mic_sd_k10": np.nan,
                "modi_local_consistency_k10": np.nan,
                "activity_cliff_neighbor_count": 0,
            })
            rows.append(row)
            continue
        sims_valid = np.asarray(DataStructs.BulkTanimotoSimilarity(fps[i], valid_fps), dtype=float)
        self_pos = pos_to_valid.get(i)
        if self_pos is not None:
            sims_valid[self_pos] = -1.0
        order = np.argsort(-sims_valid)
        order = order[sims_valid[order] >= 0]
        if len(order):
            nn_pos = int(order[0])
            nn_row = int(valid_positions[nn_pos])
            row["nearest_neighbor_row"] = nn_row
            row["nearest_neighbor_similarity"] = float(sims_valid[nn_pos])
            if np.isfinite(log10mic[i]) and np.isfinite(log10mic[nn_row]):
                row["nearest_neighbor_abs_log10mic_diff"] = float(abs(log10mic[i] - log10mic[nn_row]))
            else:
                row["nearest_neighbor_abs_log10mic_diff"] = np.nan
        else:
            row["nearest_neighbor_row"] = np.nan
            row["nearest_neighbor_similarity"] = np.nan
            row["nearest_neighbor_abs_log10mic_diff"] = np.nan

        chosen = order[: min(k, len(order))]
        chosen_rows = np.asarray([valid_positions[int(pos)] for pos in chosen], dtype=int)
        vals = log10mic[chosen_rows]
        vals = vals[np.isfinite(vals)]
        row["local_log10mic_variance_k10"] = float(np.var(vals, ddof=0)) if len(vals) else np.nan
        row["local_log10mic_sd_k10"] = float(np.std(vals, ddof=0)) if len(vals) else np.nan
        if task == "classification" and len(chosen_rows):
            row["modi_local_consistency_k10"] = float(np.mean(target[chosen_rows] == target[i]))
        else:
            row["modi_local_consistency_k10"] = np.nan

        cliff_neighbors = 0
        candidate_pos = np.where(sims_valid >= ac_sim_cutoff)[0]
        for pos in candidate_pos:
            j = int(valid_positions[int(pos)])
            if j == i or not (np.isfinite(log10mic[i]) and np.isfinite(log10mic[j])):
                continue
            if abs(log10mic[i] - log10mic[j]) >= ac_log10diff_cutoff:
                cliff_neighbors += 1
            if j > i:
                high_pair_count += 1
                if abs(log10mic[i] - log10mic[j]) >= ac_log10diff_cutoff:
                    cliff_pair_count += 1
        row["activity_cliff_neighbor_count"] = cliff_neighbors
        rows.append(row)

    molecule = pd.DataFrame(rows)
    summary = {
        "n_molecules": n,
        "valid_smiles_n": int(molecule["valid_smiles"].sum()),
        "k": k,
        "fingerprint_radius": radius,
        "fingerprint_n_bits": n_bits,
        "mean_nearest_neighbor_similarity": molecule["nearest_neighbor_similarity"].mean(),
        "mean_nearest_neighbor_abs_log10mic_diff": molecule["nearest_neighbor_abs_log10mic_diff"].mean(),
        "mean_local_log10mic_variance_k10": molecule["local_log10mic_variance_k10"].mean(),
        "median_local_log10mic_variance_k10": molecule["local_log10mic_variance_k10"].median(),
        "mean_local_log10mic_sd_k10": molecule["local_log10mic_sd_k10"].mean(),
        "MODI": molecule["modi_local_consistency_k10"].mean() if task == "classification" else np.nan,
        "landscape_roughness_1_minus_MODI": (
            1.0 - molecule["modi_local_consistency_k10"].mean()
            if task == "classification" and molecule["modi_local_consistency_k10"].notna().any()
            else np.nan
        ),
        "activity_cliff_similarity_cutoff": ac_sim_cutoff,
        "activity_cliff_log10diff_cutoff": ac_log10diff_cutoff,
        "high_similarity_pair_count": high_pair_count,
        "activity_cliff_pair_count": cliff_pair_count,
        "activity_cliff_density": cliff_pair_count / high_pair_count if high_pair_count else np.nan,
    }
    return molecule, summary


def aggregate_mean_sd(df: pd.DataFrame) -> pd.DataFrame:
    groups = ["dataset_name", "genus", "split_type"]
    numeric = [c for c in df.select_dtypes(include=[np.number]).columns if c != "seed"]
    if df.empty:
        return pd.DataFrame()
    agg = df.groupby(groups)[numeric].agg(["mean", "std", "count"]).reset_index()
    agg.columns = ["_".join(str(x) for x in tup if str(x)) for tup in agg.columns]
    return agg


def main() -> None:
    args = parse_args()
    config = load_dataset_config(args.dataset_config.resolve())
    runs = enumerate_runs(config, args.task)
    args.outdir.mkdir(parents=True, exist_ok=True)
    dataset_cache: Dict[str, Tuple[str, pd.DataFrame, Dict[str, Any], pd.DataFrame]] = {}
    summary_rows: List[Dict[str, Any]] = []
    molecule_rows: List[pd.DataFrame] = []

    for run in runs:
        paths = validate_input_run(args.data_root.resolve(), run)
        frames = {name: pd.read_csv(path) for name, path in paths.items()}
        canonical, split_rows, smiles_col, target_col = build_canonical(frames, args.task)
        content = canonical[["__smiles", "__target"]].to_csv(index=False, float_format="%.17g")
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        cache_key = f"{run['dataset_name']}:{content_hash}"
        if cache_key not in dataset_cache:
            print(f"Computing molecular roughness once for {run['dataset_name']} ({args.task})", flush=True)
            mol, summary = compute_molecule_roughness(
                canonical,
                args.task,
                args.k,
                args.radius,
                args.n_bits,
                args.ac_sim_cutoff,
                args.ac_log10diff_cutoff,
            )
            dataset_cache[cache_key] = (content_hash, mol, summary, canonical)
        _, mol, summary, canonical_cached = dataset_cache[cache_key]

        row = {
            "dataset_name": run["dataset_name"],
            "genus": run["genus"],
            "task": args.task,
            "split_type": run["split_type"],
            "seed": run["seed"],
            "threshold_for_boundary": run["threshold_for_boundary"],
            "smiles_col": smiles_col,
            "target_col": target_col,
            "dataset_content_hash": content_hash,
            **summary,
        }
        for split_name, idx in split_rows.items():
            source = canonical.iloc[idx]
            target = source["__target"].to_numpy(float)
            mic, log10mic, _ = extract_mic_arrays(source)
            if args.task == "regression":
                log10mic = target
                mic = np.power(10.0, target)
            boundary, _ = mic_range_fields(mic, float(run["threshold_for_boundary"]))
            prefix = "val" if split_name == "validation" else split_name
            row[f"{prefix}_n"] = len(idx)
            row[f"{prefix}_log10mic_mean"] = float(np.nanmean(log10mic)) if np.isfinite(log10mic).any() else np.nan
            row[f"{prefix}_log10mic_std"] = float(np.nanstd(log10mic)) if np.isfinite(log10mic).any() else np.nan
            row[f"{prefix}_boundary_ratio"] = float(np.nanmean(boundary)) if np.isfinite(boundary).any() else np.nan
            if args.task == "classification":
                row[f"{prefix}_active_ratio"] = float(np.mean(target == 1))
        summary_rows.append(row)

        mol_run = mol.copy()
        split_label = np.full(len(mol_run), "", dtype=object)
        source_row = np.full(len(mol_run), np.nan)
        for split_name, idx in split_rows.items():
            split_label[idx] = split_name
            source_row[idx] = np.arange(len(idx), dtype=float)
        mol_run.insert(0, "seed", run["seed"])
        mol_run.insert(0, "split_type", run["split_type"])
        mol_run.insert(0, "task", args.task)
        mol_run.insert(0, "genus", run["genus"])
        mol_run.insert(0, "dataset_name", run["dataset_name"])
        mol_run["split"] = split_label
        mol_run["source_row_within_split"] = source_row
        molecule_rows.append(mol_run)

    summary_df = pd.DataFrame(summary_rows)
    mean_sd = aggregate_mean_sd(summary_df)
    molecule_df = pd.concat(molecule_rows, ignore_index=True, sort=False) if molecule_rows else pd.DataFrame()
    if args.task == "classification":
        summary_df.to_csv(args.outdir / "landscape_roughness_by_split_seed.csv", index=False)
        mean_sd.to_csv(args.outdir / "landscape_roughness_mean_sd.csv", index=False)
        molecule_df.to_csv(args.outdir / "landscape_roughness_molecule_level.csv", index=False)
    else:
        summary_df.to_csv(args.outdir / "regression_landscape_roughness_by_split_seed.csv", index=False)
        mean_sd.to_csv(args.outdir / "regression_landscape_roughness_mean_sd.csv", index=False)
        molecule_df.to_csv(args.outdir / "regression_landscape_roughness_molecule_level.csv", index=False)
    print("Saved model-independent roughness outputs")


if __name__ == "__main__":
    main()
