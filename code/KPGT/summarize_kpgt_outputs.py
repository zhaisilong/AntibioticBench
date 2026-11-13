#!/usr/bin/env python3
"""Aggregate KPGT AntibioticBench per-run outputs into unified benchmark summaries."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["classification", "regression"], required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--expected-runs", type=int, default=192)
    return parser.parse_args()


def read_one_row(files: Sequence[Path], model_root: Path) -> pd.DataFrame:
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if len(df) != 1:
                print(f"WARNING: expected one row in {f.relative_to(model_root)}, got {len(df)}")
            df.insert(0, "result_dir", str(f.parent.relative_to(model_root)))
            frames.append(df)
        except Exception as e:
            print(f"WARNING: failed to read {f.relative_to(model_root)}: {type(e).__name__}")
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def mean_sd_wide(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in {"seed"}]
    rows: List[Dict[str, object]] = []
    for keys, group in df.groupby(list(group_cols), dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_runs"] = len(group)
        row["seeds"] = ",".join(str(int(x)) for x in sorted(group["seed"].dropna().unique())) if "seed" in group else ""
        for col in numeric_cols:
            values = pd.to_numeric(group[col], errors="coerce")
            row[f"{col}_mean"] = values.mean()
            row[f"{col}_sd"] = values.std(ddof=1)
            row[f"{col}_n"] = int(values.notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def ood_changes(df: pd.DataFrame, task: str) -> pd.DataFrame:
    keys = ["model", "task", "dataset_name", "genus", "seed"]
    random_df = df[df["split_type"] == "random"].copy()
    outputs = []
    if task == "classification":
        higher = [
            "test_pr_auc",
            "test_roc_auc",
            "test_mcc",
            "test_f1",
            "test_accuracy",
            "test_balanced_accuracy",
            "test_EF1%",
            "test_EF5%",
            "test_EF10%",
            "test_Hit_rate@Top1%",
            "test_Recall@Top100",
        ]
        lower: List[str] = ["test_log_loss", "test_brier"]
    else:
        higher = [
            "test_spearman",
            "test_pearson",
            "test_r2",
            "test_within_1_dilution",
            "test_within_2_dilutions",
            "test_EF1%",
            "test_EF5%",
            "test_EF10%",
        ]
        lower = ["test_rmse", "test_mae", "test_mse", "test_medae", "test_median_fold_error"]

    for split in ("scaffold", "repeated_stratified_perimeter", "strict_perimeter"):
        ood = df[df["split_type"] == split].copy()
        merged = random_df.merge(ood, on=keys, suffixes=("_random", "_ood"), how="inner")
        if merged.empty:
            continue
        out = merged[keys].copy()
        out["ood_split_type"] = split
        for metric in higher:
            r, o = f"{metric}_random", f"{metric}_ood"
            if r in merged and o in merged:
                out[f"random_{metric}"] = merged[r]
                out[f"ood_{metric}"] = merged[o]
                out[f"ood_drop_{metric}"] = pd.to_numeric(merged[r], errors="coerce") - pd.to_numeric(
                    merged[o], errors="coerce"
                )
        for metric in lower:
            r, o = f"{metric}_random", f"{metric}_ood"
            if r in merged and o in merged:
                out[f"random_{metric}"] = merged[r]
                out[f"ood_{metric}"] = merged[o]
                out[f"ood_increase_{metric}"] = pd.to_numeric(merged[o], errors="coerce") - pd.to_numeric(
                    merged[r], errors="coerce"
                )
        outputs.append(out)
    return pd.concat(outputs, ignore_index=True, sort=False) if outputs else pd.DataFrame()



def perimeter_ood_error_analysis(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Compare random and perimeter runs, including overall and diagnostic metrics.

    strict_perimeter is matched to random seed_0; repeated_stratified_perimeter is
    matched seed-by-seed. Per-run metrics_full already contains the test diagnostics.
    """
    keys = ["model", "task", "dataset_name", "genus", "seed"]
    random_df = df[df["split_type"] == "random"].copy()
    outputs: List[pd.DataFrame] = []

    if task == "classification":
        higher = [
            "test_pr_auc", "test_roc_auc", "test_mcc", "test_f1",
            "test_accuracy", "test_balanced_accuracy", "test_EF1%",
            "test_EF5%", "test_EF10%", "test_Hit_rate@Top1%",
            "test_Recall@Top100",
        ]
        lower = ["test_log_loss", "test_brier"]
        diagnostic_cols = [
            c for c in df.columns
            if c.startswith(("nn_lt_0p4_", "nn_0p4_0p7_", "nn_ge_0p7_"))
            or c in {
                "boundary_error_rate", "non_boundary_error_rate",
                "boundary_error_enrichment", "ac_error_rate",
                "non_ac_error_rate", "ac_error_enrichment",
                "ac_subgroup_pr_auc", "ac_pair_directional_accuracy",
                "ac_pair_total_count",
            }
        ]
    else:
        higher = [
            "test_spearman", "test_pearson", "test_r2",
            "test_within_1_dilution", "test_within_2_dilutions",
            "test_EF1%", "test_EF5%", "test_EF10%",
        ]
        lower = [
            "test_rmse", "test_mae", "test_mse", "test_medae",
            "test_median_fold_error",
        ]
        diagnostic_cols = [
            c for c in df.columns
            if c.startswith(("nn_lt_0p4_", "nn_0p4_0p7_", "nn_ge_0p7_"))
            or c in {
                "boundary_mae", "non_boundary_mae", "boundary_rmse",
                "non_boundary_rmse", "boundary_regression_error_enrichment",
                "ac_mae", "non_ac_mae", "ac_rmse", "non_ac_rmse",
                "ac_error_enrichment", "ac_pair_directional_accuracy",
                "ac_pair_total_count",
                "mean_local_train_log10mic_variance_k10",
                "median_local_train_log10mic_variance_k10",
                "high_local_variance_mae", "low_local_variance_mae",
                "local_variance_error_enrichment",
            }
        ]

    for split in ("repeated_stratified_perimeter", "strict_perimeter"):
        perimeter = df[df["split_type"] == split].copy()
        merged = random_df.merge(perimeter, on=keys, suffixes=("_random", "_perimeter"), how="inner")
        if merged.empty:
            continue
        out = merged[keys].copy()
        out["perimeter_split_type"] = split

        for metric in higher:
            r, pcol = f"{metric}_random", f"{metric}_perimeter"
            if r in merged.columns and pcol in merged.columns:
                out[f"random_{metric}"] = merged[r]
                out[f"perimeter_{metric}"] = merged[pcol]
                out[f"perimeter_drop_{metric}"] = (
                    pd.to_numeric(merged[r], errors="coerce")
                    - pd.to_numeric(merged[pcol], errors="coerce")
                )
        for metric in lower:
            r, pcol = f"{metric}_random", f"{metric}_perimeter"
            if r in merged.columns and pcol in merged.columns:
                out[f"random_{metric}"] = merged[r]
                out[f"perimeter_{metric}"] = merged[pcol]
                out[f"perimeter_increase_{metric}"] = (
                    pd.to_numeric(merged[pcol], errors="coerce")
                    - pd.to_numeric(merged[r], errors="coerce")
                )

        # Preserve both sides for failure diagnostics and provide a signed change.
        # For error metrics, positive change means worse under perimeter; for rates/AUC
        # users can inspect the explicit random/perimeter columns without ambiguity.
        for metric in sorted(set(diagnostic_cols)):
            r, pcol = f"{metric}_random", f"{metric}_perimeter"
            if r not in merged.columns or pcol not in merged.columns:
                continue
            out[f"random_{metric}"] = merged[r]
            out[f"perimeter_{metric}"] = merged[pcol]
            out[f"perimeter_minus_random_{metric}"] = (
                pd.to_numeric(merged[pcol], errors="coerce")
                - pd.to_numeric(merged[r], errors="coerce")
            )
        outputs.append(out)

    return pd.concat(outputs, ignore_index=True, sort=False) if outputs else pd.DataFrame()

def ranking_files(df: pd.DataFrame, task: str):
    metric = "test_pr_auc" if task == "classification" else "test_rmse"
    ascending = task == "regression"
    rank_keys = ["dataset_name", "split_type", "seed"]
    work = df.copy()
    if metric not in work:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    work["rank_metric"] = metric
    work["rank_value"] = pd.to_numeric(work[metric], errors="coerce")
    work["rank"] = work.groupby(rank_keys)["rank_value"].rank(method="min", ascending=ascending)
    per_seed = work[rank_keys + ["model", "rank_metric", "rank_value", "rank"]].copy()
    by_metric = (
        per_seed.groupby(["model", "rank_metric"], as_index=False)
        .agg(mean_rank=("rank", "mean"), median_rank=("rank", "median"), n_ranked=("rank", "count"))
        .sort_values("mean_rank")
    )
    top = (
        per_seed[per_seed["rank"] == 1]
        .groupby("model", as_index=False)
        .size()
        .rename(columns={"size": "top_model_count"})
    )
    total = len(per_seed.groupby(rank_keys))
    if not top.empty:
        top["top_model_frequency"] = top["top_model_count"] / max(total, 1)
    return per_seed, by_metric, top


def main() -> None:
    args = parse_args()
    model_root = args.out_root.expanduser().resolve() / "KPGT"
    metric_files = sorted(
        f for f in model_root.rglob("metrics_full.csv") if "_combined_summary" not in f.parts
    )
    print(f"Found {len(metric_files)} metrics_full.csv files")
    if not metric_files:
        raise SystemExit("No per-run metrics found")
    if len(metric_files) != args.expected_runs and not args.allow_incomplete:
        raise SystemExit(
            f"Expected {args.expected_runs} runs but found {len(metric_files)}. Re-run with --allow-incomplete only for interim summaries."
        )
    metrics = read_one_row(metric_files, model_root)
    metrics = metrics.sort_values(["dataset_name", "split_type", "seed"]).reset_index(drop=True)
    metrics.to_csv(model_root / "all_run_metrics.csv", index=False)

    aggregate = mean_sd_wide(metrics, ["model", "task", "dataset_name", "genus", "split_type"])
    aggregate.to_csv(model_root / "aggregate_mean_sd.csv", index=False)

    ood = ood_changes(metrics, args.task)
    ood.to_csv(model_root / "ood_change_random_vs_ood_splits.csv", index=False)

    diag_files = sorted(
        f for f in model_root.rglob("diagnostics_metrics.csv") if "_combined_summary" not in f.parts
    )
    diagnostics = read_one_row(diag_files, model_root)
    if not diagnostics.empty:
        diagnostics.to_csv(model_root / "all_diagnostics_metrics.csv", index=False)

    summary_name = "classification_summary_like_ml" if args.task == "classification" else "regression_summary_like_ml"
    summary_dir = model_root / "_combined_summary" / summary_name
    summary_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(summary_dir / "combined_all_run_metrics.csv", index=False)

    core = "test_pr_auc" if args.task == "classification" else "test_rmse"
    core_mean, core_sd = f"{core}_mean", f"{core}_sd"
    leaderboard_cols = [
        c
        for c in [
            "model",
            "task",
            "dataset_name",
            "genus",
            "split_type",
            "n_runs",
            core_mean,
            core_sd,
            "test_roc_auc_mean",
            "test_spearman_mean",
            "test_EF1%_mean",
        ]
        if c in aggregate.columns
    ]
    leaderboard = aggregate[leaderboard_cols].copy()
    if core_mean in leaderboard:
        leaderboard = leaderboard.sort_values(
            ["dataset_name", "split_type", core_mean], ascending=[True, True, args.task == "regression"]
        )
    leaderboard.to_csv(summary_dir / "leaderboard_mean_sd.csv", index=False)

    if not ood.empty:
        ood_summary = mean_sd_wide(ood, ["model", "task", "dataset_name", "genus", "ood_split_type"])
    else:
        ood_summary = pd.DataFrame()
    ood_summary.to_csv(
        summary_dir / ("ood_drop_summary.csv" if args.task == "classification" else "ood_change_summary.csv"),
        index=False,
    )

    if not diagnostics.empty:
        diagnostics.to_csv(summary_dir / "combined_diagnostics_metrics.csv", index=False)

    perimeter_analysis = perimeter_ood_error_analysis(metrics, args.task)
    perimeter_analysis.to_csv(
        summary_dir
        / (
            "perimeter_ood_error_analysis.csv"
            if args.task == "classification"
            else "perimeter_regression_ood_error_analysis.csv"
        ),
        index=False,
    )

    per_seed, by_metric, top = ranking_files(metrics, args.task)
    per_seed.to_csv(summary_dir / "model_ranking_per_task_seed.csv", index=False)
    by_metric.to_csv(summary_dir / "model_ranking_by_metric.csv", index=False)
    top.to_csv(summary_dir / "top_model_frequency.csv", index=False)

    failed = sorted(model_root.rglob("_FAILED"))
    missing_success = [f.parent for f in metric_files if not (f.parent / "_SUCCESS").is_file()]
    status = pd.DataFrame(
        [{"failed_run_dir": str(f.parent.relative_to(model_root)), "error_file": str(f.relative_to(model_root))} for f in failed]
        + [{"failed_run_dir": str(p.relative_to(model_root)), "error_file": "missing _SUCCESS"} for p in missing_success]
    )
    status.to_csv(summary_dir / "run_failures_or_incomplete.csv", index=False)

    print("Saved all-run metrics")
    print("Saved aggregate metrics")
    print("Saved combined summary")


if __name__ == "__main__":
    main()
