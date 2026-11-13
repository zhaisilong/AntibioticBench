#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Summarize AntibioticBench classification results across models.

Input:
    Result root created by train_eval_diagnose_classification_ml_v2_perimeter2.py
    Example:
        outputs_antibioticbench_classification/
          xgb/all_run_metrics.csv
          rf/all_run_metrics.csv
          lightgbm/all_run_metrics.csv

Supports both perimeter split names:
    repeated_stratified_perimeter and strict_perimeter

Outputs:
    combined_all_run_metrics.csv
    leaderboard_mean_sd.csv
    ood_drop_summary.csv
    perimeter_ood_error_analysis.csv
    model_ranking_per_task_seed.csv
    model_ranking_by_metric.csv
    top_model_frequency.csv

The dedicated perimeter_ood_error_analysis.csv implements failure diagnosis 12.4:
    - repeated_stratified_perimeter and strict_perimeter overall PR-AUC / EF1%
    - random-to-perimeter PR-AUC and EF1% drop
    - NN_sim < 0.4 subgroup error rate, PR-AUC and EF1%
    - strict_perimeter versus repeated_stratified_perimeter differences
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


DEFAULT_RANK_METRICS = [
    "test_pr_auc",
    "test_roc_auc",
    "test_mcc",
    "test_f1",
    "test_EF1%",
    "test_EF5%",
    "test_Hit_rate@Top1%",
    "test_Hit_rate@Top5%",
]

OOD_TARGET_SPLITS = [
    "scaffold",
    "repeated_stratified_perimeter",
    "strict_perimeter",
    "perimeter",
    "max_distance",
]

PERIMETER_SPLITS = [
    "repeated_stratified_perimeter",
    "strict_perimeter",
]

PERIMETER_CORE_METRICS = [
    "test_pr_auc",
    "test_EF1%",
    "test_EF5%",
    "test_Hit_rate@Top1%",
    "test_Hit_rate@Top5%",
    "test_roc_auc",
    "test_mcc",
    "test_f1",
    "test_accuracy",
    "test_balanced_accuracy",
]

LOW_NN_METRICS = [
    "nn_lt_0p4_n",
    "nn_lt_0p4_error_rate",
    "nn_lt_0p4_pr_auc",
    "nn_lt_0p4_EF1%",
    "nn_lt_0p4_EF5%",
    "nn_lt_0p4_Hit_rate@Top1%",
    "nn_lt_0p4_Hit_rate@Top5%",
    "nn_lt_0p4_top_1pct_n",
    "nn_lt_0p4_top_1pct_hits",
    "nn_lt_0p4_top_5pct_n",
    "nn_lt_0p4_top_5pct_hits",
]

NN_COMPARISON_METRICS = [
    "nn_ge_0p7_error_rate",
    "nn_0p4_0p7_error_rate",
    "nn_lt_0p4_error_rate",
    "nn_ge_0p7_pr_auc",
    "nn_0p4_0p7_pr_auc",
    "nn_lt_0p4_pr_auc",
]


def fmt_mean_sd(mean, sd):
    if pd.isna(mean):
        return ""
    if pd.isna(sd):
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {sd:.4f}"


def safe_get(row: pd.Series, col: str):
    if row is None or col not in row.index:
        return np.nan
    return row[col]


def safe_diff(a, b):
    if pd.isna(a) or pd.isna(b):
        return np.nan
    return float(a - b)


def safe_ratio(num, den):
    if pd.isna(num) or pd.isna(den) or den == 0:
        return np.nan
    return float(num / den)


def load_all_metrics(root: Path) -> pd.DataFrame:
    metric_files = list(root.glob("*/all_run_metrics.csv"))
    if not metric_files:
        metric_files = list(root.glob("*/all_run_metrics_from_current_invocation.csv"))
    if not metric_files:
        raise RuntimeError(f"No all_run_metrics.csv found under {root}")

    dfs = []
    for f in metric_files:
        df = pd.read_csv(f)
        if "model" not in df.columns:
            df["model"] = f.parent.name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def numeric_columns(df: pd.DataFrame, exclude=("seed",)):
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def write_leaderboard(all_df: pd.DataFrame, outdir: Path):
    group_cols = ["model", "dataset_name", "genus", "split_type"]
    metrics = [m for m in DEFAULT_RANK_METRICS if m in all_df.columns]
    extra_metrics = [
        c for c in all_df.columns
        if (
            c.startswith("test_")
            or c.startswith("boundary_")
            or c.startswith("non_boundary_")
            or c.startswith("nn_")
            or c.startswith("ac_")
        ) and c not in metrics
    ]
    metrics = metrics + extra_metrics
    numeric_metrics = [m for m in metrics if pd.api.types.is_numeric_dtype(all_df[m])]

    agg = all_df.groupby(group_cols)[numeric_metrics].agg(["mean", "std", "count"]).reset_index()
    agg.columns = ["_".join([str(x) for x in col if str(x)]) for col in agg.columns]
    for m in numeric_metrics:
        mean_col = f"{m}_mean"
        std_col = f"{m}_std"
        if mean_col in agg.columns and std_col in agg.columns:
            agg[f"{m}_mean±sd"] = [fmt_mean_sd(a, b) for a, b in zip(agg[mean_col], agg[std_col])]
    agg.to_csv(outdir / "leaderboard_mean_sd.csv", index=False)
    return numeric_metrics


def write_ood_drop_summary(all_df: pd.DataFrame, outdir: Path, numeric_metrics):
    mean_df = all_df.groupby(["model", "dataset_name", "genus", "split_type"])[numeric_metrics].mean().reset_index()
    drops = []
    for (model, dataset_name, genus), sub in mean_df.groupby(["model", "dataset_name", "genus"]):
        random_row = sub[sub["split_type"] == "random"]
        if random_row.empty:
            continue
        random_row = random_row.iloc[0]
        for target_split in OOD_TARGET_SPLITS:
            target_row = sub[sub["split_type"] == target_split]
            if target_row.empty:
                continue
            target_row = target_row.iloc[0]
            row = {
                "model": model,
                "dataset_name": dataset_name,
                "genus": genus,
                "reference_split": "random",
                "target_split": target_split,
            }
            for m in numeric_metrics:
                row[f"ood_drop_{m}"] = safe_diff(random_row[m], target_row[m])
            drops.append(row)
    if drops:
        pd.DataFrame(drops).to_csv(outdir / "ood_drop_summary.csv", index=False)


def write_perimeter_ood_error_analysis(all_df: pd.DataFrame, outdir: Path):
    """Create a dedicated table for diagnosis 12.4 Perimeter OOD error analysis."""
    id_cols = ["model", "dataset_name", "genus", "split_type"]
    nums = numeric_columns(all_df)
    mean_df = all_df.groupby(id_cols)[nums].mean().reset_index()
    count_df = all_df.groupby(id_cols).size().reset_index(name="split_seed_count")
    mean_df = mean_df.merge(count_df, on=id_cols, how="left")

    rows = []
    for (model, dataset_name, genus), sub in mean_df.groupby(["model", "dataset_name", "genus"]):
        random_df = sub[sub["split_type"] == "random"]
        if random_df.empty:
            continue
        random_row = random_df.iloc[0]
        repeated_df = sub[sub["split_type"] == "repeated_stratified_perimeter"]
        strict_df = sub[sub["split_type"] == "strict_perimeter"]
        repeated_row = repeated_df.iloc[0] if not repeated_df.empty else None
        strict_row = strict_df.iloc[0] if not strict_df.empty else None

        strict_minus_repeated = {}
        if repeated_row is not None and strict_row is not None:
            for m in PERIMETER_CORE_METRICS + LOW_NN_METRICS + NN_COMPARISON_METRICS:
                if m in mean_df.columns:
                    strict_minus_repeated[f"strict_minus_repeated_{m}"] = safe_diff(safe_get(strict_row, m), safe_get(repeated_row, m))

        for target_split in PERIMETER_SPLITS:
            target_df = sub[sub["split_type"] == target_split]
            if target_df.empty:
                continue
            target_row = target_df.iloc[0]
            row = {
                "model": model,
                "dataset_name": dataset_name,
                "genus": genus,
                "reference_split": "random",
                "target_split": target_split,
                "target_split_seed_count": int(safe_get(target_row, "split_seed_count")) if not pd.isna(safe_get(target_row, "split_seed_count")) else np.nan,
                "analysis_note": "positive_drop_means_random_better_than_perimeter_for_higher_is_better_metrics",
            }

            # 1. overall perimeter metrics and 2. random-to-perimeter drops.
            for m in PERIMETER_CORE_METRICS:
                if m in mean_df.columns:
                    row[f"random_{m}"] = safe_get(random_row, m)
                    row[f"perimeter_{m}"] = safe_get(target_row, m)
                    row[f"perimeter_drop_{m}"] = safe_diff(safe_get(random_row, m), safe_get(target_row, m))

            # 3-5. Low training-nearest-neighbor subgroup in perimeter test set.
            for m in LOW_NN_METRICS + NN_COMPARISON_METRICS:
                if m in mean_df.columns:
                    row[f"perimeter_{m}"] = safe_get(target_row, m)

            # Error enrichment for low-NN molecules in perimeter splits.
            target_accuracy = safe_get(target_row, "test_accuracy")
            overall_error_rate = safe_diff(1.0, target_accuracy) if not pd.isna(target_accuracy) else np.nan
            low_nn_error = safe_get(target_row, "nn_lt_0p4_error_rate")
            high_nn_error = safe_get(target_row, "nn_ge_0p7_error_rate")
            mid_nn_error = safe_get(target_row, "nn_0p4_0p7_error_rate")
            row["perimeter_overall_error_rate"] = overall_error_rate
            row["low_nn_error_enrichment_vs_overall"] = safe_ratio(low_nn_error, overall_error_rate)
            row["low_nn_error_enrichment_vs_high_nn_ge_0p7"] = safe_ratio(low_nn_error, high_nn_error)
            row["low_nn_error_enrichment_vs_mid_nn_0p4_0p7"] = safe_ratio(low_nn_error, mid_nn_error)

            # 6. strict_perimeter versus repeated_stratified_perimeter.
            row.update(strict_minus_repeated)
            rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        # Put the most important columns first if present.
        preferred = [
            "model", "dataset_name", "genus", "reference_split", "target_split", "target_split_seed_count",
            "random_test_pr_auc", "perimeter_test_pr_auc", "perimeter_drop_test_pr_auc",
            "random_test_EF1%", "perimeter_test_EF1%", "perimeter_drop_test_EF1%",
            "perimeter_nn_lt_0p4_n", "perimeter_nn_lt_0p4_error_rate", "perimeter_nn_lt_0p4_pr_auc", "perimeter_nn_lt_0p4_EF1%",
            "perimeter_overall_error_rate", "low_nn_error_enrichment_vs_overall",
            "strict_minus_repeated_test_pr_auc", "strict_minus_repeated_test_EF1%",
            "analysis_note",
        ]
        ordered = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
        df[ordered].to_csv(outdir / "perimeter_ood_error_analysis.csv", index=False)


def write_rankings(all_df: pd.DataFrame, outdir: Path, rank_metric: str, numeric_metrics):
    if rank_metric not in all_df.columns:
        raise ValueError(f"rank_metric={rank_metric} not in columns. Available examples: {numeric_metrics[:20]}")

    task_cols = ["dataset_name", "genus", "split_type", "seed"]
    rank_rows = []
    for _, sub in all_df.groupby(task_cols):
        sub = sub[["model", rank_metric] + task_cols].dropna(subset=[rank_metric]).copy()
        if sub.empty:
            continue
        # Larger is better for all default ranking metrics. If using loss, user should invert manually.
        sub["rank"] = sub[rank_metric].rank(ascending=False, method="average")
        best_value = sub[rank_metric].max()
        sub["is_top_model"] = (sub[rank_metric] == best_value).astype(int)
        rank_rows.append(sub)

    if rank_rows:
        ranks = pd.concat(rank_rows, ignore_index=True)
        ranks.to_csv(outdir / "model_ranking_per_task_seed.csv", index=False)
        avg_rank = ranks.groupby("model")["rank"].agg(["mean", "std", "count"]).reset_index().sort_values("mean")
        avg_rank.to_csv(outdir / "model_ranking_by_metric.csv", index=False)
        top_freq = ranks.groupby("model")["is_top_model"].agg(["sum", "count", "mean"]).reset_index()
        top_freq = top_freq.rename(columns={"sum": "top_model_count", "count": "total_task_seed_count", "mean": "top_model_frequency"})
        top_freq = top_freq.sort_values("top_model_frequency", ascending=False)
        top_freq.to_csv(outdir / "top_model_frequency.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True, help="Root output directory from model training")
    parser.add_argument("--outdir", default=None, help="Summary output directory")
    parser.add_argument("--rank-metric", default="test_pr_auc", help="Metric used for average ranking and top model frequency")
    args = parser.parse_args()

    root = Path(args.results_root).resolve()
    outdir = Path(args.outdir).resolve() if args.outdir else root / "_combined_summary"
    outdir.mkdir(parents=True, exist_ok=True)

    all_df = load_all_metrics(root)
    all_df.to_csv(outdir / "combined_all_run_metrics.csv", index=False)

    numeric_metrics = write_leaderboard(all_df, outdir)
    write_ood_drop_summary(all_df, outdir, numeric_metrics)
    write_perimeter_ood_error_analysis(all_df, outdir)
    write_rankings(all_df, outdir, args.rank_metric, numeric_metrics)

    print(f"Saved combined summaries to: {outdir.name}")
    print(f"Dedicated perimeter OOD diagnosis saved to: perimeter_ood_error_analysis.csv")


if __name__ == "__main__":
    main()
