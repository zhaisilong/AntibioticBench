#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Summarize AntibioticBench 1-NN/5-NN similarity baseline results in ML-compatible layouts.

This script keeps the combined similarity-baseline summary, and additionally writes
classification and regression summaries in separate subfolders using the same file
names as the traditional ML summarizers whenever possible.
"""
import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

CLASSIFICATION_DEFAULT_RANK_METRICS = [
    "test_pr_auc", "test_roc_auc", "test_mcc", "test_f1", "test_EF1%", "test_EF5%",
    "test_Hit_rate@Top1%", "test_Hit_rate@Top5%",
]
REGRESSION_DEFAULT_RANK_METRICS = [
    "test_rmse", "test_mae", "test_spearman", "test_within_1_dilution",
    "test_within_2_dilutions", "test_EF1%", "test_Hit_rate@Top1%",
]
OOD_TARGET_SPLITS = ["scaffold", "repeated_stratified_perimeter", "strict_perimeter", "perimeter", "max_distance"]
PERIMETER_SPLITS = ["repeated_stratified_perimeter", "strict_perimeter"]
LOWER_TERMS = ["loss", "mae", "rmse", "mse", "medae", "brier", "fold_error", "error_rate", "max_abs_error"]


def fmt_mean_sd(mean, sd):
    if pd.isna(mean):
        return ""
    if pd.isna(sd):
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {sd:.4f}"


def metric_direction(metric: str, task: str) -> str:
    if task == "regression":
        if any(t in metric.lower() for t in LOWER_TERMS):
            return "lower"
        return "higher"
    # Classification summary follows summarize_ml.py convention: drops are random - target.
    return "higher"


def safe_diff(a, b):
    if pd.isna(a) or pd.isna(b):
        return np.nan
    return float(a - b)


def safe_get(row, col):
    if row is None or col not in row.index:
        return np.nan
    return row[col]


def load_all_metrics(root: Path) -> pd.DataFrame:
    files = list(root.glob("*/all_run_metrics.csv"))
    if not files:
        files = list(root.glob("*/all_run_metrics_from_current_invocation.csv"))
    if not files and (root / "all_run_metrics_from_current_invocation.csv").exists():
        files = [root / "all_run_metrics_from_current_invocation.csv"]
    if not files:
        raise RuntimeError(f"No all_run_metrics.csv found under {root}")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        if "model" not in df.columns and f.parent != root:
            df["model"] = f.parent.name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True).drop_duplicates()


def numeric_columns(df: pd.DataFrame, exclude=("seed",)):
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def selected_metric_columns(df: pd.DataFrame, task: str):
    prefixes = ("test_", "val_", "boundary_", "non_boundary_", "nn_", "ac_", "range_", "mean_local_", "high_local_", "low_local_", "local_variance_")
    cols = [c for c in df.columns if c.startswith(prefixes)]
    return [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]


def write_leaderboard(df: pd.DataFrame, outdir: Path, task: str):
    group_cols = ["model", "dataset_name", "genus", "split_type"]
    metrics = selected_metric_columns(df, task)
    if not metrics:
        return []
    agg = df.groupby(group_cols)[metrics].agg(["mean", "std", "count"]).reset_index()
    agg.columns = ["_".join([str(x) for x in col if str(x)]) for col in agg.columns]
    for m in metrics:
        mean_col, std_col = f"{m}_mean", f"{m}_std"
        if mean_col in agg.columns and std_col in agg.columns:
            agg[f"{m}_mean±sd"] = [fmt_mean_sd(a, b) for a, b in zip(agg[mean_col], agg[std_col])]
    agg.to_csv(outdir / "leaderboard_mean_sd.csv", index=False)
    return metrics


def write_ood(df: pd.DataFrame, outdir: Path, task: str, metrics):
    if not metrics:
        return
    mean_df = df.groupby(["model", "dataset_name", "genus", "split_type"])[metrics].mean().reset_index()
    rows = []
    for (model, dataset_name, genus), sub in mean_df.groupby(["model", "dataset_name", "genus"]):
        random_df = sub[sub["split_type"] == "random"]
        if random_df.empty:
            continue
        random_row = random_df.iloc[0]
        for target_split in OOD_TARGET_SPLITS:
            target_df = sub[sub["split_type"] == target_split]
            if target_df.empty:
                continue
            target_row = target_df.iloc[0]
            if task == "classification":
                row = {"model": model, "dataset_name": dataset_name, "genus": genus, "reference_split": "random", "target_split": target_split}
                for m in metrics:
                    row[f"ood_drop_{m}"] = safe_diff(random_row[m], target_row[m])
                rows.append(row)
            else:
                row = {"model": model, "dataset_name": dataset_name, "genus": genus, "reference_split": "random", "target_split": target_split,
                       "analysis_note": "for error metrics: target_minus_random; for higher-is-better metrics: random_minus_target"}
                for m in metrics:
                    if metric_direction(m, task) == "lower":
                        row[f"ood_increase_{m}"] = safe_diff(target_row[m], random_row[m])
                    else:
                        row[f"ood_drop_{m}"] = safe_diff(random_row[m], target_row[m])
                rows.append(row)
    if rows:
        fname = "ood_drop_summary.csv" if task == "classification" else "ood_change_summary.csv"
        pd.DataFrame(rows).to_csv(outdir / fname, index=False)


def write_rankings(df: pd.DataFrame, outdir: Path, task: str, rank_metric: str):
    if rank_metric not in df.columns:
        return
    ascending = metric_direction(rank_metric, task) == "lower"
    task_cols = ["dataset_name", "genus", "split_type", "seed"]
    rank_rows = []
    for _, sub in df.groupby(task_cols):
        sub = sub[["model", rank_metric] + task_cols].dropna(subset=[rank_metric]).copy()
        if sub.empty:
            continue
        sub["rank"] = sub[rank_metric].rank(ascending=ascending, method="average")
        best_value = sub[rank_metric].min() if ascending else sub[rank_metric].max()
        sub["is_top_model"] = (sub[rank_metric] == best_value).astype(int)
        sub["rank_metric"] = rank_metric
        sub["rank_direction"] = "lower_is_better" if ascending else "higher_is_better"
        rank_rows.append(sub)
    if not rank_rows:
        return
    ranks = pd.concat(rank_rows, ignore_index=True)
    ranks.to_csv(outdir / "model_ranking_per_task_seed.csv", index=False)
    avg_rank = ranks.groupby("model")["rank"].agg(["mean", "std", "count"]).reset_index().sort_values("mean")
    avg_rank.to_csv(outdir / "model_ranking_by_metric.csv", index=False)
    # Also write metric-specific filename for backward compatibility with earlier similarity script.
    avg_rank.to_csv(outdir / f"model_ranking_by_{rank_metric}.csv", index=False)
    top_freq = ranks.groupby("model")["is_top_model"].agg(["sum", "count", "mean"]).reset_index()
    top_freq = top_freq.rename(columns={"sum": "top_model_count", "count": "total_task_seed_count", "mean": "top_model_frequency"})
    top_freq = top_freq.sort_values("top_model_frequency", ascending=False)
    top_freq.to_csv(outdir / "top_model_frequency.csv", index=False)
    top_freq.to_csv(outdir / f"top_model_frequency_by_{rank_metric}.csv", index=False)


def write_perimeter_classification(df: pd.DataFrame, outdir: Path):
    core = ["test_pr_auc", "test_EF1%", "test_EF5%", "test_Hit_rate@Top1%", "test_Hit_rate@Top5%", "test_roc_auc", "test_mcc", "test_f1", "test_accuracy", "test_balanced_accuracy"]
    low_nn = ["nn_lt_0p4_n", "nn_lt_0p4_error_rate", "nn_lt_0p4_pr_auc", "nn_lt_0p4_EF1%", "nn_lt_0p4_EF5%", "nn_lt_0p4_Hit_rate@Top1%", "nn_lt_0p4_Hit_rate@Top5%"]
    nums = numeric_columns(df)
    mean_df = df.groupby(["model", "dataset_name", "genus", "split_type"])[nums].mean().reset_index()
    count_df = df.groupby(["model", "dataset_name", "genus", "split_type"]).size().reset_index(name="split_seed_count")
    mean_df = mean_df.merge(count_df, on=["model", "dataset_name", "genus", "split_type"], how="left")
    rows = []
    for (model, dataset_name, genus), sub in mean_df.groupby(["model", "dataset_name", "genus"]):
        random_df = sub[sub["split_type"] == "random"]
        if random_df.empty:
            continue
        random_row = random_df.iloc[0]
        for target_split in PERIMETER_SPLITS:
            target_df = sub[sub["split_type"] == target_split]
            if target_df.empty:
                continue
            target_row = target_df.iloc[0]
            row = {"model": model, "dataset_name": dataset_name, "genus": genus, "reference_split": "random", "target_split": target_split,
                   "target_split_seed_count": int(safe_get(target_row, "split_seed_count")) if not pd.isna(safe_get(target_row, "split_seed_count")) else np.nan,
                   "analysis_note": "positive_drop_means_random_better_than_perimeter_for_higher_is_better_metrics"}
            for m in core:
                if m in mean_df.columns:
                    row[f"random_{m}"] = safe_get(random_row, m)
                    row[f"perimeter_{m}"] = safe_get(target_row, m)
                    row[f"perimeter_drop_{m}"] = safe_diff(safe_get(random_row, m), safe_get(target_row, m))
            for m in low_nn:
                if m in mean_df.columns:
                    row[f"perimeter_{m}"] = safe_get(target_row, m)
            rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(outdir / "perimeter_ood_error_analysis.csv", index=False)


def write_perimeter_regression(df: pd.DataFrame, outdir: Path):
    core = ["test_rmse", "test_mae", "test_medae", "test_spearman", "test_pearson", "test_r2", "test_within_1_dilution", "test_within_2_dilutions", "test_EF1%", "test_EF5%", "test_Hit_rate@Top1%", "test_Hit_rate@Top5%"]
    low_nn = ["nn_lt_0p4_n", "nn_lt_0p4_mae", "nn_lt_0p4_rmse", "nn_lt_0p4_spearman", "nn_lt_0p4_within_1_dilution", "nn_lt_0p4_within_2_dilutions", "nn_lt_0p4_EF1%", "nn_lt_0p4_Hit_rate@Top1%"]
    nums = numeric_columns(df)
    mean_df = df.groupby(["model", "dataset_name", "genus", "split_type"])[nums].mean().reset_index()
    count_df = df.groupby(["model", "dataset_name", "genus", "split_type"]).size().reset_index(name="split_seed_count")
    mean_df = mean_df.merge(count_df, on=["model", "dataset_name", "genus", "split_type"], how="left")
    rows = []
    for (model, dataset_name, genus), sub in mean_df.groupby(["model", "dataset_name", "genus"]):
        random_df = sub[sub["split_type"] == "random"]
        if random_df.empty:
            continue
        random_row = random_df.iloc[0]
        for target_split in PERIMETER_SPLITS:
            target_df = sub[sub["split_type"] == target_split]
            if target_df.empty:
                continue
            target_row = target_df.iloc[0]
            row = {"model": model, "dataset_name": dataset_name, "genus": genus, "reference_split": "random", "target_split": target_split,
                   "target_split_seed_count": int(safe_get(target_row, "split_seed_count")) if not pd.isna(safe_get(target_row, "split_seed_count")) else np.nan,
                   "analysis_note": "error_increase=perimeter-random; higher_metric_drop=random-perimeter"}
            for m in core:
                if m in mean_df.columns:
                    row[f"random_{m}"] = safe_get(random_row, m)
                    row[f"perimeter_{m}"] = safe_get(target_row, m)
                    if metric_direction(m, "regression") == "lower":
                        row[f"perimeter_increase_{m}"] = safe_diff(safe_get(target_row, m), safe_get(random_row, m))
                    else:
                        row[f"perimeter_drop_{m}"] = safe_diff(safe_get(random_row, m), safe_get(target_row, m))
            for m in low_nn:
                if m in mean_df.columns:
                    row[f"perimeter_{m}"] = safe_get(target_row, m)
            rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(outdir / "perimeter_regression_ood_error_analysis.csv", index=False)


def write_task_summary(all_df: pd.DataFrame, parent_outdir: Path, task: str, rank_metric: str):
    df = all_df[all_df["task"] == task].copy()
    if df.empty:
        return
    outdir = parent_outdir / f"{task}_summary_like_ml"
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "combined_all_run_metrics.csv", index=False)
    metrics = write_leaderboard(df, outdir, task)
    write_ood(df, outdir, task, metrics)
    write_rankings(df, outdir, task, rank_metric)
    if task == "classification":
        write_perimeter_classification(df, outdir)
    else:
        write_perimeter_regression(df, outdir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--classification-rank-metric", default="test_pr_auc")
    parser.add_argument("--regression-rank-metric", default="test_rmse")
    args = parser.parse_args()

    root = Path(args.results_root).resolve()
    outdir = Path(args.outdir).resolve() if args.outdir else root / "_combined_summary"
    outdir.mkdir(parents=True, exist_ok=True)

    all_df = load_all_metrics(root)
    all_df.to_csv(outdir / "combined_all_run_metrics.csv", index=False)

    # Backward-compatible combined summary with task in the grouping.
    numeric = numeric_columns(all_df)
    if numeric:
        agg = all_df.groupby(["model", "task", "dataset_name", "genus", "split_type"])[numeric].agg(["mean", "std", "count"]).reset_index()
        agg.columns = ["_".join([str(x) for x in col if str(x)]) for col in agg.columns]
        for m in CLASSIFICATION_DEFAULT_RANK_METRICS + REGRESSION_DEFAULT_RANK_METRICS:
            mean_col, std_col = f"{m}_mean", f"{m}_std"
            if mean_col in agg.columns and std_col in agg.columns:
                agg[f"{m}_mean±sd"] = [fmt_mean_sd(a, b) for a, b in zip(agg[mean_col], agg[std_col])]
        agg.to_csv(outdir / "leaderboard_mean_sd.csv", index=False)

    write_task_summary(all_df, outdir, "classification", args.classification_rank_metric)
    write_task_summary(all_df, outdir, "regression", args.regression_rank_metric)
    print(f"Saved similarity baseline combined summaries to: {outdir.name}")
    print(f"Classification ML-compatible summaries: classification_summary_like_ml")
    print(f"Regression ML-compatible summaries: regression_summary_like_ml")


if __name__ == "__main__":
    main()
