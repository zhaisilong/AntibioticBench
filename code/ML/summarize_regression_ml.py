#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Summarize AntibioticBench log10MIC regression results across models.

Input root example:
    outputs_antibioticbench_regression/
      rf/all_run_metrics.csv
      xgb/all_run_metrics.csv
      lightgbm/all_run_metrics.csv
      svm/all_run_metrics.csv

Outputs:
    combined_all_run_metrics.csv
    leaderboard_mean_sd.csv
    ood_change_summary.csv
    perimeter_regression_ood_error_analysis.csv
    model_ranking_per_task_seed.csv
    model_ranking_by_metric.csv
    top_model_frequency.csv

For lower-is-better error metrics, OOD change = target_split - random.
For higher-is-better metrics, OOD drop = random - target_split.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


DEFAULT_RANK_METRICS = [
    "test_rmse",
    "test_mae",
    "test_spearman",
    "test_within_1_dilution",
    "test_within_2_dilutions",
    "test_EF1%",
    "test_Hit_rate@Top1%",
]

LOWER_BETTER = {
    "test_mae", "test_rmse", "test_mse", "test_medae",
    "test_median_fold_error", "test_mean_fold_error", "test_max_abs_error_log10",
}

HIGHER_BETTER = {
    "test_spearman", "test_pearson", "test_r2",
    "test_within_1_dilution", "test_within_2_dilutions",
    "test_EF1%", "test_EF5%", "test_EF10%",
    "test_Hit_rate@Top1%", "test_Hit_rate@Top5%", "test_Hit_rate@Top10%",
    "test_Recall@Top50", "test_Recall@Top100", "test_Recall@Top500",
}

OOD_TARGET_SPLITS = [
    "scaffold",
    "repeated_stratified_perimeter",
    "strict_perimeter",
    "perimeter",
    "max_distance",
]

PERIMETER_SPLITS = ["repeated_stratified_perimeter", "strict_perimeter"]

PERIMETER_CORE_METRICS = [
    "test_rmse", "test_mae", "test_medae",
    "test_spearman", "test_pearson", "test_r2",
    "test_within_1_dilution", "test_within_2_dilutions",
    "test_EF1%", "test_EF5%", "test_Hit_rate@Top1%", "test_Hit_rate@Top5%",
]

LOW_NN_METRICS = [
    "nn_lt_0p4_n",
    "nn_lt_0p4_mae",
    "nn_lt_0p4_rmse",
    "nn_lt_0p4_spearman",
    "nn_lt_0p4_within_1_dilution",
    "nn_lt_0p4_within_2_dilutions",
    "nn_lt_0p4_EF1%",
    "nn_lt_0p4_Hit_rate@Top1%",
]

DIAG_METRICS = [
    "boundary_mae", "non_boundary_mae", "boundary_regression_error_enrichment",
    "ac_mae", "non_ac_mae", "ac_error_enrichment", "ac_pair_directional_accuracy",
    "mean_local_train_log10mic_variance_k10", "local_variance_error_enrichment",
    "range_strong_MIC_le_T_over_2_mae", "range_boundary_T_over_2_to_2T_mae", "range_weak_MIC_gt_2T_mae",
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


def metric_direction(metric: str) -> str:
    if metric in LOWER_BETTER or metric.endswith("_mae") or metric.endswith("_rmse") or metric.endswith("_mse") or metric.endswith("_medae") or "fold_error" in metric:
        return "lower"
    if metric in HIGHER_BETTER or metric.endswith("spearman") or metric.endswith("pearson") or "within_" in metric or "EF" in metric or "Hit_rate" in metric or "Recall" in metric or metric.endswith("r2"):
        return "higher"
    return "higher"


def write_leaderboard(all_df: pd.DataFrame, outdir: Path):
    group_cols = ["model", "dataset_name", "genus", "split_type"]
    metrics = [m for m in DEFAULT_RANK_METRICS if m in all_df.columns]
    extra_metrics = [
        c for c in all_df.columns
        if (
            c.startswith("test_")
            or c.startswith("val_")
            or c.startswith("boundary_")
            or c.startswith("non_boundary_")
            or c.startswith("nn_")
            or c.startswith("ac_")
            or c.startswith("range_")
            or c.startswith("mean_local_")
            or c.startswith("local_variance_")
        ) and c not in metrics
    ]
    metrics = metrics + extra_metrics
    numeric_metrics = [m for m in metrics if m in all_df.columns and pd.api.types.is_numeric_dtype(all_df[m])]
    agg = all_df.groupby(group_cols)[numeric_metrics].agg(["mean", "std", "count"]).reset_index()
    agg.columns = ["_".join([str(x) for x in col if str(x)]) for col in agg.columns]

    new_cols = {}
    for m in numeric_metrics:
        mean_col = f"{m}_mean"
        std_col = f"{m}_std"
        if mean_col in agg.columns and std_col in agg.columns:
            new_cols[f"{m}_mean±sd"] = [fmt_mean_sd(a, b) for a, b in zip(agg[mean_col], agg[std_col])]
    if new_cols:
        agg = pd.concat([agg, pd.DataFrame(new_cols)], axis=1)
    agg.to_csv(outdir / "leaderboard_mean_sd.csv", index=False)
    return numeric_metrics


def write_ood_change_summary(all_df: pd.DataFrame, outdir: Path, numeric_metrics):
    mean_df = all_df.groupby(["model", "dataset_name", "genus", "split_type"])[numeric_metrics].mean().reset_index()
    rows = []
    for (model, dataset_name, genus), sub in mean_df.groupby(["model", "dataset_name", "genus"]):
        random_row = sub[sub["split_type"] == "random"]
        if random_row.empty:
            continue
        random_row = random_row.iloc[0]
        for target_split in OOD_TARGET_SPLITS:
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
                "analysis_note": "for error metrics: target_minus_random; for higher-is-better metrics: random_minus_target",
            }
            for m in numeric_metrics:
                direction = metric_direction(m)
                if direction == "lower":
                    row[f"ood_increase_{m}"] = safe_diff(target_row[m], random_row[m])
                else:
                    row[f"ood_drop_{m}"] = safe_diff(random_row[m], target_row[m])
            rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(outdir / "ood_change_summary.csv", index=False)


def write_perimeter_regression_ood_analysis(all_df: pd.DataFrame, outdir: Path):
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
            for m in PERIMETER_CORE_METRICS + LOW_NN_METRICS + DIAG_METRICS:
                if m in mean_df.columns:
                    if metric_direction(m) == "lower":
                        strict_minus_repeated[f"strict_minus_repeated_{m}"] = safe_diff(safe_get(strict_row, m), safe_get(repeated_row, m))
                    else:
                        strict_minus_repeated[f"strict_drop_vs_repeated_{m}"] = safe_diff(safe_get(repeated_row, m), safe_get(strict_row, m))

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
                "analysis_note": "error_increase=perimeter-random; higher_metric_drop=random-perimeter",
            }
            for m in PERIMETER_CORE_METRICS:
                if m in mean_df.columns:
                    row[f"random_{m}"] = safe_get(random_row, m)
                    row[f"perimeter_{m}"] = safe_get(target_row, m)
                    if metric_direction(m) == "lower":
                        row[f"perimeter_increase_{m}"] = safe_diff(safe_get(target_row, m), safe_get(random_row, m))
                    else:
                        row[f"perimeter_drop_{m}"] = safe_diff(safe_get(random_row, m), safe_get(target_row, m))
            for m in LOW_NN_METRICS + DIAG_METRICS:
                if m in mean_df.columns:
                    row[f"perimeter_{m}"] = safe_get(target_row, m)
            row.update(strict_minus_repeated)
            rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        preferred = [
            "model", "dataset_name", "genus", "reference_split", "target_split", "target_split_seed_count",
            "random_test_rmse", "perimeter_test_rmse", "perimeter_increase_test_rmse",
            "random_test_mae", "perimeter_test_mae", "perimeter_increase_test_mae",
            "random_test_spearman", "perimeter_test_spearman", "perimeter_drop_test_spearman",
            "random_test_within_1_dilution", "perimeter_test_within_1_dilution", "perimeter_drop_test_within_1_dilution",
            "perimeter_nn_lt_0p4_n", "perimeter_nn_lt_0p4_mae", "perimeter_nn_lt_0p4_rmse", "perimeter_nn_lt_0p4_spearman", "perimeter_nn_lt_0p4_within_1_dilution",
            "perimeter_boundary_mae", "perimeter_ac_mae", "perimeter_ac_error_enrichment",
            "analysis_note",
        ]
        ordered = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
        df[ordered].to_csv(outdir / "perimeter_regression_ood_error_analysis.csv", index=False)


def write_rankings(all_df: pd.DataFrame, outdir: Path, rank_metric: str, numeric_metrics):
    if rank_metric not in all_df.columns:
        raise ValueError(f"rank_metric={rank_metric} not in columns. Available examples: {numeric_metrics[:20]}")
    ascending = metric_direction(rank_metric) == "lower"
    task_cols = ["dataset_name", "genus", "split_type", "seed"]
    rank_rows = []
    for _, sub in all_df.groupby(task_cols):
        sub = sub[["model", rank_metric] + task_cols].dropna(subset=[rank_metric]).copy()
        if sub.empty:
            continue
        sub["rank"] = sub[rank_metric].rank(ascending=ascending, method="average")
        best_value = sub[rank_metric].min() if ascending else sub[rank_metric].max()
        sub["is_top_model"] = (sub[rank_metric] == best_value).astype(int)
        sub["rank_direction"] = "lower_is_better" if ascending else "higher_is_better"
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
    parser.add_argument("--results-root", required=True, help="Root output directory from regression model training")
    parser.add_argument("--outdir", default=None, help="Summary output directory")
    parser.add_argument("--rank-metric", default="test_rmse", help="Metric used for average ranking and top model frequency")
    args = parser.parse_args()

    root = Path(args.results_root).resolve()
    outdir = Path(args.outdir).resolve() if args.outdir else root / "_combined_summary"
    outdir.mkdir(parents=True, exist_ok=True)

    all_df = load_all_metrics(root)
    all_df.to_csv(outdir / "combined_all_run_metrics.csv", index=False)

    numeric_metrics = write_leaderboard(all_df, outdir)
    write_ood_change_summary(all_df, outdir, numeric_metrics)
    write_perimeter_regression_ood_analysis(all_df, outdir)
    write_rankings(all_df, outdir, args.rank_metric, numeric_metrics)

    print(f"Saved regression combined summaries to: {outdir.name}")
    print(f"Dedicated regression perimeter OOD diagnosis saved to: perimeter_regression_ood_error_analysis.csv")


if __name__ == "__main__":
    main()
