#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

IDENTITY = [
    "model", "model_family", "task", "dataset_name",
    "genus", "split_type", "seed"
]

def numeric_columns(df):
    return [c for c in df.select_dtypes(include=[np.number]).columns if c != "seed"]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", required=True)
    p.add_argument("--task", choices=["classification", "regression"], required=True)
    p.add_argument("--expected-runs", type=int, default=192)
    args = p.parse_args()

    root = Path(args.out_root).resolve()
    model_root = root / "Uni-Mol"
    metric_files = sorted(model_root.glob(f"*/{args.task}/*/seed_*/metrics_full.csv"))
    if not metric_files:
        raise FileNotFoundError("No metrics_full.csv found under the output root")

    frames = []
    for f in metric_files:
        try:
            x = pd.read_csv(f)
            x["metrics_file"] = f.relative_to(root).as_posix()
            frames.append(x)
        except Exception as e:
            print(f"WARNING: cannot read {f.name}: {type(e).__name__}")
    if not frames:
        raise RuntimeError("No readable metrics files found")
    all_df = pd.concat(frames, ignore_index=True, sort=False)
    all_df.to_csv(model_root / "all_run_metrics.csv", index=False)

    if len(all_df) != args.expected_runs:
        print(f"WARNING: expected {args.expected_runs} completed runs, found {len(all_df)}")

    num = numeric_columns(all_df)
    group_cols = [c for c in ["model", "task", "dataset_name", "genus", "split_type"] if c in all_df]
    agg = all_df.groupby(group_cols, dropna=False)[num].agg(["mean", "std", "count"])
    agg.columns = ["__".join(x) for x in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(model_root / "aggregate_mean_sd.csv", index=False)

    primary = "test_pr_auc" if args.task == "classification" else "test_rmse"
    secondary = "test_EF1%" if args.task == "classification" else "test_spearman"

    ood_rows = []
    for (dataset, seed), sub in all_df.groupby(["dataset_name", "seed"], dropna=False):
        random_rows = sub[sub["split_type"] == "random"]
        if random_rows.empty:
            continue
        random_row = random_rows.iloc[0]
        for _, row in sub[sub["split_type"] != "random"].iterrows():
            rec = {
                "model": row.get("model"),
                "task": args.task,
                "dataset_name": dataset,
                "seed": seed,
                "ood_split_type": row["split_type"],
            }
            for metric in [primary, secondary]:
                if metric in all_df.columns:
                    rec[f"random_{metric}"] = random_row.get(metric)
                    rec[f"ood_{metric}"] = row.get(metric)
                    if args.task == "classification" or metric == "test_spearman":
                        rec[f"ood_drop_{metric}"] = random_row.get(metric) - row.get(metric)
                    else:
                        rec[f"ood_increase_{metric}"] = row.get(metric) - random_row.get(metric)
            ood_rows.append(rec)
    pd.DataFrame(ood_rows).to_csv(
        model_root / "ood_change_random_vs_ood_splits.csv", index=False
    )

    summary_dir = root / "_combined_summary" / (
        "classification_summary_like_ml"
        if args.task == "classification"
        else "regression_summary_like_ml"
    )
    summary_dir.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(summary_dir / "combined_all_run_metrics.csv", index=False)
    agg.to_csv(summary_dir / "leaderboard_mean_sd.csv", index=False)
    pd.DataFrame(ood_rows).to_csv(
        summary_dir / (
            "ood_drop_summary.csv"
            if args.task == "classification"
            else "ood_change_summary.csv"
        ),
        index=False,
    )

    status_files = list(model_root.glob(f"*/{args.task}/*/seed_*/status.json"))
    print(f"metrics files: {len(metric_files)}")
    print(f"status files: {len(status_files)}")
    print("summary written under the output root")

if __name__ == "__main__":
    main()
