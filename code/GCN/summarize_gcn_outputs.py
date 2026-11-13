#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize AntibioticBench GCN outputs across runs."""
from __future__ import annotations

if __name__ == "__main__":
    from privacy_utils import install_privacy_excepthook
    install_privacy_excepthook()

import argparse
import glob
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd


def ensure_dir(p: str | Path) -> Path:
    q = Path(p)
    q.mkdir(parents=True, exist_ok=True)
    return q


def read_all_metrics(model_root: Path) -> pd.DataFrame:
    files = [f for f in glob.glob(str(model_root / "**" / "metrics_full.csv"), recursive=True) if "_combined_summary" not in f]
    rows: List[pd.DataFrame] = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["metrics_file"] = str(Path(f).relative_to(model_root))
            rows.append(df)
        except Exception as e:
            print(f"[WARN] failed to read {Path(f).name}: {type(e).__name__}")
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True, sort=False)


def flatten_mean_sd(grouped: pd.core.groupby.generic.DataFrameGroupBy, numeric_cols: Sequence[str]) -> pd.DataFrame:
    mean_df = grouped[numeric_cols].mean(numeric_only=True).add_suffix("_mean")
    sd_df = grouped[numeric_cols].std(ddof=1, numeric_only=True).add_suffix("_sd")
    n_df = grouped.size().rename("n_runs")
    return pd.concat([n_df, mean_df, sd_df], axis=1).reset_index()


def pick_metric(task: str) -> str:
    return "test_pr_auc" if task == "classification" else "test_rmse"


def rank_by_metric(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame()
    for (task, dataset, split_type, seed), sub in df.groupby(["task", "dataset_name", "split_type", "seed"], dropna=False):
        metric = pick_metric(str(task))
        if metric not in sub.columns:
            continue
        higher_better = not (str(task) == "regression" and metric.endswith("rmse"))
        sub2 = sub[["model", "model_family", metric]].copy()
        sub2 = sub2.dropna(subset=[metric])
        if sub2.empty:
            continue
        sub2["rank"] = sub2[metric].rank(ascending=not higher_better, method="min")
        for _, r in sub2.iterrows():
            rows.append({
                "task": task,
                "dataset_name": dataset,
                "split_type": split_type,
                "seed": seed,
                "metric": metric,
                "model": r.get("model"),
                "model_family": r.get("model_family"),
                "value": r.get(metric),
                "rank": r.get("rank"),
            })
    return pd.DataFrame(rows)


def top_model_frequency(ranking: pd.DataFrame) -> pd.DataFrame:
    if ranking.empty:
        return pd.DataFrame()
    top = ranking[ranking["rank"] == 1].copy()
    if top.empty:
        return pd.DataFrame()
    return top.groupby(["task", "metric", "model"], dropna=False).size().rename("top1_frequency").reset_index().sort_values(["task", "metric", "top1_frequency"], ascending=[True, True, False])


def ood_change(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if df.empty or "split_type" not in df.columns:
        return pd.DataFrame()
    metrics = [c for c in df.columns if c.startswith("test_") and pd.api.types.is_numeric_dtype(df[c])]
    for (task, dataset, seed, model), sub in df.groupby(["task", "dataset_name", "seed", "model"], dropna=False):
        rand = sub[sub["split_type"].astype(str).str.lower() == "random"]
        if rand.empty:
            continue
        for split_type, ood in sub.groupby("split_type"):
            if str(split_type).lower() == "random":
                continue
            row: Dict[str, Any] = {"task": task, "dataset_name": dataset, "seed": seed, "model": model, "random_split": "random", "ood_split": split_type}
            for m in metrics:
                rv = pd.to_numeric(rand[m], errors="coerce").mean()
                ov = pd.to_numeric(ood[m], errors="coerce").mean()
                row[f"random_{m}"] = rv
                row[f"ood_{m}"] = ov
                if str(task) == "regression" and (m.endswith("rmse") or m.endswith("mae") or m.endswith("mse")):
                    row[f"change_{m}"] = ov - rv
                else:
                    row[f"drop_{m}"] = rv - ov
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True, help="Output directory used by train_gcn_antibioticbench.py; summaries are written directly under this folder")
    ap.add_argument("--model-name", default="GCN", help="Model name kept for compatibility; output root is used directly and no extra model-name subdirectory is added")
    args = ap.parse_args()

    model_root = Path(args.out_root)
    if not model_root.exists():
        raise FileNotFoundError(f"model root not found: {model_root}")
    df = read_all_metrics(model_root)
    if df.empty:
        raise RuntimeError(f"No metrics_full.csv found under {model_root}")

    # Save model-level all_run_metrics.
    df.to_csv(model_root / "all_run_metrics.csv", index=False)

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in {"seed"}]
    group_cols = [c for c in ["model", "model_family", "task", "dataset_name", "split_type", "feature_type"] if c in df.columns]
    agg = flatten_mean_sd(df.groupby(group_cols, dropna=False), numeric_cols)
    agg.to_csv(model_root / "aggregate_mean_sd.csv", index=False)

    ood = ood_change(df)
    ood.to_csv(model_root / "ood_change_random_vs_ood_splits.csv", index=False)

    ranking = rank_by_metric(df)
    ranking.to_csv(model_root / "model_ranking_per_task_seed.csv", index=False)
    ranking.to_csv(model_root / "model_ranking_by_metric.csv", index=False)
    topfreq = top_model_frequency(ranking)
    topfreq.to_csv(model_root / "top_model_frequency.csv", index=False)

    # Save task-specific combined summary folders exactly like the specification.
    for task in sorted(df["task"].dropna().unique()):
        task_df = df[df["task"] == task].copy()
        summary_dir = model_root / "_combined_summary" / f"{task}_summary_like_ml"
        ensure_dir(summary_dir)
        task_df.to_csv(summary_dir / "combined_all_run_metrics.csv", index=False)
        task_group_cols = [c for c in ["model", "model_family", "task", "dataset_name", "split_type", "feature_type"] if c in task_df.columns]
        task_numeric_cols = [c for c in task_df.columns if pd.api.types.is_numeric_dtype(task_df[c]) and c not in {"seed"}]
        leaderboard = flatten_mean_sd(task_df.groupby(task_group_cols, dropna=False), task_numeric_cols)
        leaderboard.to_csv(summary_dir / "leaderboard_mean_sd.csv", index=False)
        task_ood = ood[ood["task"] == task].copy() if not ood.empty else pd.DataFrame()
        # Naming differs in user specs: classification uses ood_drop_summary, regression uses ood_change_summary.
        if str(task) == "classification":
            task_ood.to_csv(summary_dir / "ood_drop_summary.csv", index=False)
            task_ood.to_csv(summary_dir / "perimeter_ood_error_analysis.csv", index=False)
        else:
            task_ood.to_csv(summary_dir / "ood_change_summary.csv", index=False)
            task_ood.to_csv(summary_dir / "perimeter_regression_ood_error_analysis.csv", index=False)
        task_rank = ranking[ranking["task"] == task].copy() if not ranking.empty else pd.DataFrame()
        task_rank.to_csv(summary_dir / "model_ranking_per_task_seed.csv", index=False)
        task_rank.to_csv(summary_dir / "model_ranking_by_metric.csv", index=False)
        task_top = topfreq[topfreq["task"] == task].copy() if not topfreq.empty else pd.DataFrame()
        task_top.to_csv(summary_dir / "top_model_frequency.csv", index=False)

    print("Saved summary outputs.")
    print(f"Runs summarized: {len(df)}")


if __name__ == "__main__":
    main()
