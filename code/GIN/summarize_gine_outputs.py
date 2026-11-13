#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Summarize AntibioticBench GINE outputs across runs.

Produces both generic random-vs-OOD summaries and specification-aligned
perimeter OOD tables with explicit aliases such as perimeter_test_pr_auc and
perimeter_increase_test_rmse.
"""
from __future__ import annotations

if __name__ == "__main__":
    from privacy_utils import install_privacy_excepthook
    install_privacy_excepthook()

import argparse
import glob
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd


def ensure_dir(p: str | Path) -> Path:
    q = Path(p)
    q.mkdir(parents=True, exist_ok=True)
    return q


def read_all_metrics(model_root: Path) -> pd.DataFrame:
    files = [
        f for f in glob.glob(str(model_root / "**" / "metrics_full.csv"), recursive=True)
        if "_combined_summary" not in f
    ]
    rows: List[pd.DataFrame] = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["metrics_file"] = str(Path(f).relative_to(model_root))
            rows.append(df)
        except Exception as e:
            print(f"[WARN] failed to read {Path(f).name}: {type(e).__name__}")
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def flatten_mean_sd(grouped: pd.core.groupby.generic.DataFrameGroupBy, numeric_cols: Sequence[str]) -> pd.DataFrame:
    mean_df = grouped[list(numeric_cols)].mean(numeric_only=True).add_suffix("_mean")
    sd_df = grouped[list(numeric_cols)].std(ddof=1, numeric_only=True).add_suffix("_sd")
    n_df = grouped.size().rename("n_runs")
    return pd.concat([n_df, mean_df, sd_df], axis=1).reset_index()


def pick_metric(task: str) -> str:
    return "test_pr_auc" if task == "classification" else "test_rmse"


def rank_by_metric(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame()
    for (task, dataset, split_type, seed), sub in df.groupby(
        ["task", "dataset_name", "split_type", "seed"], dropna=False
    ):
        metric = pick_metric(str(task))
        if metric not in sub.columns:
            continue
        higher_better = not (str(task) == "regression" and metric.endswith("rmse"))
        keep = [c for c in ["model", "model_family", metric] if c in sub.columns]
        sub2 = sub[keep].dropna(subset=[metric]).copy()
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
    return (
        top.groupby(["task", "metric", "model"], dropna=False)
        .size().rename("top1_frequency").reset_index()
        .sort_values(["task", "metric", "top1_frequency"], ascending=[True, True, False])
    )


def numeric_metric_columns(df: pd.DataFrame) -> List[str]:
    """Metrics worth carrying into OOD tables.

    Besides test_* metrics, include test-only NN and activity-cliff fields.
    """
    diag_prefixes = ("nn_", "ac_", "non_ac_")
    cols: List[str] = []
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if c.startswith("test_") or c.startswith(diag_prefixes):
            cols.append(c)
    return cols


def ood_change(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if df.empty or "split_type" not in df.columns:
        return pd.DataFrame()
    metrics = numeric_metric_columns(df)
    keys = ["task", "dataset_name", "seed", "model"]
    for key, sub in df.groupby(keys, dropna=False):
        task, dataset, seed, model = key
        rand = sub[sub["split_type"].astype(str).str.lower() == "random"]
        if rand.empty:
            continue
        for split_type, ood in sub.groupby("split_type", dropna=False):
            if str(split_type).lower() == "random":
                continue
            row: Dict[str, Any] = {
                "task": task, "dataset_name": dataset, "seed": seed, "model": model,
                "random_split": "random", "ood_split": split_type,
            }
            if "model_family" in sub.columns:
                row["model_family"] = sub["model_family"].dropna().iloc[0] if sub["model_family"].notna().any() else np.nan
            for m in metrics:
                rv = pd.to_numeric(rand[m], errors="coerce").mean() if m in rand else np.nan
                ov = pd.to_numeric(ood[m], errors="coerce").mean() if m in ood else np.nan
                row[f"random_{m}"] = rv
                row[f"ood_{m}"] = ov
                lower_is_better = str(task) == "regression" and (
                    m.endswith("rmse") or m.endswith("mae") or m.endswith("mse") or m.endswith("medae")
                )
                row[("change_" if lower_is_better else "drop_") + m] = (ov - rv) if lower_is_better else (rv - ov)
            rows.append(row)
    return pd.DataFrame(rows)


def _first_numeric(row: pd.Series, col: str) -> float:
    try:
        return float(pd.to_numeric(pd.Series([row.get(col, np.nan)]), errors="coerce").iloc[0])
    except Exception:
        return np.nan


def perimeter_ood_analysis(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Create the exact perimeter-oriented aliases required by the specs.

    Only repeated_stratified_perimeter and strict_perimeter are included here;
    scaffold remains in the generic OOD summary.
    """
    perimeter_splits = {"repeated_stratified_perimeter", "strict_perimeter"}
    rows: List[Dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame()
    for (dataset, seed, model), sub in df[df["task"].astype(str) == task].groupby(
        ["dataset_name", "seed", "model"], dropna=False
    ):
        rand = sub[sub["split_type"].astype(str).str.lower() == "random"]
        if rand.empty:
            continue
        r = rand.iloc[0]
        for _, p in sub[sub["split_type"].astype(str).str.lower().isin(perimeter_splits)].iterrows():
            split_name = str(p["split_type"])
            out: Dict[str, Any] = {
                "task": task,
                "dataset_name": dataset,
                "seed": seed,
                "model": model,
                "model_family": p.get("model_family", np.nan),
                "perimeter_split": split_name,
            }
            if task == "classification":
                r_pr = _first_numeric(r, "test_pr_auc")
                p_pr = _first_numeric(p, "test_pr_auc")
                r_ef = _first_numeric(r, "test_EF1%")
                p_ef = _first_numeric(p, "test_EF1%")
                out.update({
                    "random_test_pr_auc": r_pr,
                    "perimeter_test_pr_auc": p_pr,
                    "perimeter_drop_test_pr_auc": r_pr - p_pr,
                    "random_test_EF1%": r_ef,
                    "perimeter_test_EF1%": p_ef,
                    "perimeter_drop_test_EF1%": r_ef - p_ef,
                })
                # Perimeter-specific failure diagnostics.
                for c in p.index:
                    if c.startswith("nn_lt_0p4_") or c.startswith("ac_") or c.startswith("non_ac_"):
                        out[f"perimeter_{c}"] = p[c]
            else:
                r_rmse = _first_numeric(r, "test_rmse")
                p_rmse = _first_numeric(p, "test_rmse")
                r_sp = _first_numeric(r, "test_spearman")
                p_sp = _first_numeric(p, "test_spearman")
                out.update({
                    "random_test_rmse": r_rmse,
                    "perimeter_test_rmse": p_rmse,
                    "perimeter_increase_test_rmse": p_rmse - r_rmse,
                    "random_test_spearman": r_sp,
                    "perimeter_test_spearman": p_sp,
                    "perimeter_drop_test_spearman": r_sp - p_sp,
                    "perimeter_ac_mae": _first_numeric(p, "ac_mae"),
                })
                for c in p.index:
                    if c.startswith("nn_lt_0p4_") or c.startswith("ac_") or c.startswith("non_ac_"):
                        out[f"perimeter_{c}"] = p[c]
            rows.append(out)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--model-name", default="GINE", help="Kept for CLI compatibility; out-root is used directly.")
    args = ap.parse_args()

    model_root = Path(args.out_root)
    if not model_root.exists():
        raise FileNotFoundError(f"model root not found: {model_root}")
    df = read_all_metrics(model_root)
    if df.empty:
        raise RuntimeError(f"No metrics_full.csv found under {model_root}")

    df.to_csv(model_root / "all_run_metrics.csv", index=False)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != "seed"]
    group_cols = [c for c in ["model", "model_family", "task", "dataset_name", "split_type", "feature_type"] if c in df.columns]
    flatten_mean_sd(df.groupby(group_cols, dropna=False), numeric_cols).to_csv(model_root / "aggregate_mean_sd.csv", index=False)

    ood = ood_change(df)
    ood.to_csv(model_root / "ood_change_random_vs_ood_splits.csv", index=False)

    ranking = rank_by_metric(df)
    ranking.to_csv(model_root / "model_ranking_per_task_seed.csv", index=False)
    ranking.to_csv(model_root / "model_ranking_by_metric.csv", index=False)
    topfreq = top_model_frequency(ranking)
    topfreq.to_csv(model_root / "top_model_frequency.csv", index=False)

    for task in sorted(df["task"].dropna().astype(str).unique()):
        task_df = df[df["task"].astype(str) == task].copy()
        summary_dir = ensure_dir(model_root / "_combined_summary" / f"{task}_summary_like_ml")
        task_df.to_csv(summary_dir / "combined_all_run_metrics.csv", index=False)
        task_group_cols = [c for c in ["model", "model_family", "task", "dataset_name", "split_type", "feature_type"] if c in task_df.columns]
        task_numeric_cols = [c for c in task_df.columns if pd.api.types.is_numeric_dtype(task_df[c]) and c != "seed"]
        flatten_mean_sd(task_df.groupby(task_group_cols, dropna=False), task_numeric_cols).to_csv(summary_dir / "leaderboard_mean_sd.csv", index=False)

        task_ood = ood[ood["task"].astype(str) == task].copy() if not ood.empty else pd.DataFrame()
        perimeter = perimeter_ood_analysis(df, task)
        if task == "classification":
            task_ood.to_csv(summary_dir / "ood_drop_summary.csv", index=False)
            perimeter.to_csv(summary_dir / "perimeter_ood_error_analysis.csv", index=False)
        else:
            task_ood.to_csv(summary_dir / "ood_change_summary.csv", index=False)
            perimeter.to_csv(summary_dir / "perimeter_regression_ood_error_analysis.csv", index=False)

        task_rank = ranking[ranking["task"].astype(str) == task].copy() if not ranking.empty else pd.DataFrame()
        task_rank.to_csv(summary_dir / "model_ranking_per_task_seed.csv", index=False)
        task_rank.to_csv(summary_dir / "model_ranking_by_metric.csv", index=False)
        task_top = topfreq[topfreq["task"].astype(str) == task].copy() if not topfreq.empty else pd.DataFrame()
        task_top.to_csv(summary_dir / "top_model_frequency.csv", index=False)

    print("Saved summary outputs.")
    print(f"Runs summarized: {len(df)}")


if __name__ == "__main__":
    main()
