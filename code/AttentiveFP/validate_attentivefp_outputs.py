#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate AttentiveFP outputs for the test-only diagnostic policy.

Policy:
  train: predictions + overall metrics only
  validation: predictions + overall metrics only
  test: predictions + overall metrics + NN + activity-cliff diagnostics
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

COMMON_FILES = [
    "metrics_full.csv", "predictions_train.csv", "predictions_validation.csv",
    "predictions_test.csv", "diagnostics_metrics.csv", "diagnostic_subgroup_metrics.csv",
    "train_log.csv", "training_curve.csv", "train_log_long.csv", "feature_summary.json",
    "model_params.json", "run_info.json", "best_model.pth", "run.log",
]

COMMON_PRED_ID = ["split", "dataset_name", "genus", "task", "split_type", "seed", "model", "model_family", "smiles"]
DIAGNOSTIC_PREFIXES = ("nn_", "ac_")

REQ = {
    "classification": {
        "metrics_full.csv": [
            "model", "task", "dataset_name", "genus", "split_type", "seed", "feature_type",
            "threshold_used", "train_loss", "val_loss",
            "train_roc_auc", "train_pr_auc", "train_accuracy", "train_f1", "train_mcc",
            "val_roc_auc", "val_pr_auc", "val_accuracy", "val_f1", "val_mcc",
            "test_roc_auc", "test_pr_auc", "test_mcc", "test_f1", "test_accuracy",
            "test_balanced_accuracy", "test_log_loss", "test_EF1%", "test_EF5%",
            "test_EF10%", "test_Hit_rate@Top1%", "test_Hit_rate@Top5%",
            "test_Hit_rate@Top10%", "test_Recall@Top50", "test_Recall@Top100",
            "test_Recall@Top500", "nn_ge_0p7_error_rate", "nn_0p4_0p7_error_rate",
            "nn_lt_0p4_error_rate", "nn_lt_0p4_pr_auc", "nn_lt_0p4_EF1%",
            "ac_error_rate", "non_ac_error_rate", "ac_error_enrichment",
        ],
        "predictions_train.csv": COMMON_PRED_ID + [
            "active", "true_active", "pred_score", "pred_prob", "pred_label", "error",
        ],
        "predictions_validation.csv": COMMON_PRED_ID + [
            "active", "true_active", "pred_score", "pred_prob", "pred_label", "error",
        ],
        "predictions_test.csv": COMMON_PRED_ID + [
            "true_active", "pred_score", "pred_prob", "pred_label", "error",
            "nn_train_row", "nn_sim", "nn_bin", "nn_train_smiles", "nn_train_active",
            "nn_train_score", "ac_flag", "ac_neighbor_count", "ac_max_abs_log2mic_diff",
            "ac_pair_directional_accuracy_for_molecule", "ac_pair_count_for_molecule",
        ],
        "train_log.csv": [
            "epoch", "iteration", "train_loss", "val_loss", "train_pr_auc", "val_pr_auc",
            "train_auc", "val_auc", "learning_rate", "note",
        ],
    },
    "regression": {
        "metrics_full.csv": [
            "model", "task", "dataset_name", "genus", "split_type", "seed", "feature_type",
            "train_loss", "val_loss", "train_rmse", "train_mae", "train_spearman",
            "val_rmse", "val_mae", "val_spearman", "test_rmse", "test_mae", "test_mse",
            "test_medae", "test_r2", "test_pearson", "test_spearman",
            "test_within_1_dilution", "test_within_2_dilutions", "test_median_fold_error",
            "test_EF1%", "test_EF5%", "test_EF10%", "test_Hit_rate@Top1%",
            "test_Hit_rate@Top5%", "test_Hit_rate@Top10%", "test_Recall@Top50",
            "test_Recall@Top100", "test_Recall@Top500", "nn_ge_0p7_rmse",
            "nn_0p4_0p7_rmse", "nn_lt_0p4_rmse", "nn_lt_0p4_mae",
            "nn_lt_0p4_spearman", "ac_mae", "non_ac_mae", "ac_error_enrichment",
        ],
        "predictions_train.csv": COMMON_PRED_ID + [
            "true_log10MIC", "pred_log10MIC", "true_MIC", "pred_MIC", "score_reg",
            "abs_error_log10", "within_1_dilution", "within_2_dilutions",
        ],
        "predictions_validation.csv": COMMON_PRED_ID + [
            "true_log10MIC", "pred_log10MIC", "true_MIC", "pred_MIC", "score_reg",
            "abs_error_log10", "within_1_dilution", "within_2_dilutions",
        ],
        "predictions_test.csv": COMMON_PRED_ID + [
            "true_log10MIC", "pred_log10MIC", "true_MIC", "pred_MIC", "score_reg",
            "abs_error_log10", "fold_error", "within_1_dilution", "within_2_dilutions",
            "nn_train_row", "nn_sim", "nn_bin", "nn_train_smiles", "nn_train_log10mic",
            "nn_train_pred_log10mic", "ac_flag", "ac_neighbor_count",
            "ac_max_abs_log10mic_diff", "ac_pair_directional_accuracy_for_molecule",
            "ac_pair_count_for_molecule",
        ],
        "train_log.csv": [
            "epoch", "iteration", "train_loss", "val_loss", "train_rmse", "val_rmse",
            "train_mae", "val_mae", "train_spearman", "val_spearman", "learning_rate", "note",
        ],
    },
}


def find_run_dirs(out_root: Path, task: str) -> List[Path]:
    files = [Path(f) for f in glob.glob(str(out_root / "**" / "metrics_full.csv"), recursive=True)]
    dirs = []
    for f in files:
        if "_combined_summary" in str(f):
            continue
        try:
            df = pd.read_csv(f, nrows=1)
            if "task" in df.columns and not df.empty and str(df.iloc[0]["task"]) == task:
                dirs.append(f.parent)
        except Exception:
            dirs.append(f.parent)
    return sorted(set(dirs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--task", required=True, choices=["classification", "regression"])
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    root = Path(args.out_root)
    rows: List[Dict] = []
    run_dirs = find_run_dirs(root, args.task)

    for d in run_dirs:
        missing_files = [fn for fn in COMMON_FILES if not (d / fn).exists()]
        missing_cols: Dict[str, List[str]] = {}
        forbidden_cols: Dict[str, List[str]] = {}
        read_errors: Dict[str, str] = {}
        for fn, required in REQ[args.task].items():
            p = d / fn
            if not p.exists():
                continue
            try:
                cols = list(pd.read_csv(p, nrows=1).columns)
                colset = set(cols)
                miss = [c for c in required if c not in colset]
                if miss:
                    missing_cols[fn] = miss
                if fn in {"predictions_train.csv", "predictions_validation.csv"}:
                    bad = [c for c in cols if c.startswith(DIAGNOSTIC_PREFIXES)]
                    if bad:
                        forbidden_cols[fn] = bad
            except Exception as e:
                read_errors[fn] = type(e).__name__
        rows.append({
            "task": args.task,
            "run_dir": str(d.relative_to(root)),
            "missing_files": ";".join(missing_files),
            "missing_columns": json.dumps(missing_cols, ensure_ascii=False, sort_keys=True),
            "forbidden_train_val_diagnostic_columns": json.dumps(forbidden_cols, ensure_ascii=False, sort_keys=True),
            "read_errors": json.dumps(read_errors, ensure_ascii=False, sort_keys=True),
            "compliant": not missing_files and not missing_cols and not forbidden_cols and not read_errors,
        })

    summary_dir = root / "_combined_summary" / f"{args.task}_summary_like_ml"
    summary_requirements = [
        root / "all_run_metrics.csv",
        root / "aggregate_mean_sd.csv",
        root / "ood_change_random_vs_ood_splits.csv",
        summary_dir / "combined_all_run_metrics.csv",
        summary_dir / "leaderboard_mean_sd.csv",
        summary_dir / ("ood_drop_summary.csv" if args.task == "classification" else "ood_change_summary.csv"),
        summary_dir / ("perimeter_ood_error_analysis.csv" if args.task == "classification" else "perimeter_regression_ood_error_analysis.csv"),
        root / "_combined_summary" / "activity_cliff_pair_analysis" / "ac_pair_summary_all_runs.csv",
    ]
    missing_summary = [str(p.relative_to(root)) for p in summary_requirements if not p.exists()]
    report_dir = root / "_combined_summary"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame(rows)
    report.to_csv(report_dir / f"{args.task}_output_compliance_report.csv", index=False)
    overall = bool(len(report)) and bool(report["compliant"].all()) and not missing_summary
    payload = {
        "task": args.task,
        "policy": {
            "train": "predictions_and_overall_metrics_only",
            "validation": "predictions_and_overall_metrics_only",
            "test": "predictions_overall_metrics_nn_and_activity_cliff",
        },
        "out_root": ".",
        "n_runs_checked": len(report),
        "n_compliant_runs": int(report["compliant"].sum()) if len(report) else 0,
        "missing_summary_files": missing_summary,
        "overall_compliant": overall,
    }
    (report_dir / f"{args.task}_output_compliance_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.strict and not overall:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
