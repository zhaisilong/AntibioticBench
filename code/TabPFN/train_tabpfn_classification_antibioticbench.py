#!/usr/bin/env python3
"""AntibioticBench classification with standard TabPFN-3 in-context inference.

No HPO and no downstream fine-tuning. Core TabPFN model settings are fixed;
fit_mode is explicitly set to fit_with_cache to reuse the fitted train-context KV cache
during validation/test inference, and device is explicitly CUDA.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tabpfn import TabPFNClassifier

from tabpfn_benchmark_common import (
    DEFAULT_AC_FOLD_CHANGE,
    DEFAULT_AC_SIM_THRESHOLD,
    DEFAULT_NN_NBITS,
    DEFAULT_NN_RADIUS,
    FEATURE_TYPE,
    MODEL_FAMILY,
    MODEL_NAME,
    THRESHOLD_USED,
    DatasetFeatureCacheManager,
    RDKIT2D_DESCRIPTOR_NAMES,
    descriptor_missing_summary,
    best_f1_threshold,
    build_classification_prediction_table,
    classification_diagnostics_wide,
    classification_metrics,
    classification_subgroup_table,
    classification_training_log,
    compute_similarity_diagnostics,
    discover_runs,
    get_true_mic_arrays,
    load_split_dataframe,
    mark_failure,
    mark_success,
    output_run_dir,
    package_versions,
    prefix_metrics,
    predict_classifier_batched,
    save_model_reference,
    select_runs,
    setup_run_logger,
    training_log_to_long,
    write_json,
    write_summary_files,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--split-types", default="random,scaffold,repeated_stratified_perimeter,strict_perimeter")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--strict-perimeter-seeds", default="0")
    p.add_argument("--expected-datasets", type=int, default=12)
    p.add_argument("--run-mode", choices=["test", "batch"], default="batch")
    p.add_argument("--single-run-index", type=int, default=None)
    p.add_argument("--run-range", default=None, help="Inclusive run indices, e.g. 0-191")
    p.add_argument("--predict-batch-size", type=int, default=256, help="Query-row batching only; not a training hyperparameter")
    p.add_argument(
        "--feature-cache-root",
        type=Path,
        default=None,
        help="Task-specific per-dataset Morgan+RDKit2D disk cache; default: <out-root>/feature_cache",
    )
    p.add_argument(
        "--kv-cache-root",
        type=Path,
        default=None,
        help="Task-specific directory for KV-cache runtime metadata and optional persisted fitted state; default: <out-root>/kv_cache",
    )
    p.add_argument(
        "--save-fitted-kv-cache",
        action="store_true",
        help="Persist the fitted TabPFN state (including KV-cache fit state) as .tabpfn_fit. Disabled by default because files can be very large.",
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--allow-missing", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--postprocess", type=int, choices=[0, 1], default=1)
    p.add_argument("--ac-sim-threshold", type=float, default=DEFAULT_AC_SIM_THRESHOLD)
    p.add_argument("--ac-fold-change", type=float, default=DEFAULT_AC_FOLD_CHANGE)
    args = p.parse_args()
    if args.feature_cache_root is None:
        args.feature_cache_root = args.out_root / "feature_cache"
    if args.kv_cache_root is None:
        args.kv_cache_root = args.out_root / "kv_cache"
    return args


def split_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def model_params() -> dict:
    # Explicitly record the fixed TabPFN benchmark configuration for reproducibility.
    # fit_with_cache is a computational inference mode: it does not fine-tune weights.
    return {
        "model_version": "TabPFN-3_default",
        "mode": "standard_in_context_no_finetuning",
        "hyperparameter_search": False,
        "n_estimators": 8,
        "auto_scale_n_estimators": True,
        "categorical_features_indices": None,
        "softmax_temperature": 0.9,
        "balance_probabilities": False,
        "average_before_softmax": False,
        "model_path": "auto",
        "device": "cuda",
        "ignore_pretraining_limits": True,
        "inference_precision": "auto",
        "fit_mode": "fit_with_cache",
        "memory_saving_mode": "auto",
        "keep_cache_on_device": True,
        "kv_cache_precision": None,
        "random_state": 0,
        "n_preprocessing_jobs": 1,
        "inference_config": None,
        "differentiable_input": False,
        "eval_metric": None,
        "tuning_config": None,
        "show_progress_bar": False,
        "threshold_used": THRESHOLD_USED,
    }


def make_model() -> TabPFNClassifier:
    p = model_params()
    return TabPFNClassifier(
        n_estimators=p["n_estimators"],
        auto_scale_n_estimators=p["auto_scale_n_estimators"],
        categorical_features_indices=p["categorical_features_indices"],
        softmax_temperature=p["softmax_temperature"],
        balance_probabilities=p["balance_probabilities"],
        average_before_softmax=p["average_before_softmax"],
        model_path=p["model_path"],
        device=p["device"],
        ignore_pretraining_limits=p["ignore_pretraining_limits"],
        inference_precision=p["inference_precision"],
        fit_mode=p["fit_mode"],
        memory_saving_mode=p["memory_saving_mode"],
        keep_cache_on_device=p["keep_cache_on_device"],
        kv_cache_precision=p["kv_cache_precision"],
        random_state=p["random_state"],
        n_preprocessing_jobs=p["n_preprocessing_jobs"],
        inference_config=p["inference_config"],
        differentiable_input=p["differentiable_input"],
        eval_metric=p["eval_metric"],
        tuning_config=p["tuning_config"],
        show_progress_bar=p["show_progress_bar"],
    )


def run_one(index: int, total: int, spec, args, feature_cache: DatasetFeatureCacheManager) -> bool:
    run_dir = output_run_dir(args.out_root, spec)
    if args.resume and not args.force and (run_dir / "SUCCESS").exists():
        print(f"[SKIP {index+1}/{total}] {spec.id} (SUCCESS exists)")
        return True
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.force:
        for marker in (run_dir / "SUCCESS", run_dir / "FAILED.txt"):
            if marker.exists():
                marker.unlink()
    log = setup_run_logger(run_dir, "tabpfn-classification")
    started = time.time()
    try:
        log.info("=" * 100)
        log.info("RUN %d/%d | %s", index + 1, total, spec.id)
        log.info("Input directory configured")
        log.info("Output directory configured")
        log.info("No fine-tuning / no HPO. Train is context-only; validation is evaluation-only.")
        log.info("KV cache enabled via fit_mode=fit_with_cache; train prediction is disabled.")
        log.info("Model feature: Morgan radius=%d nBits=%d + RDKit2D descriptors=%d", DEFAULT_NN_RADIUS, DEFAULT_NN_NBITS, len(RDKIT2D_DESCRIPTOR_NAMES))
        log.info("Task-specific feature cache configured")
        log.info("Task-specific KV cache configured")
        log.info(
            "KV cache runtime storage: in-process %s memory; persisted .tabpfn_fit=%s",
            "GPU" if model_params()["keep_cache_on_device"] else "CPU",
            "enabled" if args.save_fitted_kv_cache else "disabled",
        )
        log.info("Test diagnostics: train-referenced Morgan/Tanimoto NN + activity cliff")

        train_df, tr_smi_col, tr_target = load_split_dataframe(spec.input_dir / "train.csv", "classification")
        val_df, va_smi_col, va_target = load_split_dataframe(spec.input_dir / "validation.csv", "classification")
        test_df, te_smi_col, te_target = load_split_dataframe(spec.input_dir / "test.csv", "classification")
        log.info("Rows | train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))
        log.info("Columns | smiles=%s target=%s", tr_smi_col, tr_target)

        train_smiles = train_df[tr_smi_col].astype(str).tolist()
        val_smiles = val_df[va_smi_col].astype(str).tolist()
        test_smiles = test_df[te_smi_col].astype(str).tolist()
        y_train = train_df[tr_target].to_numpy(int)
        y_val = val_df[va_target].to_numpy(int)
        y_test = test_df[te_target].to_numpy(int)

        t = time.time()
        required_smiles = train_smiles + val_smiles + test_smiles
        # First call for each dataset builds/loads a task-specific persistent cache shared by all classification splits/seeds.
        X_train, train_desc, train_fps, cache_info = feature_cache.get_features(
            spec.dataset_name, train_smiles, logger=log, need_bitvectors=True
        )
        X_val, val_desc, _, _ = feature_cache.get_features(
            spec.dataset_name, val_smiles, logger=log, need_bitvectors=False
        )
        X_test, test_desc, test_fps, _ = feature_cache.get_features(
            spec.dataset_name, test_smiles, logger=log, need_bitvectors=True
        )
        feature_seconds = time.time() - t
        log.info(
            "Feature retrieval finished in %.2fs | X_train=%s X_val=%s X_test=%s",
            feature_seconds, X_train.shape, X_val.shape, X_test.shape
        )
        log.info(
            "Feature split | Morgan=%d | RDKit2D=%d | total=%d",
            DEFAULT_NN_NBITS, len(RDKIT2D_DESCRIPTOR_NAMES), X_train.shape[1]
        )
        log.info(
            "RDKit2D missing | train=%s | val=%s | test=%s",
            descriptor_missing_summary(train_desc),
            descriptor_missing_summary(val_desc),
            descriptor_missing_summary(test_desc),
        )

        model = make_model()
        log.info("TabPFN fit() starting (build fitted KV cache; NO gradient training)...")
        t = time.time()
        model.fit(X_train, y_train)
        fit_seconds = time.time() - t
        effective_estimators = getattr(model, "n_estimators_", None)
        log.info("fit() finished in %.2fs | effective n_estimators=%s", fit_seconds, effective_estimators)

        # The live KV cache created by fit_with_cache is process-local memory.
        # Classification and regression launchers point to different output roots,
        # so any optional persisted fitted state is also task-isolated.
        kv_cache_run_dir = (
            args.kv_cache_root
            / spec.dataset_name
            / spec.split_type
            / f"seed_{spec.seed}"
        )
        kv_cache_run_dir.mkdir(parents=True, exist_ok=True)
        kv_state_path = kv_cache_run_dir / "fitted_model.tabpfn_fit"
        kv_runtime_info = {
            "task": "classification",
            "dataset_name": spec.dataset_name,
            "split_type": spec.split_type,
            "seed": spec.seed,
            "fit_mode": "fit_with_cache",
            "runtime_cache_location": "GPU memory" if model_params()["keep_cache_on_device"] else "CPU memory",
            "keep_cache_on_device": model_params()["keep_cache_on_device"],
            "kv_cache_precision": model_params()["kv_cache_precision"],
            "effective_n_estimators": effective_estimators,
            "fit_seconds": fit_seconds,
            "feature_cache_root": args.feature_cache_root.name,
            "kv_cache_root": args.kv_cache_root.name,
            "persist_fitted_state": bool(args.save_fitted_kv_cache),
            "persisted_state_path": kv_state_path.name if args.save_fitted_kv_cache else None,
            "note": (
                "The active TabPFN KV cache used by validation/test prediction is an in-process memory cache. "
                "The .tabpfn_fit file is an optional persisted fitted state for restart/reuse; it is not required "
                "for classification/regression process isolation."
            ),
        }
        if args.save_fitted_kv_cache:
            log.info("Persisting fitted TabPFN/KV state")
            t_save_kv = time.time()
            model.save_fit_state(kv_state_path)
            kv_runtime_info["persist_seconds"] = time.time() - t_save_kv
            log.info("Persisted fitted TabPFN/KV state in %.2fs", kv_runtime_info["persist_seconds"])
        write_json(kv_cache_run_dir / "kv_cache_runtime_info.json", kv_runtime_info)

        # Do NOT predict the labeled train set.  In standard TabPFN the same train
        # rows/labels are already part of the in-context context, so in-sample train
        # prediction is self-context evaluation and is not scientifically comparable
        # with ordinary model train performance.
        pred_train_seconds = 0.0
        log.info("Train prediction intentionally skipped (self-context evaluation is not meaningful for standard TabPFN).")

        t = time.time()
        val_prob = predict_classifier_batched(model, X_val, args.predict_batch_size, logger=log, split_name="validation")
        pred_val_seconds = time.time() - t
        log.info("Validation prediction finished in %.2fs", pred_val_seconds)
        t = time.time()
        test_prob = predict_classifier_batched(model, X_test, args.predict_batch_size, logger=log, split_name="test")
        pred_test_seconds = time.time() - t
        log.info("Test prediction finished in %.2fs", pred_test_seconds)

        val_m = classification_metrics(y_val, val_prob)
        test_m = classification_metrics(y_test, test_prob)
        # Preserve train sample statistics in metrics_full.csv, but all train
        # performance fields are NaN because predictions_train is not computed.
        train_m = {k: np.nan for k in val_m.keys()}
        train_m.update({
            "n": int(len(y_train)),
            "pos": int(np.sum(y_train == 1)),
            "neg": int(np.sum(y_train == 0)),
            "pos_ratio": float(np.mean(y_train == 1)) if len(y_train) else float("nan"),
        })
        val_best_thr = best_f1_threshold(y_val, val_prob)  # diagnostic only; never applied to test main metrics
        log.info("Overall | train prediction not computed; train performance metrics = NA")
        log.info("Overall | val   PR-AUC=%.6f ROC-AUC=%.6f logloss=%.6f", val_m["pr_auc"], val_m["roc_auc"], val_m["log_loss"])
        log.info("Overall | test  PR-AUC=%.6f ROC-AUC=%.6f logloss=%.6f MCC=%.6f", test_m["pr_auc"], test_m["roc_auc"], test_m["log_loss"], test_m["mcc"])
        log.info("Validation best-F1 threshold (sensitivity analysis only) = %.6f; benchmark threshold remains %.2f", val_best_thr, THRESHOLD_USED)

        # Only test gets NN / activity-cliff diagnostics.
        _, train_log10, _ = get_true_mic_arrays(train_df)
        _, test_log10, _ = get_true_mic_arrays(test_df)
        diag, ac_pairs = compute_similarity_diagnostics(
            train_fps=train_fps,
            query_fps=test_fps,
            train_smiles=train_smiles,
            train_true_log10=train_log10,
            query_true_log10=test_log10,
            train_pred=None,
            query_pred=test_prob,
            task="classification",
            ac_sim_threshold=args.ac_sim_threshold,
            ac_fold_change=args.ac_fold_change,
        )
        nn_idx = diag["nn_train_row"].astype(int).to_numpy()
        diag["nn_train_active"] = y_train[nn_idx]
        diag["nn_train_score"] = np.nan

        pred_val = build_classification_prediction_table(val_df, "validation", spec, val_prob, val_best_thr, va_target)
        pred_test = build_classification_prediction_table(test_df, "test", spec, test_prob, val_best_thr, te_target, diagnostics=diag)
        pred_val.to_csv(run_dir / "predictions_validation.csv", index=False)
        pred_test.to_csv(run_dir / "predictions_test.csv", index=False)
        ac_pairs.to_csv(run_dir / "activity_cliff_pairs_test.csv", index=False)

        diag_wide = classification_diagnostics_wide(pred_test, ac_pairs)
        diag_identity = {
            "model": MODEL_NAME,
            "model_family": MODEL_FAMILY,
            "task": "classification",
            "dataset_name": spec.dataset_name,
            "genus": spec.genus,
            "split_type": spec.split_type,
            "seed": spec.seed,
        }
        pd.DataFrame([{**diag_identity, **diag_wide}]).to_csv(run_dir / "diagnostics_metrics.csv", index=False)
        subgroup = classification_subgroup_table(pred_test)
        for k, v in reversed(list(diag_identity.items())):
            subgroup.insert(0, k, v)
        subgroup.to_csv(run_dir / "diagnostic_subgroup_metrics.csv", index=False)

        train_log = classification_training_log(None, val_m, fit_seconds, pred_train_seconds, pred_val_seconds)
        train_log.to_csv(run_dir / "train_log.csv", index=False)
        train_log.to_csv(run_dir / "training_curve.csv", index=False)
        training_log_to_long(train_log).to_csv(run_dir / "train_log_long.csv", index=False)

        params = model_params()
        params["effective_n_estimators_after_fit"] = effective_estimators
        params["predict_batch_size"] = args.predict_batch_size
        params["model_input_feature"] = FEATURE_TYPE
        write_json(run_dir / "model_params.json", params)
        write_json(
            run_dir / "feature_summary.json",
            {
                "model_input": FEATURE_TYPE,
                "feature_dim": int(X_train.shape[1]),
                "morgan_radius": DEFAULT_NN_RADIUS,
                "morgan_nbits": DEFAULT_NN_NBITS,
                "morgan_feature_dim": DEFAULT_NN_NBITS,
                "rdkit2d_descriptor_dim": len(RDKIT2D_DESCRIPTOR_NAMES),
                "rdkit2d_descriptor_names": list(RDKIT2D_DESCRIPTOR_NAMES),
                "rdkit2d_external_scaling": "none; TabPFN native preprocessing is used",
                "rdkit2d_nonfinite_handling": "descriptor failure/non-finite -> NaN; passed to TabPFN native preprocessing",
                "rdkit2d_missing_train": descriptor_missing_summary(train_desc),
                "rdkit2d_missing_validation": descriptor_missing_summary(val_desc),
                "rdkit2d_missing_test": descriptor_missing_summary(test_desc),
                "feature_cache": cache_info,
                "nn_diagnostic_reference": "train",
                "nn_metric": "Tanimoto",
                "nn_and_activity_cliff_representation": "Morgan radius=2, nBits=2048 only; RDKit2D descriptors are NOT used for NN/AC",
            },
        )
        save_model_reference(run_dir, spec, params)

        total_seconds = time.time() - started
        metrics = {
            "model": MODEL_NAME,
            "model_family": MODEL_FAMILY,
            "task": "classification",
            "dataset_name": spec.dataset_name,
            "genus": spec.genus,
            "split_type": spec.split_type,
            "seed": spec.seed,
            "threshold_from_dataset_name": spec.threshold,
            "threshold_for_boundary": spec.threshold,
            "threshold_mode": "fixed_0p5",
            "threshold_used": THRESHOLD_USED,
            "val_best_f1_threshold": val_best_thr,
            "feature_type": FEATURE_TYPE,
            "train_seconds": fit_seconds,
            "feature_seconds": feature_seconds,
            "predict_train_seconds": pred_train_seconds,
            "predict_val_seconds": pred_val_seconds,
            "predict_test_seconds": pred_test_seconds,
            "total_seconds": total_seconds,
            "best_epoch": np.nan,
            "best_val_metric": np.nan,
            "train_loss": np.nan,
            "val_loss": val_m["log_loss"],
            "effective_n_estimators": effective_estimators,
        }
        metrics.update(prefix_metrics("train", train_m))
        metrics.update(prefix_metrics("val", val_m))
        metrics.update(prefix_metrics("test", test_m))
        metrics.update(diag_wide)
        pd.DataFrame([metrics]).to_csv(run_dir / "metrics_full.csv", index=False)

        write_json(
            run_dir / "run_info.json",
            {
                "run_index": index,
                "run_id": spec.id,
                "input_dir": spec.input_dir.name,
                "output_dir": run_dir.name,
                "train_file": "train.csv",
                "validation_file": "validation.csv",
                "test_file": "test.csv",
                "smiles_columns": {"train": tr_smi_col, "validation": va_smi_col, "test": te_smi_col},
                "target_columns": {"train": tr_target, "validation": va_target, "test": te_target},
                "data_usage": {
                    "train": "TabPFN in-context labeled context only; no train prediction is performed",
                    "validation": "overall prediction/metrics only; no HPO, no threshold selection for main benchmark",
                    "test": "final overall metrics + NN/activity-cliff/MIC-boundary/MIC-range diagnostics",
                },
                "train_prediction_status": "not_computed_to_avoid_self_context_evaluation",
                "no_finetuning": True,
                "no_epoch_training": True,
                "loss_definition": "train_loss is NA because train prediction is intentionally omitted; val_loss is final validation log loss, not an optimization loss.",
                "activity_cliff": {
                    "reference": "test-to-train",
                    "tanimoto_threshold": args.ac_sim_threshold,
                    "fold_change_threshold": args.ac_fold_change,
                    "abs_log10mic_threshold": float(np.log10(args.ac_fold_change)),
                },
                "feature_cache": cache_info,
                "kv_cache": kv_runtime_info,
                "software": package_versions(),
                "environment": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "")},
                "timing_seconds": {
                    "feature": feature_seconds,
                    "fit": fit_seconds,
                    "predict_train": pred_train_seconds,
                    "predict_validation": pred_val_seconds,
                    "predict_test": pred_test_seconds,
                    "total": total_seconds,
                },
            },
        )
        mark_success(run_dir)
        log.info("SUCCESS | total %.2fs", total_seconds)
        return True
    except Exception as exc:
        mark_failure(run_dir, exc)
        log.error("FAILED | %s", type(exc).__name__)
        if args.fail_fast:
            raise RuntimeError(f"Run failed: {type(exc).__name__}") from None
        return False


def main() -> None:
    args = parse_args()
    splits = [x.strip() for x in args.split_types.split(",") if x.strip()]
    seeds = split_ints(args.seeds)
    strict_seeds = split_ints(args.strict_perimeter_seeds)
    runs = discover_runs(
        data_root=args.data_root,
        task="classification",
        split_types=splits,
        seeds=seeds,
        strict_seeds=strict_seeds,
        expected_datasets=args.expected_datasets,
        allow_missing=args.allow_missing,
    )
    selected = select_runs(runs, args.run_mode, args.single_run_index, args.run_range)
    print(f"Discovered {len(runs)} classification runs; selected {len(selected)}.")
    for i, r in enumerate(runs):
        print(f"RUN_INDEX {i:03d}: {r.id}")
    feature_cache = DatasetFeatureCacheManager(
        cache_root=args.feature_cache_root,
        data_root=args.data_root,
        task="classification",
        radius=DEFAULT_NN_RADIUS,
        n_bits=DEFAULT_NN_NBITS,
    )
    failures = 0
    for idx, spec in selected:
        ok = run_one(idx, len(runs), spec, args, feature_cache)
        if not ok:
            failures += 1
    if args.postprocess:
        write_summary_files(args.out_root, "classification")
        print("Classification summary files updated.")
    if failures:
        raise SystemExit(f"{failures} selected classification run(s) failed; see FAILED.txt/run.log in run directories.")


if __name__ == "__main__":
    main()
