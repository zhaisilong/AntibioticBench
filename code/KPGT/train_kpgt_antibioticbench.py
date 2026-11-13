#!/usr/bin/env python3
"""Fine-tune official KPGT on one AntibioticBench run and emit unified outputs.

Important benchmark behavior:
- train and validation are evaluated every epoch; validation selects the checkpoint.
- test is never touched during training and is evaluated once after the best checkpoint is loaded.
- train/validation predictions contain overall metrics only; test additionally receives NN and activity-cliff diagnostics.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from kpgt_benchmark_utils import (
    add_test_nn_and_activity_cliff,
    best_f1_threshold,
    build_test_diagnostics,
    classification_metrics,
    discover_kpgt_root,
    discover_pretrained_model,
    enumerate_runs,
    extract_mic_arrays,
    load_dataset_config,
    make_empty_diagnostic_columns,
    mic_range_fields,
    now_iso,
    package_versions,
    prefix_metrics,
    prepare_kpgt_dataset,
    regression_metrics,
    reorder_prediction_columns,
    save_split_metrics,
    set_global_seed,
    validate_input_run,
    write_json,
    write_train_log_files,
)


def sanitize_error_trace(text: str, args: argparse.Namespace) -> str:
    """Keep useful error context without writing configured directories to results."""
    roots = [Path.home(), Path(__file__).resolve().parent, Path(sys.executable).parent]
    for name in ("data_root", "out_root", "cache_root", "kpgt_root", "model_path", "molformer_repo", "dataset_config"):
        value = getattr(args, name, None)
        if value:
            roots.append(Path(value).expanduser().resolve())
    for name in ("BENCHMARK_ROOT", "DATA_ROOT", "OUT_ROOT", "KPGT_ROOT", "KPGT_PRETRAINED_MODEL", "MOLFORMER_ROOT", "HF_HOME", "HF_HUB_CACHE"):
        if os.environ.get(name):
            roots.append(Path(os.environ[name]).expanduser().resolve())
    for root in sorted({str(p) for p in roots if str(p) not in ("", ".", "/")}, key=len, reverse=True):
        text = text.replace(root, "<local-path>")
    return re.sub(r'File "([^"]+)"', lambda m: 'File "' + Path(m.group(1)).name + '"', text)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--task", choices=["classification", "regression"], required=True)
    parser.add_argument("--data-root", type=Path, default=os.environ.get("DATA_ROOT") or None)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--dataset-config", type=Path, default=script_dir / "dataset_config.csv")
    parser.add_argument("--single-run-index", type=int, default=None)
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--split-type", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--list-runs", action="store_true")
    parser.add_argument("--validate-inputs", action="store_true")

    parser.add_argument("--kpgt-root", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--config", type=str, default="base")
    parser.add_argument("--smiles-col", type=str, default=None)
    parser.add_argument("--target-col", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n-predictor-layers", type=int, default=2)
    parser.add_argument("--d-predictor-hidden", type=int, default=256)
    parser.add_argument("--warmup-fraction", type=float, default=0.1)
    parser.add_argument("--final-lr", type=float, default=1e-9)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))))
    parser.add_argument("--eval-num-workers", type=int, default=0)
    parser.add_argument("--preprocess-jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    parser.add_argument("--use-pos-weight", action="store_true")
    parser.add_argument("--selection-metric", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--fp-radius", type=int, default=2)
    parser.add_argument("--fp-bits", type=int, default=2048)
    parser.add_argument("--ac-sim-threshold", type=float, default=0.7)
    parser.add_argument("--ac-log10-diff-threshold", type=float, default=math.log10(16.0))
    parser.add_argument("--local-k", type=int, default=10)

    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--resume", type=int, choices=[0, 1], default=1)
    parser.add_argument("--save-last-checkpoint", action="store_true")
    return parser.parse_args()


def init_linear(module) -> None:
    import torch
    from torch import nn

    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


def make_predictor(
    d_input_feats: int,
    n_tasks: int,
    n_layers: int,
    dropout: float,
    d_hidden: int,
    device,
):
    from torch import nn

    if n_layers == 1:
        predictor = nn.Linear(d_input_feats, n_tasks)
    else:
        layers: List[nn.Module] = [nn.Linear(d_input_feats, d_hidden), nn.Dropout(dropout), nn.GELU()]
        for _ in range(n_layers - 2):
            layers.extend([nn.Linear(d_hidden, d_hidden), nn.Dropout(dropout), nn.GELU()])
        layers.append(nn.Linear(d_hidden, n_tasks))
        predictor = nn.Sequential(*layers)
    predictor.apply(init_linear)
    return predictor.to(device)


def seed_worker(worker_id: int) -> None:
    import random
    import torch

    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def unpack_batch(batch):
    if not isinstance(batch, (tuple, list)) or len(batch) != 5:
        raise ValueError(f"Expected Collator_tune batch of length 5, received {type(batch)} length={len(batch)}")
    smiles, graph, fps, mds, labels = batch
    return smiles, graph, fps, mds, labels


def masked_loss(
    logits,
    labels,
    task: str,
    loss_fn,
    label_mean=None,
    label_std=None,
):
    import torch

    labels = labels.float()
    mask = torch.isfinite(labels)
    if not mask.any():
        return None
    if task == "regression":
        target = (labels - label_mean) / label_std
    else:
        target = labels
    values = loss_fn(logits, target)
    return values[mask].mean()


def run_training_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    task: str,
    loss_fn,
    device,
    label_mean,
    label_std,
    grad_clip: float,
) -> float:
    import torch

    model.train()
    weighted_loss = 0.0
    total_items = 0
    for batch in loader:
        _, graph, fps, mds, labels = unpack_batch(batch)
        graph = graph.to(device)
        fps = fps.to(device)
        mds = mds.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model.forward_tune(graph, fps, mds)
        loss = masked_loss(logits, labels, task, loss_fn, label_mean, label_std)
        if loss is None:
            continue
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        n_items = int(torch.isfinite(labels).sum().item())
        weighted_loss += float(loss.detach().cpu()) * max(n_items, 1)
        total_items += n_items
    return weighted_loss / total_items if total_items else float("nan")


def predict_loader(
    model,
    loader,
    task: str,
    loss_fn,
    device,
    label_mean,
    label_std,
) -> Tuple[np.ndarray, np.ndarray, float]:
    import torch

    model.eval()
    truths: List[np.ndarray] = []
    preds: List[np.ndarray] = []
    weighted_loss = 0.0
    total_items = 0
    with torch.no_grad():
        for batch in loader:
            _, graph, fps, mds, labels = unpack_batch(batch)
            graph = graph.to(device)
            fps = fps.to(device)
            mds = mds.to(device)
            labels = labels.to(device).float()
            logits = model.forward_tune(graph, fps, mds)
            loss = masked_loss(logits, labels, task, loss_fn, label_mean, label_std)
            n_items = int(torch.isfinite(labels).sum().item())
            if loss is not None:
                weighted_loss += float(loss.detach().cpu()) * max(n_items, 1)
                total_items += n_items
            if task == "classification":
                pred = torch.sigmoid(logits)
            else:
                pred = logits * label_std + label_mean
            truths.append(labels.detach().cpu().numpy())
            preds.append(pred.detach().cpu().numpy())
    y_true = np.concatenate(truths, axis=0).reshape(-1)
    y_pred = np.concatenate(preds, axis=0).reshape(-1)
    return y_true, y_pred, weighted_loss / total_items if total_items else float("nan")


def metric_for_selection(task: str, metrics: Mapping[str, Any], selection_metric: str, val_loss: float) -> Tuple[float, str]:
    key = selection_metric.replace("val_", "")
    raw = metrics.get(key, float("nan"))
    raw_float = float(raw) if raw is not None else float("nan")
    maximize = task == "classification" or key in {"spearman", "pearson", "r2"}
    if math.isfinite(raw_float):
        return (raw_float if maximize else -raw_float), key
    # Stable fallback when a validation fold has only one class or a correlation is undefined.
    if task == "classification":
        for fallback in ("roc_auc", "pr_auc", "f1"):
            value = metrics.get(fallback, float("nan"))
            if value is not None and math.isfinite(float(value)):
                return float(value), f"fallback_{fallback}"
    return -float(val_loss), "fallback_negative_val_loss"


def build_prediction_dataframe(
    source_df: pd.DataFrame,
    split: str,
    task: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    run: Mapping[str, Any],
    smiles_col: str,
    threshold: float,
    val_best_threshold: float,
) -> pd.DataFrame:
    out = source_df.copy().reset_index(drop=True)
    if len(out) != len(y_true) or len(out) != len(y_pred):
        raise ValueError(f"Prediction length mismatch for {split}: source={len(out)}, true={len(y_true)}, pred={len(y_pred)}")
    if smiles_col != "smiles":
        out["smiles"] = out[smiles_col].astype(str)
    else:
        out["smiles"] = out[smiles_col].astype(str)
    identity = {
        "split": split,
        "dataset_name": run["dataset_name"],
        "genus": run["genus"],
        "task": task,
        "split_type": run["split_type"],
        "seed": run["seed"],
        "model": "KPGT",
        "model_family": "pretrained_graph_transformer",
    }
    for key, value in reversed(list(identity.items())):
        if key in out.columns:
            out[key] = value
        else:
            out.insert(0, key, value)

    mic, log10mic_diag, log2mic_diag = extract_mic_arrays(out)
    if task == "classification":
        true = np.asarray(y_true, dtype=float)
        prob = np.asarray(y_pred, dtype=float)
        pred_label = (prob >= threshold).astype(int)
        out["active"] = true
        out["true_active"] = true
        out["pred_score"] = prob
        out["pred_prob"] = prob
        out["pred_label"] = pred_label
        out["pred_label_0p5"] = (prob >= 0.5).astype(int)
        out["threshold_used"] = threshold
        out["val_best_f1_threshold"] = val_best_threshold
        out["error"] = (pred_label != true.astype(int)).astype(int)
        out["mic_numeric"] = mic
        out["log10_mic_for_diagnosis"] = log10mic_diag
        out["log2_mic_for_diagnosis"] = log2mic_diag
    else:
        true = np.asarray(y_true, dtype=float)
        pred = np.asarray(y_pred, dtype=float)
        true_mic = np.power(10.0, true)
        pred_mic = np.power(10.0, pred)
        abs_err = np.abs(pred - true)
        out["log10_mic"] = true
        out["true_log10MIC"] = true
        out["true_MIC"] = true_mic
        out["mic_numeric"] = true_mic
        out["pred_log10MIC"] = pred
        out["pred_MIC"] = pred_mic
        out["score_reg"] = -pred
        out["abs_error_log10"] = abs_err
        out["sq_error_log10"] = (pred - true) ** 2
        out["fold_error"] = np.power(10.0, abs_err)
        out["within_1_dilution"] = (abs_err <= math.log10(2.0)).astype(int)
        out["within_2_dilutions"] = (abs_err <= math.log10(4.0)).astype(int)
        t = float(run["threshold_for_boundary"])
        out["true_active_for_screening"] = (true_mic <= t).astype(int)
    out["threshold_for_boundary"] = float(run["threshold_for_boundary"])
    boundary, mic_range = mic_range_fields(
        pd.to_numeric(out["mic_numeric"], errors="coerce").to_numpy(dtype=float),
        float(run["threshold_for_boundary"]),
    )
    out["boundary_flag"] = boundary
    out["mic_range_group"] = mic_range
    out = make_empty_diagnostic_columns(out, task)
    return reorder_prediction_columns(out, task)


def long_training_row(
    epoch: int,
    train_loss: float,
    train_eval_loss: float,
    val_loss: float,
    learning_rate: float,
    train_metrics: Mapping[str, Any],
    val_metrics: Mapping[str, Any],
    best_epoch: int,
    best_metric: float,
    note: str,
    task: str,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "iteration": epoch,
        "epoch": epoch,
        "train_loss": train_eval_loss,
        "train_optimization_loss": train_loss,
        "val_loss": val_loss,
        "learning_rate": learning_rate,
        "best_epoch_so_far": best_epoch,
        "best_val_metric_so_far": best_metric,
        "note": note,
    }
    if task == "classification":
        aliases = {
            "log_loss": "log_loss",
            "roc_auc": "auc",
            "pr_auc": "pr_auc",
            "accuracy": "accuracy",
            "f1": "f1",
            "mcc": "mcc",
        }
        for source, alias in aliases.items():
            row[f"train_{alias}"] = train_metrics.get(source)
            row[f"val_{alias}"] = val_metrics.get(source)
        row["train_auc"] = train_metrics.get("roc_auc")
        row["val_auc"] = val_metrics.get("roc_auc")
    else:
        for name in (
            "mse",
            "rmse",
            "mae",
            "spearman",
            "pearson",
            "within_1_dilution",
            "within_2_dilutions",
        ):
            row[f"train_{name}"] = train_metrics.get(name)
            row[f"val_{name}"] = val_metrics.get(name)
    return row


def load_torch_checkpoint(path: Path, device):
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def run_one(args: argparse.Namespace, run: Mapping[str, Any]) -> Path:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    started = time.time()
    started_at = now_iso()
    set_global_seed(int(run["seed"]))
    kpgt_root = discover_kpgt_root(args.kpgt_root)
    model_path = discover_pretrained_model(kpgt_root, args.model_path)
    if str(kpgt_root) not in sys.path:
        sys.path.insert(0, str(kpgt_root))

    from src.data.collator import Collator_tune
    from src.data.featurizer import N_ATOM_TYPES, N_BOND_TYPES, Vocab
    from src.data.finetune_dataset import MoleculeDataset
    from src.model.light import LiGhTPredictor
    from src.model_config import config_dict

    out_root = args.out_root
    if out_root is None:
        base = Path(os.environ.get("BENCHMARK_ROOT", Path(__file__).resolve().parents[2])) / "outputs/KPGT"
        out_root = base / args.task
    out_root = Path(out_root).expanduser().resolve()
    run_dir = (
        out_root
        / "KPGT"
        / str(run["dataset_name"])
        / args.task
        / str(run["split_type"])
        / f"seed_{run['seed']}"
    )
    success_file = run_dir / "_SUCCESS"
    if args.resume and success_file.is_file() and not args.force:
        print("[resume] Complete run exists; skipping", flush=True)
        return run_dir
    if args.force and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        print("=" * 100, flush=True)
        print(f"KPGT AntibioticBench run index: {run['run_index']}", flush=True)
        print(json.dumps(dict(run), ensure_ascii=False, indent=2), flush=True)
        print("KPGT source root configured", flush=True)
        print(f"Pretrained checkpoint: {model_path.name}", flush=True)
        print("Output directory configured", flush=True)
        print(f"Python: {sys.version.split()[0]}", flush=True)
        print(f"Torch: {torch.__version__}; CUDA available={torch.cuda.is_available()}", flush=True)
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

        input_paths = validate_input_run(args.data_root.expanduser().resolve(), run)
        cache_root = args.cache_root
        if cache_root is None:
            cache_root = Path(os.environ.get("BENCHMARK_ROOT", Path(__file__).resolve().parents[2])) / ".cache/kpgt"
        cache_root = Path(cache_root).expanduser().resolve()
        prep = prepare_kpgt_dataset(
            kpgt_root=kpgt_root,
            cache_root=cache_root,
            run=run,
            input_paths=input_paths,
            task=args.task,
            smiles_col_requested=args.smiles_col,
            target_col_requested=args.target_col,
            n_jobs=args.preprocess_jobs,
            force_preprocess=args.force_preprocess,
        )
        print(f"Internal KPGT dataset id: {prep['dataset_id']}", flush=True)
        print(f"Input sizes: {prep['lengths']}", flush=True)
        print(f"SMILES column: {prep['smiles_col']}; target column: {prep['target_col']}", flush=True)

        config = copy.deepcopy(config_dict[args.config])
        vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
        collator = Collator_tune(config["path_length"])
        train_dataset = MoleculeDataset(
            root_path=str(prep["cache_root"]),
            dataset=prep["dataset_id"],
            dataset_type=args.task,
            split_name=prep["split_name"],
            split="train",
        )
        val_dataset = MoleculeDataset(
            root_path=str(prep["cache_root"]),
            dataset=prep["dataset_id"],
            dataset_type=args.task,
            split_name=prep["split_name"],
            split="val",
        )
        test_dataset = MoleculeDataset(
            root_path=str(prep["cache_root"]),
            dataset=prep["dataset_id"],
            dataset_type=args.task,
            split_name=prep["split_name"],
            split="test",
        )
        if train_dataset.n_tasks != 1:
            raise ValueError(f"Expected one task column, KPGT dataset has n_tasks={train_dataset.n_tasks}")

        generator = torch.Generator()
        generator.manual_seed(int(run["seed"]))
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            generator=generator,
            drop_last=(len(train_dataset) >= args.batch_size),
            collate_fn=collator,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(args.num_workers > 0),
        )
        train_eval_loader = DataLoader(
            train_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.eval_num_workers,
            worker_init_fn=seed_worker,
            generator=generator,
            drop_last=False,
            collate_fn=collator,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.eval_num_workers,
            worker_init_fn=seed_worker,
            generator=generator,
            drop_last=False,
            collate_fn=collator,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.eval_num_workers,
            worker_init_fn=seed_worker,
            generator=generator,
            drop_last=False,
            collate_fn=collator,
            pin_memory=torch.cuda.is_available(),
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LiGhTPredictor(
            d_node_feats=config["d_node_feats"],
            d_edge_feats=config["d_edge_feats"],
            d_g_feats=config["d_g_feats"],
            d_fp_feats=train_dataset.d_fps,
            d_md_feats=train_dataset.d_mds,
            d_hpath_ratio=config["d_hpath_ratio"],
            n_mol_layers=config["n_mol_layers"],
            path_length=config["path_length"],
            n_heads=config["n_heads"],
            n_ffn_dense_layers=config["n_ffn_dense_layers"],
            input_drop=0,
            attn_drop=args.dropout,
            feat_drop=args.dropout,
            n_node_types=vocab.vocab_size,
        ).to(device)
        pretrained = load_torch_checkpoint(model_path, device)
        if isinstance(pretrained, dict) and "state_dict" in pretrained:
            pretrained = pretrained["state_dict"]
        pretrained = {str(k).replace("module.", ""): v for k, v in pretrained.items()}
        model.load_state_dict(pretrained, strict=True)
        model.predictor = make_predictor(
            d_input_feats=config["d_g_feats"] * 3,
            n_tasks=1,
            n_layers=args.n_predictor_layers,
            dropout=args.dropout,
            d_hidden=args.d_predictor_hidden,
            device=device,
        )
        del model.md_predictor
        del model.fp_predictor
        del model.node_predictor
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: total={total_params:,}; trainable={trainable_params:,}", flush=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        steps_per_epoch = max(1, len(train_loader))
        total_steps = max(1, args.epochs * steps_per_epoch)
        warmup_steps = int(round(total_steps * args.warmup_fraction))
        try:
            from transformers import get_polynomial_decay_schedule_with_warmup

            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
                lr_end=args.final_lr,
                power=1.0,
            )
        except Exception:
            def lr_lambda(step: int) -> float:
                if warmup_steps > 0 and step < warmup_steps:
                    return max(step, 1) / warmup_steps
                progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
                end_ratio = args.final_lr / args.lr
                return max(end_ratio, (1 - progress) * (1 - end_ratio) + end_ratio)

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        if args.task == "classification":
            if args.use_pos_weight:
                pos_weight = train_dataset._task_pos_weights.float().to(device)
                print(f"Using BCE pos_weight={pos_weight.detach().cpu().numpy().tolist()}", flush=True)
                loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
            else:
                loss_fn = nn.BCEWithLogitsLoss(reduction="none")
            label_mean, label_std = None, None
            selection_metric = args.selection_metric or "val_pr_auc"
        else:
            loss_fn = nn.MSELoss(reduction="none")
            mean = train_dataset.mean.float().numpy()
            std = train_dataset.std.float().numpy()
            std = np.where(np.isfinite(std) & (std > 1e-12), std, 1.0)
            label_mean = torch.tensor(mean, dtype=torch.float32, device=device)
            label_std = torch.tensor(std, dtype=torch.float32, device=device)
            selection_metric = args.selection_metric or "val_rmse"
            print(f"Regression train-label normalization mean={mean.tolist()}, std={std.tolist()}", flush=True)

        model_params = {
            "config": args.config,
            "architecture": "LiGhT/KPGT base",
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "n_predictor_layers": args.n_predictor_layers,
            "d_predictor_hidden": args.d_predictor_hidden,
            "warmup_fraction": args.warmup_fraction,
            "final_lr": args.final_lr,
            "grad_clip": args.grad_clip,
            "use_pos_weight": bool(args.use_pos_weight),
            "num_workers": args.num_workers,
            "eval_num_workers": args.eval_num_workers,
            "preprocess_jobs": args.preprocess_jobs,
            "selection_metric": selection_metric,
            "classification_threshold": args.threshold,
            "fine_tuning_strategy": "standard_full_model_finetuning",
            "advanced_strategies": {"FLAG": False, "LLRD": False, "L2SP": False, "ReInit": False},
        }
        feature_summary = {
            "feature_type": "KPGT_LiGhT_graph+rdkfp1-7_512+molecular_descriptors",
            "molecular_graph": True,
            "line_graph": True,
            "rdkit_fingerprint_dimension": int(train_dataset.d_fps),
            "molecular_descriptor_dimension": int(train_dataset.d_mds),
            "graph_hidden_dimension": int(config["d_g_feats"]),
            "n_molecular_transformer_layers": int(config["n_mol_layers"]),
            "n_attention_heads": int(config["n_heads"]),
            "path_length": int(config["path_length"]),
        }
        write_json(run_dir / "model_params.json", model_params)
        write_json(run_dir / "feature_summary.json", feature_summary)

        history_rows: List[Dict[str, Any]] = []
        best_score = -float("inf")
        best_epoch = 0
        best_metric_raw = float("nan")
        best_selection_source = selection_metric
        epochs_without_improvement = 0
        best_checkpoint_path = run_dir / "best_model.pth"

        print("[training] Test loader is intentionally not evaluated until best checkpoint is selected.", flush=True)
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            train_optim_loss = run_training_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                args.task,
                loss_fn,
                device,
                label_mean,
                label_std,
                args.grad_clip,
            )
            train_true, train_pred, train_eval_loss = predict_loader(
                model, train_eval_loader, args.task, loss_fn, device, label_mean, label_std
            )
            val_true, val_pred, val_loss = predict_loader(
                model, val_loader, args.task, loss_fn, device, label_mean, label_std
            )
            if args.task == "classification":
                train_metrics = classification_metrics(train_true, train_pred, args.threshold)
                val_metrics = classification_metrics(val_true, val_pred, args.threshold)
            else:
                t = float(run["threshold_for_boundary"])
                train_metrics = regression_metrics(train_true, train_pred, t)
                val_metrics = regression_metrics(val_true, val_pred, t)

            selection_score, selection_source = metric_for_selection(
                args.task, val_metrics, selection_metric, val_loss
            )
            improved = selection_score > best_score + 1e-12
            if improved:
                best_score = selection_score
                best_epoch = epoch
                metric_key = selection_metric.replace("val_", "")
                best_metric_raw = float(val_metrics.get(metric_key, np.nan))
                best_selection_source = selection_source
                epochs_without_improvement = 0
                checkpoint = {
                    "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "epoch": epoch,
                    "selection_metric": selection_metric,
                    "selection_score_internal": selection_score,
                    "best_val_metric": best_metric_raw,
                    "selection_source": selection_source,
                    "label_mean": None if label_mean is None else label_mean.detach().cpu(),
                    "label_std": None if label_std is None else label_std.detach().cpu(),
                    "run": dict(run),
                    "model_params": model_params,
                    "feature_summary": feature_summary,
                }
                torch.save(checkpoint, best_checkpoint_path)
            else:
                epochs_without_improvement += 1

            current_lr = float(optimizer.param_groups[0]["lr"])
            note = "new_best" if improved else "train_val_only_test_not_evaluated"
            row = long_training_row(
                epoch,
                train_optim_loss,
                train_eval_loss,
                val_loss,
                current_lr,
                train_metrics,
                val_metrics,
                best_epoch,
                best_metric_raw,
                note,
                args.task,
            )
            row["epoch_seconds"] = time.time() - epoch_start
            row["selection_metric"] = selection_metric
            row["selection_source"] = selection_source
            history_rows.append(row)
            history = pd.DataFrame(history_rows)
            write_train_log_files(run_dir, history)

            if args.task == "classification":
                print(
                    f"Epoch {epoch:03d} | train_loss={train_eval_loss:.6f} val_loss={val_loss:.6f} "
                    f"train_PR-AUC={train_metrics.get('pr_auc', np.nan):.6f} "
                    f"val_PR-AUC={val_metrics.get('pr_auc', np.nan):.6f} "
                    f"val_ROC-AUC={val_metrics.get('roc_auc', np.nan):.6f} "
                    f"lr={current_lr:.3e} best_epoch={best_epoch} no_improve={epochs_without_improvement}",
                    flush=True,
                )
            else:
                print(
                    f"Epoch {epoch:03d} | train_loss={train_eval_loss:.6f} val_loss={val_loss:.6f} "
                    f"train_RMSE={train_metrics.get('rmse', np.nan):.6f} "
                    f"val_RMSE={val_metrics.get('rmse', np.nan):.6f} "
                    f"val_Spearman={val_metrics.get('spearman', np.nan):.6f} "
                    f"lr={current_lr:.3e} best_epoch={best_epoch} no_improve={epochs_without_improvement}",
                    flush=True,
                )
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}", flush=True)
                break

        if args.save_last_checkpoint:
            torch.save(
                {"model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "epoch": epoch},
                run_dir / "last_model.pth",
            )
        if not best_checkpoint_path.is_file():
            raise RuntimeError("No best checkpoint was saved")

        checkpoint = load_torch_checkpoint(best_checkpoint_path, device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        print(f"[final] Loaded best checkpoint from epoch {checkpoint['epoch']}", flush=True)
        # The following is the first and only test evaluation.
        train_true, train_pred, best_train_loss = predict_loader(
            model, train_eval_loader, args.task, loss_fn, device, label_mean, label_std
        )
        val_true, val_pred, best_val_loss = predict_loader(
            model, val_loader, args.task, loss_fn, device, label_mean, label_std
        )
        test_true, test_pred, test_loss = predict_loader(
            model, test_loader, args.task, loss_fn, device, label_mean, label_std
        )
        val_best_threshold = best_f1_threshold(val_true, val_pred) if args.task == "classification" else np.nan

        pred_frames: Dict[str, pd.DataFrame] = {}
        for split, true, pred in (
            ("train", train_true, train_pred),
            ("validation", val_true, val_pred),
            ("test", test_true, test_pred),
        ):
            pred_frames[split] = build_prediction_dataframe(
                prep["frames"][split],
                split,
                args.task,
                true,
                pred,
                run,
                prep["smiles_col"],
                args.threshold,
                val_best_threshold,
            )

        # Only test receives NN and Activity Cliff diagnostics.
        test_with_diag, diag_meta = add_test_nn_and_activity_cliff(
            task=args.task,
            train_pred=pred_frames["train"],
            test_pred=pred_frames["test"],
            fp_radius=args.fp_radius,
            fp_bits=args.fp_bits,
            ac_similarity_threshold=args.ac_sim_threshold,
            ac_log10_diff_threshold=args.ac_log10_diff_threshold,
            local_k=args.local_k,
        )
        pred_frames["test"] = reorder_prediction_columns(test_with_diag, args.task)
        pred_frames["train"] = reorder_prediction_columns(pred_frames["train"], args.task)
        pred_frames["validation"] = reorder_prediction_columns(pred_frames["validation"], args.task)
        for split, df in pred_frames.items():
            df.to_csv(run_dir / f"predictions_{split}.csv", index=False)

        if args.task == "classification":
            split_metrics = {
                "train": classification_metrics(train_true, train_pred, args.threshold),
                "validation": classification_metrics(val_true, val_pred, args.threshold),
                "test": classification_metrics(test_true, test_pred, args.threshold),
            }
        else:
            t = float(run["threshold_for_boundary"])
            split_metrics = {
                "train": regression_metrics(train_true, train_pred, t),
                "validation": regression_metrics(val_true, val_pred, t),
                "test": regression_metrics(test_true, test_pred, t),
            }
        save_split_metrics(run_dir, split_metrics)

        identity = {
            "model": "KPGT",
            "model_family": "pretrained_graph_transformer",
            "task": args.task,
            "dataset_name": run["dataset_name"],
            "genus": run["genus"],
            "split_type": run["split_type"],
            "seed": run["seed"],
        }
        diagnostics_wide, diagnostics_long = build_test_diagnostics(
            args.task,
            pred_frames["test"],
            identity,
            float(run["threshold_for_boundary"]),
        )
        diagnostics_wide.to_csv(run_dir / "diagnostics_metrics.csv", index=False)
        diagnostics_long.to_csv(run_dir / "diagnostic_subgroup_metrics.csv", index=False)

        final_history = pd.DataFrame(history_rows)
        best_hist = final_history.loc[final_history["epoch"] == int(checkpoint["epoch"])].iloc[0].to_dict()
        metrics_full: Dict[str, Any] = dict(identity)
        metrics_full.update(
            {
                "feature_type": feature_summary["feature_type"],
                "target_scale": "binary_active" if args.task == "classification" else "log10_mic",
                "threshold_from_dataset_name": float(run["threshold_for_boundary"]),
                "threshold_for_boundary": float(run["threshold_for_boundary"]),
                "threshold_mode": "fixed_0p5" if args.task == "classification" else "not_applicable",
                "threshold_used": args.threshold if args.task == "classification" else np.nan,
                "val_best_f1_threshold": val_best_threshold,
                "train_seconds": time.time() - started,
                "best_epoch": int(checkpoint["epoch"]),
                "best_val_metric": checkpoint.get("best_val_metric"),
                "select_metric": selection_metric,
                "selection_source": checkpoint.get("selection_source"),
                "train_loss": best_train_loss,
                "val_loss": best_val_loss,
                "test_loss": test_loss,
                "pretrained_model": model_path.name,
                "source_hash": prep["source_hash"],
                "dataset_content_hash": prep["dataset_content_hash"],
                "kpgt_internal_dataset_id": prep["dataset_id"],
            }
        )
        for split, metrics in split_metrics.items():
            prefix = "val" if split == "validation" else split
            metrics_full.update(prefix_metrics(metrics, prefix))
        if args.task == "regression":
            for split, values in (("train", train_true), ("val", val_true), ("test", test_true)):
                arr = np.asarray(values, dtype=float)
                arr = arr[np.isfinite(arr)]
                metrics_full[f"{split}_log10mic_mean"] = float(np.mean(arr)) if len(arr) else np.nan
                metrics_full[f"{split}_log10mic_std"] = float(np.std(arr, ddof=0)) if len(arr) else np.nan
        for col, value in diagnostics_wide.iloc[0].to_dict().items():
            if col not in identity:
                metrics_full[col] = value
        pd.DataFrame([metrics_full]).to_csv(run_dir / "metrics_full.csv", index=False)

        run_info = {
            **identity,
            "run_index": run["run_index"],
            "started_at": started_at,
            "finished_at": now_iso(),
            "data_root": "<DATA_ROOT>",
            "input_paths": {k: str(v.relative_to(args.data_root)) for k, v in input_paths.items()},
            "output_dir": str(run_dir.relative_to(out_root)),
            "kpgt_root": "<KPGT_ROOT>",
            "pretrained_model": model_path.name,
            "dataset_config": args.dataset_config.name,
            "smiles_column": prep["smiles_col"],
            "target_column": prep["target_col"],
            "source_hash": prep["source_hash"],
            "dataset_content_hash": prep["dataset_content_hash"],
            "split_role_policy": {
                "train": "prediction_and_overall_metrics; used for fitting",
                "validation": "prediction_and_overall_metrics; checkpoint selection",
                "test": "single final evaluation; overall metrics plus NN and activity-cliff diagnostics",
            },
            "diagnostic_definition": diag_meta,
            "software_versions": package_versions(),
            "slurm": {k: os.environ.get(k) for k in ("SLURM_CPUS_PER_TASK",)},
            "reproducibility_note": "KPGT/DGL/PyTorch GPU operations may retain some nondeterminism despite fixed seeds.",
        }
        write_json(run_dir / "run_info.json", run_info)
        shutil.copy2(Path(__file__).resolve(), run_dir / "train_kpgt_antibioticbench.used.py")
        shutil.copy2(Path(__file__).resolve().parent / "kpgt_benchmark_utils.py", run_dir / "kpgt_benchmark_utils.used.py")
        shutil.copy2(args.dataset_config.resolve(), run_dir / "dataset_config.used.csv")
        success_file.write_text(f"completed_at={now_iso()}\n", encoding="utf-8")
        failed_marker = run_dir / "_FAILED"
        if failed_marker.exists():
            failed_marker.unlink()
        print("[success] Completed", flush=True)
        print(pd.DataFrame([metrics_full]).to_string(index=False), flush=True)
        return run_dir
    except Exception:
        error_text = sanitize_error_trace(traceback.format_exc(), args)
        print(error_text, file=sys.stderr, flush=True)
        (run_dir / "_FAILED").write_text(error_text, encoding="utf-8")
        raise SystemExit("Run failed; see _FAILED for sanitized details") from None


def select_runs(args: argparse.Namespace, all_runs: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    if args.single_run_index is not None:
        if args.single_run_index < 0 or args.single_run_index >= len(all_runs):
            raise IndexError(f"single-run-index {args.single_run_index} outside [0, {len(all_runs)-1}]")
        return [all_runs[args.single_run_index]]
    selected = list(all_runs)
    if args.dataset_name is not None:
        selected = [r for r in selected if r["dataset_name"] == args.dataset_name or r["genus"] == args.dataset_name]
    if args.split_type is not None:
        selected = [r for r in selected if r["split_type"] == args.split_type]
    if args.seed is not None:
        selected = [r for r in selected if int(r["seed"]) == args.seed]
    if not selected:
        raise ValueError("No runs matched the requested filters")
    return selected


def main() -> None:
    args = parse_args()
    args.dataset_config = args.dataset_config.expanduser().resolve()
    config_df = load_dataset_config(args.dataset_config)
    all_runs = enumerate_runs(config_df, args.task)
    if args.list_runs:
        print(pd.DataFrame(all_runs).to_csv(index=False), end="")
        return
    if args.data_root is None:
        raise SystemExit("Set --data-root or DATA_ROOT to the prepared split-data directory")
    args.data_root = args.data_root.expanduser().resolve()
    if args.validate_inputs:
        failures = []
        for run in all_runs:
            try:
                validate_input_run(args.data_root, run)
            except Exception as e:
                failures.append({**run, "error": sanitize_error_trace(str(e), args)})
        if failures:
            print(pd.DataFrame(failures).to_string(index=False), file=sys.stderr)
            raise SystemExit(f"Input validation failed for {len(failures)}/{len(all_runs)} runs")
        print(f"Validated all {len(all_runs)} {args.task} runs")
        return
    selected = select_runs(args, all_runs)
    print(f"Selected {len(selected)} run(s) out of {len(all_runs)} total for task={args.task}", flush=True)
    for run in selected:
        run_one(args, run)


if __name__ == "__main__":
    main()
