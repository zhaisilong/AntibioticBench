#!/usr/bin/env python3
"""Common MoLFormer fine-tuning engine for AntibioticBench.

Protocol guarantees
-------------------
* train and validation are evaluated every epoch and their losses/metrics are logged.
* validation only selects the best checkpoint.
* test is not evaluated during training; it is evaluated once after reloading the best checkpoint.
* train/validation predictions receive overall metrics only.
* test predictions additionally receive train-reference NN and activity-cliff diagnostics.
* strict_perimeter is enumerated only for seed 0 by molformer_benchmark_utils.enumerate_runs.
"""
from __future__ import annotations

import argparse
import hashlib
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

from molformer_benchmark_utils import (
    CLASS_TARGET_CANDIDATES,
    REG_TARGET_CANDIDATES,
    SMILES_CANDIDATES,
    add_test_nn_and_activity_cliff,
    best_f1_threshold,
    build_test_diagnostics,
    classification_metrics,
    coerce_binary,
    enumerate_runs,
    extract_mic_arrays,
    find_column,
    load_dataset_config,
    make_empty_diagnostic_columns,
    mic_range_fields,
    now_iso,
    package_versions,
    prefix_metrics,
    regression_metrics,
    reorder_prediction_columns,
    save_split_metrics,
    set_global_seed,
    sha256_files,
    validate_input_run,
    write_json,
    write_train_log_files,
)

MODEL_NAME = "MoLFormer"
MODEL_FAMILY = "pretrained_smiles_language_model"


def public_model_id(model_id: str) -> str:
    """Avoid embedding local filesystem paths in shareable metadata."""
    if model_id.startswith(("/", "~", "./", "../")) or Path(model_id).is_dir():
        return "<local model>"
    return model_id


DEFAULT_MODEL_ID = "ibm-research/MoLFormer-XL-both-10pct"
DEFAULT_REVISION = "compat-v4"


def sanitize_error_trace(text: str, args: argparse.Namespace) -> str:
    """Keep useful error context without writing configured directories to results."""
    roots = [Path.home(), Path(__file__).resolve().parent, Path(sys.executable).parent]
    for name in ("data_root", "out_root", "cache_root", "kpgt_root", "model_path", "molformer_repo", "dataset_config", "model_id"):
        value = getattr(args, name, None)
        if value:
            roots.append(Path(value).expanduser().resolve())
    for name in ("BENCHMARK_ROOT", "DATA_ROOT", "OUT_ROOT", "KPGT_ROOT", "KPGT_PRETRAINED_MODEL", "MOLFORMER_ROOT", "HF_HOME", "HF_HUB_CACHE"):
        if os.environ.get(name):
            roots.append(Path(os.environ[name]).expanduser().resolve())
    for root in sorted({str(p) for p in roots if str(p) not in ("", ".", "/")}, key=len, reverse=True):
        text = text.replace(root, "<local-path>")
    return re.sub(r'File "([^"]+)"', lambda m: 'File "' + Path(m.group(1)).name + '"', text)


def build_parser(task: str) -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    benchmark_root = Path(os.environ.get("BENCHMARK_ROOT", script_dir.parents[1]))
    default_out = benchmark_root / "outputs" / "MolFormer" / task
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-root", type=Path, default=os.environ.get("DATA_ROOT") or None)
    p.add_argument("--out-root", type=Path, default=default_out)
    p.add_argument("--dataset-config", type=Path, default=script_dir / "dataset_config.csv")
    p.add_argument("--single-run-index", type=int, default=None)
    p.add_argument("--dataset-name", type=str, default=None)
    p.add_argument("--split-type", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--list-runs", action="store_true")
    p.add_argument("--validate-inputs", action="store_true")

    p.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    p.add_argument("--model-revision", type=str, default=DEFAULT_REVISION)
    p.add_argument("--local-files-only", type=int, choices=[0, 1], default=0)
    p.add_argument("--molformer-repo", type=Path, default=os.environ.get("MOLFORMER_ROOT"))
    p.add_argument("--smiles-col", type=str, default=None)
    p.add_argument("--target-col", type=str, default=None)
    p.add_argument("--canonicalize-non-isomeric", type=int, choices=[0, 1], default=1)
    p.add_argument("--max-length", type=int, default=202)

    # Fixed benchmark hyperparameters.
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.99)
    p.add_argument("--adam-eps", type=float, default=1e-8)
    p.add_argument("--head-dropout", type=float, default=0.10)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--eval-num-workers", type=int, default=0)
    p.add_argument("--selection-metric", type=str, default=("val_pr_auc" if task == "classification" else "val_rmse"))
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--min-delta", type=float, default=0.0)

    # Unified diagnostics.
    p.add_argument("--fp-radius", type=int, default=2)
    p.add_argument("--fp-bits", type=int, default=2048)
    p.add_argument("--ac-sim-threshold", type=float, default=0.7)
    p.add_argument("--ac-log10-diff-threshold", type=float, default=math.log10(16.0))
    p.add_argument("--local-k", type=int, default=10)

    p.add_argument("--resume", type=int, choices=[0, 1], default=1)
    p.add_argument("--force", action="store_true")
    p.add_argument("--save-last-checkpoint", action="store_true")
    return p


class EncodedSmilesDataset:
    def __init__(self, encoded: Mapping[str, Sequence[Sequence[int]]], labels: np.ndarray):
        self.input_ids = list(encoded["input_ids"])
        self.attention_mask = list(encoded["attention_mask"])
        self.labels = np.asarray(labels, dtype=np.float32)
        if not (len(self.input_ids) == len(self.attention_mask) == len(self.labels)):
            raise ValueError("Encoded inputs and labels have inconsistent lengths")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "label": float(self.labels[index]),
        }


class SmilesBatchCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        import torch

        features = [
            {"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]}
            for item in batch
        ]
        padded = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        padded["labels"] = torch.tensor([item["label"] for item in batch], dtype=torch.float32)
        return padded


def canonicalize_smiles_list(smiles: Sequence[str], non_isomeric: bool) -> Tuple[List[str], pd.DataFrame]:
    from rdkit import Chem

    converted: List[str] = []
    rows: List[Dict[str, Any]] = []
    failures: List[Tuple[int, str]] = []
    for i, raw in enumerate(smiles):
        text = str(raw).strip()
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            failures.append((i, text))
            converted.append("")
            rows.append({"source_row": i, "smiles": text, "molformer_smiles": None, "rdkit_valid": 0})
            continue
        canonical_isomeric = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        model_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=not non_isomeric)
        converted.append(model_smiles)
        rows.append(
            {
                "source_row": i,
                "smiles": text,
                "canonical_isomeric_smiles": canonical_isomeric,
                "molformer_smiles": model_smiles,
                "rdkit_valid": 1,
                "stereo_removed": int(non_isomeric and canonical_isomeric != model_smiles),
            }
        )
    if failures:
        preview = failures[:10]
        raise ValueError(f"Invalid SMILES found ({len(failures)} rows); first examples: {preview}")
    return converted, pd.DataFrame(rows)


def prepare_split(
    df: pd.DataFrame,
    split: str,
    task: str,
    tokenizer,
    smiles_col: str,
    target_col: str,
    max_length: int,
    canonicalize_non_isomeric: bool,
) -> Tuple[EncodedSmilesDataset, pd.DataFrame, np.ndarray, List[str]]:
    if smiles_col not in df.columns:
        raise ValueError(f"SMILES column {smiles_col!r} missing from {split}: {list(df.columns)}")
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col!r} missing from {split}: {list(df.columns)}")
    if df[smiles_col].isna().any():
        raise ValueError(f"{split} contains missing SMILES in column {smiles_col}")

    if task == "classification":
        labels = coerce_binary(df[target_col])
    else:
        labels = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(labels).all():
        bad = np.where(~np.isfinite(labels))[0][:10].tolist()
        raise ValueError(f"{split} contains missing/non-numeric targets in {target_col}; first rows={bad}")

    model_smiles, audit = canonicalize_smiles_list(
        df[smiles_col].astype(str).tolist(), non_isomeric=canonicalize_non_isomeric
    )
    encoded = tokenizer(
        model_smiles,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_attention_mask=True,
    )
    lengths = np.asarray([len(ids) for ids in encoded["input_ids"]], dtype=int)
    audit.insert(0, "split", split)
    audit["token_length"] = lengths
    audit["max_length_allowed"] = int(max_length)
    audit["over_max_length"] = (lengths > max_length).astype(int)
    audit["target"] = labels
    too_long = np.where(lengths > max_length)[0]
    if len(too_long):
        examples = [
            {"row": int(i), "token_length": int(lengths[i]), "smiles": model_smiles[int(i)]}
            for i in too_long[:10]
        ]
        raise ValueError(
            f"{split} has {len(too_long)} molecules longer than max_length={max_length}. "
            f"SMILES are not silently truncated. First examples: {examples}"
        )
    dataset = EncodedSmilesDataset(encoded, labels)
    return dataset, audit, labels.astype(float), model_smiles


def build_loaders(
    train_dataset,
    val_dataset,
    test_dataset,
    tokenizer,
    batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    eval_num_workers: int,
    seed: int,
):
    import torch
    from torch.utils.data import DataLoader

    collator = SmilesBatchCollator(tokenizer)
    generator = torch.Generator()
    generator.manual_seed(seed)
    common = dict(collate_fn=collator, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        generator=generator,
        persistent_workers=(num_workers > 0),
        **common,
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=eval_num_workers,
        persistent_workers=(eval_num_workers > 0),
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=eval_num_workers,
        persistent_workers=(eval_num_workers > 0),
        **common,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=eval_num_workers,
        persistent_workers=(eval_num_workers > 0),
        **common,
    )
    return train_loader, train_eval_loader, val_loader, test_loader


def load_hf_backbone_and_tokenizer(args: argparse.Namespace):
    import torch
    from transformers import AutoModel, AutoTokenizer

    common: Dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": bool(args.local_files_only),
    }
    if args.model_revision:
        common["revision"] = args.model_revision
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, **common)
    model_kwargs = dict(common)
    model_kwargs["deterministic_eval"] = True
    model_kwargs["torch_dtype"] = torch.float32
    try:
        backbone = AutoModel.from_pretrained(args.model_id, **model_kwargs)
    except TypeError as exc:
        # Transformers v5 renamed torch_dtype to dtype. Retain compatibility without
        # changing the benchmark numeric precision.
        if "torch_dtype" not in str(exc):
            raise
        model_kwargs.pop("torch_dtype", None)
        model_kwargs["dtype"] = torch.float32
        backbone = AutoModel.from_pretrained(args.model_id, **model_kwargs)
    if hasattr(backbone, "config"):
        backbone.config.deterministic_eval = True
    return tokenizer, backbone


def make_model(backbone, dropout: float):
    import torch
    from torch import nn

    class ResidualPredictionHead(nn.Module):
        def __init__(self, hidden_size: int, drop: float):
            super().__init__()
            self.dense = nn.Linear(hidden_size, hidden_size)
            self.dense2 = nn.Linear(hidden_size, hidden_size)
            self.dropout = nn.Dropout(drop)
            self.activation = nn.GELU()
            self.out_proj = nn.Linear(hidden_size, 1)
            for module in (self.dense, self.dense2, self.out_proj):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        def forward(self, pooled):
            hidden = self.activation(self.dropout(self.dense(pooled))) + pooled
            residual = hidden
            hidden = self.activation(self.dropout(self.dense2(hidden))) + residual
            return self.out_proj(hidden).squeeze(-1)

    class MolFormerPredictor(nn.Module):
        def __init__(self, encoder, drop: float):
            super().__init__()
            self.encoder = encoder
            hidden_size = int(getattr(encoder.config, "hidden_size", 768))
            self.head = ResidualPredictionHead(hidden_size, drop)

        def forward(self, input_ids, attention_mask):
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            pooled = getattr(outputs, "pooler_output", None)
            if pooled is None:
                hidden = outputs.last_hidden_state
                mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            return self.head(pooled)

    return MolFormerPredictor(backbone, dropout)


def compute_loss(logits, labels, task: str, loss_fn):
    if task == "classification":
        return loss_fn(logits, labels.float())
    return loss_fn(logits, labels.float())


def train_epoch(model, loader, optimizer, task: str, loss_fn, device, grad_clip: float) -> float:
    import torch

    model.train()
    total_loss = 0.0
    total_n = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = compute_loss(logits, labels, task, loss_fn)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        n = int(labels.numel())
        total_loss += float(loss.detach().cpu()) * n
        total_n += n
    return total_loss / total_n if total_n else float("nan")


def predict_loader(model, loader, task: str, loss_fn, device) -> Tuple[np.ndarray, np.ndarray, float]:
    import torch

    model.eval()
    truths: List[np.ndarray] = []
    predictions: List[np.ndarray] = []
    total_loss = 0.0
    total_n = 0
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = compute_loss(logits, labels, task, loss_fn)
            pred = torch.sigmoid(logits) if task == "classification" else logits
            n = int(labels.numel())
            total_loss += float(loss.detach().cpu()) * n
            total_n += n
            truths.append(labels.detach().cpu().numpy())
            predictions.append(pred.detach().cpu().numpy())
    return (
        np.concatenate(truths).reshape(-1),
        np.concatenate(predictions).reshape(-1),
        total_loss / total_n if total_n else float("nan"),
    )


def selection_value(
    task: str,
    metrics: Mapping[str, Any],
    selection_metric: str,
    val_loss: float,
) -> Tuple[float, float, str]:
    key = selection_metric.replace("val_", "")
    raw = metrics.get(key, float("nan"))
    try:
        raw_float = float(raw)
    except Exception:
        raw_float = float("nan")
    maximize = task == "classification" or key in {"spearman", "pearson", "r2"}
    if math.isfinite(raw_float):
        return (raw_float if maximize else -raw_float), raw_float, key
    if task == "classification":
        for fallback in ("roc_auc", "pr_auc", "f1"):
            value = metrics.get(fallback, float("nan"))
            if value is not None and math.isfinite(float(value)):
                return float(value), float(value), f"fallback_{fallback}"
    return -float(val_loss), float(val_loss), "fallback_val_loss"


def history_row(
    task: str,
    epoch: int,
    train_optimization_loss: float,
    train_eval_loss: float,
    val_loss: float,
    learning_rate: float,
    train_metrics: Mapping[str, Any],
    val_metrics: Mapping[str, Any],
    best_epoch: int,
    best_metric: float,
    note: str,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "iteration": epoch,
        "epoch": epoch,
        "train_loss": train_eval_loss,
        "train_optimization_loss": train_optimization_loss,
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


def save_checkpoint_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    import torch

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, device):
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def build_prediction_dataframe(
    source_df: pd.DataFrame,
    model_smiles: Sequence[str],
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
        raise ValueError(
            f"Prediction length mismatch for {split}: source={len(out)}, true={len(y_true)}, pred={len(y_pred)}"
        )
    out["smiles"] = out[smiles_col].astype(str)
    out["molformer_smiles"] = list(model_smiles)
    identity = {
        "split": split,
        "dataset_name": run["dataset_name"],
        "genus": run["genus"],
        "task": task,
        "split_type": run["split_type"],
        "seed": run["seed"],
        "model": MODEL_NAME,
        "model_family": MODEL_FAMILY,
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


def run_one(task: str, args: argparse.Namespace, run: Mapping[str, Any]) -> Path:
    import torch
    from torch import nn

    started = time.time()
    started_at = now_iso()
    set_global_seed(int(run["seed"]))

    out_root = args.out_root.expanduser().resolve()
    run_dir = (
        out_root
        / MODEL_NAME
        / str(run["dataset_name"])
        / task
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
        print(f"MoLFormer AntibioticBench {task} run index: {run['run_index']}", flush=True)
        print(json.dumps(dict(run), ensure_ascii=False, indent=2), flush=True)
        print("Data root configured", flush=True)
        print("Output directory configured", flush=True)
        print(f"Model: {public_model_id(args.model_id)} revision={args.model_revision!r}", flush=True)
        print(f"Python: {sys.version.split()[0]}", flush=True)
        print(f"Torch: {torch.__version__}; CUDA available={torch.cuda.is_available()}", flush=True)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; benchmark training is configured for GPU execution")
        device = torch.device("cuda:0")
        print(f"GPU: {torch.cuda.get_device_name(device)}", flush=True)

        input_paths = validate_input_run(args.data_root.expanduser().resolve(), run)
        frames = {name: pd.read_csv(path) for name, path in input_paths.items()}
        smiles_col = args.smiles_col or find_column(frames["train"], SMILES_CANDIDATES, required=True)
        target_col = args.target_col or find_column(
            frames["train"],
            CLASS_TARGET_CANDIDATES if task == "classification" else REG_TARGET_CANDIDATES,
            required=True,
        )
        for split_name, df in frames.items():
            if smiles_col not in df.columns or target_col not in df.columns:
                raise ValueError(
                    f"{input_paths[split_name]} must contain {smiles_col!r} and {target_col!r}; columns={list(df.columns)}"
                )
        print(f"Input sizes: { {k: len(v) for k, v in frames.items()} }", flush=True)
        print(f"SMILES column={smiles_col!r}; target column={target_col!r}", flush=True)

        tokenizer, backbone = load_hf_backbone_and_tokenizer(args)
        configured_max = int(getattr(backbone.config, "max_position_embeddings", args.max_length))
        if args.max_length > configured_max:
            raise ValueError(
                f"Requested max_length={args.max_length} exceeds checkpoint max_position_embeddings={configured_max}"
            )
        print(
            f"Backbone config: layers={getattr(backbone.config, 'num_hidden_layers', 'NA')}, "
            f"hidden={getattr(backbone.config, 'hidden_size', 'NA')}, "
            f"heads={getattr(backbone.config, 'num_attention_heads', 'NA')}, "
            f"max_length={configured_max}, deterministic_eval={getattr(backbone.config, 'deterministic_eval', None)}",
            flush=True,
        )

        prepared: Dict[str, Any] = {}
        audits: List[pd.DataFrame] = []
        labels: Dict[str, np.ndarray] = {}
        model_smiles: Dict[str, List[str]] = {}
        for split_name in ("train", "validation", "test"):
            ds, audit, y, m_smiles = prepare_split(
                frames[split_name],
                split_name,
                task,
                tokenizer,
                smiles_col,
                target_col,
                args.max_length,
                bool(args.canonicalize_non_isomeric),
            )
            prepared[split_name] = ds
            audits.append(audit)
            labels[split_name] = y
            model_smiles[split_name] = m_smiles
        audit_df = pd.concat(audits, ignore_index=True)
        collision_counts = audit_df.groupby("molformer_smiles", dropna=False)["smiles"].transform("nunique")
        audit_df["canonical_collision_distinct_input_count"] = collision_counts
        audit_df["canonical_collision"] = (collision_counts > 1).astype(int)
        audit_df.to_csv(run_dir / "molformer_input_audit.csv", index=False)
        print(
            f"Token length min/median/max={int(audit_df.token_length.min())}/"
            f"{float(audit_df.token_length.median()):.1f}/{int(audit_df.token_length.max())}; "
            f"stereo_removed={int(audit_df.get('stereo_removed', pd.Series(dtype=int)).fillna(0).sum())}; "
            f"canonical_collision_rows={int(audit_df.canonical_collision.sum())}",
            flush=True,
        )

        train_loader, train_eval_loader, val_loader, test_loader = build_loaders(
            prepared["train"],
            prepared["validation"],
            prepared["test"],
            tokenizer,
            args.batch_size,
            args.eval_batch_size,
            args.num_workers,
            args.eval_num_workers,
            int(run["seed"]),
        )
        model = make_model(backbone, args.head_dropout).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: total={total_params:,}; trainable={trainable_params:,}", flush=True)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )
        loss_fn = nn.BCEWithLogitsLoss(reduction="mean") if task == "classification" else nn.MSELoss(reduction="mean")
        selection_metric = args.selection_metric

        source_hash = sha256_files(list(input_paths.values()), extra=f"{smiles_col}|{target_col}|{task}")
        feature_summary = {
            "model": MODEL_NAME,
            "feature_type": "canonical_non_isomeric_SMILES_tokens" if args.canonicalize_non_isomeric else "input_SMILES_tokens",
            "input_representation": "SMILES sequence",
            "model_smiles_column": "molformer_smiles",
            "source_smiles_column": smiles_col,
            "tokenizer": public_model_id(args.model_id),
            "model_revision": args.model_revision,
            "max_length": args.max_length,
            "padding": "dynamic_per_batch",
            "silent_truncation": False,
            "canonicalize_with_rdkit": bool(args.canonicalize_non_isomeric),
            "remove_isomeric_information": bool(args.canonicalize_non_isomeric),
            "fingerprints_or_descriptors_used_by_model": False,
            "three_dimensional_coordinates_used": False,
            "diagnostic_fingerprint": f"Morgan radius={args.fp_radius}, nBits={args.fp_bits}; diagnostics only",
            "token_length_min": int(audit_df.token_length.min()),
            "token_length_median": float(audit_df.token_length.median()),
            "token_length_max": int(audit_df.token_length.max()),
        }
        model_params = {
            "architecture": "MoLFormer-XL-both-10pct encoder + masked-mean pooler + 2-layer residual prediction head",
            "model_id": public_model_id(args.model_id),
            "model_revision": args.model_revision,
            "full_finetuning": True,
            "deterministic_eval": True,
            "head_hidden_dimensions": [int(backbone.config.hidden_size), int(backbone.config.hidden_size)],
            "head_activation": "GELU",
            "head_dropout": args.head_dropout,
            "head_skip_connections": True,
            "output_dimension": 1,
            "optimizer": "AdamW",
            "learning_rate": args.lr,
            "lr_scheduler": "constant",
            "warmup_steps": 0,
            "weight_decay": args.weight_decay,
            "adam_betas": [args.adam_beta1, args.adam_beta2],
            "adam_eps": args.adam_eps,
            "epochs": args.epochs,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "gradient_clip_norm": args.grad_clip,
            "precision": "FP32",
            "classification_loss": "BCEWithLogitsLoss" if task == "classification" else None,
            "regression_loss": "MSELoss" if task == "regression" else None,
            "target_standardization": False,
            "selection_metric": selection_metric,
            "classification_threshold": args.threshold if task == "classification" else None,
            "use_pos_weight": False,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "backbone_config": backbone.config.to_dict(),
        }
        write_json(run_dir / "feature_summary.json", feature_summary)
        write_json(run_dir / "model_params.json", model_params)

        best_path = run_dir / "best_model.pth"
        last_path = run_dir / "last_model.pth"
        history_rows: List[Dict[str, Any]] = []
        best_score = -float("inf")
        best_epoch = 0
        best_raw_metric = float("nan")
        best_source = ""
        no_improvement = 0

        print("-" * 100, flush=True)
        print(
            f"Training: epochs={args.epochs}, patience={args.patience}, batch={args.batch_size}, "
            f"eval_batch={args.eval_batch_size}, lr={args.lr:g} constant, selection={selection_metric}",
            flush=True,
        )
        print("Test loader is constructed but will not be evaluated until the best validation checkpoint is reloaded.", flush=True)

        for epoch in range(1, args.epochs + 1):
            epoch_started = time.time()
            optimization_loss = train_epoch(model, train_loader, optimizer, task, loss_fn, device, args.grad_clip)
            train_true, train_pred, train_eval_loss = predict_loader(model, train_eval_loader, task, loss_fn, device)
            val_true, val_pred, val_loss = predict_loader(model, val_loader, task, loss_fn, device)
            if task == "classification":
                train_metrics = classification_metrics(train_true, train_pred, args.threshold)
                val_metrics = classification_metrics(val_true, val_pred, args.threshold)
            else:
                threshold_for_screening = float(run["threshold_for_boundary"])
                train_metrics = regression_metrics(train_true, train_pred, threshold_for_screening)
                val_metrics = regression_metrics(val_true, val_pred, threshold_for_screening)
            score, raw_metric, source = selection_value(task, val_metrics, selection_metric, val_loss)
            improved = score > best_score + args.min_delta
            if improved:
                best_score = score
                best_raw_metric = raw_metric
                best_epoch = epoch
                best_source = source
                no_improvement = 0
                save_checkpoint_atomic(
                    best_path,
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "best_score": best_score,
                        "best_val_metric": best_raw_metric,
                        "selection_source": best_source,
                        "task": task,
                        "run": dict(run),
                        "model_id": public_model_id(args.model_id),
                        "model_revision": args.model_revision,
                        "model_params": model_params,
                    },
                )
                note = "best_checkpoint"
            else:
                no_improvement += 1
                note = f"no_improvement_{no_improvement}_of_{args.patience}"

            row = history_row(
                task,
                epoch,
                optimization_loss,
                train_eval_loss,
                val_loss,
                float(optimizer.param_groups[0]["lr"]),
                train_metrics,
                val_metrics,
                best_epoch,
                best_raw_metric,
                note,
            )
            row["epoch_seconds"] = time.time() - epoch_started
            history_rows.append(row)
            write_train_log_files(run_dir, pd.DataFrame(history_rows))

            if task == "classification":
                print(
                    f"epoch={epoch:03d} opt_loss={optimization_loss:.6f} train_loss={train_eval_loss:.6f} "
                    f"val_loss={val_loss:.6f} train_pr_auc={train_metrics.get('pr_auc', np.nan):.6f} "
                    f"val_pr_auc={val_metrics.get('pr_auc', np.nan):.6f} "
                    f"train_roc_auc={train_metrics.get('roc_auc', np.nan):.6f} "
                    f"val_roc_auc={val_metrics.get('roc_auc', np.nan):.6f} "
                    f"train_f1={train_metrics.get('f1', np.nan):.6f} val_f1={val_metrics.get('f1', np.nan):.6f} "
                    f"lr={optimizer.param_groups[0]['lr']:.3e} best_epoch={best_epoch} "
                    f"patience={no_improvement}/{args.patience} note={note}",
                    flush=True,
                )
            else:
                print(
                    f"epoch={epoch:03d} opt_loss={optimization_loss:.6f} train_loss={train_eval_loss:.6f} "
                    f"val_loss={val_loss:.6f} train_rmse={train_metrics.get('rmse', np.nan):.6f} "
                    f"val_rmse={val_metrics.get('rmse', np.nan):.6f} "
                    f"train_mae={train_metrics.get('mae', np.nan):.6f} val_mae={val_metrics.get('mae', np.nan):.6f} "
                    f"train_spearman={train_metrics.get('spearman', np.nan):.6f} "
                    f"val_spearman={val_metrics.get('spearman', np.nan):.6f} "
                    f"lr={optimizer.param_groups[0]['lr']:.3e} best_epoch={best_epoch} "
                    f"patience={no_improvement}/{args.patience} note={note}",
                    flush=True,
                )
            if args.save_last_checkpoint:
                save_checkpoint_atomic(
                    last_path,
                    {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()},
                )
            if no_improvement >= args.patience:
                print(
                    f"Early stopping after epoch {epoch}; best epoch={best_epoch}, "
                    f"best {selection_metric}={best_raw_metric}",
                    flush=True,
                )
                break

        if not best_path.is_file():
            raise RuntimeError("No best checkpoint was written")
        checkpoint = load_checkpoint(best_path, device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        print(
            f"Reloaded best checkpoint: epoch={checkpoint['epoch']}, "
            f"best_val_metric={checkpoint.get('best_val_metric')}, source={checkpoint.get('selection_source')}",
            flush=True,
        )

        # Final predictions: test is touched for the first time here.
        train_true, train_pred, best_train_loss = predict_loader(model, train_eval_loader, task, loss_fn, device)
        val_true, val_pred, best_val_loss = predict_loader(model, val_loader, task, loss_fn, device)
        test_true, test_pred, test_loss = predict_loader(model, test_loader, task, loss_fn, device)
        val_best_threshold = best_f1_threshold(val_true, val_pred) if task == "classification" else np.nan

        pred_frames: Dict[str, pd.DataFrame] = {}
        for split_name, true_values, pred_values in (
            ("train", train_true, train_pred),
            ("validation", val_true, val_pred),
            ("test", test_true, test_pred),
        ):
            pred_frames[split_name] = build_prediction_dataframe(
                frames[split_name],
                model_smiles[split_name],
                split_name,
                task,
                true_values,
                pred_values,
                run,
                smiles_col,
                args.threshold,
                val_best_threshold,
            )

        test_with_diag, diag_meta = add_test_nn_and_activity_cliff(
            task=task,
            train_pred=pred_frames["train"],
            test_pred=pred_frames["test"],
            fp_radius=args.fp_radius,
            fp_bits=args.fp_bits,
            ac_similarity_threshold=args.ac_sim_threshold,
            ac_log10_diff_threshold=args.ac_log10_diff_threshold,
            local_k=args.local_k,
        )
        pred_frames["test"] = reorder_prediction_columns(test_with_diag, task)
        pred_frames["train"] = reorder_prediction_columns(pred_frames["train"], task)
        pred_frames["validation"] = reorder_prediction_columns(pred_frames["validation"], task)
        for split_name, frame in pred_frames.items():
            frame.to_csv(run_dir / f"predictions_{split_name}.csv", index=False)

        if task == "classification":
            split_metrics = {
                "train": classification_metrics(train_true, train_pred, args.threshold),
                "validation": classification_metrics(val_true, val_pred, args.threshold),
                "test": classification_metrics(test_true, test_pred, args.threshold),
            }
        else:
            threshold_for_screening = float(run["threshold_for_boundary"])
            split_metrics = {
                "train": regression_metrics(train_true, train_pred, threshold_for_screening),
                "validation": regression_metrics(val_true, val_pred, threshold_for_screening),
                "test": regression_metrics(test_true, test_pred, threshold_for_screening),
            }
        save_split_metrics(run_dir, split_metrics)

        identity = {
            "model": MODEL_NAME,
            "model_family": MODEL_FAMILY,
            "task": task,
            "dataset_name": run["dataset_name"],
            "genus": run["genus"],
            "split_type": run["split_type"],
            "seed": run["seed"],
        }
        diagnostics_wide, diagnostics_long = build_test_diagnostics(
            task,
            pred_frames["test"],
            identity,
            float(run["threshold_for_boundary"]),
        )
        diagnostics_wide.to_csv(run_dir / "diagnostics_metrics.csv", index=False)
        diagnostics_long.to_csv(run_dir / "diagnostic_subgroup_metrics.csv", index=False)

        history_df = pd.DataFrame(history_rows)
        best_hist = history_df.loc[history_df["epoch"] == int(checkpoint["epoch"])].iloc[0].to_dict()
        metrics_full: Dict[str, Any] = dict(identity)
        metrics_full.update(
            {
                "feature_type": feature_summary["feature_type"],
                "target_scale": "binary_active" if task == "classification" else "raw_log10_mic",
                "threshold_from_dataset_name": float(run["threshold_for_boundary"]),
                "threshold_for_boundary": float(run["threshold_for_boundary"]),
                "threshold_mode": "fixed_0p5" if task == "classification" else "not_applicable",
                "threshold_used": args.threshold if task == "classification" else np.nan,
                "val_best_f1_threshold": val_best_threshold,
                "train_seconds": time.time() - started,
                "best_epoch": int(checkpoint["epoch"]),
                "stop_epoch": int(history_df["epoch"].max()),
                "epochs_trained": int(len(history_df)),
                "best_val_metric": checkpoint.get("best_val_metric"),
                "select_metric": selection_metric,
                "selection_source": checkpoint.get("selection_source"),
                "train_loss": best_train_loss,
                "val_loss": best_val_loss,
                "test_loss": test_loss,
                "learning_rate_at_best_epoch": best_hist.get("learning_rate"),
                "model_id": public_model_id(args.model_id),
                "model_revision": args.model_revision,
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "source_hash": source_hash,
            }
        )
        for split_name, metrics in split_metrics.items():
            prefix = "val" if split_name == "validation" else split_name
            metrics_full.update(prefix_metrics(metrics, prefix))
        if task == "regression":
            for prefix, values in (("train", train_true), ("val", val_true), ("test", test_true)):
                arr = np.asarray(values, dtype=float)
                arr = arr[np.isfinite(arr)]
                metrics_full[f"{prefix}_log10mic_mean"] = float(np.mean(arr)) if len(arr) else np.nan
                metrics_full[f"{prefix}_log10mic_std"] = float(np.std(arr, ddof=0)) if len(arr) else np.nan
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
            "molformer_source_repo": "<MOLFORMER_ROOT>" if args.molformer_repo else None,
            "model_id": public_model_id(args.model_id),
            "model_revision": args.model_revision,
            "dataset_config": args.dataset_config.name,
            "smiles_column": smiles_col,
            "target_column": target_col,
            "source_hash": source_hash,
            "split_role_policy": {
                "train": "fitting plus predictions and overall metrics only",
                "validation": "checkpoint selection plus predictions and overall metrics only",
                "test": "single final evaluation after loading best checkpoint; overall metrics plus NN and activity-cliff diagnostics",
            },
            "diagnostic_definition": diag_meta,
            "software_versions": package_versions(),
            "environment": {
                k: os.environ.get(k)
                for k in (
                    "CUDA_VISIBLE_DEVICES",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                )
            },
            "slurm": {k: os.environ.get(k) for k in ("SLURM_CPUS_PER_TASK",)},
            "reproducibility_note": (
                "Seeds and deterministic evaluation are fixed. Some CUDA operations can retain hardware/library-level nondeterminism."
            ),
        }
        write_json(run_dir / "run_info.json", run_info)
        shutil.copy2(Path(__file__).resolve(), run_dir / "molformer_train_common.used.py")
        entry_name = (
            "train_molformer_classification_antibioticbench.py"
            if task == "classification"
            else "train_molformer_regression_antibioticbench.py"
        )
        entry_path = Path(__file__).resolve().parent / entry_name
        if entry_path.is_file():
            shutil.copy2(entry_path, run_dir / f"{entry_name}.used.py")
        shutil.copy2(Path(__file__).resolve().parent / "molformer_benchmark_utils.py", run_dir / "molformer_benchmark_utils.used.py")
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


def main(task: str) -> None:
    args = build_parser(task).parse_args()
    args.out_root = args.out_root.expanduser().resolve()
    args.dataset_config = args.dataset_config.expanduser().resolve()
    config_df = load_dataset_config(args.dataset_config)
    all_runs = enumerate_runs(config_df, task)
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
            except Exception as exc:
                failures.append({**run, "error": sanitize_error_trace(str(exc), args)})
        if failures:
            print(pd.DataFrame(failures).to_string(index=False), file=sys.stderr)
            raise SystemExit(f"Input validation failed for {len(failures)}/{len(all_runs)} runs")
        print(f"Validated all {len(all_runs)} {task} runs")
        return
    selected = select_runs(args, all_runs)
    print(f"Selected {len(selected)} run(s) out of {len(all_runs)} total for task={task}", flush=True)
    for run in selected:
        run_one(task, args, run)
