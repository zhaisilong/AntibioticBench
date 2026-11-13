#!/usr/bin/env python3
"""Common utilities for AntibioticBench TabPFN classification/regression runs.

Design choices
--------------
* Standard TabPFN-3 in-context inference only; NO downstream gradient fine-tuning.
* Official TabPFN inference defaults are fixed in the task scripts.
* Model input representation is Morgan/ECFP4 (radius=2, 2048 bits) + full RDKit 2D descriptors.
* Train and validation receive only overall prediction/metrics.
* Test receives overall metrics + train-referenced NN, MIC-boundary, activity-cliff,
  MIC-range diagnostics; regression additionally receives local roughness.
* Activity cliff definition used here: Tanimoto >= 0.90 and |Δlog10(MIC)| >= log10(16).

The code is intentionally explicit and verbose to support benchmark reproducibility.
"""
from __future__ import annotations

import json
import logging
import math
import os
import platform
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

try:
    from rdkit import Chem, DataStructs, rdBase
    from rdkit.Chem import Descriptors, rdFingerprintGenerator
except Exception as exc:  # pragma: no cover - clear import error on server
    raise RuntimeError(
        "RDKit is required. Install it in the tabpfn environment, e.g. `pip install rdkit`."
    ) from exc


MODEL_NAME = "TabPFN"
MODEL_FAMILY = "tabular_foundation_model_in_context"
FEATURE_TYPE = "Morgan_ECFP4_radius2_nBits2048_plus_RDKit2D"
THRESHOLD_USED = 0.5
DEFAULT_AC_SIM_THRESHOLD = 0.90
DEFAULT_AC_FOLD_CHANGE = 16.0
DEFAULT_NN_RADIUS = 2
DEFAULT_NN_NBITS = 2048
DEFAULT_NN_K = 10
DILUTION_1_LOG10 = math.log10(2.0)
DILUTION_2_LOG10 = math.log10(4.0)

RDKIT2D_DESCRIPTOR_NAMES = tuple(name for name, _ in Descriptors._descList)
RDKIT2D_DESCRIPTOR_FUNCS = tuple(func for _, func in Descriptors._descList)
FEATURE_CACHE_FORMAT_VERSION = 1



@dataclass(frozen=True)
class RunSpec:
    dataset_name: str
    genus: str
    threshold: float | None
    task: str
    split_type: str
    seed: int
    input_dir: Path

    @property
    def id(self) -> str:
        return f"{self.dataset_name}|{self.task}|{self.split_type}|seed_{self.seed}"


class FingerprintCache:
    """In-process Morgan fingerprint cache; can be seeded from disk-cached arrays."""

    def __init__(self, radius: int = DEFAULT_NN_RADIUS, n_bits: int = DEFAULT_NN_NBITS):
        self.radius = radius
        self.n_bits = n_bits
        self.generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits
        )
        self._bitvect: dict[str, Any] = {}
        self._array: dict[str, np.ndarray] = {}

    def seed_array(self, smiles: str, arr: np.ndarray) -> None:
        self._array[str(smiles)] = np.asarray(arr, dtype=np.uint8)

    def get(self, smiles: str) -> tuple[Any, np.ndarray]:
        smiles = str(smiles)
        if smiles in self._bitvect:
            return self._bitvect[smiles], self._array[smiles]
        if smiles in self._array:
            arr = self._array[smiles]
            fp = DataStructs.ExplicitBitVect(self.n_bits)
            fp.SetBitsFromList([int(x) for x in np.flatnonzero(arr)])
            self._bitvect[smiles] = fp
            return fp, arr
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles!r}")
        fp = self.generator.GetFingerprint(mol)
        arr = np.zeros((self.n_bits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        self._bitvect[smiles] = fp
        self._array[smiles] = arr
        return fp, arr

    def transform(self, smiles: Sequence[str]) -> tuple[np.ndarray, list[Any]]:
        arrays: list[np.ndarray] = []
        bitvects: list[Any] = []
        bad: list[tuple[int, str, str]] = []
        for i, smi in enumerate(smiles):
            try:
                fp, arr = self.get(str(smi))
                arrays.append(arr)
                bitvects.append(fp)
            except Exception as exc:
                bad.append((i, str(smi), str(exc)))
        if bad:
            preview = "; ".join(f"row={i} smiles={s!r} err={e}" for i, s, e in bad[:10])
            raise ValueError(f"Found {len(bad)} invalid SMILES. First errors: {preview}")
        X = np.asarray(arrays, dtype=np.float32)
        return X, bitvects


class DescriptorCache:
    """In-process RDKit 2D descriptor cache.

    Full ``Descriptors._descList`` is used. Descriptor failures/non-finite values
    are represented as NaN and left for TabPFN native preprocessing.
    """

    def __init__(self):
        self.names = RDKIT2D_DESCRIPTOR_NAMES
        self.funcs = RDKIT2D_DESCRIPTOR_FUNCS
        self._array: dict[str, np.ndarray] = {}

    def get(self, smiles: str) -> np.ndarray:
        smiles = str(smiles)
        if smiles in self._array:
            return self._array[smiles]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles!r}")
        values = np.empty(len(self.funcs), dtype=np.float32)
        for j, func in enumerate(self.funcs):
            try:
                v = func(mol)
                v = float(v) if v is not None else float("nan")
                values[j] = v if np.isfinite(v) else np.nan
            except Exception:
                values[j] = np.nan
        self._array[smiles] = values
        return values

    def transform(self, smiles: Sequence[str]) -> np.ndarray:
        return np.asarray([self.get(str(s)) for s in smiles], dtype=np.float32)


def descriptor_missing_summary(desc: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(desc, dtype=np.float32)
    if arr.ndim != 2:
        return {
            "n_values": int(arr.size),
            "n_missing": int(np.sum(~np.isfinite(arr))),
            "missing_fraction": float(np.mean(~np.isfinite(arr))) if arr.size else 0.0,
            "descriptor_columns_with_any_missing": None,
            "descriptor_columns_all_missing": None,
        }
    missing = ~np.isfinite(arr)
    return {
        "n_values": int(arr.size),
        "n_missing": int(missing.sum()),
        "missing_fraction": float(missing.mean()) if arr.size else 0.0,
        "descriptor_columns_with_any_missing": int(np.any(missing, axis=0).sum()) if arr.shape[1] else 0,
        "descriptor_columns_all_missing": int(np.all(missing, axis=0).sum()) if arr.shape[1] else 0,
    }


def build_tabpfn_molecular_features(
    smiles: Sequence[str],
    fp_cache: FingerprintCache,
    descriptor_cache: DescriptorCache,
) -> tuple[np.ndarray, list[Any], np.ndarray]:
    morgan, bitvects = fp_cache.transform(smiles)
    desc = descriptor_cache.transform(smiles)
    X = np.hstack([morgan.astype(np.float32, copy=False), desc.astype(np.float32, copy=False)])
    return X, bitvects, desc


def _read_smiles_from_csv(path: Path) -> list[str]:
    header = pd.read_csv(path, nrows=0)
    col = infer_smiles_column(header)
    vals = pd.read_csv(path, usecols=[col])[col]
    return vals.loc[vals.notna()].astype(str).tolist()


def _representative_dataset_smiles(dataset_dir: Path, task: str) -> list[str]:
    """Collect the molecule universe from one complete partition for one task.

    Classification and regression intentionally use separate persistent feature
    caches.  Only the requested task is scanned so simultaneously running task
    launchers never contend for, extend, or lock the same cache files.
    """
    if task not in {"classification", "regression"}:
        raise ValueError(f"Unsupported task for feature cache: {task!r}")
    collected: set[str] = set()
    task_dir = dataset_dir / task
    if not task_dir.is_dir():
        return []
    candidates: list[Path] = []
    preferred = task_dir / "random" / "seed_0"
    if all((preferred / f).exists() for f in ("train.csv", "validation.csv", "test.csv")):
        candidates = [preferred / f for f in ("train.csv", "validation.csv", "test.csv")]
    else:
        for train_file in sorted(task_dir.rglob("train.csv")):
            d = train_file.parent
            trio = [d / f for f in ("train.csv", "validation.csv", "test.csv")]
            if all(x.exists() for x in trio):
                candidates = trio
                break
    for path in candidates:
        collected.update(_read_smiles_from_csv(path))
    return sorted(collected)


class DatasetFeatureCacheManager:
    """Persistent per-dataset molecular-feature cache for one benchmark task.

    Disk layout::

        <task_output_root>/feature_cache/<dataset_name>/molecular_features.npz
        <task_output_root>/feature_cache/<dataset_name>/metadata.json
        <task_output_root>/feature_cache/<dataset_name>/.cache.lock

    Classification and regression intentionally use different cache roots and
    different manager instances.  Within one task, all random/scaffold/seeds
    reuse the same label-independent Morgan + RDKit2D molecular features.
    """

    def __init__(
        self,
        cache_root: Path,
        data_root: Path,
        task: str,
        radius: int = DEFAULT_NN_RADIUS,
        n_bits: int = DEFAULT_NN_NBITS,
    ):
        if task not in {"classification", "regression"}:
            raise ValueError(f"Unsupported task for feature cache: {task!r}")
        self.cache_root = Path(cache_root)
        self.data_root = Path(data_root)
        self.task = task
        self.radius = int(radius)
        self.n_bits = int(n_bits)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self._dataset_name: str | None = None
        self._smiles: np.ndarray | None = None
        self._index: dict[str, int] = {}
        self._morgan: np.ndarray | None = None
        self._desc: np.ndarray | None = None
        self._X: np.ndarray | None = None
        self._fp_cache: FingerprintCache | None = None
        self._last_cache_info: dict[str, Any] = {}

    def _paths(self, dataset_name: str) -> tuple[Path, Path, Path]:
        d = self.cache_root / dataset_name
        return d / "molecular_features.npz", d / "metadata.json", d / ".cache.lock"

    def _expected_meta(self, dataset_name: str) -> dict[str, Any]:
        return {
            "cache_format_version": FEATURE_CACHE_FORMAT_VERSION,
            "dataset_name": dataset_name,
            "task": self.task,
            "feature_type": FEATURE_TYPE,
            "morgan_radius": self.radius,
            "morgan_nbits": self.n_bits,
            "rdkit_version": getattr(rdBase, "rdkitVersion", "unknown"),
            "rdkit2d_descriptor_names": list(RDKIT2D_DESCRIPTOR_NAMES),
            "rdkit2d_descriptor_dim": len(RDKIT2D_DESCRIPTOR_NAMES),
        }

    def _metadata_compatible(self, meta: dict[str, Any], dataset_name: str) -> bool:
        expected = self._expected_meta(dataset_name)
        return all(meta.get(k) == v for k, v in expected.items())

    def _load_disk(self, dataset_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        cache_path, meta_path, _ = self._paths(dataset_name)
        if not cache_path.exists() or not meta_path.exists():
            raise FileNotFoundError(cache_path)
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        if not self._metadata_compatible(meta, dataset_name):
            raise ValueError("feature-cache metadata/configuration mismatch")
        with np.load(cache_path, allow_pickle=False) as z:
            smiles = np.asarray(z["smiles"]).astype(str)
            morgan = np.asarray(z["morgan"], dtype=np.uint8)
            desc = np.asarray(z["rdkit2d"], dtype=np.float32)
        if morgan.shape != (len(smiles), self.n_bits):
            raise ValueError(f"invalid cached Morgan shape: {morgan.shape}")
        if desc.shape != (len(smiles), len(RDKIT2D_DESCRIPTOR_NAMES)):
            raise ValueError(f"invalid cached RDKit2D shape: {desc.shape}")
        return smiles, morgan, desc, meta

    def _compute_rows(
        self,
        smiles: Sequence[str],
        logger: logging.Logger | None,
        dataset_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=self.radius, fpSize=self.n_bits)
        n = len(smiles)
        morgan = np.empty((n, self.n_bits), dtype=np.uint8)
        desc = np.empty((n, len(RDKIT2D_DESCRIPTOR_FUNCS)), dtype=np.float32)
        log_every = max(1, math.ceil(n / 20))
        t0 = time.time()
        for i, smi in enumerate(smiles):
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                raise ValueError(f"Invalid SMILES while building cache: {smi!r}")
            fp = fp_gen.GetFingerprint(mol)
            arr = np.zeros((self.n_bits,), dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            morgan[i] = arr
            for j, func in enumerate(RDKIT2D_DESCRIPTOR_FUNCS):
                try:
                    v = func(mol)
                    v = float(v) if v is not None else float("nan")
                    desc[i, j] = v if np.isfinite(v) else np.nan
                except Exception:
                    desc[i, j] = np.nan
            done = i + 1
            if logger is not None and (done == 1 or done == n or done % log_every == 0):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else float("nan")
                eta = (n - done) / rate if rate and np.isfinite(rate) and rate > 0 else float("nan")
                logger.info(
                    "Feature cache build | %s | %d/%d (%.1f%%) | elapsed %.1fs | ETA %.1fs",
                    dataset_name, done, n, 100.0 * done / n, elapsed, eta,
                )
        return morgan, desc

    def _atomic_save(
        self,
        dataset_name: str,
        smiles: Sequence[str],
        morgan: np.ndarray,
        desc: np.ndarray,
        source: str,
    ) -> dict[str, Any]:
        cache_path, meta_path, _ = self._paths(dataset_name)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_cache = cache_path.with_name(cache_path.stem + ".tmp.npz")
        tmp_meta = meta_path.with_name(meta_path.name + ".tmp")
        np.savez(
            tmp_cache,
            smiles=np.asarray(smiles, dtype=str),
            morgan=np.asarray(morgan, dtype=np.uint8),
            rdkit2d=np.asarray(desc, dtype=np.float32),
        )
        meta = {
            **self._expected_meta(dataset_name),
            "n_smiles": int(len(smiles)),
            "source": source,
            "created_at_unix": time.time(),
            "cache_file": cache_path.name,
            "morgan_dtype": "uint8",
            "rdkit2d_dtype": "float32",
            "rdkit2d_missing": descriptor_missing_summary(desc),
        }
        with tmp_meta.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2, default=json_default)
        os.replace(tmp_cache, cache_path)
        os.replace(tmp_meta, meta_path)
        return meta

    def _ensure_disk_cache(
        self,
        dataset_name: str,
        required_smiles: Sequence[str],
        logger: logging.Logger | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], bool]:
        import fcntl

        cache_path, meta_path, lock_path = self._paths(dataset_name)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+") as lock_f:
            if logger is not None:
                logger.info("Feature cache | acquiring lock")
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            built_or_extended = False
            try:
                try:
                    smiles, morgan, desc, meta = self._load_disk(dataset_name)
                    cached_set = set(smiles.tolist())
                    missing = sorted(set(map(str, required_smiles)) - cached_set)
                    if not missing:
                        if logger is not None:
                            logger.info(
                                "Feature cache HIT | %s | molecules=%d | file=%s",
                                dataset_name, len(smiles), cache_path.name,
                            )
                        return smiles, morgan, desc, meta, built_or_extended
                    if logger is not None:
                        logger.info(
                            "Feature cache EXTEND | %s | %d requested SMILES missing; computing only missing rows",
                            dataset_name, len(missing),
                        )
                    new_morgan, new_desc = self._compute_rows(missing, logger, dataset_name)
                    smiles = np.concatenate([smiles, np.asarray(missing, dtype=str)])
                    morgan = np.vstack([morgan, new_morgan])
                    desc = np.vstack([desc, new_desc])
                    meta = self._atomic_save(dataset_name, smiles, morgan, desc, "existing_cache_plus_required_smiles")
                    built_or_extended = True
                    return smiles, morgan, desc, meta, built_or_extended
                except Exception as load_exc:
                    if logger is not None:
                        logger.info(
                            "Feature cache MISS/REBUILD | %s | reason=%s",
                            dataset_name, type(load_exc).__name__,
                        )
                    dataset_dir = self.data_root / dataset_name
                    universe = set(_representative_dataset_smiles(dataset_dir, self.task))
                    universe.update(map(str, required_smiles))
                    all_smiles = sorted(universe)
                    if not all_smiles:
                        raise RuntimeError(f"Could not collect any SMILES for dataset {dataset_name}")
                    if logger is not None:
                        logger.info(
                            "Feature cache BUILD | %s | unique molecules=%d | Morgan=%d + RDKit2D=%d",
                            dataset_name, len(all_smiles), self.n_bits, len(RDKIT2D_DESCRIPTOR_NAMES),
                        )
                    morgan, desc = self._compute_rows(all_smiles, logger, dataset_name)
                    meta = self._atomic_save(
                        dataset_name,
                        all_smiles,
                        morgan,
                        desc,
                        f"representative_full_partition_{self.task}",
                    )
                    built_or_extended = True
                    if logger is not None:
                        logger.info("Feature cache SAVED | %s", cache_path.name)
                    return np.asarray(all_smiles, dtype=str), morgan, desc, meta, built_or_extended
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

    def _activate(
        self,
        dataset_name: str,
        required_smiles: Sequence[str],
        logger: logging.Logger | None,
    ) -> None:
        # Fast path: current dataset already in RAM and contains all requested molecules.
        if self._dataset_name == dataset_name and all(str(s) in self._index for s in required_smiles):
            self._last_cache_info = {
                **self._last_cache_info,
                "cache_hit_memory": True,
                "cache_built_or_extended_this_call": False,
            }
            return

        t0 = time.time()
        smiles, morgan, desc, meta, built = self._ensure_disk_cache(dataset_name, required_smiles, logger)
        self._dataset_name = dataset_name
        self._smiles = smiles
        self._index = {str(s): i for i, s in enumerate(smiles.tolist())}
        self._morgan = morgan
        self._desc = desc
        self._X = np.hstack([
            morgan.astype(np.float32, copy=False),
            desc.astype(np.float32, copy=False),
        ])
        self._fp_cache = FingerprintCache(radius=self.radius, n_bits=self.n_bits)
        for smi, arr in zip(smiles.tolist(), morgan):
            self._fp_cache.seed_array(str(smi), arr)
        cache_path, meta_path, _ = self._paths(dataset_name)
        self._last_cache_info = {
            "cache_root": self.cache_root.name,
            "cache_file": cache_path.name,
            "metadata_file": meta_path.name,
            "dataset_name": dataset_name,
            "task": self.task,
            "n_cached_molecules": int(len(smiles)),
            "cache_built_or_extended_this_call": bool(built),
            "cache_hit_memory": False,
            "load_activate_seconds": float(time.time() - t0),
            "metadata": meta,
        }
        if logger is not None:
            logger.info(
                "Feature cache READY | %s | molecules=%d | X_all=%s | load/activate %.2fs",
                dataset_name, len(smiles), self._X.shape, self._last_cache_info["load_activate_seconds"],
            )

    def get_features(
        self,
        dataset_name: str,
        smiles: Sequence[str],
        logger: logging.Logger | None = None,
        need_bitvectors: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, list[Any] | None, dict[str, Any]]:
        smiles = [str(s) for s in smiles]
        self._activate(dataset_name, smiles, logger)
        assert self._X is not None and self._desc is not None and self._fp_cache is not None
        idx = np.asarray([self._index[s] for s in smiles], dtype=int)
        X = np.asarray(self._X[idx], dtype=np.float32)
        desc = np.asarray(self._desc[idx], dtype=np.float32)
        fps = None
        if need_bitvectors:
            fps = [self._fp_cache.get(s)[0] for s in smiles]
        return X, desc, fps, dict(self._last_cache_info)


# ----------------------------- logging / json -----------------------------


def setup_run_logger(run_dir: Path, name: str) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"{name}:{run_dir}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # Avoid duplicate handlers when re-entering in the same process.
    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(run_dir / "run.log", mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=json_default)


def safe_div(a: float, b: float) -> float:
    if b is None or not np.isfinite(b) or b == 0:
        return float("nan")
    return float(a / b)


def finite_or_nan(x: Any) -> float:
    try:
        y = float(x)
        return y if np.isfinite(y) else float("nan")
    except Exception:
        return float("nan")


# ----------------------------- run discovery ------------------------------


def parse_dataset_metadata(dataset_name: str) -> tuple[str, float | None]:
    """Parse genus and MIC threshold from names like 1_Staphylococcus_median_16_..."""
    no_index = re.sub(r"^\d+_", "", dataset_name)
    genus = no_index.split("_median_")[0].split("_")[0]
    m = re.search(r"_median_([0-9]+(?:\.[0-9]+)?)_", dataset_name)
    threshold = float(m.group(1)) if m else None
    return genus, threshold


def discover_runs(
    data_root: Path,
    task: str,
    split_types: Sequence[str],
    seeds: Sequence[int],
    strict_seeds: Sequence[int] = (0,),
    expected_datasets: int | None = 12,
    allow_missing: bool = False,
) -> list[RunSpec]:
    def natural_dataset_key(p: Path):
        m = re.match(r"^(\d+)_", p.name)
        return (int(m.group(1)) if m else 10**9, p.name)

    dataset_dirs = sorted(
        [p for p in data_root.iterdir() if p.is_dir() and (p / task).is_dir()],
        key=natural_dataset_key,
    )
    if expected_datasets is not None and len(dataset_dirs) != expected_datasets:
        msg = (
            f"Expected {expected_datasets} datasets containing task={task!r}, "
            f"but found {len(dataset_dirs)} under {data_root}."
        )
        if not allow_missing:
            raise RuntimeError(msg)
        print("WARNING:", msg, file=sys.stderr)

    runs: list[RunSpec] = []
    missing: list[Path] = []
    for ds in dataset_dirs:
        genus, threshold = parse_dataset_metadata(ds.name)
        for split in split_types:
            use_seeds = strict_seeds if split == "strict_perimeter" else seeds
            for seed in use_seeds:
                inp = ds / task / split / f"seed_{seed}"
                expected = [inp / "train.csv", inp / "validation.csv", inp / "test.csv"]
                if all(x.exists() for x in expected):
                    runs.append(
                        RunSpec(
                            dataset_name=ds.name,
                            genus=genus,
                            threshold=threshold,
                            task=task,
                            split_type=split,
                            seed=int(seed),
                            input_dir=inp,
                        )
                    )
                else:
                    missing.extend([x for x in expected if not x.exists()])
    if missing and not allow_missing:
        preview = "\n".join(str(p) for p in missing[:20])
        raise FileNotFoundError(
            f"Missing {len(missing)} expected split files. First missing files:\n{preview}"
        )
    return runs


def select_runs(
    runs: Sequence[RunSpec],
    run_mode: str,
    single_run_index: int | None,
    run_range: str | None,
) -> list[tuple[int, RunSpec]]:
    indexed = list(enumerate(runs))
    if single_run_index is not None:
        if not 0 <= single_run_index < len(indexed):
            raise IndexError(f"single_run_index={single_run_index} outside 0..{len(indexed)-1}")
        return [indexed[single_run_index]]
    if run_range:
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", run_range)
        if not m:
            raise ValueError("--run-range must look like 0-191")
        lo, hi = map(int, m.groups())
        if lo < 0 or hi < lo or hi >= len(indexed):
            raise IndexError(f"Invalid run range {run_range}; available 0..{len(indexed)-1}")
        return indexed[lo : hi + 1]
    if run_mode == "test":
        return indexed[:1]
    return indexed


def output_run_dir(out_root: Path, spec: RunSpec) -> Path:
    return (
        out_root
        / MODEL_NAME
        / spec.dataset_name
        / spec.task
        / spec.split_type
        / f"seed_{spec.seed}"
    )


# ------------------------------- columns ----------------------------------


def find_column(df: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> str | None:
    exact = {c: c for c in df.columns}
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c in exact:
            return exact[c]
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise KeyError(f"Could not find any of columns {list(candidates)}. Available: {list(df.columns)}")
    return None


def infer_smiles_column(df: pd.DataFrame) -> str:
    return str(
        find_column(
            df,
            ["smiles", "smiles_std", "SMILES", "SMILES_std", "canonical_smiles", "canonical_smiles_std"],
            required=True,
        )
    )


def infer_classification_target(df: pd.DataFrame) -> str:
    col = find_column(df, ["active", "label", "target", "y", "class"], required=True)
    return str(col)


def infer_regression_target(df: pd.DataFrame) -> str:
    col = find_column(
        df,
        ["log10_mic", "log10MIC", "log10_MIC", "log10mic", "target", "y"],
        required=True,
    )
    return str(col)


def infer_mic_numeric_column(df: pd.DataFrame) -> str | None:
    return find_column(
        df,
        ["mic_numeric", "MIC_numeric", "mic", "MIC", "mic_value", "MIC_value"],
        required=False,
    )


def infer_log10mic_column(df: pd.DataFrame) -> str | None:
    return find_column(
        df,
        ["log10_mic", "log10MIC", "log10_MIC", "log10mic"],
        required=False,
    )


def infer_log2mic_column(df: pd.DataFrame) -> str | None:
    return find_column(df, ["log2_mic", "log2MIC", "log2_MIC", "log2mic"], required=False)


def infer_active_column(df: pd.DataFrame) -> str | None:
    return find_column(df, ["active", "true_active", "label", "class"], required=False)


def load_split_dataframe(path: Path, task: str) -> tuple[pd.DataFrame, str, str]:
    df = pd.read_csv(path)
    smiles_col = infer_smiles_column(df)
    target_col = infer_classification_target(df) if task == "classification" else infer_regression_target(df)
    before = len(df)
    df = df.loc[df[smiles_col].notna() & df[target_col].notna()].copy().reset_index(drop=False)
    df.rename(columns={"index": "source_row"}, inplace=True)
    if len(df) != before:
        logging.getLogger(__name__).warning(
            "Dropped %d rows with missing smiles/target from %s", before - len(df), path
        )
    if task == "classification":
        vals = pd.to_numeric(df[target_col], errors="raise").astype(int)
        unique = sorted(pd.unique(vals).tolist())
        if not set(unique).issubset({0, 1}):
            raise ValueError(f"Classification target must be binary 0/1; got {unique} in {path}")
        df[target_col] = vals
    else:
        df[target_col] = pd.to_numeric(df[target_col], errors="raise").astype(float)
    return df, smiles_col, target_col


def get_true_mic_arrays(df: pd.DataFrame, regression_target_col: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mic_numeric, log10_mic, log2_mic; NaN when unavailable."""
    n = len(df)
    mic = np.full(n, np.nan, dtype=float)
    log10 = np.full(n, np.nan, dtype=float)
    log2 = np.full(n, np.nan, dtype=float)

    mic_col = infer_mic_numeric_column(df)
    l10_col = infer_log10mic_column(df)
    l2_col = infer_log2mic_column(df)

    if mic_col is not None:
        mic = pd.to_numeric(df[mic_col], errors="coerce").to_numpy(float)
    if regression_target_col is not None:
        log10 = pd.to_numeric(df[regression_target_col], errors="coerce").to_numpy(float)
    elif l10_col is not None:
        log10 = pd.to_numeric(df[l10_col], errors="coerce").to_numpy(float)
    if l2_col is not None:
        log2 = pd.to_numeric(df[l2_col], errors="coerce").to_numpy(float)

    # Fill mutually derivable quantities without overwriting source values.
    valid = np.isfinite(mic) & (mic > 0) & ~np.isfinite(log10)
    log10[valid] = np.log10(mic[valid])
    valid = np.isfinite(log10) & ~np.isfinite(mic)
    mic[valid] = np.power(10.0, log10[valid])
    valid = np.isfinite(mic) & (mic > 0) & ~np.isfinite(log2)
    log2[valid] = np.log2(mic[valid])
    valid = np.isfinite(log10) & ~np.isfinite(log2)
    log2[valid] = log10[valid] / math.log10(2.0)
    return mic, log10, log2


# ------------------------------ predictions -------------------------------


def _prediction_progress_log(
    logger: logging.Logger | None,
    split_name: str,
    done_batches: int,
    total_batches: int,
    done_rows: int,
    total_rows: int,
    started: float,
) -> None:
    if logger is None:
        return
    elapsed = time.time() - started
    row_rate = done_rows / elapsed if elapsed > 0 else float("nan")
    eta = (total_rows - done_rows) / row_rate if row_rate and np.isfinite(row_rate) and row_rate > 0 else float("nan")
    logger.info(
        "Prediction progress | %-10s | batch %d/%d | rows %d/%d (%.1f%%) | elapsed %.1fs | ETA %.1fs",
        split_name,
        done_batches,
        total_batches,
        done_rows,
        total_rows,
        100.0 * done_rows / max(1, total_rows),
        elapsed,
        eta,
    )


def predict_classifier_batched(
    model: Any,
    X: np.ndarray,
    batch_size: int,
    logger: logging.Logger | None = None,
    split_name: str = "data",
) -> np.ndarray:
    n = len(X)
    if n == 0:
        return np.empty(0, dtype=float)
    if batch_size <= 0:
        batch_size = n
    total_batches = math.ceil(n / batch_size)
    log_every = max(1, math.ceil(total_batches / 20))  # about every 5%
    if logger is not None:
        logger.info(
            "Prediction starting | %-10s | rows=%d | batch_size=%d | batches=%d",
            split_name, n, batch_size, total_batches,
        )
    started = time.time()
    out: list[np.ndarray] = []
    for bi, start in enumerate(range(0, n, batch_size), start=1):
        stop = min(start + batch_size, n)
        out.append(np.asarray(model.predict_proba(X[start:stop]))[:, 1].astype(float))
        if bi == 1 or bi == total_batches or bi % log_every == 0:
            _prediction_progress_log(logger, split_name, bi, total_batches, stop, n, started)
    return np.concatenate(out)


def predict_regressor_batched(
    model: Any,
    X: np.ndarray,
    batch_size: int,
    logger: logging.Logger | None = None,
    split_name: str = "data",
) -> np.ndarray:
    n = len(X)
    if n == 0:
        return np.empty(0, dtype=float)
    if batch_size <= 0:
        batch_size = n
    total_batches = math.ceil(n / batch_size)
    log_every = max(1, math.ceil(total_batches / 20))  # about every 5%
    if logger is not None:
        logger.info(
            "Prediction starting | %-10s | rows=%d | batch_size=%d | batches=%d",
            split_name, n, batch_size, total_batches,
        )
    started = time.time()
    out: list[np.ndarray] = []
    for bi, start in enumerate(range(0, n, batch_size), start=1):
        stop = min(start + batch_size, n)
        out.append(np.asarray(model.predict(X[start:stop]), dtype=float).reshape(-1))
        if bi == 1 or bi == total_batches or bi % log_every == 0:
            _prediction_progress_log(logger, split_name, bi, total_batches, stop, n, started)
    return np.concatenate(out)


# ------------------------------ core metrics ------------------------------


def safe_roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score)) if len(np.unique(y)) >= 2 else float("nan")


def safe_pr_auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(average_precision_score(y, score)) if np.sum(y == 1) > 0 else float("nan")


def safe_log_loss(y: np.ndarray, prob: np.ndarray) -> float:
    p = np.clip(prob, 1e-7, 1 - 1e-7)
    return float(log_loss(y, p, labels=[0, 1]))


def screening_metrics(y_active: np.ndarray, score: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_active, dtype=int)
    s = np.asarray(score, dtype=float)
    n = len(y)
    total_hits = int(np.sum(y == 1))
    prevalence = total_hits / n if n else float("nan")
    order = np.argsort(-s, kind="mergesort")
    out: dict[str, float] = {}
    for pct in (1, 5, 10):
        top_n = max(1, int(math.ceil(n * pct / 100.0))) if n else 0
        idx = order[:top_n]
        hits = int(y[idx].sum()) if top_n else 0
        hit_rate = hits / top_n if top_n else float("nan")
        ef = hit_rate / prevalence if prevalence and prevalence > 0 else float("nan")
        out[f"top_{pct}pct_n"] = top_n
        out[f"top_{pct}pct_hits"] = hits
        out[f"hit_rate_top_{pct}pct"] = hit_rate
        out[f"ef_{pct}pct"] = ef
        out[f"Hit_rate@Top{pct}%"] = hit_rate
        out[f"EF{pct}%"] = ef
    for k in (50, 100, 500):
        kk = min(k, n)
        hits = int(y[order[:kk]].sum()) if kk else 0
        recall = hits / total_hits if total_hits > 0 else float("nan")
        out[f"Top{k}_n"] = kk
        out[f"Top{k}_hits"] = hits
        out[f"Recall@Top{k}"] = recall
    return out


def classification_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = THRESHOLD_USED) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out: dict[str, float] = {
        "n": int(len(y)),
        "pos": int(np.sum(y == 1)),
        "neg": int(np.sum(y == 0)),
        "pos_ratio": float(np.mean(y == 1)) if len(y) else float("nan"),
        "roc_auc": safe_roc_auc(y, p),
        "pr_auc": safe_pr_auc(y, p),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)) if len(np.unique(y)) >= 2 and len(np.unique(pred)) >= 2 else float(matthews_corrcoef(y, pred)),
        "specificity": safe_div(tn, tn + fp),
        "sensitivity": safe_div(tp, tp + fn),
        "log_loss": safe_log_loss(y, p),
        "brier": float(brier_score_loss(y, p)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }
    out.update(screening_metrics(y, p))
    return out


def best_f1_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob, dtype=float)
    if len(np.unique(y)) < 2:
        return THRESHOLD_USED
    precision, recall, thresholds = precision_recall_curve(y, p)
    if len(thresholds) == 0:
        return THRESHOLD_USED
    f1s = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    if not np.isfinite(f1s).any():
        return THRESHOLD_USED
    return float(thresholds[int(np.nanargmax(f1s))])


def safe_corr(fn: Any, y: np.ndarray, p: np.ndarray) -> float:
    if len(y) < 2 or np.nanstd(y) == 0 or np.nanstd(p) == 0:
        return float("nan")
    try:
        return float(fn(y, p).statistic)
    except Exception:
        try:
            return float(fn(y, p)[0])
        except Exception:
            return float("nan")


def regression_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(pred, dtype=float)
    err = p - y
    abs_err = np.abs(err)
    mse = float(mean_squared_error(y, p))
    rmse = float(math.sqrt(mse))
    fold_error = np.power(10.0, np.clip(abs_err, 0, 300))
    out: dict[str, float] = {
        "n": int(len(y)),
        "log10mic_mean": float(np.mean(y)) if len(y) else float("nan"),
        "log10mic_std": float(np.std(y, ddof=1)) if len(y) > 1 else float("nan"),
        "mae": float(mean_absolute_error(y, p)),
        "mse": mse,
        "rmse": rmse,
        "medae": float(median_absolute_error(y, p)),
        "r2": float(r2_score(y, p)) if len(y) >= 2 else float("nan"),
        "pearson": safe_corr(pearsonr, y, p),
        "spearman": safe_corr(spearmanr, y, p),
        "within_1_dilution": float(np.mean(abs_err <= DILUTION_1_LOG10)),
        "within_2_dilutions": float(np.mean(abs_err <= DILUTION_2_LOG10)),
        "median_fold_error": float(np.median(fold_error)),
        "mean_fold_error": float(np.mean(fold_error)),
        "max_abs_error_log10": float(np.max(abs_err)) if len(abs_err) else float("nan"),
    }
    return out


def add_screening_to_regression_metrics(
    metrics: dict[str, float],
    true_active: np.ndarray | None,
    pred_log10: np.ndarray,
) -> dict[str, float]:
    if true_active is not None and np.isfinite(np.asarray(true_active, dtype=float)).all():
        metrics.update(screening_metrics(np.asarray(true_active, dtype=int), -np.asarray(pred_log10, dtype=float)))
    else:
        for pct in (1, 5, 10):
            metrics[f"EF{pct}%"] = float("nan")
            metrics[f"Hit_rate@Top{pct}%"] = float("nan")
        for k in (50, 100, 500):
            metrics[f"Recall@Top{k}"] = float("nan")
    return metrics


def prefix_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{k}": v for k, v in metrics.items()}


# ----------------------------- MIC diagnostics ----------------------------


def boundary_and_range(mic: np.ndarray, threshold: float | None) -> tuple[np.ndarray, np.ndarray]:
    n = len(mic)
    boundary = np.full(n, np.nan, dtype=float)
    ranges = np.full(n, "unknown", dtype=object)
    if threshold is None or not np.isfinite(threshold) or threshold <= 0:
        return boundary, ranges
    valid = np.isfinite(mic) & (mic > 0)
    boundary[valid] = ((mic[valid] >= threshold / 2.0) & (mic[valid] <= 2.0 * threshold)).astype(int)
    ranges[valid & (mic <= threshold / 2.0)] = "strong_MIC_le_T_over_2"
    ranges[valid & (mic > threshold / 2.0) & (mic <= 2.0 * threshold)] = "boundary_T_over_2_to_2T"
    ranges[valid & (mic > 2.0 * threshold)] = "weak_MIC_gt_2T"
    return boundary, ranges


def true_active_for_screening(df: pd.DataFrame, mic: np.ndarray, threshold: float | None) -> np.ndarray | None:
    ac = infer_active_column(df)
    if ac is not None:
        vals = pd.to_numeric(df[ac], errors="coerce").to_numpy(float)
        if np.isfinite(vals).all() and set(np.unique(vals).astype(int).tolist()).issubset({0, 1}):
            return vals.astype(int)
    if threshold is not None and np.isfinite(mic).all():
        return (mic <= threshold).astype(int)
    return None


def nn_bin_from_similarity(sim: float) -> str:
    if not np.isfinite(sim):
        return "unknown"
    if sim >= 0.7:
        return ">=0.7"
    if sim >= 0.4:
        return "0.4-0.7"
    return "<0.4"


def compute_similarity_diagnostics(
    train_fps: Sequence[Any],
    query_fps: Sequence[Any],
    train_smiles: Sequence[str],
    train_true_log10: np.ndarray,
    query_true_log10: np.ndarray,
    train_pred: np.ndarray | None,
    query_pred: np.ndarray,
    task: str,
    ac_sim_threshold: float = DEFAULT_AC_SIM_THRESHOLD,
    ac_fold_change: float = DEFAULT_AC_FOLD_CHANGE,
    k_local: int = DEFAULT_NN_K,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute query-to-train NN, local roughness, and structural activity cliffs.

    ``train_pred`` is optional.  When train predictions are intentionally omitted
    (the TabPFN benchmark default), structural NN/AC diagnostics are still fully
    computed from train fingerprints and true MIC values, while prediction-based
    train-neighbor fields and pair directional accuracy are left as NaN.
    """
    ac_log10_threshold = math.log10(ac_fold_change)
    rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    tr_y = np.asarray(train_true_log10, dtype=float)
    q_y = np.asarray(query_true_log10, dtype=float)
    q_pred = np.asarray(query_pred, dtype=float)
    tr_pred = None if train_pred is None else np.asarray(train_pred, dtype=float)
    train_fps_list = list(train_fps)

    for qi, qfp in enumerate(query_fps):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(qfp, train_fps_list), dtype=float)
        if len(sims) == 0:
            rows.append({})
            continue
        ni = int(np.argmax(sims))
        nn_sim = float(sims[ni])
        k = min(k_local, len(sims))
        top_idx = np.argpartition(-sims, kth=k - 1)[:k] if k < len(sims) else np.arange(len(sims))
        top_y = tr_y[top_idx]
        top_y = top_y[np.isfinite(top_y)]
        local_var = float(np.var(top_y, ddof=1)) if len(top_y) > 1 else float("nan")
        local_sd = float(np.std(top_y, ddof=1)) if len(top_y) > 1 else float("nan")

        cliff_idx = np.array([], dtype=int)
        can_compute_ac = bool(np.isfinite(q_y[qi]) and np.isfinite(tr_y).any())
        if can_compute_ac:
            dy = np.abs(tr_y - q_y[qi])
            cliff_idx = np.where(
                (sims >= ac_sim_threshold)
                & np.isfinite(tr_y)
                & (dy >= ac_log10_threshold)
            )[0]

        directional: list[float] = []
        max_diff_log10 = float("nan")
        max_diff_log2 = float("nan")
        if len(cliff_idx):
            diffs = np.abs(tr_y[cliff_idx] - q_y[qi])
            max_diff_log10 = float(np.nanmax(diffs))
            max_diff_log2 = max_diff_log10 / math.log10(2.0)
            for tj in cliff_idx:
                true_delta = q_y[qi] - tr_y[tj]
                train_pred_value = (
                    float(tr_pred[tj])
                    if tr_pred is not None and np.isfinite(tr_pred[tj])
                    else float("nan")
                )
                if tr_pred is None or not np.isfinite(train_pred_value):
                    correct = float("nan")
                else:
                    if task == "regression":
                        pred_delta = q_pred[qi] - train_pred_value
                    else:
                        # Higher classification score = more active = lower MIC, so sign reverses.
                        pred_delta = -(q_pred[qi] - train_pred_value)
                    correct = float(np.sign(pred_delta) == np.sign(true_delta)) if true_delta != 0 else float("nan")
                    directional.append(correct)
                pair_rows.append(
                    {
                        "query_row": qi,
                        "train_row": int(tj),
                        "tanimoto": float(sims[tj]),
                        "query_log10mic": float(q_y[qi]),
                        "train_log10mic": float(tr_y[tj]),
                        "abs_log10mic_diff": float(abs(true_delta)),
                        "query_pred": float(q_pred[qi]),
                        "train_pred": train_pred_value,
                        "directional_correct": correct,
                    }
                )

        nn_train_pred = (
            float(tr_pred[ni])
            if tr_pred is not None and np.isfinite(tr_pred[ni])
            else float("nan")
        )
        rows.append(
            {
                "nn_train_row": ni,
                "nn_sim": nn_sim,
                "nn_bin": nn_bin_from_similarity(nn_sim),
                "nn_train_smiles": str(train_smiles[ni]),
                "nn_train_true": float(tr_y[ni]) if np.isfinite(tr_y[ni]) else float("nan"),
                "nn_train_pred": nn_train_pred,
                "local_train_log10mic_variance_k10": local_var,
                "local_train_log10mic_sd_k10": local_sd,
                "ac_flag": int(len(cliff_idx) > 0) if can_compute_ac else float("nan"),
                "ac_neighbor_count": int(len(cliff_idx)) if can_compute_ac else float("nan"),
                "ac_max_abs_log10mic_diff": max_diff_log10,
                "ac_max_abs_log2mic_diff": max_diff_log2,
                "ac_pair_directional_accuracy_for_molecule": float(np.nanmean(directional)) if directional else float("nan"),
                "ac_pair_count_for_molecule": int(len(cliff_idx)) if can_compute_ac else float("nan"),
            }
        )

    diag_columns = [
        "nn_train_row", "nn_sim", "nn_bin", "nn_train_smiles", "nn_train_true",
        "nn_train_pred", "local_train_log10mic_variance_k10", "local_train_log10mic_sd_k10",
        "ac_flag", "ac_neighbor_count", "ac_max_abs_log10mic_diff", "ac_max_abs_log2mic_diff",
        "ac_pair_directional_accuracy_for_molecule", "ac_pair_count_for_molecule",
    ]
    pair_columns = [
        "query_row", "train_row", "tanimoto", "query_log10mic", "train_log10mic",
        "abs_log10mic_diff", "query_pred", "train_pred", "directional_correct",
    ]
    return pd.DataFrame(rows, columns=diag_columns), pd.DataFrame(pair_rows, columns=pair_columns)


# --------------------------- prediction tables ----------------------------


def build_classification_prediction_table(
    df: pd.DataFrame,
    split_name: str,
    spec: RunSpec,
    prob: np.ndarray,
    val_best_threshold: float,
    target_col: str,
    diagnostics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = df.copy()
    y = pd.to_numeric(out[target_col], errors="raise").astype(int).to_numpy()
    pred = (prob >= THRESHOLD_USED).astype(int)
    mic, l10, l2 = get_true_mic_arrays(out)
    if split_name == "test":
        boundary, ranges = boundary_and_range(mic, spec.threshold)
    else:
        boundary = np.full(len(out), np.nan, dtype=float)
        ranges = np.full(len(out), "not_computed_non_test", dtype=object)

    out["split"] = split_name
    out["dataset_name"] = spec.dataset_name
    out["genus"] = spec.genus
    out["task"] = "classification"
    out["split_type"] = spec.split_type
    out["seed"] = spec.seed
    out["model"] = MODEL_NAME
    out["model_family"] = MODEL_FAMILY
    if "smiles" not in out.columns:
        out["smiles"] = out[infer_smiles_column(out)].astype(str)
    out["active"] = y
    out["true_active"] = y
    out["pred_score"] = prob
    out["pred_prob"] = prob
    out["pred_label"] = pred
    out["pred_label_0p5"] = pred
    out["threshold_used"] = THRESHOLD_USED
    out["val_best_f1_threshold"] = val_best_threshold
    out["error"] = (pred != y).astype(int)
    out["boundary_flag"] = boundary
    out["threshold_for_boundary"] = spec.threshold
    out["mic_range_group"] = ranges
    out["log2_mic_for_diagnosis"] = l2
    out["log10_mic_for_diagnosis"] = l10
    out["mic_numeric"] = mic

    diag_cols = [
        "nn_train_row", "nn_sim", "nn_bin", "nn_train_smiles", "nn_train_active",
        "nn_train_score", "ac_flag", "ac_neighbor_count", "ac_max_abs_log2mic_diff",
        "ac_pair_directional_accuracy_for_molecule", "ac_pair_count_for_molecule",
    ]
    for c in diag_cols:
        if c not in out.columns:
            out[c] = np.nan
    if diagnostics is not None and len(diagnostics) == len(out):
        out["nn_train_row"] = diagnostics["nn_train_row"].to_numpy()
        out["nn_sim"] = diagnostics["nn_sim"].to_numpy()
        out["nn_bin"] = diagnostics["nn_bin"].to_numpy()
        out["nn_train_smiles"] = diagnostics["nn_train_smiles"].to_numpy()
        # For classification, nn_train_true in helper is log10MIC; active is added by caller if available.
        if "nn_train_active" in diagnostics:
            out["nn_train_active"] = diagnostics["nn_train_active"].to_numpy()
        if "nn_train_score" in diagnostics:
            out["nn_train_score"] = diagnostics["nn_train_score"].to_numpy()
        out["ac_flag"] = diagnostics["ac_flag"].to_numpy()
        out["ac_neighbor_count"] = diagnostics["ac_neighbor_count"].to_numpy()
        out["ac_max_abs_log2mic_diff"] = diagnostics["ac_max_abs_log2mic_diff"].to_numpy()
        out["ac_pair_directional_accuracy_for_molecule"] = diagnostics[
            "ac_pair_directional_accuracy_for_molecule"
        ].to_numpy()
        out["ac_pair_count_for_molecule"] = diagnostics["ac_pair_count_for_molecule"].to_numpy()
    return out


def build_regression_prediction_table(
    df: pd.DataFrame,
    split_name: str,
    spec: RunSpec,
    pred_log10: np.ndarray,
    target_col: str,
    diagnostics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = df.copy()
    y = pd.to_numeric(out[target_col], errors="raise").astype(float).to_numpy()
    mic, l10, _ = get_true_mic_arrays(out, regression_target_col=target_col)
    pred_mic = np.power(10.0, np.clip(pred_log10, -300, 300))
    abs_err = np.abs(pred_log10 - y)
    if split_name == "test":
        boundary, ranges = boundary_and_range(mic, spec.threshold)
    else:
        boundary = np.full(len(out), np.nan, dtype=float)
        ranges = np.full(len(out), "not_computed_non_test", dtype=object)
    active = true_active_for_screening(out, mic, spec.threshold)

    out["split"] = split_name
    out["dataset_name"] = spec.dataset_name
    out["genus"] = spec.genus
    out["task"] = "regression"
    out["split_type"] = spec.split_type
    out["seed"] = spec.seed
    out["model"] = MODEL_NAME
    out["model_family"] = MODEL_FAMILY
    if "smiles" not in out.columns:
        out["smiles"] = out[infer_smiles_column(out)].astype(str)
    out["log10_mic"] = y
    out["true_log10MIC"] = y
    out["true_MIC"] = mic
    out["mic_numeric"] = mic
    out["pred_log10MIC"] = pred_log10
    out["pred_MIC"] = pred_mic
    out["score_reg"] = -pred_log10
    out["abs_error_log10"] = abs_err
    out["sq_error_log10"] = np.square(pred_log10 - y)
    out["fold_error"] = np.power(10.0, np.clip(abs_err, 0, 300))
    out["within_1_dilution"] = (abs_err <= DILUTION_1_LOG10).astype(int)
    out["within_2_dilutions"] = (abs_err <= DILUTION_2_LOG10).astype(int)
    out["true_active_for_screening"] = active if active is not None else np.nan
    out["threshold_for_boundary"] = spec.threshold
    out["boundary_flag"] = boundary
    out["mic_range_group"] = ranges

    diag_cols = [
        "nn_train_row", "nn_sim", "nn_bin", "nn_train_smiles", "nn_train_log10mic",
        "nn_train_pred_log10mic", "local_train_log10mic_variance_k10",
        "local_train_log10mic_sd_k10", "ac_flag", "ac_neighbor_count",
        "ac_max_abs_log10mic_diff", "ac_pair_directional_accuracy_for_molecule",
        "ac_pair_count_for_molecule",
    ]
    for c in diag_cols:
        if c not in out.columns:
            out[c] = np.nan
    if diagnostics is not None and len(diagnostics) == len(out):
        out["nn_train_row"] = diagnostics["nn_train_row"].to_numpy()
        out["nn_sim"] = diagnostics["nn_sim"].to_numpy()
        out["nn_bin"] = diagnostics["nn_bin"].to_numpy()
        out["nn_train_smiles"] = diagnostics["nn_train_smiles"].to_numpy()
        out["nn_train_log10mic"] = diagnostics["nn_train_true"].to_numpy()
        out["nn_train_pred_log10mic"] = diagnostics["nn_train_pred"].to_numpy()
        out["local_train_log10mic_variance_k10"] = diagnostics[
            "local_train_log10mic_variance_k10"
        ].to_numpy()
        out["local_train_log10mic_sd_k10"] = diagnostics[
            "local_train_log10mic_sd_k10"
        ].to_numpy()
        out["ac_flag"] = diagnostics["ac_flag"].to_numpy()
        out["ac_neighbor_count"] = diagnostics["ac_neighbor_count"].to_numpy()
        out["ac_max_abs_log10mic_diff"] = diagnostics["ac_max_abs_log10mic_diff"].to_numpy()
        out["ac_pair_directional_accuracy_for_molecule"] = diagnostics[
            "ac_pair_directional_accuracy_for_molecule"
        ].to_numpy()
        out["ac_pair_count_for_molecule"] = diagnostics["ac_pair_count_for_molecule"].to_numpy()
    return out


# ----------------------------- subgroup metrics ---------------------------


def _classification_subgroup_row(
    df: pd.DataFrame, group_type: str, group_col: str, group_value: Any
) -> dict[str, Any]:
    if len(df) == 0:
        return {"group_type": group_type, "group_col": group_col, "group_value": group_value, "n": 0}
    y = df["true_active"].to_numpy(int)
    p = df["pred_prob"].to_numpy(float)
    m = classification_metrics(y, p)
    row = {"group_type": group_type, "group_col": group_col, "group_value": group_value}
    row.update(m)
    row["error_rate"] = float(df["error"].mean())
    return row


def classification_subgroup_table(test_pred: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = [
        ("MIC_boundary", "boundary_flag", [0, 1]),
        ("NN_similarity", "nn_bin", [">=0.7", "0.4-0.7", "<0.4"]),
        ("Activity_cliff", "ac_flag", [0, 1]),
        (
            "MIC_range",
            "mic_range_group",
            ["strong_MIC_le_T_over_2", "boundary_T_over_2_to_2T", "weak_MIC_gt_2T"],
        ),
    ]
    for gt, col, vals in groups:
        for v in vals:
            if col not in test_pred.columns:
                continue
            mask = test_pred[col] == v
            rows.append(_classification_subgroup_row(test_pred.loc[mask], gt, col, v))
    return pd.DataFrame(rows)


def _regression_subgroup_row(
    df: pd.DataFrame, group_type: str, group_col: str, group_value: Any
) -> dict[str, Any]:
    if len(df) == 0:
        return {"group_type": group_type, "group_col": group_col, "group_value": group_value, "n": 0}
    y = df["true_log10MIC"].to_numpy(float)
    p = df["pred_log10MIC"].to_numpy(float)
    m = regression_metrics(y, p)
    active = None
    if "true_active_for_screening" in df and df["true_active_for_screening"].notna().all():
        active = df["true_active_for_screening"].to_numpy(int)
    add_screening_to_regression_metrics(m, active, p)
    row = {"group_type": group_type, "group_col": group_col, "group_value": group_value}
    row.update(m)
    return row


def regression_subgroup_table(test_pred: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if "local_train_log10mic_variance_k10" in test_pred:
        vals = test_pred["local_train_log10mic_variance_k10"].to_numpy(float)
        med = float(np.nanmedian(vals)) if np.isfinite(vals).any() else float("nan")
        test_pred = test_pred.copy()
        test_pred["local_variance_group"] = np.where(
            np.isfinite(vals), np.where(vals >= med, "high_local_variance", "low_local_variance"), "unknown"
        )
    groups = [
        ("MIC_boundary", "boundary_flag", [0, 1]),
        ("NN_similarity", "nn_bin", [">=0.7", "0.4-0.7", "<0.4"]),
        ("Activity_cliff", "ac_flag", [0, 1]),
        (
            "MIC_range",
            "mic_range_group",
            ["strong_MIC_le_T_over_2", "boundary_T_over_2_to_2T", "weak_MIC_gt_2T"],
        ),
        ("Local_variance", "local_variance_group", ["high_local_variance", "low_local_variance"]),
    ]
    for gt, col, vals in groups:
        for v in vals:
            if col not in test_pred.columns:
                continue
            rows.append(_regression_subgroup_row(test_pred.loc[test_pred[col] == v], gt, col, v))
    return pd.DataFrame(rows)


# ---------------------------- diagnostic widths ---------------------------


def classification_diagnostics_wide(test_pred: pd.DataFrame, pair_df: pd.DataFrame) -> dict[str, Any]:
    d: dict[str, Any] = {}
    for flag, name in [(1, "boundary"), (0, "non_boundary")]:
        sub = test_pred.loc[test_pred["boundary_flag"] == flag]
        d[f"{name}_n"] = int(len(sub))
        d[f"{name}_error_rate"] = float(sub["error"].mean()) if len(sub) else float("nan")
    d["boundary_error_enrichment"] = safe_div(d.get("boundary_error_rate", np.nan), d.get("non_boundary_error_rate", np.nan))

    for label, prefix in [(">=0.7", "nn_ge_0p7"), ("0.4-0.7", "nn_0p4_0p7"), ("<0.4", "nn_lt_0p4")]:
        sub = test_pred.loc[test_pred["nn_bin"] == label]
        d[f"{prefix}_n"] = int(len(sub))
        if len(sub):
            m = classification_metrics(sub["true_active"].to_numpy(int), sub["pred_prob"].to_numpy(float))
            d[f"{prefix}_error_rate"] = float(sub["error"].mean())
            for k in ["roc_auc", "pr_auc", "accuracy", "f1", "mcc", "EF1%", "EF5%", "EF10%", "Hit_rate@Top1%", "Recall@Top100"]:
                d[f"{prefix}_{k}"] = m.get(k, float("nan"))
        else:
            d[f"{prefix}_error_rate"] = float("nan")

    for flag, name in [(1, "ac"), (0, "non_ac")]:
        sub = test_pred.loc[test_pred["ac_flag"] == flag]
        d[f"{name}_n"] = int(len(sub))
        d[f"{name}_error_rate"] = float(sub["error"].mean()) if len(sub) else float("nan")
    d["ac_error_enrichment"] = safe_div(d.get("ac_error_rate", np.nan), d.get("non_ac_error_rate", np.nan))
    ac_sub = test_pred.loc[test_pred["ac_flag"] == 1]
    d["ac_subgroup_pr_auc"] = safe_pr_auc(ac_sub["true_active"].to_numpy(int), ac_sub["pred_prob"].to_numpy(float)) if len(ac_sub) else float("nan")
    d["ac_pair_directional_accuracy"] = float(pair_df["directional_correct"].mean()) if len(pair_df) else float("nan")
    d["ac_pair_total_count"] = int(len(pair_df))
    return d


def regression_diagnostics_wide(test_pred: pd.DataFrame, pair_df: pd.DataFrame) -> dict[str, Any]:
    d: dict[str, Any] = {}
    for flag, name in [(1, "boundary"), (0, "non_boundary")]:
        sub = test_pred.loc[test_pred["boundary_flag"] == flag]
        d[f"{name}_n"] = int(len(sub))
        if len(sub):
            m = regression_metrics(sub["true_log10MIC"].to_numpy(float), sub["pred_log10MIC"].to_numpy(float))
            d[f"{name}_mae"] = m["mae"]
            d[f"{name}_rmse"] = m["rmse"]
            d[f"{name}_within_1_dilution"] = m["within_1_dilution"]
        else:
            d[f"{name}_mae"] = float("nan")
            d[f"{name}_rmse"] = float("nan")
            d[f"{name}_within_1_dilution"] = float("nan")
    d["boundary_regression_error_enrichment"] = safe_div(d.get("boundary_mae", np.nan), d.get("non_boundary_mae", np.nan))

    for label, prefix in [(">=0.7", "nn_ge_0p7"), ("0.4-0.7", "nn_0p4_0p7"), ("<0.4", "nn_lt_0p4")]:
        sub = test_pred.loc[test_pred["nn_bin"] == label]
        d[f"{prefix}_n"] = int(len(sub))
        if len(sub):
            m = regression_metrics(sub["true_log10MIC"].to_numpy(float), sub["pred_log10MIC"].to_numpy(float))
            for k in ["mae", "rmse", "mse", "medae", "r2", "pearson", "spearman", "within_1_dilution", "within_2_dilutions"]:
                d[f"{prefix}_{k}"] = m[k]
        else:
            for k in ["mae", "rmse", "mse", "medae", "r2", "pearson", "spearman", "within_1_dilution", "within_2_dilutions"]:
                d[f"{prefix}_{k}"] = float("nan")

    for flag, name in [(1, "ac"), (0, "non_ac")]:
        sub = test_pred.loc[test_pred["ac_flag"] == flag]
        d[f"{name}_n"] = int(len(sub))
        if len(sub):
            m = regression_metrics(sub["true_log10MIC"].to_numpy(float), sub["pred_log10MIC"].to_numpy(float))
            d[f"{name}_mae"] = m["mae"]
            d[f"{name}_rmse"] = m["rmse"]
        else:
            d[f"{name}_mae"] = float("nan")
            d[f"{name}_rmse"] = float("nan")
    d["ac_error_enrichment"] = safe_div(d.get("ac_mae", np.nan), d.get("non_ac_mae", np.nan))
    d["ac_pair_directional_accuracy"] = float(pair_df["directional_correct"].mean()) if len(pair_df) else float("nan")
    d["ac_pair_total_count"] = int(len(pair_df))

    vals = test_pred["local_train_log10mic_variance_k10"].to_numpy(float)
    d["mean_local_train_log10mic_variance_k10"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
    d["median_local_train_log10mic_variance_k10"] = float(np.nanmedian(vals)) if np.isfinite(vals).any() else float("nan")
    if np.isfinite(vals).any():
        med = np.nanmedian(vals)
        high = test_pred.loc[vals >= med]
        low = test_pred.loc[vals < med]
        high_mae = float(np.mean(high["abs_error_log10"])) if len(high) else float("nan")
        low_mae = float(np.mean(low["abs_error_log10"])) if len(low) else float("nan")
    else:
        high_mae = low_mae = float("nan")
    d["high_local_variance_mae"] = high_mae
    d["low_local_variance_mae"] = low_mae
    d["local_variance_error_enrichment"] = safe_div(high_mae, low_mae)
    return d


# ----------------------------- training logs ------------------------------


def classification_training_log(
    train_metrics: dict[str, Any] | None,
    val_metrics: dict[str, Any],
    fit_seconds: float,
    predict_train_seconds: float,
    predict_val_seconds: float,
) -> pd.DataFrame:
    # Standard TabPFN uses the labeled train set as in-context context.  Predicting
    # that same train set is intentionally omitted because it is a self-context
    # prediction and is not comparable with ordinary train-set performance.
    row = {
        "iteration": 1,
        "epoch": np.nan,
        "train_loss": np.nan,
        "val_loss": val_metrics["log_loss"],
        "learning_rate": np.nan,
        "best_epoch_so_far": np.nan,
        "best_val_metric_so_far": np.nan,
        "train_log_loss": np.nan,
        "val_log_loss": val_metrics["log_loss"],
        "train_auc": np.nan,
        "val_auc": val_metrics["roc_auc"],
        "train_pr_auc": np.nan,
        "val_pr_auc": val_metrics["pr_auc"],
        "train_accuracy": np.nan,
        "val_accuracy": val_metrics["accuracy"],
        "train_f1": np.nan,
        "val_f1": val_metrics["f1"],
        "train_mcc": np.nan,
        "val_mcc": val_metrics["mcc"],
        "fit_seconds": fit_seconds,
        "predict_train_seconds": predict_train_seconds,
        "predict_val_seconds": predict_val_seconds,
        "note": "final_only_no_epoch_training_for_tabpfn_in_context; train prediction intentionally not computed to avoid self-context evaluation; val_loss is a final diagnostic, not an optimization loss",
    }
    return pd.DataFrame([row])


def regression_training_log(
    train_metrics: dict[str, Any] | None,
    val_metrics: dict[str, Any],
    fit_seconds: float,
    predict_train_seconds: float,
    predict_val_seconds: float,
) -> pd.DataFrame:
    row = {
        "iteration": 1,
        "epoch": np.nan,
        "train_loss": np.nan,
        "val_loss": val_metrics["mse"],
        "learning_rate": np.nan,
        "best_epoch_so_far": np.nan,
        "best_val_metric_so_far": np.nan,
        "train_mse": np.nan,
        "val_mse": val_metrics["mse"],
        "train_rmse": np.nan,
        "val_rmse": val_metrics["rmse"],
        "train_mae": np.nan,
        "val_mae": val_metrics["mae"],
        "train_spearman": np.nan,
        "val_spearman": val_metrics["spearman"],
        "train_pearson": np.nan,
        "val_pearson": val_metrics["pearson"],
        "train_within_1_dilution": np.nan,
        "val_within_1_dilution": val_metrics["within_1_dilution"],
        "train_within_2_dilution": np.nan,
        "val_within_2_dilution": val_metrics["within_2_dilutions"],
        "fit_seconds": fit_seconds,
        "predict_train_seconds": predict_train_seconds,
        "predict_val_seconds": predict_val_seconds,
        "note": "final_only_no_epoch_training_for_tabpfn_in_context; train prediction intentionally not computed to avoid self-context evaluation; val_loss is a final MSE diagnostic, not an optimization loss",
    }
    return pd.DataFrame([row])


def training_log_to_long(df: pd.DataFrame) -> pd.DataFrame:
    id_cols = [c for c in ["iteration", "epoch", "learning_rate", "note"] if c in df.columns]
    metric_cols = [c for c in df.columns if c not in id_cols]
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        for c in metric_cols:
            split = "train" if c.startswith("train_") else "validation" if c.startswith("val_") else "run"
            rows.append(
                {
                    "iteration": r.get("iteration", np.nan),
                    "epoch": r.get("epoch", np.nan),
                    "split": split,
                    "metric": c,
                    "value": r[c],
                    "learning_rate": r.get("learning_rate", np.nan),
                    "note": r.get("note", ""),
                }
            )
    return pd.DataFrame(rows)


# --------------------------- reproducibility files ------------------------


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "rdkit": getattr(rdBase, "rdkitVersion", "unknown"),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    try:
        import sklearn
        versions["scikit_learn"] = sklearn.__version__
    except Exception:
        pass
    try:
        import scipy
        versions["scipy"] = scipy.__version__
    except Exception:
        pass
    try:
        import torch
        versions["torch"] = torch.__version__
        versions["torch_cuda"] = str(torch.version.cuda)
        versions["cuda_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            versions["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        from importlib.metadata import version
        versions["tabpfn"] = version("tabpfn")
    except Exception:
        pass
    return versions


def save_model_reference(run_dir: Path, spec: RunSpec, model_params: dict[str, Any]) -> None:
    """Small joblib reference; standard TabPFN has no run-specific trained checkpoint."""
    payload = {
        "kind": "reference_only_no_finetuning",
        "model": MODEL_NAME,
        "run_id": spec.id,
        "params": model_params,
        "note": (
            "Standard TabPFN in-context inference does not create dataset-specific trained weights. "
            "This file stores the model/checkpoint reference and fixed parameters only."
        ),
    }
    joblib.dump(payload, run_dir / "model.joblib")


# ------------------------------- summaries --------------------------------


def collect_metrics(out_root: Path, task: str) -> pd.DataFrame:
    base = out_root / MODEL_NAME
    paths = sorted(base.glob(f"*/{task}/*/seed_*/metrics_full.csv"))
    frames = []
    for p in paths:
        try:
            x = pd.read_csv(p)
            x["metrics_path"] = str(p.relative_to(out_root))
            frames.append(x)
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def aggregate_mean_sd(all_metrics: pd.DataFrame) -> pd.DataFrame:
    if all_metrics.empty:
        return pd.DataFrame()
    group_cols = [c for c in ["model", "task", "dataset_name", "genus", "split_type"] if c in all_metrics.columns]
    numeric = [c for c in all_metrics.select_dtypes(include=[np.number]).columns if c not in {"seed"}]
    rows: list[dict[str, Any]] = []
    for keys, g in all_metrics.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_runs"] = len(g)
        for c in numeric:
            row[f"{c}_mean"] = float(g[c].mean())
            row[f"{c}_sd"] = float(g[c].std(ddof=1)) if len(g) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def ood_change_table(all_metrics: pd.DataFrame, task: str) -> pd.DataFrame:
    if all_metrics.empty:
        return pd.DataFrame()
    if task == "classification":
        metric_cols = [
            "test_pr_auc", "test_roc_auc", "test_EF1%", "test_Hit_rate@Top1%",
            "nn_lt_0p4_n", "nn_lt_0p4_error_rate", "nn_lt_0p4_pr_auc",
            "boundary_error_rate", "ac_error_rate",
        ]
    else:
        metric_cols = [
            "test_rmse", "test_mae", "test_spearman", "test_EF1%", "test_Hit_rate@Top1%",
            "nn_lt_0p4_rmse", "nn_lt_0p4_mae", "boundary_mae", "ac_mae",
        ]
    rows: list[dict[str, Any]] = []
    keys = ["dataset_name", "seed"]
    for (ds, seed), g in all_metrics.groupby(keys, dropna=False):
        r = g.loc[g["split_type"] == "random"]
        if len(r) == 0:
            continue
        r0 = r.iloc[0]
        for _, o in g.loc[g["split_type"] != "random"].iterrows():
            row = {"dataset_name": ds, "seed": seed, "ood_split": o["split_type"]}
            for c in metric_cols:
                if c not in all_metrics.columns:
                    continue
                rv, ov = finite_or_nan(r0.get(c)), finite_or_nan(o.get(c))
                row[f"random_{c}"] = rv
                row[f"ood_{c}"] = ov
                if task == "regression" and c in {"test_rmse", "test_mae"}:
                    row[f"increase_{c}"] = ov - rv if np.isfinite(rv) and np.isfinite(ov) else float("nan")
                else:
                    row[f"drop_{c}"] = rv - ov if np.isfinite(rv) and np.isfinite(ov) else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def write_summary_files(out_root: Path, task: str) -> None:
    base = out_root / MODEL_NAME
    base.mkdir(parents=True, exist_ok=True)
    all_metrics = collect_metrics(out_root, task)
    if all_metrics.empty:
        return
    all_metrics.to_csv(base / "all_run_metrics.csv", index=False)
    agg = aggregate_mean_sd(all_metrics)
    agg.to_csv(base / "aggregate_mean_sd.csv", index=False)
    ood = ood_change_table(all_metrics, task)
    ood.to_csv(base / "ood_change_random_vs_ood_splits.csv", index=False)

    combined_dir = out_root / "_combined_summary" / f"{task}_summary_like_ml"
    combined_dir.mkdir(parents=True, exist_ok=True)
    all_metrics.to_csv(combined_dir / "combined_all_run_metrics.csv", index=False)
    agg.to_csv(combined_dir / "leaderboard_mean_sd.csv", index=False)
    ood.to_csv(combined_dir / "ood_change_summary.csv", index=False)
    if task == "classification":
        ood.to_csv(combined_dir / "ood_drop_summary.csv", index=False)
        ood.to_csv(combined_dir / "perimeter_ood_error_analysis.csv", index=False)
    else:
        ood.to_csv(combined_dir / "perimeter_regression_ood_error_analysis.csv", index=False)

    # Single-model benchmark still emits ranking-shaped files for schema compatibility.
    rank_metric = "test_pr_auc" if task == "classification" else "test_rmse"
    ranking = all_metrics[[c for c in ["model", "dataset_name", "split_type", "seed", rank_metric] if c in all_metrics.columns]].copy()
    if rank_metric in ranking:
        ranking["rank"] = 1
    ranking.to_csv(combined_dir / "model_ranking_per_task_seed.csv", index=False)
    ranking.to_csv(combined_dir / "model_ranking_by_metric.csv", index=False)
    pd.DataFrame([{"model": MODEL_NAME, "top_model_count": len(ranking)}]).to_csv(
        combined_dir / "top_model_frequency.csv", index=False
    )


def mark_success(run_dir: Path) -> None:
    (run_dir / "SUCCESS").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"), encoding="utf-8")


def mark_failure(run_dir: Path, exc: BaseException) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "FAILED.txt").write_text(
        f"{type(exc).__name__}: See local execution context for details.\n", encoding="utf-8"
    )
