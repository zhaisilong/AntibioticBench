#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Precompute one shared Uni-Mol v1 no-H multi-conformer cache per dataset.

The cache is keyed by the exact SMILES strings found across classification and
regression train/validation/test CSVs. Labels are never read into the cache.
All split methods and training seeds reuse the same cached coordinates.

For conf_size=11 and conformer_seed=0, RDKit embedding seeds are 0..10.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import pickle
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


def package_version(name: str):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def choose_smiles_column(df: pd.DataFrame) -> str:
    candidates = [
        "smiles", "SMILES", "smiles_std", "SMILES_std", "canonical_smiles"
    ]
    lower = {str(c).lower(): str(c) for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    raise ValueError(
        f"No SMILES column found. Available columns={list(df.columns)}"
    )


def source_split_files(dataset_dir: Path) -> List[Path]:
    files: List[Path] = []
    for task in ("classification", "regression"):
        task_dir = dataset_dir / task
        if not task_dir.is_dir():
            continue
        for name in ("train.csv", "validation.csv", "test.csv"):
            files.extend(sorted(task_dir.glob(f"*/seed_*/{name}")))
    return sorted(set(x.resolve() for x in files))


def public_source_path(path: Path, dataset_dir: Path) -> str:
    """Describe a split without exposing the local dataset root."""
    try:
        return path.relative_to(dataset_dir).as_posix()
    except ValueError:
        # Resolved symlinks may point outside the dataset root.
        return path.name


def collect_unique_smiles(dataset_dir: Path) -> Tuple[List[str], List[Path]]:
    files = source_split_files(dataset_dir)
    if not files:
        raise FileNotFoundError(
            "No classification/regression split CSVs found in the dataset directory"
        )
    seen = set()
    smiles: List[str] = []
    for path in files:
        df = pd.read_csv(path)
        col = choose_smiles_column(df)
        if df[col].isna().any():
            raise ValueError(f"Missing SMILES in {path}")
        for smi in df[col].astype(str):
            if smi not in seen:
                seen.add(smi)
                smiles.append(smi)
    return smiles, files


def smiles_sha256(smiles: Sequence[str]) -> str:
    h = hashlib.sha256()
    for smi in smiles:
        h.update(smi.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _coordinate_worker(task: Tuple[int, int, str, int, str]):
    molecule_index, conformer_index, smiles, rdkit_seed, mode = task
    # Keep numerical libraries from creating extra threads inside each process.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    try:
        from unimol_tools.data.conformer import inner_smi2coords
        atoms, coordinates, _ = inner_smi2coords(
            smiles,
            seed=int(rdkit_seed),
            mode=mode,
            remove_hs=True,
        )
        coordinates = np.asarray(coordinates, dtype=np.float32)
        is_all_zero = bool((coordinates == 0.0).all())
        is_2d = bool(
            len(coordinates) > 0
            and np.allclose(coordinates[:, 2], 0.0)
        )
        return (
            molecule_index,
            conformer_index,
            atoms,
            coordinates,
            is_all_zero,
            is_2d,
            None,
        )
    except Exception as e:
        return (
            molecule_index,
            conformer_index,
            None,
            None,
            True,
            True,
            type(e).__name__,
        )


def validate_existing_cache(
    cache_dir: Path,
    dataset_name: str,
    conf_size: int,
    conformer_seed: int,
    smiles_hash: str,
) -> bool:
    info_path = cache_dir / "cache_info.json"
    feature_path = cache_dir / "unimol_v1_noh_conformers.pkl"
    completed_path = cache_dir / "completed.flag"
    if not all(x.is_file() for x in (info_path, feature_path, completed_path)):
        return False
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        return (
            info.get("dataset_name") == dataset_name
            and info.get("model_name") == "unimolv1"
            and info.get("remove_hs") is True
            and int(info.get("conf_size", -1)) == conf_size
            and int(info.get("conformer_seed", -1)) == conformer_seed
            and info.get("smiles_sha256") == smiles_hash
        )
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--conf-size", type=int, default=11)
    p.add_argument("--conformer-seed", type=int, default=0)
    p.add_argument("--cpu-workers", type=int, default=15)
    p.add_argument("--mode", choices=["fast", "heavy"], default="fast")
    p.add_argument("--max-atoms", type=int, default=256)
    p.add_argument("--chunksize", type=int, default=8)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    dataset_name = dataset_dir.name

    if args.conf_size < 1:
        raise ValueError("--conf-size must be >= 1")
    if args.cpu_workers < 1:
        raise ValueError("--cpu-workers must be >= 1")

    started = time.time()
    smiles, source_files = collect_unique_smiles(dataset_dir)
    smiles_hash = smiles_sha256(smiles)
    workers = min(
        int(args.cpu_workers),
        os.cpu_count() or int(args.cpu_workers),
    )

    print("=" * 100, flush=True)
    print("Uni-Mol shared 3D cache precomputation", flush=True)
    print(f"dataset_name={dataset_name}", flush=True)
    print(f"unique_smiles={len(smiles)}", flush=True)
    print(f"source_split_files={len(source_files)}", flush=True)
    print(f"remove_hs=True", flush=True)
    print(f"conf_size={args.conf_size}", flush=True)
    print(f"conformer_seed={args.conformer_seed}", flush=True)
    print(
        f"rdkit_seeds={list(range(args.conformer_seed, args.conformer_seed + args.conf_size))}",
        flush=True,
    )
    print(f"cpu_workers={workers}", flush=True)
    print("=" * 100, flush=True)

    if (
        not args.force
        and validate_existing_cache(
            cache_dir, dataset_name, args.conf_size,
            args.conformer_seed, smiles_hash,
        )
    ):
        print("CACHE_ALREADY_COMPLETE_AND_VALID; nothing to do.", flush=True)
        return

    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{dataset_name}.cache_tmp_",
            dir=str(cache_dir.parent),
        )
    )
    try:
        from unimol_tools.data.conformer import ConformerGen, coords2unimol

        # Load the official no-H dictionary once in the parent process.
        feature_generator = ConformerGen(
            seed=args.conformer_seed,
            max_atoms=args.max_atoms,
            data_type="molecule",
            method="rdkit_random",
            mode=args.mode,
            remove_hs=True,
            multi_process=False,
        )
        dictionary = feature_generator.dictionary

        conformers: List[List[Any]] = [
            [None for _ in range(args.conf_size)] for _ in smiles
        ]
        all_zero_counts = np.zeros(len(smiles), dtype=int)
        fallback_2d_counts = np.zeros(len(smiles), dtype=int)
        errors: List[Dict[str, Any]] = []

        def tasks():
            for molecule_index, smi in enumerate(smiles):
                for conformer_index in range(args.conf_size):
                    yield (
                        molecule_index,
                        conformer_index,
                        smi,
                        args.conformer_seed + conformer_index,
                        args.mode,
                    )

        total = len(smiles) * args.conf_size
        done = 0
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers
        ) as executor:
            for result in executor.map(
                _coordinate_worker,
                tasks(),
                chunksize=max(1, args.chunksize),
            ):
                (
                    molecule_index,
                    conformer_index,
                    atoms,
                    coordinates,
                    is_all_zero,
                    is_2d,
                    error,
                ) = result
                if error is not None:
                    errors.append({
                        "molecule_index": molecule_index,
                        "conformer_index": conformer_index,
                        "smiles": smiles[molecule_index],
                        "error": error,
                    })
                    raise RuntimeError(
                        f"Conformer generation failed for molecule_index="
                        f"{molecule_index}, conformer_index={conformer_index}, "
                        f"smiles={smiles[molecule_index]}: {error}"
                    )

                # coords2unimol may crop molecules above max_atoms using
                # NumPy randomness. Seed that rare path deterministically too.
                np_state = np.random.get_state()
                np.random.seed(
                    (
                        args.conformer_seed
                        + conformer_index
                        + molecule_index * 1000003
                    ) % (2**32 - 1)
                )
                try:
                    feat = coords2unimol(
                        atoms,
                        coordinates,
                        dictionary,
                        args.max_atoms,
                        remove_hs=True,
                        data_type="molecule",
                    )
                finally:
                    np.random.set_state(np_state)
                conformers[molecule_index][conformer_index] = feat
                all_zero_counts[molecule_index] += int(is_all_zero)
                fallback_2d_counts[molecule_index] += int(is_2d)

                done += 1
                if done == 1 or done % 500 == 0 or done == total:
                    elapsed = time.time() - started
                    rate = done / elapsed if elapsed > 0 else float("nan")
                    print(
                        f"progress={done}/{total} "
                        f"({100.0*done/total:.2f}%) "
                        f"elapsed_s={elapsed:.1f} rate={rate:.2f}_confs_per_s",
                        flush=True,
                    )

        for i, confs in enumerate(conformers):
            if len(confs) != args.conf_size or any(x is None for x in confs):
                raise RuntimeError(
                    f"Incomplete conformers for molecule index {i}"
                )

        payload = {
            "format_version": 2,
            "dataset_name": dataset_name,
            "smiles": smiles,
            "conformers": conformers,
            "model_name": "unimolv1",
            "remove_hs": True,
            "conf_size": args.conf_size,
            "conformer_seed": args.conformer_seed,
            "conformer_seeds": list(
                range(args.conformer_seed, args.conformer_seed + args.conf_size)
            ),
        }
        feature_path = tmp_dir / "unimol_v1_noh_conformers.pkl"
        with feature_path.open("wb") as f:
            pickle.dump(payload, f, protocol=5)

        molecules_df = pd.DataFrame({
            "molecule_index": np.arange(len(smiles)),
            "smiles": smiles,
            "conformer_count": args.conf_size,
            "all_zero_conformer_count": all_zero_counts,
            "fallback_2d_conformer_count": fallback_2d_counts,
        })
        molecules_df.to_csv(tmp_dir / "molecules.csv", index=False)

        info = {
            "format_version": 2,
            "dataset_name": dataset_name,
            "dataset_dir": dataset_dir.name,
            "model_name": "unimolv1",
            "feature_type": "Uni-Mol_v1_noH_shared_multi_conformer",
            "remove_hs": True,
            "conf_size": args.conf_size,
            "conformer_seed": args.conformer_seed,
            "conformer_seeds": list(
                range(args.conformer_seed, args.conformer_seed + args.conf_size)
            ),
            "conformer_generation_mode": args.mode,
            "max_atoms": args.max_atoms,
            "unique_smiles": len(smiles),
            "smiles_sha256": smiles_hash,
            "source_split_files": [public_source_path(x, dataset_dir) for x in source_files],
            "labels_used_for_cache": False,
            "shared_across": [
                "classification",
                "regression",
                "split_types",
                "training_seeds",
            ],
            "cpu_workers": workers,
            "elapsed_seconds": time.time() - started,
            "all_zero_conformer_total": int(all_zero_counts.sum()),
            "fallback_2d_conformer_total": int(fallback_2d_counts.sum()),
            "software": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": package_version("numpy"),
                "pandas": package_version("pandas"),
                "rdkit": package_version("rdkit"),
                "unimol_tools": package_version("unimol_tools"),
            },
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        }
        (tmp_dir / "cache_info.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (tmp_dir / "completed.flag").write_text(
            f"completed_at_unix={time.time()}\n",
            encoding="utf-8",
        )

        if cache_dir.exists():
            if args.force:
                backup = cache_dir.with_name(
                    cache_dir.name + f".bak_{int(time.time())}"
                )
                cache_dir.rename(backup)
                print("Existing cache moved to a local backup", flush=True)
            else:
                raise FileExistsError(
                    f"Cache directory exists but is not valid/complete: {cache_dir}. "
                    "Use --force only after inspection."
                )
        tmp_dir.rename(cache_dir)
        print("CACHE_COMPLETED", flush=True)
        print(
            f"feature_file_size_GiB="
            f"{(cache_dir/'unimol_v1_noh_conformers.pkl').stat().st_size/1024**3:.3f}",
            flush=True,
        )
        print(f"elapsed_seconds={time.time()-started:.1f}", flush=True)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
