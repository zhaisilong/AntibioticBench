#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import argparse
import importlib
import inspect
import os
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-root",
        default=os.environ.get("DATA_ROOT"), help="Prepared split data directory (or set DATA_ROOT)"
    )
    args = p.parse_args()
    if not args.data_root:
        p.error("--data-root or DATA_ROOT is required")

    # Allow the same optional source checkout used by the training launchers.
    repo = os.environ.get("UNIMOL_REPO")
    if repo:
        repo_path = Path(repo).expanduser().resolve()
        if not repo_path.is_dir():
            p.error("UNIMOL_REPO must name an existing source directory")
        sys.path.insert(0, str(repo_path))

    print("===== PYTHON =====")
    print(sys.version)

    print("===== IMPORTS =====")
    for name in [
        "torch", "numpy", "pandas", "scipy", "sklearn",
        "rdkit", "unimol_tools"
    ]:
        try:
            mod = importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if name == "unimol_tools" and exc.name == name:
                p.error("unimol_tools is unavailable; install it or set UNIMOL_REPO")
            raise
        print(name, getattr(mod, "__version__", "version-not-exposed"))

    import torch
    print("===== CUDA =====")
    print("available:", torch.cuda.is_available())
    print("device_count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))

    print("===== CPU =====")
    print("os.cpu_count:", os.cpu_count())
    print("SLURM_CPUS_PER_TASK:", os.environ.get("SLURM_CPUS_PER_TASK"))
    print("OMP_NUM_THREADS:", os.environ.get("OMP_NUM_THREADS"))

    print("===== UNIMOL INTERNAL API =====")
    from unimol_tools.tasks import Trainer
    from unimol_tools.models import NNModel
    from unimol_tools.models.unimol import UniMolModel, LinearHead
    from unimol_tools.data.conformer import (
        ConformerGen,
        inner_smi2coords,
        coords2unimol,
    )
    print("Trainer:", inspect.signature(Trainer))
    print("NNModel:", inspect.signature(NNModel))
    print("UniMolModel:", inspect.signature(UniMolModel))
    print("LinearHead:", inspect.signature(LinearHead))
    print("ConformerGen:", inspect.signature(ConformerGen))
    print("inner_smi2coords:", inspect.signature(inner_smi2coords))
    print("coords2unimol:", inspect.signature(coords2unimol))

    print("===== SMALL CONFORMER TEST =====")
    atoms0, coords0, _ = inner_smi2coords(
        "CCO", seed=0, mode="fast", remove_hs=True
    )
    atoms1, coords1, _ = inner_smi2coords(
        "CCO", seed=1, mode="fast", remove_hs=True
    )
    print("seed0_atoms:", atoms0)
    print("seed0_coord_shape:", coords0.shape)
    print("seed1_coord_shape:", coords1.shape)

    print("===== DATA ROOT =====")
    root = Path(args.data_root)
    print("exists:", root.exists())
    datasets = sorted(
        x for x in root.iterdir()
        if x.is_dir()
        and ((x / "classification").is_dir() or (x / "regression").is_dir())
    ) if root.exists() else []
    print("dataset_count:", len(datasets))
    for d in datasets:
        print(" -", d.name)

    print("===== NO-H WEIGHT CHECK =====")
    try:
        from unimol_tools.config import MODEL_CONFIG
        try:
            from unimol_tools.weights import WEIGHT_DIR
        except Exception:
            from unimol_tools.weights import get_weight_dir
            WEIGHT_DIR = get_weight_dir()
        print("model config no-H:", MODEL_CONFIG["weight"]["molecule_no_h"])
        print(
            "weight exists:",
            Path(WEIGHT_DIR, MODEL_CONFIG["weight"]["molecule_no_h"]).exists()
        )
        print("dict:", MODEL_CONFIG["dict"]["molecule_no_h"])
        print(
            "dict exists:",
            Path(WEIGHT_DIR, MODEL_CONFIG["dict"]["molecule_no_h"]).exists()
        )
    except Exception as e:
        print("Could not inspect weight config:", repr(e))

    print("PREFLIGHT_OK")


if __name__ == "__main__":
    main()
