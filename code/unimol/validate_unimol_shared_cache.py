#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--cache-root",
        default=os.environ.get("SHARED_CACHE_ROOT", str(Path(__file__).resolve().parents[2] / "outputs/unimol/shared_3d_cache/UniMol_v1_noH_conf11_seed0")),
    )
    p.add_argument("--expected-datasets", type=int, default=12)
    p.add_argument("--conf-size", type=int, default=11)
    p.add_argument("--conformer-seed", type=int, default=0)
    p.add_argument(
        "--deep",
        action="store_true",
        help="Also open every pickle and validate each molecule's conformer count.",
    )
    args = p.parse_args()

    root = Path(args.cache_root).resolve()
    dataset_dirs = sorted(
        x for x in root.iterdir()
        if x.is_dir() and not x.name.startswith("_") and not x.name.startswith(".")
    )
    failures = []
    for d in dataset_dirs:
        info_path = d / "cache_info.json"
        pkl_path = d / "unimol_v1_noh_conformers.pkl"
        flag_path = d / "completed.flag"
        if not all(x.is_file() for x in (info_path, pkl_path, flag_path)):
            failures.append(f"{d.name}: missing required cache files")
            continue
        info = json.loads(info_path.read_text(encoding="utf-8"))
        expected = {
            "dataset_name": d.name,
            "model_name": "unimolv1",
            "remove_hs": True,
            "conf_size": args.conf_size,
            "conformer_seed": args.conformer_seed,
        }
        for key, value in expected.items():
            if info.get(key) != value:
                failures.append(
                    f"{d.name}: {key}={info.get(key)!r}, expected={value!r}"
                )
        if args.deep:
            with pkl_path.open("rb") as f:
                payload = pickle.load(f)
            smiles = payload["smiles"]
            conformers = payload["conformers"]
            if len(smiles) != len(conformers):
                failures.append(f"{d.name}: smiles/conformers length mismatch")
            for i, confs in enumerate(conformers):
                if len(confs) != args.conf_size:
                    failures.append(
                        f"{d.name}: molecule {i} has {len(confs)} conformers"
                    )
                    break
        print(
            f"OK {d.name}: molecules={info.get('unique_smiles')} "
            f"conf_size={info.get('conf_size')} "
            f"file_GiB={pkl_path.stat().st_size/1024**3:.3f}"
        )

    if len(dataset_dirs) != args.expected_datasets:
        failures.append(
            f"dataset cache count={len(dataset_dirs)}, "
            f"expected={args.expected_datasets}"
        )
    if failures:
        print("\nCACHE VALIDATION FAILED:")
        for x in failures:
            print(" -", x)
        raise SystemExit(1)
    print(
        f"\nCACHE_VALIDATION_OK datasets={len(dataset_dirs)} "
        f"conf_size={args.conf_size} conformer_seed={args.conformer_seed}"
    )


if __name__ == "__main__":
    main()
