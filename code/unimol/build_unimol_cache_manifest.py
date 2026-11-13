#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--cache-root", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--expected-datasets", type=int, default=12)
    args = p.parse_args()

    data_root = Path(args.data_root).resolve()
    cache_root = Path(args.cache_root).resolve()
    manifest = Path(args.manifest).resolve()
    manifest.parent.mkdir(parents=True, exist_ok=True)

    datasets = sorted(
        x for x in data_root.iterdir()
        if x.is_dir()
        and ((x / "classification").is_dir() or (x / "regression").is_dir())
    )
    if len(datasets) != args.expected_datasets:
        raise RuntimeError(
            f"Expected {args.expected_datasets} datasets, found {len(datasets)}: "
            f"{[x.name for x in datasets]}"
        )

    rows = []
    for dataset_dir in datasets:
        rows.append({
            "run_index": len(rows),
            "dataset_name": dataset_dir.name,
            "dataset_dir": str(dataset_dir.resolve()),
            "cache_dir": str((cache_root / dataset_dir.name).resolve()),
        })

    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote cache manifest: {manifest.name}")
    print(f"Datasets: {len(rows)}")


if __name__ == "__main__":
    main()
