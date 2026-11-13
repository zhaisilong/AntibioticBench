#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
import csv
from pathlib import Path

ALLOWED_SPLITS = [
    "random",
    "scaffold",
    "repeated_stratified_perimeter",
    "strict_perimeter",
]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--task", choices=["classification", "regression"], required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--model-dir-name", default="Uni-Mol")
    p.add_argument("--expected-datasets", type=int, default=12)
    args = p.parse_args()

    data_root = Path(args.data_root).resolve()
    out_root = Path(args.out_root).resolve()
    manifest = Path(args.manifest).resolve()
    manifest.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    dataset_dirs = sorted(
        p for p in data_root.iterdir()
        if p.is_dir() and (p / args.task).is_dir()
    )
    for dataset_dir in dataset_dirs:
        task_dir = dataset_dir / args.task
        for split_type in ALLOWED_SPLITS:
            split_dir = task_dir / split_type
            if not split_dir.is_dir():
                continue
            seed_dirs = sorted(
                (p for p in split_dir.glob("seed_*") if p.is_dir()),
                key=lambda x: int(x.name.split("_")[-1]),
            )
            for seed_dir in seed_dirs:
                seed = int(seed_dir.name.split("_")[-1])
                if split_type == "strict_perimeter" and seed != 0:
                    continue
                if split_type != "strict_perimeter" and seed not in {0,1,2,3,4}:
                    continue
                train = seed_dir / "train.csv"
                val = seed_dir / "validation.csv"
                test = seed_dir / "test.csv"
                missing = [str(x) for x in (train, val, test) if not x.is_file()]
                if missing:
                    raise FileNotFoundError(
                        f"Missing split files for {dataset_dir.name}/{args.task}/"
                        f"{split_type}/seed_{seed}: {missing}"
                    )
                out_dir = (
                    out_root / args.model_dir_name / dataset_dir.name /
                    args.task / split_type / f"seed_{seed}"
                )
                rows.append({
                    "run_index": len(rows),
                    "task": args.task,
                    "dataset_name": dataset_dir.name,
                    "split_type": split_type,
                    "seed": seed,
                    "train_csv": str(train.resolve()),
                    "validation_csv": str(val.resolve()),
                    "test_csv": str(test.resolve()),
                    "out_dir": str(out_dir.resolve()),
                })

    if args.expected_datasets and len(dataset_dirs) != args.expected_datasets:
        raise RuntimeError(
            f"Expected {args.expected_datasets} datasets, found {len(dataset_dirs)}: "
            f"{[x.name for x in dataset_dirs]}"
        )

    expected_runs = args.expected_datasets * (5 + 5 + 5 + 1)
    if len(rows) != expected_runs:
        raise RuntimeError(
            f"Expected {expected_runs} runs ({args.expected_datasets} datasets × 16), "
            f"found {len(rows)}. Check split/seed directory completeness."
        )

    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote manifest: {manifest.name}")
    print(f"Datasets: {len(dataset_dirs)}")
    print(f"Runs: {len(rows)}")
    print("strict_perimeter: seed_0 only")

if __name__ == "__main__":
    main()
