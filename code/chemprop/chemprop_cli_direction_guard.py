#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run Chemprop CLI with explicit metric optimization directions.

Chemprop derives Lightning early-stopping/checkpoint mode from each metric class'
``higher_is_better`` attribute. Some Chemprop/TorchMetrics combinations leave
AUPRC/PRC direction unset, which can make PRC behave as a minimization metric.
This wrapper sets the benchmark-required directions before importing the CLI:

* classification PRC/AUPRC: maximize
* regression RMSE: minimize

It also checks the remaining benchmark metrics for internally consistent
directions. It does not modify files in the installed Chemprop package.
"""
from __future__ import annotations

import json
import sys
from typing import Any, Dict

POLICY_VERSION = "antibioticbench_v11_metric_direction_guard"
EXPECTED_DIRECTIONS = {
    "prc": True,
    "roc": True,
    "accuracy": True,
    "f1": True,
    "binary-mcc": True,
    "r2": True,
    "rmse": False,
    "mae": False,
    "mse": False,
}


def apply_metric_direction_guard() -> Dict[str, Dict[str, Any]]:
    from chemprop.nn import MetricRegistry

    report: Dict[str, Dict[str, Any]] = {}
    missing = []
    for alias, expected in EXPECTED_DIRECTIONS.items():
        try:
            metric_cls = MetricRegistry[alias]
        except Exception:
            missing.append(alias)
            continue
        before = getattr(metric_cls, "higher_is_better", None)
        # Set the class attribute before Chemprop constructs Lightning callbacks.
        setattr(metric_cls, "higher_is_better", expected)
        after = getattr(metric_cls, "higher_is_better", None)
        if bool(after) is not expected:
            raise RuntimeError(
                f"Failed to set metric direction: alias={alias}, expected={expected}, observed={after}"
            )
        report[alias] = {
            "class": f"{metric_cls.__module__}.{metric_cls.__name__}",
            "before": before,
            "after": after,
            "mode": "max" if expected else "min",
        }

    for required in ("prc", "rmse"):
        if required not in report:
            raise RuntimeError(
                f"Installed Chemprop does not expose required MetricRegistry alias: {required}. "
                f"Missing aliases seen by guard: {missing}"
            )

    print(
        "EARLY_STOP_DIRECTION_GUARD="
        + json.dumps(
            {
                "policy_version": POLICY_VERSION,
                "classification": {"tracking_metric": "prc", "mode": "max"},
                "regression": {"tracking_metric": "rmse", "mode": "min"},
                "registry": report,
            },
            ensure_ascii=False,
            default=str,
        ),
        flush=True,
    )
    return report


def main() -> None:
    check_only = len(sys.argv) == 2 and sys.argv[1] == "--check-only"
    apply_metric_direction_guard()
    if check_only:
        return

    # sys.argv remains: wrapper.py train|predict ...; Chemprop parses it normally.
    from chemprop.cli.main import main as chemprop_main

    chemprop_main()


if __name__ == "__main__":
    main()
