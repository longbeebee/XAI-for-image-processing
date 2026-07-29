"""Visible whole-job DirectML-to-CPU fallback."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Callable, TypeVar

import pandas as pd

from .device import DeviceManager
from .utils import utc_now

T = TypeVar("T")


def run_with_device_fallback(
    config: dict[str, Any],
    job_name: str,
    action: Callable[[dict[str, Any]], T],
) -> T:
    """Retry an entire failed DirectML job on CPU and record the transition."""
    manager = DeviceManager(
        config["device"].get("preferred", "auto"),
        bool(config["device"].get("allow_cpu_fallback", False)),
    )
    try:
        return action(config)
    except Exception as exc:
        if (
            manager.get_logical_device_name() != "directml"
            or not bool(config["device"].get("allow_cpu_fallback", False))
        ):
            raise
        manager.release_resources()
        fallback_config = copy.deepcopy(config)
        fallback_config["device"]["preferred"] = "cpu"
        fallback_config["device"]["allow_cpu_fallback"] = False
        metrics_dir = Path(config["paths"]["output_dir"]) / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        target = metrics_dir / "device_fallbacks.csv"
        row = pd.DataFrame(
            [
                {
                    "timestamp": utc_now(),
                    "job": job_name,
                    "requested_device": config["device"].get("preferred"),
                    "failed_actual_device": "directml",
                    "actual_device": "cpu_fallback",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            ]
        )
        existing = pd.read_csv(target) if target.exists() else pd.DataFrame()
        temporary = target.with_suffix(".csv.tmp")
        pd.concat([existing, row], ignore_index=True).to_csv(temporary, index=False)
        os.replace(temporary, target)
        return action(fallback_config)

