"""Backend-aware timing helpers."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from ..device import DeviceManager


def benchmark(
    function: Callable[[], Any],
    manager: DeviceManager,
    repetitions: int,
    warmup: int = 1,
) -> dict[str, Any]:
    """Benchmark callable with CUDA synchronization only on CUDA."""
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(repetitions):
        manager.synchronize_if_supported()
        started = time.perf_counter()
        function()
        manager.synchronize_if_supported()
        samples.append((time.perf_counter() - started) * 1000)
    values = np.asarray(samples)
    return {
        "requested_device": manager.preferred,
        "actual_device": manager.get_logical_device_name(),
        "fallback_used": manager.preferred not in {"auto", manager.get_logical_device_name()},
        "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "standard_deviation_ms": float(values.std()),
        "sample_count": repetitions,
    }

