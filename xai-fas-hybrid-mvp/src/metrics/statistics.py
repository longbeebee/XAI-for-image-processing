"""Bootstrap confidence intervals and paired tests."""

from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def bootstrap_summary(
    values: np.ndarray, iterations: int = 500, confidence: float = 0.95, seed: int = 42
) -> dict[str, float]:
    """Summarize finite values with a percentile bootstrap CI."""
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if not len(data):
        raise ValueError("Cannot summarize an empty finite sample.")
    generator = np.random.default_rng(seed)
    means = np.asarray(
        [generator.choice(data, len(data), replace=True).mean() for _ in range(iterations)]
    )
    alpha = (1 - confidence) / 2
    return {
        "mean": float(data.mean()),
        "standard_deviation": float(data.std(ddof=1)) if len(data) > 1 else 0.0,
        "median": float(np.median(data)),
        "ci_low": float(np.quantile(means, alpha)),
        "ci_high": float(np.quantile(means, 1 - alpha)),
    }


def paired_wilcoxon(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    """Run a paired Wilcoxon test on finite pairs."""
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if not valid.any():
        raise ValueError("No finite pairs are available.")
    if np.allclose(a[valid], b[valid]):
        return {"statistic": 0.0, "p_value": 1.0}
    result = wilcoxon(a[valid], b[valid])
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}
