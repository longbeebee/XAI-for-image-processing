"""Canonical normalized-map stability metrics."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Cosine similarity with defined zero-map behavior."""
    a, b = np.asarray(left, dtype=float).ravel(), np.asarray(right, dtype=float).ravel()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator <= 1e-12:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.dot(a, b) / denominator)


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Spearman correlation, mapping identical constants to one."""
    a, b = np.asarray(left, dtype=float).ravel(), np.asarray(right, dtype=float).ravel()
    if np.allclose(a, b):
        return 1.0
    value = spearmanr(a, b, nan_policy="omit").statistic
    return 0.0 if not np.isfinite(value) else float(value)


def topk_iou(left: np.ndarray, right: np.ndarray, fraction: float) -> float:
    """IoU between deterministic top-k pixel masks."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1].")
    a, b = np.asarray(left).ravel(), np.asarray(right).ravel()
    count = max(1, int(np.ceil(len(a) * fraction)))
    a_indices = np.argpartition(a, -count)[-count:]
    b_indices = np.argpartition(b, -count)[-count:]
    intersection = len(set(a_indices) & set(b_indices))
    union = len(set(a_indices) | set(b_indices))
    return float(intersection / union)


def pces(values: np.ndarray, prediction_unchanged: np.ndarray) -> float:
    """Prediction-conditioned explanation similarity."""
    selected = np.asarray(values, dtype=float)[np.asarray(prediction_unchanged, dtype=bool)]
    return float(np.mean(selected)) if len(selected) else float("nan")

