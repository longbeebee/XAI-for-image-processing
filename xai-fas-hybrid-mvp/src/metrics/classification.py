"""Binary classification metrics with canonical real/spoof semantics."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def error_rates(y_true: np.ndarray, p_spoof: np.ndarray, threshold: float) -> dict[str, float]:
    """Compute APCER, BPCER, and ACER (0=real, 1=spoof)."""
    predicted = (p_spoof >= threshold).astype(int)
    spoof = y_true == 1
    real = y_true == 0
    apcer = float(np.mean(predicted[spoof] == 0)) if spoof.any() else float("nan")
    bpcer = float(np.mean(predicted[real] == 1)) if real.any() else float("nan")
    return {"apcer": apcer, "bpcer": bpcer, "acer": float((apcer + bpcer) / 2)}


def select_threshold(
    y_true: np.ndarray, p_spoof: np.ndarray, strategy: str = "min_acer"
) -> float:
    """Select a threshold using validation data only."""
    if strategy == "fixed_0_5":
        return 0.5
    candidates = np.unique(np.concatenate(([0.0], p_spoof, [1.0])))
    scored = [(error_rates(y_true, p_spoof, float(value))["acer"], float(value)) for value in candidates]
    if strategy == "min_acer":
        return min(scored, key=lambda item: (item[0], abs(item[1] - 0.5)))[1]
    if strategy == "eer":
        differences = []
        for value in candidates:
            rates = error_rates(y_true, p_spoof, float(value))
            differences.append((abs(rates["apcer"] - rates["bpcer"]), float(value)))
        return min(differences)[1]
    raise ValueError(f"Unknown threshold strategy: {strategy}")


def classification_metrics(
    y_true: np.ndarray, p_spoof: np.ndarray, threshold: float
) -> dict[str, Any]:
    """Compute standard and anti-spoofing classification metrics."""
    predicted = (p_spoof >= threshold).astype(int)
    result: dict[str, Any] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, p_spoof)),
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=[0, 1]).tolist(),
    }
    result["roc_auc"] = float(roc_auc_score(y_true, p_spoof)) if len(np.unique(y_true)) > 1 else None
    result.update(error_rates(y_true, p_spoof, threshold))
    return result

