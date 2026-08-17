"""Classification confidence intervals and per-spoof-type analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrics.classification import classification_metrics


METRIC_NAMES = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "apcer",
    "bpcer",
    "acer",
    "far",
    "ffr",
    "tar",
    "spoof_detection_rate",
)


def confidence_intervals(
    frame: pd.DataFrame,
    threshold: float,
    iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Bootstrap classification metrics by resampling prediction rows."""
    if frame.empty:
        raise ValueError("Cannot bootstrap an empty prediction table.")
    generator = np.random.default_rng(seed)
    values = {name: [] for name in METRIC_NAMES}
    labels = frame["true_label"].to_numpy(dtype=int)
    scores = frame["p_spoof"].to_numpy(dtype=float)
    for _ in range(iterations):
        indices = generator.integers(0, len(frame), size=len(frame))
        metrics = classification_metrics(labels[indices], scores[indices], threshold)
        for name in METRIC_NAMES:
            value = metrics.get(name)
            if value is not None and np.isfinite(value):
                values[name].append(float(value))
    alpha = (1.0 - confidence) / 2.0
    result: dict[str, dict[str, float]] = {}
    for name, samples in values.items():
        if not samples:
            continue
        array = np.asarray(samples, dtype=float)
        result[name] = {
            "estimate": float(classification_metrics(labels, scores, threshold).get(name, np.nan)),
            "bootstrap_mean": float(array.mean()),
            "ci_low": float(np.quantile(array, alpha)),
            "ci_high": float(np.quantile(array, 1.0 - alpha)),
        }
    return result


def per_spoof_type(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Summarize attack acceptance/detection for each spoof type."""
    attacks = frame[frame["true_label"].astype(int) == 1].copy()
    if "spoof_type" not in attacks:
        attacks["spoof_type"] = "unknown"
    attacks["spoof_type"] = attacks["spoof_type"].replace("", "unknown").fillna("unknown")
    rows = []
    for spoof_type, group in attacks.groupby("spoof_type", sort=True):
        scores = group["p_spoof"].to_numpy(dtype=float)
        detected = scores >= threshold
        rows.append(
            {
                "spoof_type": spoof_type,
                "sample_count": int(len(group)),
                "far_apcer": float(np.mean(~detected)),
                "spoof_detection_rate": float(np.mean(detected)),
                "mean_p_spoof": float(scores.mean()),
                "median_p_spoof": float(np.median(scores)),
            }
        )
    return pd.DataFrame(rows)


def analyze(
    predictions_path: str | Path,
    output_dir: str | Path,
    threshold: float,
    iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[Path, Path]:
    """Write classification CI JSON and spoof-type CSV."""
    output = Path(output_dir)
    metrics_dir = output / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(predictions_path)
    ci = confidence_intervals(frame, threshold, iterations, confidence, seed)
    ci_path = metrics_dir / "classification_confidence_intervals.json"
    ci_path.write_text(json.dumps(ci, indent=2), encoding="utf-8")
    type_path = metrics_dir / "spoof_type_metrics.csv"
    per_spoof_type(frame, threshold).to_csv(type_path, index=False)
    return ci_path, type_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    analyze(args.predictions, args.output_dir, args.threshold, args.iterations)


if __name__ == "__main__":
    main()
