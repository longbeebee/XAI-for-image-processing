"""Generate core result figures from produced metric files."""

from __future__ import annotations

import argparse
import json
from typing import Any

import pandas as pd

from .config import ensure_output_layout, load_config
from .visualization.plots import plot_confusion_matrix, plot_metric_by_severity


def generate(config: dict[str, Any]) -> None:
    """Generate only figures whose source results exist."""
    layout = ensure_output_layout(config)
    classification = layout["metrics"] / "classification_metrics.json"
    if classification.exists():
        metrics = json.loads(classification.read_text(encoding="utf-8"))
        plot_confusion_matrix(
            metrics["confusion_matrix"], layout["figures"] / "confusion_matrix.png"
        )
    prediction = layout["metrics"] / "prediction_stability.csv"
    if prediction.exists():
        frame = pd.read_csv(prediction)
        plot_metric_by_severity(
            frame, "prediction_stability", layout["figures"] / "prediction_stability.png"
        )
    explanation = layout["metrics"] / "explanation_stability.csv"
    if explanation.exists():
        frame = pd.read_csv(explanation)
        plot_metric_by_severity(
            frame,
            "cosine_similarity",
            layout["figures"] / "explanation_stability.png",
        )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    generate(load_config(args.config))


if __name__ == "__main__":
    main()
