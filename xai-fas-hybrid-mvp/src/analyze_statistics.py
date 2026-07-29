"""Paired XAI comparison with bootstrap summaries and Holm correction."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ensure_output_layout, load_config
from .metrics.statistics import bootstrap_summary, paired_wilcoxon


def _holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(p_values) - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def analyze(config: dict[str, Any]) -> pd.DataFrame:
    """Compare methods only on exactly paired prediction-unchanged records."""
    layout = ensure_output_layout(config)
    frame = pd.read_csv(layout["metrics"] / "explanation_stability.csv")
    frame = frame[frame["prediction_unchanged"].astype(bool)]
    keys = ["image_id", "perturbation", "severity"]
    rows: list[dict[str, Any]] = []
    for metric in ("cosine_similarity", "spearman_correlation", "top10_iou", "top20_iou"):
        pivot = frame.pivot_table(index=keys, columns="xai_method", values=metric, aggfunc="first")
        if not {"gradcam", "integrated_gradients"}.issubset(pivot.columns):
            continue
        paired = pivot[["gradcam", "integrated_gradients"]].dropna()
        if paired.empty:
            continue
        test = paired_wilcoxon(
            paired["gradcam"].to_numpy(), paired["integrated_gradients"].to_numpy()
        )
        for method in ("gradcam", "integrated_gradients"):
            summary = bootstrap_summary(
                paired[method].to_numpy(),
                int(config.get("statistics", {}).get("bootstrap_iterations", 500)),
                float(config.get("statistics", {}).get("confidence_level", 0.95)),
                int(config.get("seed", 42)),
            )
            rows.append(
                {
                    "metric": metric,
                    "method": method,
                    "paired_sample_count": len(paired),
                    **summary,
                    **test,
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        unique = result.drop_duplicates("metric")
        correction = dict(zip(unique["metric"], _holm(unique["p_value"].tolist())))
        result["holm_adjusted_p_value"] = result["metric"].map(correction)
    result.to_csv(layout["metrics"] / "statistical_tests.csv", index=False)
    return result


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    analyze(load_config(args.config))


if __name__ == "__main__":
    main()

