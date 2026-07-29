"""Small reusable Matplotlib plotting helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_confusion_matrix(matrix: list[list[int]], output: str | Path) -> None:
    """Plot a real/spoof confusion matrix without seaborn."""
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(np.asarray(matrix), cmap="Blues")
    axis.set_xticks([0, 1], ["real", "spoof"])
    axis.set_yticks([0, 1], ["real", "spoof"])
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row][column]), ha="center", va="center")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def plot_metric_by_severity(
    frame: pd.DataFrame, metric: str, output: str | Path
) -> None:
    """Plot mean metric by perturbation and severity."""
    figure, axis = plt.subplots(figsize=(7, 4))
    for name, group in frame.groupby("perturbation"):
        summary = group.groupby("severity")[metric].mean()
        axis.plot(summary.index, summary.values, marker="o", label=name)
    axis.set_xlabel("Severity")
    axis.set_ylabel(metric.replace("_", " ").title())
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)

