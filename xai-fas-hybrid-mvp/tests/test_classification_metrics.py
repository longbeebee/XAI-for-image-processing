from __future__ import annotations

import numpy as np

from src.metrics.classification import classification_metrics, error_rates


def test_real_spoof_direction_and_error_rates() -> None:
    y_true = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray([0.1, 0.8, 0.2, 0.9])
    rates = error_rates(y_true, probabilities, 0.5)
    assert rates == {"apcer": 0.5, "bpcer": 0.5, "acer": 0.5}
    metrics = classification_metrics(y_true, probabilities, 0.5)
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]

