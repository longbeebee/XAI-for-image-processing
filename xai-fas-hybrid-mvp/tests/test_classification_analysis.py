from __future__ import annotations

import pandas as pd

from src.analyze_classification import confidence_intervals, per_spoof_type


def test_classification_confidence_intervals_and_spoof_types() -> None:
    frame = pd.DataFrame(
        {
            "true_label": [0, 0, 1, 1, 1, 1],
            "p_spoof": [0.1, 0.2, 0.9, 0.8, 0.3, 0.4],
            "spoof_type": ["live", "live", "mask", "mask", "replay", "replay"],
        }
    )
    ci = confidence_intervals(frame, 0.5, iterations=30, seed=42)
    assert "far" in ci
    assert 0.0 <= ci["far"]["ci_low"] <= ci["far"]["ci_high"] <= 1.0
    by_type = per_spoof_type(frame, 0.5)
    assert set(by_type["spoof_type"]) == {"mask", "replay"}
    assert set(by_type.columns) >= {"far_apcer", "spoof_detection_rate"}
