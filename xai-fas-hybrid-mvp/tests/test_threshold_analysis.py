from __future__ import annotations

import pandas as pd

from src.analyze_thresholds import _summary, operating_points


def test_operating_points_expose_far_ffr_and_tar() -> None:
    frame = pd.DataFrame(
        {
            "true_label": [0, 0, 1, 1],
            "p_spoof": [0.1, 0.8, 0.2, 0.9],
        }
    )
    points = operating_points(frame)
    point = points.loc[(points["threshold"] - 0.5).abs().idxmin()]
    assert point["far"] == 0.5
    assert point["ffr"] == 0.5
    assert point["tar"] == 0.5
    assert point["spoof_detection_rate"] == 0.5


def test_target_far_uses_highest_threshold_within_budget() -> None:
    frame = pd.DataFrame(
        {
            "true_label": [0, 0, 1, 1],
            "p_spoof": [0.1, 0.8, 0.2, 0.9],
        }
    )
    points = operating_points(frame)
    summary = _summary(points, selected_threshold=0.5)
    chosen = summary["target_far_operating_points"]["far_le_0.20"]
    assert chosen["threshold"] == 0.2
