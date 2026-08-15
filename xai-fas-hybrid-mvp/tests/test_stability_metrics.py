from __future__ import annotations

import numpy as np

from src.metrics.stability import cosine_similarity, pces, spearman_correlation, topk_iou


def test_identical_maps() -> None:
    heatmap = np.arange(100).reshape(10, 10)
    assert np.isclose(cosine_similarity(heatmap, heatmap), 1)
    assert np.isclose(spearman_correlation(heatmap, heatmap), 1)
    assert topk_iou(heatmap, heatmap, 0.1) == 1


def test_zero_and_pces() -> None:
    assert cosine_similarity(np.zeros(4), np.zeros(4)) == 1
    assert pces(np.asarray([0.2, 0.8]), np.asarray([False, True])) == 0.8


def test_constant_map_semantics() -> None:
    constant = np.ones((4, 4))
    different = np.zeros((4, 4))
    varying = np.arange(16).reshape(4, 4)

    assert spearman_correlation(constant, constant) == 1.0
    assert spearman_correlation(constant, different) == 0.0
    assert spearman_correlation(constant, varying) == 0.0

