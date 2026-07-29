from __future__ import annotations

import numpy as np
from PIL import Image

from src.datasets.perturbations import apply_perturbation


def test_shifts_are_deterministic_and_non_mutating() -> None:
    image = Image.fromarray(np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3))
    original = np.asarray(image).copy()
    for kind, value in (("brightness", 0.6), ("blur", 2.0), ("jpeg", 20)):
        left = np.asarray(apply_perturbation(image, kind, value))
        right = np.asarray(apply_perturbation(image, kind, value))
        assert left.shape == original.shape
        assert np.array_equal(left, right)
        assert not np.array_equal(left, original)
    assert np.array_equal(np.asarray(image), original)

