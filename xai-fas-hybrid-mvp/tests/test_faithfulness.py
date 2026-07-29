from __future__ import annotations

import torch

from src.metrics.faithfulness import normalized_auc, patch_curve


def test_patch_curve_shape_range_and_input_unchanged() -> None:
    image = torch.rand(3, 32, 32)
    original = image.clone()
    heatmap = torch.rand(32, 32)

    def predict(batch: torch.Tensor) -> torch.Tensor:
        value = batch.mean(dim=(1, 2, 3))
        return torch.stack([-value, value], dim=1)

    fractions, probabilities = patch_curve(
        image, heatmap, predict, 1, "deletion", patch_size=8, steps=5
    )
    assert len(fractions) == 6
    assert fractions[0] == 0 and fractions[-1] == 1
    assert 0 <= normalized_auc(fractions, probabilities) <= 1
    assert torch.equal(image, original)

