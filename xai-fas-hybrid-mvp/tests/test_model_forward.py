from __future__ import annotations

import torch

from src.models import build_model


def test_model_output_convention() -> None:
    model = build_model(False).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 224, 224))
    assert output.shape == (1, 2)

