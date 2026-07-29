from __future__ import annotations

import torch

from src.metrics.sanity import randomized_model
from src.models import build_model


def test_randomization_does_not_modify_original() -> None:
    model = build_model(False)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    for level in range(4):
        clone = randomized_model(model, level)
        assert clone is not model
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())

