"""Model parameter-randomization helpers."""

from __future__ import annotations

import copy

import torch


def randomized_model(model: torch.nn.Module, level: int) -> torch.nn.Module:
    """Deep-copy and randomize the configured model scope."""
    if level not in {0, 1, 2, 3}:
        raise ValueError("Randomization level must be 0, 1, 2, or 3.")
    clone = copy.deepcopy(model)
    if level == 0:
        return clone
    modules: list[torch.nn.Module]
    if level == 1:
        modules = [clone.classifier]
    elif level == 2:
        modules = [clone.features[-1], clone.classifier]
    else:
        modules = [clone]
    for root in modules:
        for module in root.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
    return clone

