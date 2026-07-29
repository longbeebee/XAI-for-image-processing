"""MobileNetV3-Small binary face anti-spoofing model."""

from __future__ import annotations

import torch
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


def build_model(pretrained: bool = True, num_classes: int = 2) -> torch.nn.Module:
    """Create MobileNetV3-Small with logits [real, spoof]."""
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    final = model.classifier[-1]
    model.classifier[-1] = torch.nn.Linear(final.in_features, num_classes)
    return model


def set_training_stage(
    model: torch.nn.Module,
    head_only: bool,
    unfreeze_last_blocks: int = 3,
) -> None:
    """Freeze features or unfreeze only the final requested blocks."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    if not head_only and unfreeze_last_blocks > 0:
        for block in list(model.features)[-unfreeze_last_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True


def find_last_conv_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Find the final Conv2d layer and validate its module type."""
    candidates = [module for module in model.features.modules() if isinstance(module, torch.nn.Conv2d)]
    if not candidates:
        raise ValueError("MobileNet features contain no Conv2d layer.")
    return candidates[-1]

