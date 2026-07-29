"""Captum LayerGradCam for MobileNetV3."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from captum.attr import LayerGradCam

from ..models.mobilenet_fas import find_last_conv_layer
from .normalization import normalize_heatmap
from .types import Explanation


class GradCAMExplainer:
    """Generate ReLU Grad-CAM maps against a validated convolutional layer."""

    def __init__(self, model: torch.nn.Module, layer: torch.nn.Module | None = None) -> None:
        self.model = model
        self.layer = layer or find_last_conv_layer(model)
        self.explainer = LayerGradCam(model, self.layer)

    def explain(self, inputs: torch.Tensor, targets: torch.Tensor) -> list[Explanation]:
        """Explain a batch using fixed target classes."""
        attribution = self.explainer.attribute(inputs, target=targets)
        if attribution.ndim != 4:
            raise RuntimeError(f"Grad-CAM layer output must be 4D, got {attribution.shape}")
        upsampled = F.interpolate(
            attribution,
            size=inputs.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        spatial = torch.relu(upsampled.sum(dim=1))
        results = []
        for index in range(len(inputs)):
            normalized, status = normalize_heatmap(spatial[index])
            results.append(
                Explanation(
                    raw_attribution=attribution[index].detach().cpu(),
                    spatial_map=spatial[index].detach().cpu(),
                    normalized_map=normalized.cpu(),
                    target_class=int(targets[index]),
                    method="gradcam",
                    status=status,
                )
            )
        return results

