"""Differentiable Grad-CAM consistency loss for training."""

from __future__ import annotations

import torch


def differentiable_gradcam(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Return normalized Grad-CAM maps while preserving gradients for the loss."""
    activations = []
    layer = [module for module in model.features.modules() if isinstance(module, torch.nn.Conv2d)][-1]
    handle = layer.register_forward_hook(lambda _module, _inputs, output: activations.append(output))
    outputs = model(images)
    handle.remove()
    target = outputs["logits"].max(dim=1).values.sum()
    gradients = torch.autograd.grad(target, activations[-1], create_graph=True, retain_graph=True)[0]
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    maps = torch.relu((weights * activations[-1]).sum(dim=1))
    maps = torch.nn.functional.interpolate(maps.unsqueeze(1), size=images.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
    flat = maps.flatten(1)
    return flat / (flat.norm(dim=1, keepdim=True) + 1e-8)


def consistency_loss(original_map: torch.Tensor, degraded_map: torch.Tensor) -> torch.Tensor:
    return 1.0 - (original_map * degraded_map).sum(dim=1).mean()
