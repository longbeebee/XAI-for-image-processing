"""Patch-based deletion and insertion in RGB [0,1] space."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def normalized_auc(fractions: np.ndarray, probabilities: np.ndarray) -> float:
    """Integrate probability over modified area in [0,1]."""
    # ``trapz`` was removed from NumPy 2.0. Keep compatibility with older
    # NumPy versions while preferring the current API.
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapezoid(probabilities, fractions))


def _blur_baseline(image: torch.Tensor) -> torch.Tensor:
    batch = image.unsqueeze(0)
    return F.avg_pool2d(batch, kernel_size=31, stride=1, padding=15).squeeze(0)


def patch_curve(
    image_rgb: torch.Tensor,
    heatmap: torch.Tensor,
    predict: Callable[[torch.Tensor], torch.Tensor],
    target_class: int,
    mode: str,
    patch_size: int = 32,
    steps: int = 10,
    baseline: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a deletion/insertion curve without modifying the input."""
    if mode not in {"deletion", "insertion"}:
        raise ValueError("mode must be deletion or insertion")
    image = image_rgb.detach().cpu().clone()
    if image.ndim != 3 or image.shape[0] != 3 or image.min() < 0 or image.max() > 1:
        raise ValueError("image_rgb must have shape [3,H,W] and range [0,1]")
    if baseline == "mean":
        base = image.mean(dim=(1, 2), keepdim=True).expand_as(image).clone()
    elif baseline == "blur":
        base = _blur_baseline(image)
    else:
        raise ValueError(f"Unsupported baseline: {baseline}")
    _, height, width = image.shape
    patches: list[tuple[float, int, int, int, int]] = []
    for top in range(0, height, patch_size):
        for left in range(0, width, patch_size):
            bottom, right = min(top + patch_size, height), min(left + patch_size, width)
            score = float(heatmap[top:bottom, left:right].mean())
            patches.append((score, top, bottom, left, right))
    patches.sort(key=lambda item: item[0], reverse=True)
    fractions = np.linspace(0.0, 1.0, min(steps, len(patches)) + 1)
    versions: list[torch.Tensor] = []
    for fraction in fractions:
        count = int(round(fraction * len(patches)))
        current = image.clone() if mode == "deletion" else base.clone()
        for _, top, bottom, left, right in patches[:count]:
            replacement = base if mode == "deletion" else image
            current[:, top:bottom, left:right] = replacement[:, top:bottom, left:right]
        versions.append(current)
    batch = torch.stack(versions)
    normalized = (batch - IMAGENET_MEAN) / IMAGENET_STD
    probabilities = predict(normalized).softmax(dim=1)[:, target_class].detach().cpu().numpy()
    return fractions, probabilities
