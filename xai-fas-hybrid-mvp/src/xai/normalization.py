"""NaN-safe attribution normalization."""

from __future__ import annotations

import torch


def normalize_heatmap(spatial_map: torch.Tensor) -> tuple[torch.Tensor, str]:
    """Normalize to [0,1], returning a visible status for degenerate maps."""
    value = spatial_map.detach()
    if not torch.isfinite(value).all():
        return torch.zeros_like(value), "degenerate_heatmap"
    minimum, maximum = value.min(), value.max()
    if float(maximum - minimum) <= 1e-8:
        return torch.zeros_like(value), "degenerate_heatmap"
    return (value - minimum) / (maximum - minimum + 1e-8), "ok"

