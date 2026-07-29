"""Canonical explanation result."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Explanation:
    """Attribution and canonical normalized spatial map for one sample."""

    raw_attribution: torch.Tensor
    spatial_map: torch.Tensor
    normalized_map: torch.Tensor
    target_class: int
    method: str
    status: str = "ok"
    convergence_delta: float | None = None

