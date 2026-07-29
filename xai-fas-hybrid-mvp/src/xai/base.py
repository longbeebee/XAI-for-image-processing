"""Common explainer protocol."""

from __future__ import annotations

from typing import Protocol

import torch

from .types import Explanation


class BaseExplainer(Protocol):
    """Explainer interface shared by Grad-CAM and IG."""

    def explain(self, inputs: torch.Tensor, targets: torch.Tensor) -> list[Explanation]:
        """Explain every item in a batch."""
        raise NotImplementedError
