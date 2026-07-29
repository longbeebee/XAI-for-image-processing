"""Captum Integrated Gradients with canonical magnitude maps."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients

from .normalization import normalize_heatmap
from .types import Explanation


class IntegratedGradientsExplainer:
    """Generate signed IG attributions and channel-summed magnitude maps."""

    def __init__(
        self,
        model: torch.nn.Module,
        n_steps: int = 24,
        internal_batch_size: int | None = 8,
        baseline: str = "zero",
    ) -> None:
        self.model = model
        self.n_steps = n_steps
        self.internal_batch_size = internal_batch_size
        self.baseline = baseline
        self.explainer = IntegratedGradients(model)

    def _baseline(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.baseline == "zero":
            return torch.zeros_like(inputs)
        if self.baseline == "mean":
            return inputs.mean(dim=(-2, -1), keepdim=True).expand_as(inputs)
        if self.baseline == "blur":
            return F.avg_pool2d(inputs, kernel_size=31, stride=1, padding=15)
        raise ValueError(f"Unsupported IG baseline: {self.baseline}")

    def explain(self, inputs: torch.Tensor, targets: torch.Tensor) -> list[Explanation]:
        """Explain a batch and expose convergence deltas."""
        attribution, deltas = self.explainer.attribute(
            inputs,
            baselines=self._baseline(inputs),
            target=targets,
            n_steps=self.n_steps,
            internal_batch_size=self.internal_batch_size,
            return_convergence_delta=True,
        )
        spatial = attribution.abs().sum(dim=1)
        results = []
        for index in range(len(inputs)):
            normalized, status = normalize_heatmap(spatial[index])
            results.append(
                Explanation(
                    raw_attribution=attribution[index].detach().cpu(),
                    spatial_map=spatial[index].detach().cpu(),
                    normalized_map=normalized.cpu(),
                    target_class=int(targets[index]),
                    method="integrated_gradients",
                    status=status,
                    convergence_delta=float(deltas[index].detach().cpu()),
                )
            )
        return results

