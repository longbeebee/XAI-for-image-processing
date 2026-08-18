"""Losses used by staged quality-aware experiments."""

from __future__ import annotations

import torch


def quality_aware_loss(outputs: dict[str, torch.Tensor], labels: torch.Tensor, quality_targets: torch.Tensor, quality_weight: float = 0.2, consistency_weight: float = 0.0, uncertainty_weight: float = 0.0) -> tuple[torch.Tensor, dict[str, float]]:
    classification = torch.nn.functional.cross_entropy(outputs["logits"], labels)
    quality = torch.nn.functional.mse_loss(outputs["quality"], quality_targets)
    consistency = outputs["representation"].new_zeros(())
    probabilities = outputs["logits"].softmax(dim=1)[:, 1]
    error_target = (probabilities.detach() - labels.float()).abs().clamp(0.0, 1.0)
    uncertainty_score = 1.0 - torch.exp(-outputs["uncertainty"])
    uncertainty = torch.nn.functional.mse_loss(uncertainty_score, error_target)
    total = classification + quality_weight * quality + consistency_weight * consistency + uncertainty_weight * uncertainty
    return total, {"classification": float(classification.detach()), "quality": float(quality.detach()), "consistency": float(consistency.detach()), "uncertainty": float(uncertainty.detach()), "total": float(total.detach())}
