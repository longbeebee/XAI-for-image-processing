"""Quality-aware, uncertainty-aware MobileNet model for the new experiment."""

from __future__ import annotations

import torch
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class QualityAwareFAS(torch.nn.Module):
    def __init__(self, pretrained: bool = True, quality_dimensions: int = 3, dropout: float = 0.2) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.features = backbone.features
        self.pool = backbone.avgpool
        feature_dim = backbone.classifier[0].in_features
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(feature_dim, 2)
        self.quality_head = torch.nn.Sequential(torch.nn.Linear(feature_dim, 64), torch.nn.Hardswish(), torch.nn.Linear(64, quality_dimensions), torch.nn.Sigmoid())
        self.uncertainty_head = torch.nn.Sequential(torch.nn.Linear(feature_dim, 64), torch.nn.Hardswish(), torch.nn.Linear(64, 1), torch.nn.Softplus())

    def representation(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        features = self.pool(features).flatten(1)
        return self.dropout(features)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        representation = self.representation(images)
        return {"logits": self.classifier(representation), "quality": self.quality_head(representation), "uncertainty": self.uncertainty_head(representation).squeeze(1), "representation": representation}
