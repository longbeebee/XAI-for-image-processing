"""Shared model, image, and explainer evaluation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision import transforms

from .device import DeviceManager
from .models import build_model

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_trained_model(
    config: dict[str, Any], manager: DeviceManager
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load best checkpoint on CPU before moving it to the target backend."""
    checkpoint_path = Path(config["paths"]["output_dir"]) / "checkpoints" / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(False, 2)
    model.load_state_dict(checkpoint["model_state"])
    return manager.move_model(model).eval(), checkpoint


def normalized_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert one ImageNet-normalized tensor back to an RGB PIL image."""
    rgb = (tensor.detach().cpu() * STD + MEAN).clamp(0, 1)
    return transforms.ToPILImage()(rgb)


def pil_to_normalized(image: Image.Image, image_size: int = 224) -> torch.Tensor:
    """Resize and ImageNet-normalize a PIL image."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN.flatten().tolist(), STD.flatten().tolist()),
        ]
    )(image.convert("RGB"))

