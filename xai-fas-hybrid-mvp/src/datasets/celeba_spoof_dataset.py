"""PyTorch dataset backed by a portable Parquet manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

from .paths import resolve_dataset_path


def build_transform(image_size: int = 224, training: bool = False) -> Callable[[Image.Image], Any]:
    """Build pickle-safe preprocessing and optional mild training augmentation."""
    operations: list[Any] = []
    if training:
        operations.extend(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(0.1, 0.1, 0.05, 0.02),
            ]
        )
    else:
        operations.append(transforms.Resize((image_size, image_size)))
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return transforms.Compose(operations)


class CelebASpoofDataset(Dataset[dict[str, Any]]):
    """Read RGB, grayscale, or RGBA images without mutating source files."""

    def __init__(
        self,
        metadata: str | Path | pd.DataFrame,
        dataset_root: str | Path,
        image_size: int = 224,
        training: bool = False,
        transform: Callable[[Image.Image], Any] | None = None,
    ) -> None:
        self.frame = (
            pd.read_parquet(metadata) if isinstance(metadata, (str, Path)) else metadata.copy()
        )
        self.dataset_root = Path(dataset_root)
        self.transform = transform or build_transform(image_size, training)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        path = resolve_dataset_path(self.dataset_root, str(row["relative_path"]))
        try:
            with Image.open(path) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                tensor = self.transform(image)
        except Exception as exc:
            raise RuntimeError(f"Cannot read dataset image: {path}") from exc
        return {
            "image": tensor,
            "label": int(row["label_id"]),
            "image_id": str(row["image_id"]),
            "relative_path": str(row["relative_path"]),
            "subject_id": None if pd.isna(row.get("subject_id")) else str(row.get("subject_id")),
            "spoof_type": str(row.get("spoof_type", "")),
            "illumination": str(row.get("illumination", "")),
            "environment": str(row.get("environment", "")),
        }

