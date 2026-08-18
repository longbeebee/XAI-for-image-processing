"""Dataset for quality-aware training; it never modifies source images."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

from src.datasets.paths import resolve_dataset_path
from src.datasets.perturbations import apply_perturbation


class QualityAwareDataset(Dataset[dict[str, Any]]):
    def __init__(self, metadata: str | Path, dataset_root: str | Path, image_size: int = 224, training: bool = False, seed: int = 42) -> None:
        self.frame = pd.read_parquet(metadata)
        self.root = Path(dataset_root)
        self.training = training
        self.seed = seed
        self.transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        path = resolve_dataset_path(self.root, str(row["relative_path"]))
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        quality = [1.0, 1.0, 1.0]
        if self.training:
            perturbations = [("brightness", 0.8, [0.8, 1.0, 1.0]), ("brightness", 0.6, [0.6, 1.0, 1.0]), ("blur", 1.0, [1.0, 0.5, 1.0]), ("blur", 2.0, [1.0, 0.0, 1.0]), ("jpeg", 40, [1.0, 1.0, 0.4]), ("jpeg", 20, [1.0, 1.0, 0.2])]
            perturbation, value, quality = perturbations[(index + self.seed) % len(perturbations)]
            image = apply_perturbation(image, perturbation, value)
        return {"image": self.transform(image), "label": int(row["label_id"]), "quality": torch.tensor(quality, dtype=torch.float32), "image_id": str(row["image_id"]), "subject_id": str(row["subject_id"])}
