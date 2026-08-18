"""Original/degraded image pairs for explanation-consistency training."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

from src.datasets.paths import resolve_dataset_path
from src.datasets.perturbations import apply_perturbation


class PairedQualityDataset(Dataset):
    def __init__(self, metadata: str | Path, dataset_root: str | Path, image_size: int = 224, seed: int = 42) -> None:
        self.frame = pd.read_parquet(metadata); self.root = Path(dataset_root); self.seed = seed
        self.transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def __len__(self): return len(self.frame)

    def __getitem__(self, index):
        row=self.frame.iloc[index]; path=resolve_dataset_path(self.root,str(row["relative_path"]))
        with Image.open(path) as opened: original=ImageOps.exif_transpose(opened).convert("RGB")
        choices=[("brightness",0.6),("blur",1.0),("jpeg",40)]; kind,value=choices[(index+self.seed)%len(choices)]
        degraded=apply_perturbation(original,kind,value)
        return {"original": self.transform(original), "degraded": self.transform(degraded), "label": int(row["label_id"]), "image_id": str(row["image_id"])}
