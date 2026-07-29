from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from src.datasets import CelebASpoofDataset


def test_rgb_grayscale_rgba_and_unicode_path(mini_dataset: Path) -> None:
    paths = sorted((mini_dataset / "Data").rglob("*.png"))[:3]
    frame = pd.DataFrame(
        [
            {
                "image_id": hashlib.sha256(str(path).encode()).hexdigest(),
                "relative_path": path.relative_to(mini_dataset).as_posix(),
                "label_id": index % 2,
                "subject_id": None,
                "spoof_type": "live" if index % 2 == 0 else "photo",
                "illumination": "normal",
                "environment": "indoor",
            }
            for index, path in enumerate(paths)
        ]
    )
    dataset = CelebASpoofDataset(frame, mini_dataset)
    for index in range(len(dataset)):
        item = dataset[index]
        assert item["image"].shape == (3, 224, 224)
        assert item["label"] in {0, 1}

