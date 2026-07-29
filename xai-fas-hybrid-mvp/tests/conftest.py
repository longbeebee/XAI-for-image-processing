"""Synthetic CelebA-Spoof fixture; no restricted dataset content."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image


def annotation(spoof_type: int, raw_label: int, length: int = 44) -> list[int]:
    """Create a minimally meaningful CelebA-Spoof annotation."""
    values = [0] * length
    if length >= 44:
        values[40:44] = [spoof_type, 1 if spoof_type else 0, 1 if spoof_type else 0, raw_label]
    return values


@pytest.fixture()
def mini_dataset(tmp_path: Path) -> Path:
    """Create RGB/grayscale/RGBA, corrupt, missing, short, and conflict records."""
    root = tmp_path / "mini_celeba_spoof"
    (root / "Data" / "train").mkdir(parents=True)
    (root / "Data" / "test").mkdir(parents=True)
    train: dict[str, list[int]] = {}
    test: dict[str, list[int]] = {}
    modes = ["RGB", "L", "RGBA", "RGB", "RGB", "RGB", "RGB", "RGB"]
    for index, mode in enumerate(modes):
        split = "train" if index < 6 else "test"
        relative = f"Data/{split}/subject_{index}/image {index}.png"
        path = root.joinpath(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        color = 90 if mode == "L" else (90, 120, 150, 255) if mode == "RGBA" else (90, 120, 150)
        Image.new(mode, (32, 32), color).save(path)
        spoof = 0 if index % 2 == 0 else 1 + index % 2
        (train if split == "train" else test)[relative] = annotation(spoof, int(spoof > 0))
    corrupt = "Data/train/subject_bad/corrupt.png"
    corrupt_path = root.joinpath(*corrupt.split("/"))
    corrupt_path.parent.mkdir(parents=True)
    corrupt_path.write_bytes(b"not an image")
    train[corrupt] = annotation(1, 1)
    train["Data/train/subject_bad/missing.png"] = annotation(1, 1)
    train["Data/train/subject_bad/short.png"] = annotation(1, 1, 12)
    train["Data/train/subject_bad/conflict.png"] = annotation(1, 0)
    (root / "train_label.json").write_text(json.dumps(train), encoding="utf-8")
    (root / "test_label.json").write_text(json.dumps(test), encoding="utf-8")
    return root

