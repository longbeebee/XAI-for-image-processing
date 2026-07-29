from __future__ import annotations

import pytest

from src.datasets.paths import normalize_relative_path, resolve_dataset_path


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Data/train/a b/ảnh.jpg", "Data/train/a b/ảnh.jpg"),
        ("Data\\train\\a b\\ảnh.jpg", "Data/train/a b/ảnh.jpg"),
    ],
)
def test_portable_relative_paths(raw: str, expected: str) -> None:
    assert normalize_relative_path(raw) == expected


@pytest.mark.parametrize("raw", ["D:/Dataset/image.jpg", "D:\\Dataset\\image.jpg", "/data/a.jpg"])
def test_absolute_paths_rejected(raw: str) -> None:
    with pytest.raises(ValueError):
        normalize_relative_path(raw)


def test_resolve_portable_path() -> None:
    assert str(resolve_dataset_path("/data/root", "Data/train/a.jpg")).endswith(
        "data/root/Data/train/a.jpg"
    )

