from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.prepare_celeba_spoof import prepare


def test_fixture_inspection_metadata_and_subset(mini_dataset: Path, tmp_path: Path) -> None:
    config = {
        "seed": 42,
        "paths": {
            "dataset_root": str(mini_dataset),
            "processed_dir": str(tmp_path / "processed"),
            "output_dir": str(tmp_path / "results"),
            "cache_dir": str(tmp_path / "cache"),
        },
        "device": {"preferred": "cpu", "allow_cpu_fallback": False},
        "dataset": {
            "name": "celeba_spoof",
            "subset": {
                "train_samples": 3,
                "val_samples": 1,
                "test_samples": 2,
                "seed": 42,
            },
        },
        "dataset_preparation": {"batch_size": 3},
        "model": {"pretrained": False, "num_classes": 2},
        "training": {"batch_size": 2},
    }
    manifest = prepare(config)
    metadata = pd.read_parquet(tmp_path / "processed" / "celeba_spoof_metadata.parquet")
    assert len(metadata) >= 10
    assert (metadata["relative_path"].str.contains(r"^[A-Za-z]:", regex=True)).sum() == 0
    assert "missing_image" in set(metadata["error_message"].dropna())
    assert "corrupted_image" in set(metadata["error_message"].dropna())
    assert sum(manifest["counts"].values()) > 0

