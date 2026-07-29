from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.export_aws_bundle import export_bundle
from src.verify_aws_bundle import verify_bundle


def test_small_shards_verify_and_missing_shard_detected(
    mini_dataset: Path, tmp_path: Path
) -> None:
    image_paths = sorted(path for path in (mini_dataset / "Data").rglob("*.png") if path.is_file())
    processed = tmp_path / "processed"
    processed.mkdir()
    splits = {
        "train": image_paths[:4],
        "val": image_paths[4:6],
        "test": image_paths[6:8],
    }
    for split, paths in splits.items():
        pd.DataFrame(
            {
                "relative_path": [
                    path.relative_to(mini_dataset).as_posix() for path in paths
                ]
            }
        ).to_parquet(processed / f"{split}_subset.parquet", index=False)
    (processed / "subset_manifest.json").write_text(
        json.dumps({"counts": {name: len(paths) for name, paths in splits.items()}}),
        encoding="utf-8",
    )
    config = {
        "paths": {
            "dataset_root": str(mini_dataset),
            "processed_dir": str(processed),
        }
    }
    bundle = tmp_path / "bundle"
    manifest = export_bundle(config, bundle, max_shard_bytes=200)
    assert manifest["number_of_shards"] > 1
    assert verify_bundle(bundle)["status"] == "passed"
    (bundle / "shards" / manifest["shards"][0]["name"]).unlink()
    with pytest.raises(FileNotFoundError):
        verify_bundle(bundle)

