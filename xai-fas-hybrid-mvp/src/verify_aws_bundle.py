"""Validate AWS bundle checksums, TAR paths, counts, and overlaps."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Any

import pandas as pd

from .datasets.paths import normalize_relative_path
from .utils import sha256_file


def verify_bundle(bundle_dir: str | Path) -> dict[str, Any]:
    """Raise on any bundle integrity or portability violation."""
    root = Path(bundle_dir)
    manifest = json.loads((root / "bundle_manifest.json").read_text(encoding="utf-8"))
    declared = manifest["shards"]
    archive_paths: list[str] = []
    for item in declared:
        shard = root / "shards" / item["name"]
        if not shard.is_file():
            raise FileNotFoundError(f"Missing shard: {shard}")
        if sha256_file(shard) != item["sha256"]:
            raise ValueError(f"Checksum mismatch: {shard.name}")
        with tarfile.open(shard, "r") as archive:
            members = [member for member in archive.getmembers() if member.isfile()]
            if len(members) != item["image_count"]:
                raise ValueError(f"Image count mismatch: {shard.name}")
            for member in members:
                archive_paths.append(normalize_relative_path(member.name))
    if len(archive_paths) != len(set(archive_paths)):
        raise ValueError("Duplicate path found across TAR shards.")
    split_paths: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        frame = pd.read_parquet(root / "manifests" / f"{split}_subset.parquet")
        split_paths[split] = set(frame["relative_path"].map(normalize_relative_path))
        if len(frame) != manifest[f"{split}_count"]:
            raise ValueError(f"{split} manifest count mismatch.")
    if (
        split_paths["train"] & split_paths["val"]
        or split_paths["train"] & split_paths["test"]
        or split_paths["val"] & split_paths["test"]
    ):
        raise ValueError("Train/validation/test overlap detected.")
    expected = set().union(*split_paths.values())
    if set(archive_paths) != expected:
        raise ValueError("Archive paths do not exactly match subset manifests.")
    return {
        "status": "passed",
        "number_of_shards": len(declared),
        "number_of_images": len(archive_paths),
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(verify_bundle(args.bundle_dir), indent=2))


if __name__ == "__main__":
    main()

