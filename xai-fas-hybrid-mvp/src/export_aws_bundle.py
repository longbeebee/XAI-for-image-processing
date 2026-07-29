"""Export only subset images into resumable portable TAR shards."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_config
from .datasets.paths import normalize_relative_path, resolve_dataset_path
from .utils import atomic_json, sha256_file, utc_now


def export_bundle(
    config: dict[str, Any],
    output_dir: str | Path,
    max_shard_bytes: int = 2_000_000_000,
    force: bool = False,
) -> dict[str, Any]:
    """Create manifest and TAR shards without copying unrelated dataset files."""
    destination = Path(output_dir)
    complete_manifest = destination / "bundle_manifest.json"
    if complete_manifest.exists() and not force:
        raise FileExistsError("A complete bundle already exists; use --force.")
    manifests = destination / "manifests"
    shards = destination / "shards"
    manifests.mkdir(parents=True, exist_ok=True)
    shards.mkdir(parents=True, exist_ok=True)
    if force:
        for shard in shards.glob("images-*.tar"):
            shard.unlink()
    processed = Path(config["paths"]["processed_dir"])
    frames = {}
    for split in ("train", "val", "test"):
        source = processed / f"{split}_subset.parquet"
        shutil.copy2(source, manifests / source.name)
        frames[split] = pd.read_parquet(source)
    subset_manifest = processed / "subset_manifest.json"
    shutil.copy2(subset_manifest, manifests / subset_manifest.name)
    combined = pd.concat(frames.values(), ignore_index=True)
    if combined["relative_path"].duplicated().any():
        raise ValueError("Subset contains duplicate paths.")

    shard_records = []
    current_paths: list[str] = []
    current_bytes = 0
    groups: list[list[str]] = []
    root = Path(config["paths"]["dataset_root"])
    for raw in combined["relative_path"]:
        relative = normalize_relative_path(str(raw))
        image = resolve_dataset_path(root, relative)
        if not image.is_file():
            raise FileNotFoundError(image)
        size = image.stat().st_size
        if current_paths and current_bytes + size > max_shard_bytes:
            groups.append(current_paths)
            current_paths, current_bytes = [], 0
        current_paths.append(relative)
        current_bytes += size
    if current_paths:
        groups.append(current_paths)
    for index, paths in enumerate(groups):
        shard = shards / f"images-{index:05d}.tar"
        temporary = shard.with_suffix(".tar.tmp")
        with tarfile.open(temporary, "w") as archive:
            for relative in paths:
                archive.add(resolve_dataset_path(root, relative), arcname=relative, recursive=False)
        os.replace(temporary, shard)
        shard_records.append(
            {
                "name": shard.name,
                "sha256": sha256_file(shard),
                "image_count": len(paths),
                "bytes": shard.stat().st_size,
            }
        )
    checksums = "\n".join(f"{item['sha256']}  {item['name']}" for item in shard_records) + "\n"
    (manifests / "checksums.sha256").write_text(checksums, encoding="utf-8")
    manifest = {
        "bundle_version": 1,
        "created_at": utc_now(),
        "subset_manifest_hash": sha256_file(subset_manifest),
        "number_of_shards": len(shard_records),
        "number_of_images": len(combined),
        "total_bytes": sum(item["bytes"] for item in shard_records),
        "train_count": len(frames["train"]),
        "val_count": len(frames["val"]),
        "test_count": len(frames["test"]),
        "shards": shard_records,
    }
    atomic_json(complete_manifest, manifest)
    return manifest


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-shard-bytes", type=int, default=2_000_000_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    export_bundle(load_config(args.config), args.output_dir, args.max_shard_bytes, args.force)


if __name__ == "__main__":
    main()

