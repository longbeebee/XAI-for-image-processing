"""Resumable metadata creation and deterministic subset preparation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
from PIL import Image

from .config import config_hash, ensure_output_layout, load_config
from .datasets.celeba_spoof_adapter import CelebASpoofAdapter
from .datasets.paths import normalize_relative_path, resolve_dataset_path
from .datasets.subset import create_subsets, validate_no_overlap, write_subsets
from .utils import atomic_json, sha256_file, utc_now, write_status


def _annotation_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.json") if "label" in path.name.lower())


def _load_annotations(paths: list[Path]) -> list[tuple[Path, dict[str, Any]]]:
    loaded: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        try:
            with path.open("r", encoding="utf-8-sig") as stream:
                data = json.load(stream)
            if isinstance(data, dict) and data:
                loaded.append((path, data))
        except (OSError, json.JSONDecodeError):
            continue
    if not loaded:
        raise ValueError("No usable label JSON mapping was discovered.")
    return loaded


def _annotation_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def _records(
    root: Path,
    annotations: list[tuple[Path, dict[str, Any]]],
    adapter: CelebASpoofAdapter,
) -> Iterator[dict[str, Any]]:
    for annotation_path, mapping in annotations:
        name = annotation_path.name.lower()
        split = "test" if "test" in name else "train" if "train" in name else "unknown"
        for raw_path, annotation in mapping.items():
            try:
                relative = normalize_relative_path(str(raw_path))
            except ValueError:
                continue
            subject_id = None
            parts = Path(relative).parts
            if len(parts) >= 3 and parts[-2].isdigit():
                subject_id = parts[-2]
            record = adapter.parse(relative, annotation, split, subject_id)
            image_path = resolve_dataset_path(root, relative)
            if record["is_valid"]:
                if not image_path.is_file():
                    record.update(is_valid=False, error_message="missing_image")
                else:
                    try:
                        with Image.open(image_path) as image:
                            image.verify()
                    except Exception:
                        record.update(is_valid=False, error_message="corrupted_image")
            yield record


def prepare(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    """Build atomic Parquet parts, compact metadata, and reproducible subsets."""
    root = Path(config["paths"]["dataset_root"])
    processed = Path(config["paths"]["processed_dir"])
    output = ensure_output_layout(config)["output"]
    processed.mkdir(parents=True, exist_ok=True)
    parts_dir = processed / "metadata_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    paths = _annotation_files(root)
    annotations = _load_annotations(paths)
    annotation_digest = _annotation_hash(paths)
    current_config_hash = config_hash(config)
    manifest_path = parts_dir / "manifest.json"
    old_manifest = {}
    if manifest_path.exists():
        old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    compatible = (
        old_manifest.get("config_hash") == current_config_hash
        and old_manifest.get("annotation_hash") == annotation_digest
    )
    if not compatible and any(parts_dir.glob("part-*.parquet")) and not force:
        raise RuntimeError("Existing metadata parts do not match config/annotations; use --force.")
    if force:
        for part in parts_dir.glob("part-*.parquet"):
            part.unlink()

    vectors = [
        value
        for _, mapping in annotations
        for value in mapping.values()
        if isinstance(value, list) and len(value) >= 44
    ]
    label_mapping = CelebASpoofAdapter.infer_label_mapping(vectors)
    adapter = CelebASpoofAdapter(label_mapping)
    batch_size = int(config.get("dataset_preparation", {}).get("batch_size", 5000))
    completed = {path.name for path in parts_dir.glob("part-*.parquet")} if compatible else set()
    batch: list[dict[str, Any]] = []
    part_index = 0
    record_count = 0
    for record in _records(root, annotations, adapter):
        batch.append(record)
        if len(batch) >= batch_size:
            name = f"part-{part_index:05d}.parquet"
            if name not in completed:
                target = parts_dir / name
                temporary = target.with_suffix(".parquet.tmp")
                pd.DataFrame(batch).to_parquet(temporary, index=False)
                os.replace(temporary, target)
            record_count += len(batch)
            batch.clear()
            part_index += 1
    if batch:
        name = f"part-{part_index:05d}.parquet"
        if name not in completed:
            target = parts_dir / name
            temporary = target.with_suffix(".parquet.tmp")
            pd.DataFrame(batch).to_parquet(temporary, index=False)
            os.replace(temporary, target)
        record_count += len(batch)

    manifest = {
        "completed_parts": sorted(path.name for path in parts_dir.glob("part-*.parquet")),
        "record_count": record_count,
        "config_hash": current_config_hash,
        "annotation_hash": annotation_digest,
        "label_mapping": label_mapping,
        "last_updated": utc_now(),
    }
    atomic_json(manifest_path, manifest)
    frames = [pd.read_parquet(path) for path in sorted(parts_dir.glob("part-*.parquet"))]
    metadata = pd.concat(frames, ignore_index=True)
    compact = processed / "celeba_spoof_metadata.parquet"
    temporary = compact.with_suffix(".parquet.tmp")
    metadata.to_parquet(temporary, index=False)
    os.replace(temporary, compact)

    subsets = create_subsets(metadata, config["dataset"]["subset"])
    validate_no_overlap(subsets)
    write_subsets(subsets, processed)
    subset_manifest = {
        "seed": config["dataset"]["subset"].get("seed", config.get("seed", 42)),
        "counts": {name: len(frame) for name, frame in subsets.items()},
        "label_counts": {
            name: frame["label"].value_counts().to_dict() for name, frame in subsets.items()
        },
        "label_mapping": label_mapping,
        "metadata_hash": sha256_file(compact),
        "created_at": utc_now(),
    }
    atomic_json(processed / "subset_manifest.json", subset_manifest)
    atomic_json(output / "dataset_manifest.json", manifest)
    atomic_json(output / "subset_manifest.json", subset_manifest)
    leakage = {
        "duplicate_relative_paths": int(metadata["relative_path"].duplicated().sum()),
        "duplicate_image_ids": int(metadata["image_id"].duplicated().sum()),
        "split_overlap": False,
        "subject_id_available": bool(metadata["subject_id"].notna().any()),
    }
    atomic_json(output / "subject_leakage_report.json", leakage)
    return subset_manifest


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    try:
        prepare(config, args.force)
        write_status(
            config["paths"]["output_dir"],
            "dataset",
            "completed",
            config_hash=config_hash(config),
        )
    except Exception as exc:
        write_status(
            config["paths"]["output_dir"],
            "dataset",
            "failed",
            config_hash=config_hash(config),
            error_message=f"{type(exc).__name__}: {exc}",
        )
        raise


if __name__ == "__main__":
    main()

