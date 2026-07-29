"""Read-only CelebA-Spoof structure and annotation inspection."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from .config import config_hash, ensure_output_layout, load_config
from .datasets.paths import normalize_relative_path, resolve_dataset_path
from .utils import atomic_json, write_status

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _tree(root: Path, max_depth: int = 3) -> list[str]:
    lines = [root.name + "/"]
    for path in sorted(root.rglob("*")):
        try:
            depth = len(path.relative_to(root).parts)
        except ValueError:
            continue
        if depth <= max_depth:
            suffix = "/" if path.is_dir() else ""
            lines.append("  " * depth + path.name + suffix)
    return lines


def _json_summary(path: Path, root: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
    }
    try:
        with path.open("r", encoding="utf-8-sig") as stream:
            data = json.load(stream)
        summary["root_type"] = type(data).__name__
        summary["record_count"] = len(data) if hasattr(data, "__len__") else None
        sample = next(iter(data.items())) if isinstance(data, dict) and data else None
        summary["sample_key"] = str(sample[0]) if sample else None
        summary["sample_annotation_length"] = (
            len(sample[1]) if sample and isinstance(sample[1], list) else None
        )
        if sample:
            try:
                relative = normalize_relative_path(str(sample[0]))
                summary["sample_image_resolves"] = resolve_dataset_path(root, relative).is_file()
            except ValueError:
                summary["sample_image_resolves"] = False
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
    return summary


def inspect_dataset(config: dict[str, Any]) -> dict[str, Any]:
    """Inspect a dataset incrementally and return a JSON-serializable report."""
    root = Path(config["paths"]["dataset_root"])
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    output = ensure_output_layout(config)["output"]
    counts: Counter[str] = Counter()
    json_files: list[Path] = []
    other_metadata: list[str] = []
    sample_images: list[dict[str, Any]] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            counts[suffix] += 1
            if len(sample_images) < 10:
                record = {"relative_path": path.relative_to(root).as_posix()}
                try:
                    with Image.open(path) as image:
                        image.verify()
                    record["readable"] = True
                except Exception as exc:
                    record.update(readable=False, error=f"{type(exc).__name__}: {exc}")
                sample_images.append(record)
        elif suffix == ".json":
            json_files.append(path)
        elif suffix in {".txt", ".csv", ".parquet"}:
            other_metadata.append(path.relative_to(root).as_posix())
    tree_lines = _tree(root)
    (output / "dataset_tree.txt").write_text("\n".join(tree_lines), encoding="utf-8")
    report = {
        "dataset_root": str(root),
        "image_counts": dict(counts),
        "total_images": sum(counts.values()),
        "json_candidates": [_json_summary(path, root) for path in json_files],
        "other_metadata_files": other_metadata,
        "sample_images": sample_images,
        "tree_max_depth": 3,
        "config_hash": config_hash(config),
    }
    atomic_json(output / "dataset_inspection.json", report)
    return report


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    try:
        inspect_dataset(config)
        write_status(
            config["paths"]["output_dir"],
            "dataset",
            "passed",
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

