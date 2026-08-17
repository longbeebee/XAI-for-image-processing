"""Create report provenance from the portable AWS subset bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import config_hash, ensure_output_layout, load_config
from .utils import atomic_json, sha256_file, utc_now, write_status


def create(config: dict[str, Any]) -> dict[str, Any]:
    """Write subset, dataset, and leakage manifests without raw annotations."""
    layout = ensure_output_layout(config)
    processed = Path(config["paths"]["processed_dir"])
    frames = {
        split: pd.read_parquet(processed / f"{split}_subset.parquet")
        for split in ("train", "val", "test")
    }
    combined = pd.concat(frames.values(), ignore_index=True)
    subset_source = processed / "subset_manifest.json"
    if subset_source.exists():
        subset_manifest = json.loads(subset_source.read_text(encoding="utf-8"))
    else:
        subset_manifest = {
            "created_from": "portable_aws_bundle",
            "counts": {name: len(frame) for name, frame in frames.items()},
            "label_counts": {
                name: {str(k): int(v) for k, v in frame["label_id"].value_counts().items()}
                for name, frame in frames.items()
            },
            "created_at": utc_now(),
        }
    atomic_json(layout["output"] / "subset_manifest.json", subset_manifest)
    atomic_json(
        layout["output"] / "dataset_manifest.json",
        {
            "created_from": "portable_aws_bundle",
            "config_hash": config_hash(config),
            "counts": {name: len(frame) for name, frame in frames.items()},
            "parquet_hashes": {
                name: sha256_file(processed / f"{name}_subset.parquet")
                for name in frames
            },
            "created_at": utc_now(),
        },
    )
    split_paths = {name: set(frame["relative_path"].astype(str)) for name, frame in frames.items()}
    split_subjects = {
        name: set(frame["subject_id"].dropna().astype(str))
        for name, frame in frames.items()
        if "subject_id" in frame
    }
    overlap = any(
        split_paths[left] & split_paths[right]
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    )
    subject_overlap = any(
        split_subjects.get(left, set()) & split_subjects.get(right, set())
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    )
    leakage = {
        "duplicate_relative_paths": int(combined["relative_path"].duplicated().sum()),
        "duplicate_image_ids": int(combined["image_id"].duplicated().sum()),
        "split_overlap": bool(overlap),
        "subject_overlap": bool(subject_overlap),
        "subject_id_available": bool(
            "subject_id" in combined and combined["subject_id"].notna().any()
        ),
        "source": "portable_aws_bundle",
    }
    atomic_json(layout["output"] / "subject_leakage_report.json", leakage)
    return leakage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    try:
        create(config)
        write_status(config["paths"]["output_dir"], "provenance", "completed")
    except Exception as exc:
        write_status(
            config["paths"]["output_dir"],
            "provenance",
            "failed",
            error_message=f"{type(exc).__name__}: {exc}",
        )
        raise


if __name__ == "__main__":
    main()
