"""Create a subject-disjoint protocol without touching the frozen manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    """Load full metadata or the subset manifests shipped in the baseline bundle."""
    if metadata_path.is_file():
        return pd.read_parquet(metadata_path)
    parent = metadata_path.parent
    parts = [parent / "train_subset.parquet", parent / "val_subset.parquet", parent / "test_subset.parquet"]
    available = [path for path in parts if path.is_file()]
    if len(available) != len(parts):
        raise FileNotFoundError(f"Need full metadata or all subset Parquets under {parent}")
    return pd.concat([pd.read_parquet(path) for path in available], ignore_index=True)


def make_protocol(metadata_path: Path, output_dir: Path, seed: int = 42) -> dict:
    frame = load_metadata(metadata_path)
    required = {"subject_id", "label_id", "relative_path", "image_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Metadata is missing required columns: {missing}")
    if frame["subject_id"].isna().any():
        raise ValueError("Subject-disjoint protocol requires subject_id for every row.")
    frame = frame.drop_duplicates("relative_path").reset_index(drop=True)
    groups = frame["subject_id"].astype(str)
    first, rest = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(frame, frame["label_id"], groups))
    train_val, test = frame.iloc[first].copy(), frame.iloc[rest].copy()
    groups_tv = train_val["subject_id"].astype(str)
    train_idx, val_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=seed + 1).split(train_val, train_val["label_id"], groups_tv))
    splits = {"train": train_val.iloc[train_idx], "val": train_val.iloc[val_idx], "test": test}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_parquet(output_dir / f"{name}_subject_disjoint.parquet", index=False)
    overlap = set(splits["train"].subject_id) & set(splits["val"].subject_id) | set(splits["train"].subject_id) & set(splits["test"].subject_id) | set(splits["val"].subject_id) & set(splits["test"].subject_id)
    manifest = {"seed": seed, "subject_disjoint": not overlap, "counts": {k: len(v) for k, v in splits.items()}, "label_counts": {k: v["label_id"].value_counts().to_dict() for k, v in splits.items()}, "subject_counts": {k: int(v["subject_id"].nunique()) for k, v in splits.items()}, "overlap_count": len(overlap)}
    (output_dir / "subject_disjoint_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(make_protocol(args.metadata, args.output_dir, args.seed), indent=2))
