"""Deterministic split-aware subset sampling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _sample_balanced(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    """Sample approximately equal labels while preserving deterministic order."""
    if count <= 0 or frame.empty:
        return frame.iloc[:0].copy()
    pieces: list[pd.DataFrame] = []
    labels = sorted(frame["label_id"].dropna().unique())
    base, remainder = divmod(count, max(len(labels), 1))
    for index, label in enumerate(labels):
        group = frame[frame["label_id"] == label]
        target = min(len(group), base + int(index < remainder))
        pieces.append(group.sample(n=target, random_state=seed + index))
    sampled = pd.concat(pieces) if pieces else frame.iloc[:0]
    missing = min(count - len(sampled), len(frame) - len(sampled))
    if missing > 0:
        remainder_frame = frame.drop(sampled.index)
        sampled = pd.concat(
            [sampled, remainder_frame.sample(n=missing, random_state=seed + 97)]
        )
    return sampled.sort_values("image_id").reset_index(drop=True)


def create_subsets(
    metadata: pd.DataFrame,
    subset_config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """Create train/validation/test subsets without cross-split path overlap."""
    valid = metadata[
        metadata["is_valid"].astype(bool) & metadata["annotation_consistent"].astype(bool)
    ].drop_duplicates("relative_path")
    seed = int(subset_config.get("seed", 42))
    official_test = valid[valid["source_split"].str.lower().eq("test")]
    training_pool = valid[~valid.index.isin(official_test.index)]

    subjects_available = (
        "subject_id" in training_pool
        and training_pool["subject_id"].notna().all()
        and training_pool["subject_id"].astype(str).ne("").all()
    )
    if subjects_available:
        subjects = sorted(training_pool["subject_id"].astype(str).unique())
        subject_series = pd.Series(subjects).sample(frac=1, random_state=seed)
        val_target = int(subset_config.get("val_samples", 1000))
        val_subjects: set[str] = set()
        selected = 0
        for subject in subject_series:
            val_subjects.add(str(subject))
            selected += int((training_pool["subject_id"].astype(str) == str(subject)).sum())
            if selected >= val_target:
                break
        val_pool = training_pool[training_pool["subject_id"].astype(str).isin(val_subjects)]
        train_pool = training_pool[~training_pool["subject_id"].astype(str).isin(val_subjects)]
    else:
        val_count = min(int(subset_config.get("val_samples", 1000)), len(training_pool))
        val_pool = _sample_balanced(training_pool, val_count, seed + 11)
        train_pool = training_pool[
            ~training_pool["image_id"].isin(set(val_pool["image_id"].astype(str)))
        ]

    return {
        "train": _sample_balanced(
            train_pool, int(subset_config.get("train_samples", 10000)), seed
        ),
        "val": _sample_balanced(
            val_pool, int(subset_config.get("val_samples", 1000)), seed + 1
        ),
        "test": _sample_balanced(
            official_test, int(subset_config.get("test_samples", 1000)), seed + 2
        ),
    }


def validate_no_overlap(subsets: dict[str, pd.DataFrame]) -> None:
    """Raise when any relative path occurs in multiple splits."""
    names = list(subsets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = set(subsets[left]["relative_path"]) & set(subsets[right]["relative_path"])
            if overlap:
                raise ValueError(f"{left}/{right} overlap contains {len(overlap)} paths")


def write_subsets(subsets: dict[str, pd.DataFrame], processed_dir: str | Path) -> None:
    """Write subset Parquet files atomically."""
    destination = Path(processed_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for name, frame in subsets.items():
        target = destination / f"{name}_subset.parquet"
        temporary = target.with_suffix(".parquet.tmp")
        frame.to_parquet(temporary, index=False)
        temporary.replace(target)
