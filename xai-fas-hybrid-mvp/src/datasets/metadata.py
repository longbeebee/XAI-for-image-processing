"""Canonical CelebA-Spoof metadata schema."""

from __future__ import annotations

import hashlib
from typing import Any

METADATA_COLUMNS = [
    "image_id",
    "relative_path",
    "source_split",
    "subject_id",
    "label",
    "label_id",
    "spoof_type_id",
    "spoof_type",
    "illumination_id",
    "illumination",
    "environment_id",
    "environment",
    "annotation_length",
    "annotation_consistent",
    "is_valid",
    "error_message",
]


def stable_image_id(relative_path: str) -> str:
    """Derive a stable ID from a normalized relative path."""
    return hashlib.sha256(relative_path.encode("utf-8")).hexdigest()


def empty_record(relative_path: str, source_split: str) -> dict[str, Any]:
    """Create a complete invalid record ready for error annotation."""
    return {
        "image_id": stable_image_id(relative_path),
        "relative_path": relative_path,
        "source_split": source_split,
        "subject_id": None,
        "label": None,
        "label_id": None,
        "spoof_type_id": None,
        "spoof_type": None,
        "illumination_id": None,
        "illumination": None,
        "environment_id": None,
        "environment": None,
        "annotation_length": None,
        "annotation_consistent": False,
        "is_valid": False,
        "error_message": None,
    }

