"""Cross-platform dataset path conversion."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath


def normalize_relative_path(value: str | Path) -> str:
    """Normalize a dataset-relative path to POSIX and reject traversal/absolute paths."""
    raw = str(value).replace("\\", "/")
    if re.match(r"^[A-Za-z]:/", raw) or raw.startswith("/"):
        raise ValueError(f"Absolute path is not portable metadata: {value}")
    relative = PurePosixPath(raw)
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"Invalid relative path: {value}")
    return relative.as_posix()


def resolve_dataset_path(dataset_root: str | Path, relative_path: str) -> Path:
    """Resolve portable metadata against a Windows or Linux root."""
    relative = PurePosixPath(normalize_relative_path(relative_path))
    return Path(dataset_root).joinpath(*relative.parts)

