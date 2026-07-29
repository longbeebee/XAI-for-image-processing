"""CPU-only NPZ explanation cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .types import Explanation


def cache_key(**fields: Any) -> str:
    """Hash checkpoint, image, target, XAI, and preprocessing identity."""
    payload = json.dumps(fields, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def save_cached(path: str | Path, explanation: Explanation, metadata: dict[str, Any]) -> None:
    """Save an explanation with CPU NumPy arrays."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        raw_attribution=explanation.raw_attribution.detach().cpu().numpy(),
        spatial_map=explanation.spatial_map.detach().cpu().numpy(),
        normalized_map=explanation.normalized_map.detach().cpu().numpy(),
        metadata=json.dumps(metadata, sort_keys=True),
    )

