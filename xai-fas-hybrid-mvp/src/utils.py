"""Atomic files, hashes, status records, and portable checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    """Atomically write JSON in the target directory."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, default=str)
    os.replace(temporary, target)


def write_status(
    output_dir: str | Path,
    stage: str,
    status: str,
    **details: Any,
) -> None:
    """Write a stage status record rather than relying on output existence."""
    now = utc_now()
    record = {
        "stage": stage,
        "status": status,
        "started_at": details.pop("started_at", now),
        "completed_at": now if status in {"passed", "failed", "completed"} else None,
        "config_hash": details.pop("config_hash", None),
        "checkpoint_hash": details.pop("checkpoint_hash", None),
        "dataset_manifest_hash": details.pop("dataset_manifest_hash", None),
        "actual_backend": details.pop("actual_backend", None),
        "error_message": details.pop("error_message", None),
        **details,
    }
    atomic_json(Path(output_dir) / "status" / f"{stage}.json", record)


def portable_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Copy every checkpoint tensor to CPU."""
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def save_checkpoint(path: str | Path, model: torch.nn.Module, **state: Any) -> None:
    """Atomically save a portable checkpoint."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save({"model_state": portable_state_dict(model), **state}, temporary)
    os.replace(temporary, target)

