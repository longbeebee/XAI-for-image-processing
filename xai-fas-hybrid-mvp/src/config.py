"""YAML configuration loading and stable hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a configuration file is invalid."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and resolve no paths implicitly."""
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"Configuration file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ConfigError("Top-level YAML value must be a mapping.")
    for section in ("paths", "device", "dataset", "training"):
        if section not in value:
            raise ConfigError(f"Missing required configuration section: {section}")
    return value


def config_hash(config: dict[str, Any]) -> str:
    """Return a deterministic SHA-256 hash of a configuration."""
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ensure_output_layout(config: dict[str, Any]) -> dict[str, Path]:
    """Create standard writable output directories."""
    output = Path(config["paths"]["output_dir"])
    paths = {
        "output": output,
        "status": output / "status",
        "logs": output / "logs",
        "checkpoints": output / "checkpoints",
        "metrics": output / "metrics",
        "figures": output / "figures",
        "predictions": output / "predictions",
        "explanations": output / "explanations",
        "curves": output / "curves",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths

