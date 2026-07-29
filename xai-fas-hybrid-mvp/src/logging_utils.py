"""Consistent console and file logging."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(output_dir: str | Path, name: str) -> logging.Logger:
    """Configure an idempotent stage logger."""
    logger = logging.getLogger(f"xai_fas.{name}")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger

