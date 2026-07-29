"""Create and update auditable validation-gate records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import config_hash, load_config
from .gates import REQUIRED_GATES
from .utils import atomic_json, utc_now


def update_gate(
    output_dir: str | Path,
    name: str,
    status: str,
    current_config_hash: str | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    """Set one gate while preserving every other gate."""
    if name not in REQUIRED_GATES:
        raise ValueError(f"Unknown gate: {name}")
    if status not in {"not_run", "passed", "failed"}:
        raise ValueError(f"Invalid gate status: {status}")
    target = Path(output_dir) / "validation_gates.json"
    if target.exists():
        document = json.loads(target.read_text(encoding="utf-8"))
    else:
        document = {
            "gates": [
                {
                    "name": gate,
                    "status": "not_run",
                    "started_at": None,
                    "completed_at": None,
                    "config_hash": None,
                    "error_message": None,
                }
                for gate in REQUIRED_GATES
            ]
        }
    for record in document["gates"]:
        if record["name"] == name:
            now = utc_now()
            record.update(
                {
                    "status": status,
                    "started_at": record["started_at"] or now,
                    "completed_at": now if status in {"passed", "failed"} else None,
                    "config_hash": current_config_hash,
                    "error_message": error_message,
                }
            )
    atomic_json(target, document)
    return document


def main() -> None:
    """CLI entry point for CI/AWS scripts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--status", choices=["not_run", "passed", "failed"], required=True)
    parser.add_argument("--error-message")
    args = parser.parse_args()
    config = load_config(args.config)
    update_gate(
        config["paths"]["output_dir"],
        args.gate,
        args.status,
        config_hash(config),
        args.error_message,
    )


if __name__ == "__main__":
    main()
