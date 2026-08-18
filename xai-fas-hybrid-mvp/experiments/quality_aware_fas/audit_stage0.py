"""Audit the frozen baseline without modifying its code or result artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED = (
    "metrics/classification_metrics.json",
    "metrics/prediction_stability.csv",
    "metrics/pces_summary.csv",
    "metrics/faithfulness.csv",
    "metrics/sanity.csv",
    "metrics/runtime.csv",
    "environment_report.json",
    "validation_gates.json",
)


def audit(baseline_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Record a read-only baseline audit in the new experiment output area."""
    missing = [relative for relative in REQUIRED if not (baseline_dir / relative).is_file()]
    classification = {}
    metrics_path = baseline_dir / "metrics/classification_metrics.json"
    if metrics_path.is_file():
        classification = json.loads(metrics_path.read_text(encoding="utf-8"))
    gates_path = baseline_dir / "validation_gates.json"
    gates_passed = None
    if gates_path.is_file():
        gates = json.loads(gates_path.read_text(encoding="utf-8")).get("gates", [])
        gates_passed = all(item.get("status") == "passed" for item in gates)
    result = {
        "stage": "stage0_frozen_baseline_audit",
        "baseline_dir": str(baseline_dir),
        "baseline_artifacts_missing": missing,
        "validation_gates_all_passed": gates_passed,
        "classification_metrics": classification,
        "read_only": True,
        "ready_for_stage1": not missing and gates_passed is True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stage0_baseline_audit.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.baseline_dir, args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["ready_for_stage1"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
