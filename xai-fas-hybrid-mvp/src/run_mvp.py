"""Gate-protected full MVP orchestration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyze_statistics import analyze
from .analyze_classification import analyze as analyze_classification
from .analyze_thresholds import analyze as analyze_thresholds
from .config import load_config
from .create_provenance import create as create_provenance
from .check_environment import check_environment
from .evaluate_classifier import evaluate as evaluate_classifier
from .evaluate_explanation_stability import evaluate as evaluate_explanations
from .evaluate_faithfulness import evaluate as evaluate_faithfulness
from .evaluate_prediction_stability import evaluate as evaluate_predictions
from .evaluate_runtime import evaluate as evaluate_runtime
from .evaluate_sanity import evaluate as evaluate_sanity
from .export_validation_predictions import export as export_validation_predictions
from .export_explanation_evidence import export as export_explanation_evidence
from .generate_figures import generate as generate_figures
from .generate_report import generate
from .gates import REQUIRED_GATES
from .train import train


def require_gates(output_dir: str | Path) -> None:
    """Prevent full MVP execution unless every required gate is passed."""
    path = Path(output_dir) / "validation_gates.json"
    if not path.is_file():
        raise RuntimeError(f"Validation gates file is missing: {path}")
    records = json.loads(path.read_text(encoding="utf-8"))
    statuses = {record["name"]: record["status"] for record in records["gates"]}
    failed = [name for name in REQUIRED_GATES if statuses.get(name) != "passed"]
    if failed:
        raise RuntimeError("Full MVP blocked by gates: " + ", ".join(failed))


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    require_gates(config["paths"]["output_dir"])
    create_provenance(config)
    check_environment(config)
    train(config, args.resume)
    classification = evaluate_classifier(config)
    export_validation_predictions(config)
    analyze_classification(
        Path(config["paths"]["output_dir"])
        / "predictions"
        / "original_predictions.parquet",
        config["paths"]["output_dir"],
        float(classification["threshold"]),
        int(config.get("statistics", {}).get("bootstrap_iterations", 1000)),
        float(config.get("statistics", {}).get("confidence_level", 0.95)),
        int(config.get("seed", 42)),
    )
    analyze_thresholds(
        Path(config["paths"]["output_dir"])
        / "predictions"
        / "original_predictions.parquet",
        config["paths"]["output_dir"],
        float(classification["threshold"]),
    )
    evaluate_predictions(config)
    evaluate_explanations(config)
    export_explanation_evidence(config)
    evaluate_faithfulness(config)
    evaluate_sanity(config)
    evaluate_runtime(config)
    analyze(config)
    generate_figures(config)
    generate(config)


if __name__ == "__main__":
    main()
