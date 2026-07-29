"""Small end-to-end smoke pipeline."""

from __future__ import annotations

import argparse
from typing import Any

from .check_environment import check_environment
from .config import config_hash, load_config
from .evaluate_classifier import evaluate as evaluate_classifier
from .evaluate_explanation_stability import evaluate as evaluate_explanations
from .evaluate_faithfulness import evaluate as evaluate_faithfulness
from .evaluate_prediction_stability import evaluate as evaluate_predictions
from .evaluate_runtime import evaluate as evaluate_runtime
from .evaluate_sanity import evaluate as evaluate_sanity
from .generate_report import generate
from .train import train
from .utils import atomic_json, utc_now


def run(config: dict[str, Any]) -> dict[str, Any]:
    """Execute all smoke stages and stop at the first error."""
    stages = [
        ("environment", lambda: check_environment(config)),
        ("training", lambda: train(config, resume=False)),
        ("classifier", lambda: evaluate_classifier(config)),
        ("prediction_stability", lambda: evaluate_predictions(config)),
        ("explanation_stability", lambda: evaluate_explanations(config)),
        ("faithfulness", lambda: evaluate_faithfulness(config)),
        ("sanity", lambda: evaluate_sanity(config)),
        ("runtime", lambda: evaluate_runtime(config)),
        ("report", lambda: generate(config)),
    ]
    results = []
    for name, action in stages:
        started = utc_now()
        try:
            action()
            results.append({"stage": name, "status": "passed", "started_at": started})
        except Exception as exc:
            results.append(
                {
                    "stage": name,
                    "status": "failed",
                    "started_at": started,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            break
    report = {
        "status": "passed" if len(results) == len(stages) else "failed",
        "config_hash": config_hash(config),
        "stages": results,
    }
    atomic_json(
        f"{config['paths']['output_dir']}/smoke_test_report.json",
        report,
    )
    return report


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    report = run(load_config(args.config))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
