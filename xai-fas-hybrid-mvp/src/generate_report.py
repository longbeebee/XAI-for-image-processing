"""Generate a conservative Markdown report from available artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ensure_output_layout, load_config
from .utils import write_status


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def generate(config: dict[str, Any]) -> Path:
    """Summarize available results without inventing scientific conclusions."""
    layout = ensure_output_layout(config)
    output = layout["output"]
    environment = _json(output / "environment_report.json")
    subset = _json(output / "subset_manifest.json")
    classification = _json(layout["metrics"] / "classification_metrics.json")
    threshold_summary = _json(layout["metrics"] / "threshold_summary.json")
    backends: set[str] = set()
    for filename in ("explanation_stability.csv", "faithfulness.csv", "sanity.csv"):
        path = layout["metrics"] / filename
        if path.exists():
            frame = pd.read_csv(path)
            if "actual_backend" in frame:
                backends.update(frame["actual_backend"].dropna().astype(str))
    warning = ""
    if len(backends) > 1:
        warning = (
            "\n> **Warning:** Results were generated on different numerical backends and "
            "must not be treated as one homogeneous experiment.\n"
        )
    lines = [
        "# XAI Face Anti-Spoofing MVP Report",
        warning,
        "## Dataset and subset",
        f"Subset manifest: `{json.dumps(subset, ensure_ascii=False)}`" if subset else "Not run.",
        "## Data leakage analysis",
        json.dumps(_json(output / "subject_leakage_report.json"), ensure_ascii=False)
        if (output / "subject_leakage_report.json").exists()
        else "Not run.",
        "## Hardware and software environment",
        f"Selected backend: `{environment.get('selected_backend')}`"
        if environment
        else "Environment check not run.",
        "## Classification results",
        f"`{json.dumps(classification)}`" if classification else "Not run.",
        "## Classification uncertainty and subgroup analysis",
        "See `metrics/classification_confidence_intervals.json`, "
        "`metrics/spoof_type_metrics.csv`, and "
        "`predictions/validation_predictions.parquet` when present.",
        "## Threshold and anti-spoofing operating points",
        (
            "Definitions: FAR = APCER (spoof accepted as real), FFR = BPCER "
            "(real rejected as spoof), TAR = bona-fide acceptance rate (1 - FFR).\n\n"
            f"Validation-selected threshold analysis: `{json.dumps(threshold_summary['selected_threshold_from_validation'])}`\n\n"
            f"Diagnostic test-only minimum ACER: `{json.dumps(threshold_summary['diagnostic_min_acer_on_test'])}`\n\n"
            f"Diagnostic test-only closest EER: `{json.dumps(threshold_summary['diagnostic_eer_on_test'])}`\n\n"
            f"> {threshold_summary['warning']}"
        )
        if threshold_summary
        else "Not run. See `metrics/threshold_operating_points.csv` when present.",
        "## Prediction and explanation stability",
        "See `metrics/prediction_stability.csv`, `metrics/explanation_stability.csv`, and "
        "`metrics/pces_summary.csv` when present.",
        "## Visual explanation evidence",
        "See `explanations/evidence/` for paired original/perturbed panels containing "
        "Grad-CAM and Integrated Gradients overlays, with metrics in "
        "`explanations/evidence/evidence_manifest.csv` when present.",
        "## Faithfulness and sanity",
        "See `metrics/faithfulness.csv` and `metrics/sanity.csv` when present.",
        "## Limitations",
        "- Stability does not imply faithfulness.",
        "- Faithfulness does not imply classifier correctness.",
        "- Passing sanity checks does not prove that an explanation is perfect.",
        "- Grad-CAM and Integrated Gradients are not causal explanations.",
        "- Official results require one homogeneous CUDA run and all validation gates.",
        "## Reproducibility",
        f"Seed: `{config.get('seed', 42)}`. Paths stored in metadata are relative POSIX paths.",
    ]
    target = output / "report.md"
    target.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    return target


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    generate(config)
    write_status(config["paths"]["output_dir"], "report", "completed")


if __name__ == "__main__":
    main()
