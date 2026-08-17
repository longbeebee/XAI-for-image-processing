"""Threshold operating-point analysis for anti-spoofing predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrics.classification import error_rates


def operating_points(frame: pd.DataFrame) -> pd.DataFrame:
    """Return all score-derived threshold operating points for a prediction table."""
    required = {"true_label", "p_spoof"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    y_true = frame["true_label"].to_numpy(dtype=int)
    scores = frame["p_spoof"].to_numpy(dtype=float)
    candidates = np.unique(np.concatenate(([0.0, 0.5], scores, [1.0])))
    rows: list[dict[str, Any]] = []
    real = y_true == 0
    spoof = y_true == 1
    for threshold in candidates:
        predicted = (scores >= threshold).astype(int)
        tp = int(np.sum((predicted == 1) & spoof))
        tn = int(np.sum((predicted == 0) & real))
        fp = int(np.sum((predicted == 1) & real))
        fn = int(np.sum((predicted == 0) & spoof))
        rates = error_rates(y_true, scores, float(threshold))
        rows.append(
            {
                "threshold": float(threshold),
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "far": rates["far"],
                "ffr": rates["ffr"],
                "tar": rates["tar"],
                "spoof_detection_rate": rates["spoof_detection_rate"],
                "apcer": rates["apcer"],
                "bpcer": rates["bpcer"],
                "acer": rates["acer"],
                "accuracy": float(np.mean(predicted == y_true)),
            }
        )
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def _nearest(points: pd.DataFrame, threshold: float) -> pd.Series:
    index = (points["threshold"] - float(threshold)).abs().idxmin()
    return points.loc[index]


def _summary(points: pd.DataFrame, selected_threshold: float) -> dict[str, Any]:
    """Build named operating points; test-derived choices are diagnostic only."""
    min_acer = points.loc[points["acer"].idxmin()]
    eer = points.loc[(points["far"] - points["ffr"]).abs().idxmin()]
    summary: dict[str, Any] = {
        "definitions": {
            "far": "spoof accepted as real (APCER)",
            "ffr": "real rejected as spoof (BPCER)",
            "tar": "real accepted as real (1 - FFR)",
            "spoof_detection_rate": "spoof detected as spoof (1 - FAR)",
        },
        "selected_threshold_from_validation": _nearest(points, selected_threshold).to_dict(),
        "diagnostic_min_acer_on_test": min_acer.to_dict(),
        "diagnostic_eer_on_test": eer.to_dict(),
        "target_far_operating_points": {},
        "warning": (
            "The selected threshold must come from validation data. Test-derived minimum ACER "
            "and EER points are diagnostic and must not be used as unbiased final estimates."
        ),
    }
    for target in (0.01, 0.05, 0.10, 0.20):
        eligible = points[points["far"] <= target]
        if eligible.empty:
            continue
        # Highest threshold satisfying the FAR constraint gives the lowest FFR
        # while remaining within the requested FAR budget.
        chosen = eligible.sort_values(["threshold", "ffr"], ascending=[False, True]).iloc[0]
        summary["target_far_operating_points"][f"far_le_{target:.2f}"] = chosen.to_dict()
    return summary


def analyze(
    predictions_path: str | Path,
    output_dir: str | Path,
    selected_threshold: float,
) -> tuple[Path, Path, Path]:
    """Write operating points, JSON summary, and a human-readable analysis note."""
    output = Path(output_dir)
    metrics_dir = output / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(predictions_path)
    points = operating_points(frame)
    points_path = metrics_dir / "threshold_operating_points.csv"
    summary_path = metrics_dir / "threshold_summary.json"
    note_path = output / "threshold_analysis.md"
    points.to_csv(points_path, index=False)
    summary = _summary(points, selected_threshold)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    selected = summary["selected_threshold_from_validation"]
    min_acer = summary["diagnostic_min_acer_on_test"]
    eer = summary["diagnostic_eer_on_test"]
    lines = [
        "# Threshold and anti-spoofing operating-point analysis",
        "",
        "Definitions: FAR = APCER (spoof accepted as real), FFR = BPCER "
        "(real rejected as spoof), TAR = bona-fide acceptance rate (1 - FFR).",
        "",
        "## Validation-selected threshold evaluated on test",
        "",
        f"- Threshold: `{selected['threshold']:.6f}`",
        f"- FAR/APCER: `{selected['far']:.4f}`",
        f"- FFR/BPCER: `{selected['ffr']:.4f}`",
        f"- TAR (real acceptance): `{selected['tar']:.4f}`",
        f"- Spoof detection rate: `{selected['spoof_detection_rate']:.4f}`",
        f"- ACER: `{selected['acer']:.4f}`",
        "",
        "## Diagnostic test-only reference points",
        "",
        "These points are descriptive only; selecting them on test labels would be optimistic.",
        "",
        f"- Minimum test ACER: threshold `{min_acer['threshold']:.6f}`, "
        f"FAR `{min_acer['far']:.4f}`, FFR `{min_acer['ffr']:.4f}`, "
        f"TAR `{min_acer['tar']:.4f}`, ACER `{min_acer['acer']:.4f}`",
        f"- Closest test EER: threshold `{eer['threshold']:.6f}`, "
        f"FAR `{eer['far']:.4f}`, FFR `{eer['ffr']:.4f}`, "
        f"TAR `{eer['tar']:.4f}`, ACER `{eer['acer']:.4f}`",
        "",
        "Full operating points are in `metrics/threshold_operating_points.csv`; "
        "the machine-readable summary is `metrics/threshold_summary.json`.",
    ]
    note_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return points_path, summary_path, note_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selected-threshold", type=float, required=True)
    args = parser.parse_args()
    analyze(args.predictions, args.output_dir, args.selected_threshold)


if __name__ == "__main__":
    main()
