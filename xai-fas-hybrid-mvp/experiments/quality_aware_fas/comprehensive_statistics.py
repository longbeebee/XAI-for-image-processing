"""Comprehensive baseline-versus-proposed statistics for the final study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _calibration(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> dict[str, float]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (scores >= left) & (scores <= right if right == 1.0 else scores < right)
        if mask.any():
            ece += float(mask.mean()) * abs(float(scores[mask].mean()) - float(labels[mask].mean()))
    return {"ece": float(ece), "brier": float(brier_score_loss(labels, scores))}


def _uncertainty(labels: np.ndarray, predictions: np.ndarray, uncertainty: np.ndarray) -> dict[str, float]:
    errors = (predictions != labels).astype(int)
    result: dict[str, float] = {
        "mean_uncertainty": float(uncertainty.mean()),
        "error_rate": float(errors.mean()),
        "uncertainty_error_correlation": float(np.corrcoef(uncertainty, errors)[0, 1]) if errors.std() and uncertainty.std() else 0.0,
    }
    if len(np.unique(errors)) > 1:
        result["error_detection_auroc"] = float(roc_auc_score(errors, uncertainty))
        result["error_detection_auprc"] = float(average_precision_score(errors, uncertainty))
    else:
        result["error_detection_auroc"] = float("nan")
        result["error_detection_auprc"] = float("nan")
    order = np.argsort(uncertainty)
    sorted_errors = errors[order]
    coverage = np.arange(1, len(errors) + 1) / len(errors)
    risk = np.cumsum(sorted_errors) / np.arange(1, len(errors) + 1)
    result["aurc"] = float(np.trapezoid(risk, coverage))
    for fraction in (0.8, 0.9, 0.95):
        count = max(1, int(len(errors) * fraction))
        result[f"risk_at_{int(fraction * 100)}pct_coverage"] = float(risk[count - 1])
    return result


def _bootstrap_delta(values: np.ndarray, iterations: int = 2000, seed: int = 42) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    return {
        "mean_delta": float(values.mean()),
        "std_delta": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
    }


def _proposed_records(root: Path) -> list[tuple[str, dict[str, Any], pd.DataFrame]]:
    records = []
    for path in sorted(root.glob("seed_*/final_evaluation/final_evaluation.json")):
        prediction_path = path.parent / "test_predictions.parquet"
        if not prediction_path.is_file():
            raise FileNotFoundError(f"Missing proposed prediction table: {prediction_path}")
        records.append((path.parent.parent.name, _json(path), pd.read_parquet(prediction_path)))
    if not records:
        raise FileNotFoundError("No proposed final_evaluation.json files found.")
    return records


def summarize(baseline_dir: Path, proposed_root: Path, output_dir: Path) -> dict[str, Any]:
    baseline = _json(baseline_dir / "baseline_control_manifest.json")
    baseline_predictions = pd.read_parquet(baseline_dir / "predictions/test_predictions.parquet")
    base_labels = baseline_predictions["true_label"].to_numpy(int)
    base_scores = baseline_predictions["p_spoof"].to_numpy(float)
    base_pred = (base_scores >= float(baseline["threshold"])).astype(int)
    base_calibration = _calibration(base_labels, base_scores)
    base_uncertainty = _uncertainty(base_labels, base_pred, 1.0 - np.maximum(base_scores, 1.0 - base_scores))
    records = _proposed_records(proposed_root)
    proposed_summary = []
    paired = []
    for seed, result, frame in records:
        labels = frame["label_id"].to_numpy(int)
        scores = frame["score"].to_numpy(float)
        predictions = frame["prediction"].to_numpy(int)
        uncertainty = frame["uncertainty"].to_numpy(float)
        proposed_summary.append({
            "seed": seed,
            "classification": result["classification"],
            "calibration": _calibration(labels, scores),
            "uncertainty": _uncertainty(labels, predictions, uncertainty),
            "quality_mean": result.get("quality_mean", []),
            "xai_consistency": result.get("xai_consistency", {}),
            "faithfulness": result.get("faithfulness", {}),
            "sanity": result.get("sanity", []),
            "runtime": result.get("runtime", {}),
        })
        aligned = baseline_predictions[["image_id", "true_label"]].merge(
            frame[["image_id", "prediction"]], on="image_id", how="inner", suffixes=("_baseline", "_proposed")
        )
        paired.append((aligned["prediction_baseline"].to_numpy() != aligned["true_label"].to_numpy()) - (aligned["prediction_proposed"].to_numpy() != aligned["true_label"].to_numpy()))

    base = {
        "classification": baseline["classification"],
        "calibration": base_calibration,
        "uncertainty": base_uncertainty,
        "xai_consistency": baseline.get("xai_consistency", {}),
        "faithfulness": baseline.get("faithfulness", {}),
        "sanity": baseline.get("sanity", {}),
        "runtime": baseline.get("runtime", {}),
    }
    result = {
        "protocol": {"baseline_test_samples": len(base_labels), "proposed_seed_count": len(records), "paired_sample_counts": [len(values) for values in paired]},
        "baseline": base,
        "proposed_runs": proposed_summary,
        "paired_error_reduction": [_bootstrap_delta(values) for values in paired],
        "note": "Descriptive comparison uses one frozen baseline run and three proposed seeds; paired deltas are per-image bootstrap intervals, not per-seed Wilcoxon tests.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comprehensive_statistics.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    rows = []
    for record in proposed_summary:
        row = {"model": "proposed", "seed": record["seed"], **record["classification"], **record["calibration"], **record["uncertainty"]}
        rows.append(row)
    rows.append({"model": "baseline", "seed": "frozen", **base["classification"], **base["calibration"], **base["uncertainty"]})
    pd.DataFrame(rows).to_csv(output_dir / "classification_calibration_uncertainty.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--proposed-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.baseline_dir, args.proposed_root, args.output_dir), indent=2, default=str))


if __name__ == "__main__":
    main()
