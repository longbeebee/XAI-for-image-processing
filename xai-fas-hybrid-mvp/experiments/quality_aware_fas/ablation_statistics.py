"""Aggregate the staged ablation results without modifying old outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .comprehensive_statistics import _calibration, _uncertainty


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bootstrap(values: np.ndarray, seed: int = 42, iterations: int = 2000) -> dict[str, float]:
    if len(values) == 0:
        return {"mean": float("nan"), "std": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    boot = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0, "ci_low": float(np.quantile(boot, 0.025)), "ci_high": float(np.quantile(boot, 0.975))}


def summarize(baseline_dir: Path, ablation_root: Path, output_dir: Path) -> dict[str, Any]:
    baseline = _read(baseline_dir / "baseline_control_manifest.json")
    base_frame = pd.read_parquet(baseline_dir / "predictions/test_predictions.parquet")
    base_labels = base_frame["true_label"].to_numpy(int)
    base_scores = base_frame["p_spoof"].to_numpy(float)
    base_pred = (base_scores >= float(baseline["threshold"])).astype(int)
    base_frame["prediction"] = base_pred
    variants: dict[str, Any] = {
        "baseline": {
            "runs": [baseline],
            "classification": baseline.get("classification", {}),
            "xai_consistency": baseline.get("xai_consistency", {}),
            "faithfulness_by_method": baseline.get("faithfulness_by_method", {}),
            "sanity_by_method": baseline.get("sanity_by_method", {}),
            "uncertainty_status": "confidence_proxy",
        }
    }
    paired_rows = []
    for variant in ("stage1_only", "stage1_stage2", "full_model"):
        runs = []
        for path in sorted((ablation_root / variant).glob("seed_*/final_evaluation/final_evaluation.json")):
            result = _read(path)
            frame = pd.read_parquet(path.parent / "test_predictions.parquet")
            labels = frame["label_id"].to_numpy(int)
            scores = frame["score"].to_numpy(float)
            predictions = frame["prediction"].to_numpy(int)
            uncertainty_status = _read(path.parent / "ablation_metadata.json").get("uncertainty_status", "unknown")
            run = {
                "seed": path.parent.parent.name,
                "classification": result.get("classification", {}),
                "calibration": _calibration(labels, scores),
                "uncertainty": _uncertainty(labels, predictions, frame["uncertainty"].to_numpy(float)) if uncertainty_status != "not_calibrated" else None,
                "uncertainty_status": uncertainty_status,
                "xai_consistency": result.get("xai_consistency", {}),
                "faithfulness_by_method": result.get("faithfulness_by_method", {}),
                "sanity_by_method": result.get("sanity_by_method", {}),
                "runtime": result.get("runtime", {}),
            }
            runs.append(run)
            aligned = base_frame[["image_id", "true_label", "prediction"]].merge(frame[["image_id", "prediction"]], on="image_id", suffixes=("_baseline", "_variant"))
            paired_rows.append({"variant": variant, "seed": run["seed"], "error_reduction": float(((aligned["prediction_baseline"] != aligned["true_label"]).astype(int) - (aligned["prediction_variant"] != aligned["true_label"]).astype(int)).mean())})
        variants[variant] = {"runs": runs, "run_count": len(runs), "uncertainty_status": "not_calibrated" if variant == "stage1_only" else "learned_error_aware"}
    summary = {"protocol": {"baseline_test_samples": len(base_frame), "subject_disjoint": True, "seed_count": 3}, "variants": variants, "paired_error_reduction": paired_rows}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ablation_statistics.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    pd.DataFrame(paired_rows).to_csv(output_dir / "ablation_paired_error_reduction.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--ablation-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.baseline_dir, args.ablation_root, args.output_dir), indent=2, default=str))


if __name__ == "__main__":
    main()
