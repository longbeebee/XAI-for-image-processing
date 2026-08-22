"""Summarize degradation sweep curves and robustness areas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _auc(levels: list[dict[str, Any]], getter) -> float:
    ordered = sorted(levels, key=lambda row: float(row["level"]))
    x = np.asarray([float(row["level"]) for row in ordered])
    y = np.asarray([float(getter(row)) for row in ordered])
    if len(x) < 2 or x[-1] == x[0]: return float(y.mean()) if len(y) else float("nan")
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def summarize(root: Path, output_dir: Path) -> dict[str, Any]:
    records = []
    for path in sorted(root.glob("**/*_sweep.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        levels = data.get("levels", [])
        for row in levels:
            classification = row["classification"]
            xai = row.get("xai_consistency", {})
            records.append({"file": str(path), "model_type": data.get("model_type"), "kind": data.get("kind"), "level": row["level"], "accuracy": classification.get("accuracy"), "acer": classification.get("acer"), "apcer": classification.get("apcer"), "bpcer": classification.get("bpcer"), "roc_auc": classification.get("roc_auc"), "ece": row.get("calibration", {}).get("ece"), "brier": row.get("calibration", {}).get("brier"), "mean_uncertainty": row.get("mean_uncertainty"), "gradcam_cosine": xai.get("gradcam", {}).get("cosine_similarity"), "gradcam_top10_iou": xai.get("gradcam", {}).get("top10_iou"), "ig_cosine": xai.get("integrated_gradients", {}).get("cosine_similarity"), "ig_top10_iou": xai.get("integrated_gradients", {}).get("top10_iou")})
    if not records:
        output_dir.mkdir(parents=True, exist_ok=True)
        empty = {"sweep_count": 0, "curves": [], "robustness_summary": []}
        (output_dir / "degradation_statistics.json").write_text(json.dumps(empty, indent=2) + "\n", encoding="utf-8")
        pd.DataFrame().to_csv(output_dir / "degradation_curves.csv", index=False)
        pd.DataFrame().to_csv(output_dir / "robustness_auc.csv", index=False)
        return empty
    frame = pd.DataFrame(records)
    robustness = []
    for (model_type, kind), group in frame.groupby(["model_type", "kind"], dropna=False):
        levels = group.to_dict("records")
        robustness.append({"model_type": model_type, "kind": kind, "robustness_auc_accuracy": _auc(levels, lambda row: row["accuracy"]), "robustness_auc_one_minus_acer": _auc(levels, lambda row: 1.0 - row["acer"]), "robustness_auc_gradcam_cosine": _auc(levels, lambda row: row["gradcam_cosine"]), "robustness_auc_ig_cosine": _auc(levels, lambda row: row["ig_cosine"])})
    result = {"sweep_count": len(records), "curves": records, "robustness_summary": robustness}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "degradation_statistics.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    frame.to_csv(output_dir / "degradation_curves.csv", index=False)
    pd.DataFrame(robustness).to_csv(output_dir / "robustness_auc.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", required=True, type=Path); parser.add_argument("--output-dir", required=True, type=Path); args = parser.parse_args(); print(json.dumps(summarize(args.root, args.output_dir), indent=2, default=str))


if __name__ == "__main__": main()
