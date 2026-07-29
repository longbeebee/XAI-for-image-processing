"""Prediction stability under brightness, blur, and JPEG shifts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import ensure_output_layout, load_config
from .datasets import CelebASpoofDataset
from .datasets.perturbations import apply_perturbation
from .device import DeviceManager
from .evaluation_utils import load_trained_model, normalized_to_pil, pil_to_normalized
from .job_fallback import run_with_device_fallback
from .utils import write_status


def evaluate(config: dict[str, Any]) -> pd.DataFrame:
    """Evaluate unchanged predictions and confidence deltas for quality shifts."""
    layout = ensure_output_layout(config)
    manager = DeviceManager(
        config["device"].get("preferred", "auto"),
        bool(config["device"].get("allow_cpu_fallback", False)),
    )
    model, checkpoint = load_trained_model(config, manager)
    threshold = float(checkpoint["selected_threshold"])
    dataset = CelebASpoofDataset(
        Path(config["paths"]["processed_dir"]) / "test_subset.parquet",
        config["paths"]["dataset_root"],
        int(config["training"].get("image_size", 224)),
    )
    limit = min(len(dataset), int(config["evaluation_limits"]["prediction_stability_samples"]))
    rows: list[dict[str, Any]] = []
    for index in range(limit):
        item = dataset[index]
        original = item["image"].unsqueeze(0)
        with torch.no_grad():
            original_probability = float(
                model(manager.move_tensor(original)).softmax(1)[0, 1].detach().cpu()
            )
        original_prediction = int(original_probability >= threshold)
        pil = normalized_to_pil(item["image"])
        for perturbation, severities in config["perturbations"].items():
            for severity, value in severities.items():
                shifted = pil_to_normalized(
                    apply_perturbation(pil, perturbation, value),
                    int(config["training"].get("image_size", 224)),
                ).unsqueeze(0)
                with torch.no_grad():
                    shifted_probability = float(
                        model(manager.move_tensor(shifted)).softmax(1)[0, 1].detach().cpu()
                    )
                shifted_prediction = int(shifted_probability >= threshold)
                rows.append(
                    {
                        "image_id": item["image_id"],
                        "relative_path": item["relative_path"],
                        "true_label": item["label"],
                        "spoof_type": item["spoof_type"],
                        "illumination": item["illumination"],
                        "environment": item["environment"],
                        "perturbation": perturbation,
                        "severity": int(severity),
                        "original_prediction": original_prediction,
                        "shifted_prediction": shifted_prediction,
                        "original_probability": original_probability,
                        "shifted_probability": shifted_probability,
                        "confidence_delta": shifted_probability - original_probability,
                        "prediction_unchanged": original_prediction == shifted_prediction,
                        "requested_backend": manager.preferred,
                        "actual_backend": manager.get_logical_device_name(),
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_parquet(layout["predictions"] / "shifted_predictions.parquet", index=False)
    summary = (
        frame.groupby(["perturbation", "severity"], as_index=False)["prediction_unchanged"]
        .mean()
        .rename(columns={"prediction_unchanged": "prediction_stability"})
    )
    summary.to_csv(layout["metrics"] / "prediction_stability.csv", index=False)
    return frame


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    try:
        run_with_device_fallback(config, "prediction_stability", evaluate)
        write_status(config["paths"]["output_dir"], "prediction_stability", "completed")
    except Exception as exc:
        write_status(
            config["paths"]["output_dir"],
            "prediction_stability",
            "failed",
            error_message=f"{type(exc).__name__}: {exc}",
        )
        raise


if __name__ == "__main__":
    main()
