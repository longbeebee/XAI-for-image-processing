"""Classifier evaluation using a validation-selected checkpoint threshold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import ensure_output_layout, load_config
from .datasets import CelebASpoofDataset
from .device import DeviceManager
from .metrics.classification import classification_metrics
from .models import build_model
from .job_fallback import run_with_device_fallback
from .utils import atomic_json, write_status


def evaluate(config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the official test subset without threshold tuning."""
    layout = ensure_output_layout(config)
    manager = DeviceManager(
        config["device"].get("preferred", "auto"),
        bool(config["device"].get("allow_cpu_fallback", False)),
    )
    checkpoint_path = layout["checkpoints"] / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(False, 2)
    model.load_state_dict(checkpoint["model_state"])
    model = manager.move_model(model).eval()
    dataset = CelebASpoofDataset(
        Path(config["paths"]["processed_dir"]) / "test_subset.parquet",
        config["paths"]["dataset_root"],
        int(config["training"].get("image_size", 224)),
    )
    loader = DataLoader(dataset, batch_size=int(config["training"]["batch_size"]))
    rows = []
    with torch.no_grad():
        for batch in loader:
            logits = model(manager.move_tensor(batch["image"]))
            probabilities = logits.softmax(dim=1).detach().cpu().numpy()
            for index in range(len(probabilities)):
                rows.append(
                    {
                        "image_id": batch["image_id"][index],
                        "relative_path": batch["relative_path"][index],
                        "true_label": int(batch["label"][index]),
                        "p_real": float(probabilities[index, 0]),
                        "p_spoof": float(probabilities[index, 1]),
                    }
                )
    frame = pd.DataFrame(rows)
    target = layout["predictions"] / "original_predictions.parquet"
    frame.to_parquet(target, index=False)
    metrics = classification_metrics(
        frame["true_label"].to_numpy(),
        frame["p_spoof"].to_numpy(),
        float(checkpoint["selected_threshold"]),
    )
    atomic_json(layout["metrics"] / "classification_metrics.json", metrics)
    return metrics


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    try:
        metrics = run_with_device_fallback(config, "prediction", evaluate)
        print(json.dumps(metrics, indent=2))
        write_status(config["paths"]["output_dir"], "classifier", "completed")
    except Exception as exc:
        write_status(
            config["paths"]["output_dir"],
            "classifier",
            "failed",
            error_message=f"{type(exc).__name__}: {exc}",
        )
        raise


if __name__ == "__main__":
    main()
