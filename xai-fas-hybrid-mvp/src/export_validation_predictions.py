"""Export validation-set probabilities used for threshold selection."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import ensure_output_layout, load_config
from .datasets import CelebASpoofDataset
from .device import DeviceManager
from .evaluation_utils import load_trained_model
from .job_fallback import run_with_device_fallback
from .utils import write_status


def export(config: dict[str, Any]) -> pd.DataFrame:
    """Run the best checkpoint on validation data and save probabilities."""
    layout = ensure_output_layout(config)
    manager = DeviceManager(
        config["device"].get("preferred", "auto"),
        bool(config["device"].get("allow_cpu_fallback", False)),
    )
    model, _ = load_trained_model(config, manager)
    dataset = CelebASpoofDataset(
        Path(config["paths"]["processed_dir"]) / "val_subset.parquet",
        config["paths"]["dataset_root"],
        int(config["training"].get("image_size", 224)),
    )
    loader = DataLoader(dataset, batch_size=int(config["training"]["batch_size"]))
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            probabilities = model(manager.move_tensor(batch["image"])).softmax(1)
            probabilities = probabilities.detach().cpu().numpy()
            for index in range(len(probabilities)):
                rows.append(
                    {
                        "image_id": batch["image_id"][index],
                        "relative_path": batch["relative_path"][index],
                        "subject_id": batch["subject_id"][index],
                        "spoof_type": batch["spoof_type"][index],
                        "illumination": batch["illumination"][index],
                        "environment": batch["environment"][index],
                        "true_label": int(batch["label"][index]),
                        "p_real": float(probabilities[index, 0]),
                        "p_spoof": float(probabilities[index, 1]),
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_parquet(layout["predictions"] / "validation_predictions.parquet", index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    try:
        run_with_device_fallback(config, "validation_predictions", export)
        write_status(config["paths"]["output_dir"], "validation_predictions", "completed")
    except Exception as exc:
        write_status(
            config["paths"]["output_dir"],
            "validation_predictions",
            "failed",
            error_message=f"{type(exc).__name__}: {exc}",
        )
        raise


if __name__ == "__main__":
    main()
