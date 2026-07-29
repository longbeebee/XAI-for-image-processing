"""Patch deletion/insertion faithfulness evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .config import ensure_output_layout, load_config
from .datasets import CelebASpoofDataset
from .device import DeviceManager
from .evaluation_utils import MEAN, STD, load_trained_model
from .metrics.faithfulness import normalized_auc, patch_curve
from .job_fallback import run_with_device_fallback
from .utils import write_status
from .xai import GradCAMExplainer, IntegratedGradientsExplainer


def evaluate(config: dict[str, Any]) -> pd.DataFrame:
    """Compute faithfulness curves with one fixed original predicted target."""
    layout = ensure_output_layout(config)
    manager = DeviceManager(
        config["device"].get("preferred", "auto"),
        bool(config["device"].get("allow_cpu_fallback", False)),
    )
    model, checkpoint = load_trained_model(config, manager)
    ig = config["xai"]["integrated_gradients"]
    explainers = {
        "gradcam": GradCAMExplainer(model),
        "integrated_gradients": IntegratedGradientsExplainer(
            model, int(ig["n_steps"]), int(ig["internal_batch_size"]), str(ig["baseline"])
        ),
    }
    dataset = CelebASpoofDataset(
        Path(config["paths"]["processed_dir"]) / "test_subset.parquet",
        config["paths"]["dataset_root"],
        int(config["training"].get("image_size", 224)),
    )
    limit = min(len(dataset), int(config["evaluation_limits"]["faithfulness_samples"]))
    rows = []
    for index in range(limit):
        item = dataset[index]
        normalized = item["image"].unsqueeze(0)
        device_input = manager.move_tensor(normalized)
        target_class = int(model(device_input).argmax(1).detach().cpu())
        target = torch.tensor([target_class], device=manager.get_torch_device())
        rgb = (item["image"] * STD + MEAN).clamp(0, 1)

        def predict(batch: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return model(manager.move_tensor(batch))

        for method, explainer in explainers.items():
            heatmap = explainer.explain(device_input, target)[0].normalized_map
            for baseline in config["faithfulness"]["baselines"]:
                for mode in ("deletion", "insertion"):
                    fractions, probabilities = patch_curve(
                        rgb,
                        heatmap,
                        predict,
                        target_class,
                        mode,
                        int(config["faithfulness"]["patch_size"]),
                        int(config["faithfulness"]["steps"]),
                        str(baseline),
                    )
                    rows.append(
                        {
                            "image_id": item["image_id"],
                            "xai_method": method,
                            "baseline": baseline,
                            "curve": mode,
                            "auc": normalized_auc(fractions, probabilities),
                            "target_class": target_class,
                            "fractions": fractions.tolist(),
                            "probabilities": probabilities.tolist(),
                            "actual_backend": manager.get_logical_device_name(),
                        }
                    )
    frame = pd.DataFrame(rows)
    frame.to_csv(layout["metrics"] / "faithfulness.csv", index=False)
    return frame


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    try:
        run_with_device_fallback(config, "faithfulness", evaluate)
        write_status(config["paths"]["output_dir"], "faithfulness", "completed")
    except Exception as exc:
        write_status(
            config["paths"]["output_dir"],
            "faithfulness",
            "failed",
            error_message=f"{type(exc).__name__}: {exc}",
        )
        raise


if __name__ == "__main__":
    main()
