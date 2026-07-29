"""Warm-up and benchmark classifier, explainers, deletion, and insertion."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch

from .config import ensure_output_layout, load_config
from .datasets import CelebASpoofDataset
from .device import DeviceManager
from .evaluation_utils import MEAN, STD, load_trained_model
from .metrics.faithfulness import patch_curve
from .metrics.runtime import benchmark
from .utils import write_status
from .xai import GradCAMExplainer, IntegratedGradientsExplainer


def evaluate(config: dict[str, Any]) -> pd.DataFrame:
    """Benchmark computation only; exclude DataLoader and image-file I/O."""
    layout = ensure_output_layout(config)
    manager = DeviceManager(
        config["device"].get("preferred", "auto"),
        bool(config["device"].get("allow_cpu_fallback", False)),
    )
    model, _ = load_trained_model(config, manager)
    dataset = CelebASpoofDataset(
        Path(config["paths"]["processed_dir"]) / "test_subset.parquet",
        config["paths"]["dataset_root"],
        int(config["training"].get("image_size", 224)),
    )
    item = dataset[0]
    image = manager.move_tensor(item["image"].unsqueeze(0))
    target_class = int(model(image).argmax(1).detach().cpu())
    target = torch.tensor([target_class], device=manager.get_torch_device())
    ig_config = config["xai"]["integrated_gradients"]
    gradcam = GradCAMExplainer(model)
    ig = IntegratedGradientsExplainer(
        model,
        int(ig_config["n_steps"]),
        int(ig_config["internal_batch_size"]),
        str(ig_config["baseline"]),
    )
    heatmap = gradcam.explain(image, target)[0].normalized_map
    rgb = (item["image"] * STD + MEAN).clamp(0, 1)

    def predict(batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model(manager.move_tensor(batch))

    functions: dict[str, Callable[[], Any]] = {
        "classifier_prediction": lambda: model(image),
        "gradcam": lambda: gradcam.explain(image, target),
        "integrated_gradients": lambda: ig.explain(image, target),
        "deletion": lambda: patch_curve(
            rgb, heatmap, predict, target_class, "deletion",
            int(config["faithfulness"]["patch_size"]),
            int(config["faithfulness"]["steps"]),
            "mean",
        ),
        "insertion": lambda: patch_curve(
            rgb, heatmap, predict, target_class, "insertion",
            int(config["faithfulness"]["patch_size"]),
            int(config["faithfulness"]["steps"]),
            "blur",
        ),
    }
    repetitions = int(config["evaluation_limits"]["runtime_samples"])
    rows = []
    for name, function in functions.items():
        result = benchmark(function, manager, repetitions, warmup=1)
        result.update(operation=name, batch_size=1)
        rows.append(result)
    frame = pd.DataFrame(rows)
    frame.to_csv(layout["metrics"] / "runtime.csv", index=False)
    return frame


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    try:
        evaluate(config)
        write_status(config["paths"]["output_dir"], "runtime", "completed")
    except Exception as exc:
        write_status(
            config["paths"]["output_dir"],
            "runtime",
            "failed",
            error_message=f"{type(exc).__name__}: {exc}",
        )
        raise


if __name__ == "__main__":
    main()

