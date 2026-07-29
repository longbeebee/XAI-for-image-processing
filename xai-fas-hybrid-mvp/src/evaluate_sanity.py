"""Parameter-randomization sanity evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .config import ensure_output_layout, load_config
from .datasets import CelebASpoofDataset
from .device import DeviceManager
from .evaluation_utils import load_trained_model
from .metrics.sanity import randomized_model
from .job_fallback import run_with_device_fallback
from .metrics.stability import cosine_similarity, spearman_correlation, topk_iou
from .utils import write_status
from .xai import GradCAMExplainer, IntegratedGradientsExplainer


def _explainers(model: torch.nn.Module, config: dict[str, Any]) -> dict[str, Any]:
    ig = config["xai"]["integrated_gradients"]
    return {
        "gradcam": GradCAMExplainer(model),
        "integrated_gradients": IntegratedGradientsExplainer(
            model, int(ig["n_steps"]), int(ig["internal_batch_size"]), str(ig["baseline"])
        ),
    }


def evaluate(config: dict[str, Any]) -> pd.DataFrame:
    """Compare trained maps with levels 1-3 while preserving the main model."""
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
    limit = min(len(dataset), int(config["evaluation_limits"]["sanity_samples"]))
    originals = _explainers(model, config)
    randomized = {
        level: manager.move_model(randomized_model(model, level)).eval() for level in (1, 2, 3)
    }
    randomized_explainers = {
        level: _explainers(randomized[level], config) for level in randomized
    }
    rows = []
    for index in range(limit):
        item = dataset[index]
        image = manager.move_tensor(item["image"].unsqueeze(0))
        target_class = int(model(image).argmax(1).detach().cpu())
        target = torch.tensor([target_class], device=manager.get_torch_device())
        reference = {
            method: explainer.explain(image, target)[0].normalized_map.numpy()
            for method, explainer in originals.items()
        }
        for level, explainers in randomized_explainers.items():
            for method, explainer in explainers.items():
                shifted = explainer.explain(image, target)[0].normalized_map.numpy()
                rows.append(
                    {
                        "image_id": item["image_id"],
                        "xai_method": method,
                        "randomization_level": level,
                        "cosine_similarity": cosine_similarity(reference[method], shifted),
                        "spearman_correlation": spearman_correlation(reference[method], shifted),
                        "top10_iou": topk_iou(reference[method], shifted, 0.1),
                        "actual_backend": manager.get_logical_device_name(),
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(layout["metrics"] / "sanity.csv", index=False)
    return frame


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    try:
        run_with_device_fallback(config, "sanity", evaluate)
        write_status(config["paths"]["output_dir"], "sanity", "completed")
    except Exception as exc:
        write_status(
            config["paths"]["output_dir"],
            "sanity",
            "failed",
            error_message=f"{type(exc).__name__}: {exc}",
        )
        raise


if __name__ == "__main__":
    main()
