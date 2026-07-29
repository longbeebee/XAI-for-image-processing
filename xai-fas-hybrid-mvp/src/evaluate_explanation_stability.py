"""Prediction-conditioned Grad-CAM and IG stability evaluation."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .config import ensure_output_layout, load_config
from .datasets import CelebASpoofDataset
from .datasets.perturbations import apply_perturbation
from .device import DeviceManager
from .evaluation_utils import load_trained_model, normalized_to_pil, pil_to_normalized
from .metrics.stability import cosine_similarity, spearman_correlation, topk_iou
from .job_fallback import run_with_device_fallback
from .utils import write_status
from .xai import GradCAMExplainer, IntegratedGradientsExplainer


def evaluate(config: dict[str, Any]) -> pd.DataFrame:
    """Compare canonical normalized maps using the original predicted target."""
    layout = ensure_output_layout(config)
    manager = DeviceManager(
        config["device"].get("preferred", "auto"),
        bool(config["device"].get("allow_cpu_fallback", False)),
    )
    model, checkpoint = load_trained_model(config, manager)
    threshold = float(checkpoint["selected_threshold"])
    ig_config = config["xai"]["integrated_gradients"]
    explainers = {
        "gradcam": GradCAMExplainer(model),
        "integrated_gradients": IntegratedGradientsExplainer(
            model,
            int(ig_config["n_steps"]),
            int(ig_config["internal_batch_size"]),
            str(ig_config.get("baseline", "zero")),
        ),
    }
    dataset = CelebASpoofDataset(
        Path(config["paths"]["processed_dir"]) / "test_subset.parquet",
        config["paths"]["dataset_root"],
        int(config["training"].get("image_size", 224)),
    )
    limit = min(len(dataset), int(config["evaluation_limits"]["explanation_stability_samples"]))
    rows: list[dict[str, Any]] = []
    for index in range(limit):
        item = dataset[index]
        original = manager.move_tensor(item["image"].unsqueeze(0))
        original_probability = float(model(original).softmax(1)[0, 1].detach().cpu())
        original_prediction = int(original_probability >= threshold)
        target = torch.tensor([original_prediction], device=manager.get_torch_device())
        pil = normalized_to_pil(item["image"])
        originals = {
            name: explainer.explain(original, target)[0] for name, explainer in explainers.items()
        }
        for perturbation, severities in config["perturbations"].items():
            for severity, value in severities.items():
                shifted = manager.move_tensor(
                    pil_to_normalized(
                        apply_perturbation(pil, perturbation, value),
                        int(config["training"].get("image_size", 224)),
                    ).unsqueeze(0)
                )
                shifted_probability = float(model(shifted).softmax(1)[0, 1].detach().cpu())
                shifted_prediction = int(shifted_probability >= threshold)
                for method, explainer in explainers.items():
                    started = time.perf_counter()
                    shifted_explanation = explainer.explain(shifted, target)[0]
                    manager.synchronize_if_supported()
                    runtime_ms = (time.perf_counter() - started) * 1000
                    a = originals[method].normalized_map.numpy()
                    b = shifted_explanation.normalized_map.numpy()
                    rows.append(
                        {
                            "image_id": item["image_id"],
                            "relative_path": item["relative_path"],
                            "subject_id": item["subject_id"],
                            "true_label": item["label"],
                            "spoof_type": item["spoof_type"],
                            "illumination": item["illumination"],
                            "environment": item["environment"],
                            "xai_method": method,
                            "perturbation": perturbation,
                            "severity": int(severity),
                            "target_class": original_prediction,
                            "original_prediction": original_prediction,
                            "shifted_prediction": shifted_prediction,
                            "original_confidence": original_probability,
                            "shifted_confidence": shifted_probability,
                            "confidence_delta": shifted_probability - original_probability,
                            "prediction_unchanged": original_prediction == shifted_prediction,
                            "cosine_similarity": cosine_similarity(a, b),
                            "spearman_correlation": spearman_correlation(a, b),
                            "top10_iou": topk_iou(a, b, 0.1),
                            "top20_iou": topk_iou(a, b, 0.2),
                            "heatmap_status": shifted_explanation.status,
                            "requested_backend": manager.preferred,
                            "actual_backend": manager.get_logical_device_name(),
                            "runtime_ms": runtime_ms,
                        }
                    )
    frame = pd.DataFrame(rows)
    frame.to_parquet(layout["metrics"] / "explanation_stability.parquet", index=False)
    frame.to_csv(layout["metrics"] / "explanation_stability.csv", index=False)
    conditioned = frame[frame["prediction_unchanged"]]
    summary = conditioned.groupby(
        ["xai_method", "perturbation", "severity"], as_index=False
    )[["cosine_similarity", "spearman_correlation", "top10_iou", "top20_iou"]].mean()
    summary.to_csv(layout["metrics"] / "pces_summary.csv", index=False)
    return frame


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    try:
        run_with_device_fallback(config, "explanation_stability", evaluate)
        write_status(config["paths"]["output_dir"], "gradcam", "completed")
        write_status(config["paths"]["output_dir"], "integrated_gradients", "completed")
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        write_status(config["paths"]["output_dir"], "gradcam", "failed", error_message=message)
        write_status(
            config["paths"]["output_dir"], "integrated_gradients", "failed", error_message=message
        )
        raise


if __name__ == "__main__":
    main()
