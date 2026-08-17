"""Export image-level visual evidence for Grad-CAM and Integrated Gradients."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

from .config import ensure_output_layout, load_config
from .datasets import CelebASpoofDataset
from .datasets.perturbations import apply_perturbation
from .device import DeviceManager
from .evaluation_utils import load_trained_model, normalized_to_pil, pil_to_normalized
from .job_fallback import run_with_device_fallback
from .metrics.stability import cosine_similarity, spearman_correlation, topk_iou
from .utils import write_status
from .xai import GradCAMExplainer, IntegratedGradientsExplainer


def _heatmap_overlay(image: Image.Image, heatmap: torch.Tensor) -> Image.Image:
    """Overlay a normalized heatmap on an RGB image without changing its size."""
    values = heatmap.detach().cpu().numpy().clip(0.0, 1.0)
    # A small hand-built blue-to-red map avoids requiring a plotting backend.
    red = np.clip(2.0 * values, 0.0, 1.0)
    blue = np.clip(2.0 * (1.0 - values), 0.0, 1.0)
    green = 1.0 - np.abs(2.0 * values - 1.0)
    colors = (np.stack([red, green, blue], axis=-1) * 255).astype(np.uint8)
    color_image = Image.fromarray(colors, mode="RGB").resize(image.size, Image.Resampling.BILINEAR)
    return Image.blend(image.convert("RGB"), color_image, alpha=0.45)


def _panel(images: list[tuple[str, Image.Image]]) -> Image.Image:
    """Create a labeled horizontal evidence panel."""
    tile_width, tile_height, label_height = 224, 224, 28
    panel = Image.new("RGB", (tile_width * len(images), tile_height + label_height), "white")
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    for index, (label, image) in enumerate(images):
        x = index * tile_width
        panel.paste(image.convert("RGB").resize((tile_width, tile_height)), (x, label_height))
        draw.text((x + 4, 7), label, fill="black", font=font)
    return panel


def _explainers(model: torch.nn.Module, config: dict[str, Any]) -> dict[str, Any]:
    ig = config["xai"]["integrated_gradients"]
    return {
        "gradcam": GradCAMExplainer(model),
        "integrated_gradients": IntegratedGradientsExplainer(
            model,
            int(ig["n_steps"]),
            int(ig["internal_batch_size"]),
            str(ig.get("baseline", "zero")),
        ),
    }


def export(config: dict[str, Any]) -> pd.DataFrame:
    """Save paired original/perturbed evidence panels and a manifest."""
    layout = ensure_output_layout(config)
    evidence_dir = layout["explanations"] / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    manager = DeviceManager(
        config["device"].get("preferred", "auto"),
        bool(config["device"].get("allow_cpu_fallback", False)),
    )
    model, checkpoint = load_trained_model(config, manager)
    threshold = float(checkpoint["selected_threshold"])
    explainers = _explainers(model, config)
    dataset = CelebASpoofDataset(
        Path(config["paths"]["processed_dir"]) / "test_subset.parquet",
        config["paths"]["dataset_root"],
        int(config["training"].get("image_size", 224)),
    )
    visualization = config.get("visualization", {})
    sample_limit = min(len(dataset), int(visualization.get("evidence_samples", 6)))
    selected_severities = {int(value) for value in visualization.get("evidence_severities", [1, 3])}
    rows: list[dict[str, Any]] = []
    for index in range(sample_limit):
        item = dataset[index]
        original_tensor = manager.move_tensor(item["image"].unsqueeze(0))
        original_probability = float(model(original_tensor).softmax(1)[0, 1].detach().cpu())
        original_prediction = int(original_probability >= threshold)
        target = torch.tensor([original_prediction], device=manager.get_torch_device())
        original_pil = normalized_to_pil(item["image"])
        original_explanations = {
            method: explainer.explain(original_tensor, target)[0]
            for method, explainer in explainers.items()
        }
        for perturbation, severities in config["perturbations"].items():
            for severity, value in severities.items():
                severity = int(severity)
                if severity not in selected_severities:
                    continue
                shifted_pil = apply_perturbation(original_pil, perturbation, value)
                shifted_tensor = manager.move_tensor(
                    pil_to_normalized(
                        shifted_pil, int(config["training"].get("image_size", 224))
                    ).unsqueeze(0)
                )
                shifted_probability = float(model(shifted_tensor).softmax(1)[0, 1].detach().cpu())
                shifted_prediction = int(shifted_probability >= threshold)
                panels: list[tuple[str, Image.Image]] = [
                    ("original", original_pil),
                    (f"{perturbation} s{severity}", shifted_pil),
                ]
                method_metrics: dict[str, dict[str, Any]] = {}
                for method, explainer in explainers.items():
                    started = time.perf_counter()
                    shifted_explanation = explainer.explain(shifted_tensor, target)[0]
                    manager.synchronize_if_supported()
                    original_explanation = original_explanations[method]
                    original_overlay = _heatmap_overlay(original_pil, original_explanation.normalized_map)
                    shifted_overlay = _heatmap_overlay(shifted_pil, shifted_explanation.normalized_map)
                    panels.extend(
                        [
                            (f"{method} original", original_overlay),
                            (f"{method} shifted", shifted_overlay),
                        ]
                    )
                    a = original_explanation.normalized_map.numpy()
                    b = shifted_explanation.normalized_map.numpy()
                    method_metrics[method] = {
                        "heatmap_status": shifted_explanation.status,
                        "cosine_similarity": cosine_similarity(a, b),
                        "spearman_correlation": spearman_correlation(a, b),
                        "top10_iou": topk_iou(a, b, 0.1),
                        "top20_iou": topk_iou(a, b, 0.2),
                        "runtime_ms": (time.perf_counter() - started) * 1000,
                    }
                filename = (
                    f"{item['image_id']}__{perturbation}__severity-{severity}.png"
                )
                panel_path = evidence_dir / filename
                _panel(panels).save(panel_path, format="PNG", optimize=True)
                for method, metrics in method_metrics.items():
                    rows.append(
                        {
                            "image_id": item["image_id"],
                            "relative_path": item["relative_path"],
                            "true_label": int(item["label"]),
                            "spoof_type": item["spoof_type"],
                            "perturbation": perturbation,
                            "severity": severity,
                            "target_class": original_prediction,
                            "original_prediction": original_prediction,
                            "shifted_prediction": shifted_prediction,
                            "original_confidence": original_probability,
                            "shifted_confidence": shifted_probability,
                            "xai_method": method,
                            "panel_path": str(panel_path.relative_to(layout["output"])),
                            **metrics,
                        }
                    )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(evidence_dir / "evidence_manifest.csv", index=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    try:
        run_with_device_fallback(config, "explanation_evidence", export)
        write_status(config["paths"]["output_dir"], "explanation_evidence", "completed")
    except Exception as exc:
        write_status(
            config["paths"]["output_dir"],
            "explanation_evidence",
            "failed",
            error_message=f"{type(exc).__name__}: {exc}",
        )
        raise


if __name__ == "__main__":
    main()
