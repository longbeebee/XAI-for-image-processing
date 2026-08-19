"""Re-evaluate the frozen baseline on the new subject-disjoint protocol.

This runner is intentionally isolated from ``results_aws_mvp``.  It loads the
frozen baseline checkpoint, selects a threshold on the new validation split,
and evaluates the new test split without modifying the original baseline
artifacts.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageOps
from torchvision import transforms

from src.datasets import CelebASpoofDataset
from src.datasets.paths import resolve_dataset_path
from src.datasets.perturbations import apply_perturbation
from src.metrics.classification import classification_metrics, select_threshold
from src.models import build_model


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def _predict(
    model: torch.nn.Module,
    metadata: Path,
    dataset_root: Path,
    image_size: int,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> tuple[pd.DataFrame, float]:
    dataset = CelebASpoofDataset(metadata, dataset_root, image_size, training=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    rows: list[dict[str, Any]] = []
    elapsed: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            probabilities = model(images).softmax(dim=1)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed.append((time.perf_counter() - started) * 1000.0 / len(images))
            probabilities = probabilities.detach().cpu().numpy()
            for index in range(len(probabilities)):
                rows.append(
                    {
                        "image_id": str(batch["image_id"][index]),
                        "relative_path": str(batch["relative_path"][index]),
                        "subject_id": str(batch["subject_id"][index]),
                        "spoof_type": str(batch["spoof_type"][index]),
                        "true_label": int(batch["label"][index]),
                        "p_real": float(probabilities[index, 0]),
                        "p_spoof": float(probabilities[index, 1]),
                    }
                )
    return pd.DataFrame(rows), float(np.mean(elapsed)) if elapsed else float("nan")


def _spoof_type_metrics(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    attacks = frame[frame["true_label"] == 1].copy()
    attacks["spoof_type"] = attacks["spoof_type"].replace("", "unknown").fillna("unknown")
    attacks["predicted_spoof"] = attacks["p_spoof"] >= threshold
    rows = []
    for spoof_type, group in attacks.groupby("spoof_type", sort=True):
        rows.append(
            {
                "spoof_type": str(spoof_type),
                "sample_count": int(len(group)),
                "false_acceptance_rate": float((~group["predicted_spoof"]).mean()),
                "spoof_detection_rate": float(group["predicted_spoof"].mean()),
                "mean_spoof_score": float(group["p_spoof"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _normalize_maps(values: torch.Tensor) -> torch.Tensor:
    values = values.flatten(1)
    return values / (values.norm(dim=1, keepdim=True) + 1e-8)


def _gradcam(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    layer = [module for module in model.features.modules() if isinstance(module, torch.nn.Conv2d)][-1]
    forward_handle = layer.register_forward_hook(lambda _, __, output: activations.append(output))
    backward_handle = layer.register_full_backward_hook(lambda _, __, output: gradients.append(output[0]))
    model.zero_grad(set_to_none=True)
    model(images).max(dim=1).values.sum().backward()
    forward_handle.remove()
    backward_handle.remove()
    weights = gradients[-1].mean(dim=(2, 3), keepdim=True)
    maps = torch.relu((weights * activations[-1]).sum(dim=1, keepdim=True))
    maps = torch.nn.functional.interpolate(maps, size=images.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
    return _normalize_maps(maps)


def _integrated_gradients(model: torch.nn.Module, images: torch.Tensor, steps: int = 24) -> torch.Tensor:
    baseline = torch.zeros_like(images)
    total = torch.zeros_like(images)
    for alpha in torch.linspace(0.0, 1.0, steps, device=images.device):
        point = (baseline + alpha * (images - baseline)).detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        score = model(point).max(dim=1).values.sum()
        total += torch.autograd.grad(score, point)[0]
    maps = ((images - baseline) * total / steps).abs().mean(dim=1)
    return _normalize_maps(maps)


def _attribution(model: torch.nn.Module, images: torch.Tensor, method: str) -> torch.Tensor:
    if method == "gradcam":
        return _gradcam(model, images)
    if method == "integrated_gradients":
        return _integrated_gradients(model, images)
    raise ValueError(f"Unsupported attribution method: {method}")


def _summarize_maps(first: torch.Tensor, second: torch.Tensor) -> dict[str, float]:
    cosine = (first * second).sum(dim=1)
    count = max(1, int(first.shape[1] * 0.1))
    first_top = first.topk(count, dim=1).indices
    second_top = second.topk(count, dim=1).indices
    ious = []
    for left, right in zip(first_top, second_top):
        a, b = set(left.tolist()), set(right.tolist())
        ious.append(len(a & b) / max(len(a | b), 1))
    return {"cosine_similarity": float(cosine.mean().detach()), "top10_iou": float(np.mean(ious))}


def _quality_pair(frame: pd.DataFrame, root: Path, index: int, image_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    row = frame.iloc[index]
    path = resolve_dataset_path(root, str(row["relative_path"]))
    with Image.open(path) as opened:
        original = ImageOps.exif_transpose(opened).convert("RGB")
    kind, value = [("brightness", 0.6), ("blur", 1.0), ("jpeg", 40)][index % 3]
    degraded = apply_perturbation(original, kind, value)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return transform(original), transform(degraded)


def _faithfulness(
    model: torch.nn.Module,
    dataset: CelebASpoofDataset,
    device: torch.device,
    samples: int,
    method: str = "gradcam",
) -> dict[str, float]:
    rows = []
    for index in range(min(samples, len(dataset))):
        image = dataset[index]["image"].unsqueeze(0).to(device)
        saliency = _attribution(model, image, method).detach()
        order = saliency[0].argsort(descending=True)
        total = image.shape[-1] * image.shape[-2]
        deletion, insertion = [], []
        for fraction in np.linspace(0, 1, 11):
            count = int(fraction * total)
            original = image.flatten(2)
            deleted = original.clone()
            inserted = torch.zeros_like(original)
            deleted[:, :, order[:count]] = 0.0
            inserted[:, :, order[:count]] = original[:, :, order[:count]]
            with torch.inference_mode():
                deletion.append(float(model(deleted.view_as(image)).softmax(1)[0, 1]))
                insertion.append(float(model(inserted.view_as(image)).softmax(1)[0, 1]))
        x = np.linspace(0, 1, len(deletion))
        rows.append({"deletion_auc": float(np.trapezoid(deletion, x)), "insertion_auc": float(np.trapezoid(insertion, x)), "original_probability": deletion[0]})
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]} if rows else {}


def _sanity(
    model: torch.nn.Module,
    dataset: CelebASpoofDataset,
    device: torch.device,
    samples: int,
    method: str = "gradcam",
) -> list[dict[str, float]]:
    original_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    references = [(index, _attribution(model, dataset[index]["image"].unsqueeze(0).to(device), method).detach()) for index in range(min(samples, len(dataset)))]
    results = []
    for level in (1, 2, 3):
        model.load_state_dict(original_state)
        targets = [model.classifier] if level == 1 else list(model.features)[-3:] if level == 2 else list(model.children())
        for target in targets:
            for parameter in target.parameters():
                parameter.data.normal_(0.0, 0.02)
        similarities = []
        for index, reference in references:
            current = _attribution(model, dataset[index]["image"].unsqueeze(0).to(device), method).detach()
            similarities.append(float((reference * current).sum()))
        results.append({"randomization_level": level, "cosine_similarity": float(np.mean(similarities))})
    model.load_state_dict(original_state)
    return results


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir)
    metrics_dir = output / "metrics"
    predictions_dir = output / "predictions"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    val_frame = pd.read_parquet(args.validation_split)
    test_frame = pd.read_parquet(args.test_split)
    val_subjects = set(val_frame["subject_id"].astype(str))
    test_subjects = set(test_frame["subject_id"].astype(str))
    overlap = sorted(val_subjects & test_subjects)
    if overlap:
        raise RuntimeError(f"Validation/test subject overlap detected: {len(overlap)} subjects")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(pretrained=False, num_classes=2)
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device(args.device)
    model.to(device).eval()

    val_predictions, val_runtime = _predict(
        model, args.validation_split, args.dataset_root, args.image_size,
        args.batch_size, args.workers, device,
    )
    threshold = select_threshold(
        val_predictions["true_label"].to_numpy(),
        val_predictions["p_spoof"].to_numpy(),
        args.threshold_strategy,
    )
    test_predictions, test_runtime = _predict(
        model, args.test_split, args.dataset_root, args.image_size,
        args.batch_size, args.workers, device,
    )

    val_predictions.to_parquet(predictions_dir / "validation_predictions.parquet", index=False)
    test_predictions.to_parquet(predictions_dir / "test_predictions.parquet", index=False)
    classification = classification_metrics(
        test_predictions["true_label"].to_numpy(),
        test_predictions["p_spoof"].to_numpy(),
        float(threshold),
    )
    _write_json(metrics_dir / "classification_metrics.json", classification)
    _spoof_type_metrics(test_predictions, float(threshold)).to_csv(
        metrics_dir / "spoof_type_metrics.csv", index=False
    )
    test_dataset = CelebASpoofDataset(args.test_split, args.dataset_root, args.image_size, training=False)
    gradcam_rows: list[dict[str, float]] = []
    integrated_gradients_rows: list[dict[str, float]] = []
    xai_started = time.perf_counter()
    for index in range(min(args.xai_samples, len(test_frame))):
        original, degraded = _quality_pair(test_frame, args.dataset_root, index, args.image_size)
        original_tensor = original.unsqueeze(0).to(device)
        degraded_tensor = degraded.unsqueeze(0).to(device)
        gradcam_rows.append(_summarize_maps(
            _gradcam(model, original_tensor),
            _gradcam(model, degraded_tensor),
        ))
        integrated_gradients_rows.append(_summarize_maps(
            _integrated_gradients(model, original_tensor),
            _integrated_gradients(model, degraded_tensor),
        ))
    xai_runtime = (time.perf_counter() - xai_started) * 1000.0 / max(len(gradcam_rows), 1)
    gradcam_consistency = {
        key: float(np.mean([row[key] for row in gradcam_rows]))
        for key in gradcam_rows[0]
    } if gradcam_rows else {}
    integrated_gradients_consistency = {
        key: float(np.mean([row[key] for row in integrated_gradients_rows]))
        for key in integrated_gradients_rows[0]
    } if integrated_gradients_rows else {}
    faithfulness_gradcam = _faithfulness(model, test_dataset, device, args.xai_samples, method="gradcam")
    faithfulness_ig = _faithfulness(model, test_dataset, device, args.xai_samples, method="integrated_gradients")
    sanity_gradcam = _sanity(model, test_dataset, device, args.xai_samples, method="gradcam")
    sanity_ig = _sanity(model, test_dataset, device, args.xai_samples, method="integrated_gradients")
    faithfulness_by_method = {
        "gradcam": faithfulness_gradcam,
        "integrated_gradients": faithfulness_ig,
    }
    sanity_by_method = {
        "gradcam": sanity_gradcam,
        "integrated_gradients": sanity_ig,
    }
    xai_consistency = {
        "gradcam": gradcam_consistency,
        "integrated_gradients": integrated_gradients_consistency,
    }
    _write_json(metrics_dir / "xai_consistency.json", xai_consistency)
    _write_json(metrics_dir / "faithfulness.json", faithfulness_by_method)
    _write_json(metrics_dir / "sanity.json", sanity_by_method)
    _write_json(
        metrics_dir / "runtime.json",
        {
            "device": str(device),
            "validation_mean_ms_per_image": val_runtime,
            "test_mean_ms_per_image": test_runtime,
            "gradcam_mean_ms_per_sample": xai_runtime,
            "batch_size": args.batch_size,
            "test_sample_count": int(len(test_predictions)),
            "xai_sample_count": int(len(gradcam_rows)),
        },
    )
    manifest = {
        "stage": "baseline_control_subject_disjoint_evaluation",
        "checkpoint": str(args.checkpoint),
        "validation_split": str(args.validation_split),
        "test_split": str(args.test_split),
        "threshold_source": "validation_subject_disjoint",
        "threshold": float(threshold),
        "validation_subject_count": len(val_subjects),
        "test_subject_count": len(test_subjects),
        "subject_overlap_count": len(overlap),
        "test_sample_count": int(len(test_predictions)),
        "classification": classification,
        "xai_consistency": xai_consistency,
        "xai_sample_count": int(args.xai_samples),
        "integrated_gradients_steps": 24,
        "faithfulness": faithfulness_gradcam,
        "sanity": {"gradcam": sanity_gradcam},
        "faithfulness_by_method": faithfulness_by_method,
        "sanity_by_method": sanity_by_method,
    }
    _write_json(output / "baseline_control_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--validation-split", required=True, type=Path)
    parser.add_argument("--test-split", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threshold-strategy", default="min_acer")
    parser.add_argument("--xai-samples", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(evaluate(args), indent=2, default=str))


if __name__ == "__main__":
    main()
