"""Evaluate classification, uncertainty, and XAI over quality-degradation levels."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.datasets.paths import resolve_dataset_path
from src.datasets.perturbations import apply_perturbation
from src.models import build_model
from .evaluation import classification_summary, threshold_min_acer
from .model import QualityAwareFAS
from .xai_evaluation import normalize


LEVELS = {
    "brightness": [1.0, 0.8, 0.6, 0.4],
    "blur": [0.0, 0.5, 1.0, 2.0, 4.0],
    "jpeg": [100, 80, 60, 40, 20],
}


class QualityDataset(Dataset):
    def __init__(self, metadata: Path, root: Path, image_size: int, kind: str | None = None, value: float | int | None = None):
        self.frame = pd.read_parquet(metadata).reset_index(drop=True)
        self.root, self.kind, self.value = root, kind, value
        self.transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = resolve_dataset_path(self.root, str(row["relative_path"]))
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        if self.kind is not None and self.value is not None:
            image = apply_perturbation(image, self.kind, self.value)
        return {"image": self.transform(image), "label": int(row["label_id"]), "image_id": str(row["image_id"]), "subject_id": str(row["subject_id"])}


def _load_model(config: dict, checkpoint: Path, model_type: str, device: torch.device):
    if model_type == "baseline":
        model = build_model(pretrained=False, num_classes=2)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state"])
    else:
        model = QualityAwareFAS(bool(config["model"].get("pretrained", True)))
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state"])
    return model.to(device).eval()


def _logits(model, images):
    output = model(images)
    return output["logits"] if isinstance(output, dict) else output


def _collect(model, loader, device):
    logits, labels, uncertainties = [], [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            raw = model(images)
            values = raw["logits"] if isinstance(raw, dict) else raw
            score = values.softmax(1)
            logits.append(values.cpu().numpy())
            labels.append(batch["label"].numpy())
            if isinstance(raw, dict):
                uncertainties.append(raw["uncertainty"].cpu().numpy())
            else:
                uncertainties.append((1.0 - score.max(1).values).cpu().numpy())
    return np.concatenate(logits), np.concatenate(labels), np.concatenate(uncertainties)


def _attribution(model, images, method: str, steps: int = 24):
    activations, gradients = [], []
    layer = [m for m in model.features.modules() if isinstance(m, torch.nn.Conv2d)][-1]
    if method == "gradcam":
        handle_a = layer.register_forward_hook(lambda _, __, output: activations.append(output))
        handle_g = layer.register_full_backward_hook(lambda _, __, output: gradients.append(output[0]))
        model.zero_grad(set_to_none=True)
        _logits(model, images).max(1).values.sum().backward()
        handle_a.remove(); handle_g.remove()
        weights = gradients[-1].mean(dim=(2, 3), keepdim=True)
        maps = torch.relu((weights * activations[-1]).sum(1, keepdim=True))
        maps = torch.nn.functional.interpolate(maps, size=images.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        return normalize(maps)
    baseline = torch.zeros_like(images); total = torch.zeros_like(images)
    for alpha in torch.linspace(0.0, 1.0, steps, device=images.device):
        point = (baseline + alpha * (images - baseline)).detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        score = _logits(model, point).max(1).values.sum()
        total += torch.autograd.grad(score, point)[0]
    return normalize(((images - baseline) * total / steps).abs().mean(1))


def _xai_consistency(model, clean: Dataset, degraded: Dataset, device: torch.device, samples: int):
    values = {"gradcam": [], "integrated_gradients": []}
    with torch.enable_grad():
        for index in range(min(samples, len(clean))):
            original = clean[index]["image"].unsqueeze(0).to(device)
            changed = degraded[index]["image"].unsqueeze(0).to(device)
            for method in values:
                first = _attribution(model, original, method)
                second = _attribution(model, changed, method)
                cosine = float((first * second).sum().detach().cpu())
                k = max(1, int(first.shape[1] * 0.1))
                left = set(first[0].topk(k).indices.detach().cpu().tolist())
                right = set(second[0].topk(k).indices.detach().cpu().tolist())
                values[method].append({"cosine_similarity": cosine, "top10_iou": len(left & right) / max(len(left | right), 1)})
    return {method: {key: float(np.mean([row[key] for row in rows])) for key in rows[0]} for method, rows in values.items() if rows}


def _calibration(logits, labels):
    probabilities = torch.softmax(torch.tensor(logits), 1).numpy()
    confidence = probabilities.max(1); predictions = probabilities.argmax(1)
    ece = 0.0
    for low, high in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any(): ece += float(mask.mean()) * abs(float(confidence[mask].mean()) - float((predictions[mask] == labels[mask]).mean()))
    one_hot = np.eye(2)[labels]
    return {"ece": ece, "brier": float(np.mean((probabilities - one_hot) ** 2))}


def evaluate(config: dict, checkpoint: Path, protocol_dir: Path, dataset_root: Path, output_dir: Path, model_type: str, kind: str, xai_samples: int) -> dict:
    if kind not in LEVELS: raise ValueError(f"Unknown degradation kind: {kind}")
    device = torch.device("cuda" if torch.cuda.is_available() and config["device"]["preferred"] == "cuda" else "cpu")
    if device.type != "cuda" and not config["device"].get("allow_cpu_fallback", False): raise RuntimeError("CUDA is required for degradation sweep.")
    model = _load_model(config, checkpoint, model_type, device)
    image_size = int(config["training"]["image_size"])
    val = QualityDataset(protocol_dir / "val_subject_disjoint.parquet", dataset_root, image_size)
    clean = QualityDataset(protocol_dir / "test_subject_disjoint.parquet", dataset_root, image_size)
    batch_size = int(config["training"].get("batch_size", 32))
    val_logits, val_labels, _ = _collect(model, DataLoader(val, batch_size=batch_size), device)
    threshold = threshold_min_acer(val_labels, torch.softmax(torch.tensor(val_logits), 1)[:, 1].numpy())
    records = []
    for value in LEVELS[kind]:
        degraded = QualityDataset(protocol_dir / "test_subject_disjoint.parquet", dataset_root, image_size, None if (kind == "brightness" and value == 1.0) or (kind == "blur" and value == 0.0) or (kind == "jpeg" and value == 100) else kind, value)
        started = time.perf_counter(); logits, labels, uncertainty = _collect(model, DataLoader(degraded, batch_size=batch_size), device)
        scores = torch.softmax(torch.tensor(logits), 1)[:, 1].numpy()
        row = {"degradation": kind, "level": value, "threshold_from_clean_validation": threshold, "classification": classification_summary(labels, scores, threshold), "calibration": _calibration(logits, labels), "mean_uncertainty": float(uncertainty.mean()), "runtime_ms_per_image": float((time.perf_counter() - started) * 1000 / max(len(degraded), 1))}
        row["xai_consistency"] = _xai_consistency(model, clean, degraded, device, xai_samples)
        records.append(row)
    result = {"checkpoint": str(checkpoint), "model_type": model_type, "kind": kind, "xai_sample_count": xai_samples, "levels": records}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{model_type}_{kind}_sweep.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path); parser.add_argument("--checkpoint", required=True, type=Path); parser.add_argument("--protocol-dir", required=True, type=Path); parser.add_argument("--dataset-root", required=True, type=Path); parser.add_argument("--output-dir", required=True, type=Path); parser.add_argument("--model-type", choices=["baseline", "quality_aware"], required=True); parser.add_argument("--kind", choices=list(LEVELS), required=True); parser.add_argument("--xai-samples", type=int, default=300)
    args = parser.parse_args(); config = yaml.safe_load(args.config.read_text(encoding="utf-8")); print(json.dumps(evaluate(config, args.checkpoint, args.protocol_dir, args.dataset_root, args.output_dir, args.model_type, args.kind, args.xai_samples), indent=2, default=str))


if __name__ == "__main__": main()
