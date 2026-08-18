"""Fine-tune the uncertainty head against validation-free training error targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from .dataset import QualityAwareDataset
from .losses import quality_aware_loss
from .model import QualityAwareFAS


def run(config: dict, checkpoint: Path, output_dir: Path, seed: int = 42) -> dict:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and config["device"]["preferred"] == "cuda" else "cpu")
    if device.type != "cuda" and not config["device"].get("allow_cpu_fallback", False):
        raise RuntimeError("CUDA is required by this experiment.")
    protocol = Path(config["paths"].get("protocol_dir", config["paths"]["processed_dir"]))
    dataset = QualityAwareDataset(protocol / "train_subject_disjoint.parquet", config["paths"]["dataset_root"], int(config["training"]["image_size"]), True, seed)
    loader = DataLoader(dataset, batch_size=int(config["training"]["batch_size"]), shuffle=True, num_workers=int(config["training"].get("num_workers", 0)), pin_memory=True)
    model = QualityAwareFAS(bool(config["model"].get("pretrained", True))).to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"])
    for parameter in model.features.parameters(): parameter.requires_grad = False
    for parameter in model.classifier.parameters(): parameter.requires_grad = False
    for parameter in model.quality_head.parameters(): parameter.requires_grad = False
    optimizer = torch.optim.AdamW(model.uncertainty_head.parameters(), lr=float(config["training"]["uncertainty_learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    history = []
    for epoch in range(int(config["training"].get("uncertainty_epochs", 3))):
        model.train(); total = 0.0
        for batch in loader:
            images = batch["image"].to(device); labels = batch["label"].to(device); quality = batch["quality"].to(device)
            optimizer.zero_grad(set_to_none=True); outputs = model(images)
            loss, details = quality_aware_loss(outputs, labels, quality, quality_weight=0.0, uncertainty_weight=1.0)
            loss.backward(); optimizer.step(); total += details["uncertainty"]
        history.append({"epoch": epoch, "uncertainty_loss": total / max(len(loader), 1)})
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "uncertainty_calibrated_model.pt"
    torch.save({"model_state": model.state_dict(), "seed": seed, "history": history}, target)
    result = {"stage": "uncertainty_calibration", "seed": seed, "checkpoint": str(target), "device": str(device), "history": history}
    (output_dir / "stage2_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True, type=Path); parser.add_argument("--checkpoint", required=True, type=Path); parser.add_argument("--output-dir", required=True, type=Path); parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(); config = yaml.safe_load(args.config.read_text(encoding="utf-8")); print(json.dumps(run(config, args.checkpoint, args.output_dir, args.seed), indent=2))


if __name__ == "__main__": main()
