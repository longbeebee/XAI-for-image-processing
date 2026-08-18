"""CUDA smoke checks for the new quality-aware model only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from .gradcam_training import differentiable_gradcam
from .model import QualityAwareFAS


def check(config: dict) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("The quality-aware experiment requires CUDA.")
    device = torch.device("cuda")
    model = QualityAwareFAS(bool(config["model"].get("pretrained", False))).to(device)
    images = torch.randn(2, 3, int(config["training"]["image_size"]), int(config["training"]["image_size"]), device=device, requires_grad=True)
    outputs = model(images)
    loss = outputs["logits"].mean() + outputs["quality"].mean() + outputs["uncertainty"].mean()
    loss.backward()
    model.zero_grad(set_to_none=True)
    maps = differentiable_gradcam(model, images)
    report = {"status": "passed", "device": str(device), "gpu_name": torch.cuda.get_device_name(0), "torch_version": torch.__version__, "cuda_version": torch.version.cuda, "logits_shape": list(outputs["logits"].shape), "quality_shape": list(outputs["quality"].shape), "uncertainty_shape": list(outputs["uncertainty"].shape), "gradcam_shape": list(maps.shape)}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True, type=Path); parser.add_argument("--output", required=True, type=Path); args = parser.parse_args()
    result = check(yaml.safe_load(args.config.read_text(encoding="utf-8"))); args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
