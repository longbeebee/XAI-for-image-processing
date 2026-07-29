"""Backend and dependency compatibility checks."""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import psutil
import torch

from .config import ensure_output_layout, load_config
from .device import DeviceManager
from .models import build_model
from .utils import atomic_json, save_checkpoint, write_status
from .xai import GradCAMExplainer, IntegratedGradientsExplainer


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _check(name: str, function: Callable[[], None], requested: str, actual: str) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        function()
        return {
            "name": name,
            "status": "passed",
            "requested_device": requested,
            "actual_device": actual,
            "error_type": None,
            "error_message": None,
            "runtime": time.perf_counter() - started,
        }
    except Exception as exc:
        return {
            "name": name,
            "status": "failed",
            "requested_device": requested,
            "actual_device": actual,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "runtime": time.perf_counter() - started,
        }


def check_environment(config: dict[str, Any]) -> dict[str, Any]:
    """Run tensor, model, optimizer, XAI, and checkpoint compatibility tests."""
    layout = ensure_output_layout(config)
    requested = config["device"].get("preferred", "auto")
    manager = DeviceManager(requested, bool(config["device"].get("allow_cpu_fallback", False)))
    actual = manager.get_logical_device_name()
    device = manager.get_torch_device()

    def tensor_test() -> None:
        tensor = torch.ones(2, device=device)
        if not torch.equal((tensor + 1).cpu(), torch.full((2,), 2.0)):
            raise RuntimeError("Tensor arithmetic returned an invalid result.")

    def model_test() -> None:
        model = build_model(False).to(device)
        inputs = torch.randn(2, 3, 224, 224, device=device)
        labels = torch.tensor([0, 1], device=device)
        optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3)
        optimizer.zero_grad()
        loss = torch.nn.CrossEntropyLoss()(model(inputs), labels)
        loss.backward()
        optimizer.step()

    def gradcam_test() -> None:
        model = build_model(False).to(device).eval()
        inputs = torch.randn(1, 3, 224, 224, device=device)
        result = GradCAMExplainer(model).explain(inputs, torch.tensor([1], device=device))
        if result[0].normalized_map.shape != (224, 224):
            raise RuntimeError("Grad-CAM map shape mismatch.")

    def ig_test() -> None:
        model = build_model(False).to(device).eval()
        inputs = torch.randn(1, 3, 224, 224, device=device)
        result = IntegratedGradientsExplainer(model, n_steps=4, internal_batch_size=2).explain(
            inputs, torch.tensor([1], device=device)
        )
        if result[0].normalized_map.shape != (224, 224):
            raise RuntimeError("IG map shape mismatch.")

    def checkpoint_test() -> None:
        model = build_model(False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "portable.pt"
            save_checkpoint(path, model, epoch=0)
            saved = torch.load(path, map_location="cpu", weights_only=False)
            restored = build_model(False)
            restored.load_state_dict(saved["model_state"])

    checks = [
        _check("tensor_operation", tensor_test, requested, actual),
        _check("model_forward_backward_optimizer", model_test, requested, actual),
        _check("gradcam_dummy", gradcam_test, requested, actual),
        _check("integrated_gradients_dummy", ig_test, requested, actual),
        _check("portable_checkpoint", checkpoint_test, requested, actual),
    ]
    report = {
        "python_version": sys.version,
        "operating_system": platform.platform(),
        "cpu": platform.processor(),
        "ram_bytes": psutil.virtual_memory().total,
        "pytorch_version": torch.__version__,
        "torchvision_version": _version("torchvision"),
        "captum_version": _version("captum"),
        "torch_directml_version": _version("torch-directml"),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if actual == "cuda" else None,
        "selected_backend": actual,
        "dataset_root": config["paths"]["dataset_root"],
        "available_disk_space": shutil.disk_usage(layout["output"]).free,
        "checks": checks,
    }
    atomic_json(layout["output"] / "environment_report.json", report)
    return report


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    report = check_environment(config)
    failed = [check for check in report["checks"] if check["status"] != "passed"]
    write_status(
        config["paths"]["output_dir"],
        "environment",
        "failed" if failed else "passed",
        actual_backend=report["selected_backend"],
        error_message=f"{len(failed)} checks failed" if failed else None,
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

