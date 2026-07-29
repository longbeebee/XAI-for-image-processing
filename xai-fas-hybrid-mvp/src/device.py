"""Backend resolution without unsafe implicit fallback."""

from __future__ import annotations

import gc
import platform
from typing import Any

import torch


class DeviceManager:
    """Resolve and safely operate a CPU, CUDA, or DirectML backend."""

    def __init__(self, preferred: str = "auto", allow_cpu_fallback: bool = False) -> None:
        self.preferred = preferred.lower()
        self.allow_cpu_fallback = allow_cpu_fallback
        self._device: Any
        self._logical_name: str
        self._device, self._logical_name = self.resolve_device(self.preferred)

    def resolve_device(self, preferred: str) -> tuple[Any, str]:
        """Resolve a backend and validate it with a tensor operation."""
        preferred = preferred.lower()
        if preferred not in {"auto", "cpu", "cuda", "directml"}:
            raise ValueError(f"Unsupported device preference: {preferred}")
        if preferred == "cpu":
            return torch.device("cpu"), "cpu"
        if preferred in {"auto", "cuda"} and torch.cuda.is_available():
            try:
                test = torch.tensor([1.0], device="cuda")
                _ = (test + 1).cpu()
                return torch.device("cuda"), "cuda"
            except Exception as exc:
                if preferred == "cuda" and not self.allow_cpu_fallback:
                    raise RuntimeError("CUDA was explicitly requested but failed validation.") from exc
        elif preferred == "cuda" and not self.allow_cpu_fallback:
            raise RuntimeError("CUDA was explicitly requested but is unavailable.")
        if preferred in {"auto", "directml"} and platform.system().lower() == "windows":
            try:
                import torch_directml

                directml_device = torch_directml.device()
                test = torch.tensor([1.0]).to(directml_device)
                _ = (test + 1).cpu()
                return directml_device, "directml"
            except Exception as exc:
                if preferred == "directml" and not self.allow_cpu_fallback:
                    raise RuntimeError("DirectML was explicitly requested but failed.") from exc
        elif preferred == "directml" and not self.allow_cpu_fallback:
            raise RuntimeError("DirectML was explicitly requested but is unavailable.")
        return torch.device("cpu"), "cpu"

    def get_logical_device_name(self) -> str:
        """Return cpu, cuda, or directml independently from device.type."""
        return self._logical_name

    def get_torch_device(self) -> Any:
        """Return the underlying torch-compatible device."""
        return self._device

    def move_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Move a model to the selected device."""
        return model.to(self._device)

    def move_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move a tensor to the selected device."""
        return tensor.to(self._device)

    def synchronize_if_supported(self) -> None:
        """Synchronize only the actual CUDA backend."""
        if self._logical_name == "cuda":
            torch.cuda.synchronize()

    def release_resources(self) -> None:
        """Release Python and backend caches safely."""
        gc.collect()
        if self._logical_name == "cuda":
            torch.cuda.empty_cache()

    def supports_mixed_precision(self) -> bool:
        """MVP AMP is intentionally restricted to CUDA."""
        return self._logical_name == "cuda"

