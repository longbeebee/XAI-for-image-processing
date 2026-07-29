from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from src.device import DeviceManager
from src.models import build_model
from src.utils import portable_state_dict


def test_cpu_resolve() -> None:
    manager = DeviceManager("cpu")
    assert manager.get_logical_device_name() == "cpu"
    assert manager.supports_mixed_precision() is False


def test_explicit_cuda_unavailable_errors() -> None:
    with patch("torch.cuda.is_available", return_value=False), pytest.raises(RuntimeError):
        DeviceManager("cuda")


def test_auto_falls_back_to_cpu() -> None:
    with patch("torch.cuda.is_available", return_value=False):
        assert DeviceManager("auto").get_logical_device_name() == "cpu"


def test_checkpoint_tensors_are_cpu() -> None:
    state = portable_state_dict(build_model(False))
    assert state and all(tensor.device.type == "cpu" for tensor in state.values())

