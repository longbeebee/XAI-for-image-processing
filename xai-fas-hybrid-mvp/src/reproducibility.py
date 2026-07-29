"""Reproducibility helpers for Python, NumPy, and PyTorch."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed all supported random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker from PyTorch's worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """Return a deterministically seeded DataLoader generator."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator

