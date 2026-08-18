"""Validation-only calibration and uncertainty metrics."""

from __future__ import annotations

import numpy as np


def fit_temperature(logits: np.ndarray, labels: np.ndarray, steps: int = 200, learning_rate: float = 0.05) -> float:
    """Fit one positive temperature on validation logits only."""
    import torch

    values = torch.tensor(logits, dtype=torch.float32)
    targets = torch.tensor(labels, dtype=torch.long)
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.Adam([log_temperature], lr=learning_rate)
    for _ in range(steps):
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(0.05, 20.0)
        loss = torch.nn.functional.cross_entropy(values / temperature, targets)
        loss.backward()
        optimizer.step()
    return float(torch.exp(log_temperature).clamp(0.05, 20.0).detach())


def probabilities(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    shifted = (logits - logits.max(axis=1, keepdims=True)) / max(float(temperature), 1e-6)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def expected_calibration_error(probabilities_: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    confidence = probabilities_.max(axis=1)
    predictions = probabilities_.argmax(axis=1)
    error = 0.0
    for lower, upper in zip(np.linspace(0.0, 1.0, bins + 1)[:-1], np.linspace(0.0, 1.0, bins + 1)[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            error += selected.mean() * abs(float(confidence[selected].mean()) - float((predictions[selected] == labels[selected]).mean()))
    return float(error)


def calibration_summary(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> dict[str, float]:
    probs = probabilities(logits, temperature)
    one_hot = np.eye(probs.shape[1])[labels]
    return {"temperature": float(temperature), "ece": expected_calibration_error(probs, labels), "brier": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))), "mean_confidence": float(probs.max(axis=1).mean())}
