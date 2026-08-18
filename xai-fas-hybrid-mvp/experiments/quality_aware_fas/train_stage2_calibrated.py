"""Improved uncertainty calibration on a frozen classifier.

The classifier and representation are frozen.  The uncertainty head is trained
against both continuous prediction error and a binary error indicator, with a
pairwise ranking term so that harder examples receive larger uncertainty.
Validation AUROC for error detection selects the saved checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from .dataset import QualityAwareDataset
from .model import QualityAwareFAS


def uncertainty_score(raw: torch.Tensor) -> torch.Tensor:
    return 1.0 - torch.exp(-raw.clamp_min(0.0))


def ranking_loss(score: torch.Tensor, error: torch.Tensor, margin: float = 0.05) -> torch.Tensor:
    order = error[:, None] - error[None, :]
    valid = order > 0.05
    if not valid.any():
        return score.new_zeros(())
    score_gap = score[:, None] - score[None, :]
    return torch.relu(margin - score_gap[valid]).mean()


def calibration_loss(raw: torch.Tensor, probabilities: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    score = uncertainty_score(raw)
    continuous_error = (probabilities.detach() - labels.float()).abs().clamp(0.0, 1.0)
    binary_error = (probabilities.detach().round() != labels).float()
    regression = torch.nn.functional.smooth_l1_loss(score, continuous_error)
    binary = torch.nn.functional.binary_cross_entropy(score.clamp(1e-5, 1.0 - 1e-5), binary_error)
    ranking = ranking_loss(score, continuous_error)
    total = 0.5 * regression + 0.3 * binary + 0.2 * ranking
    return total, {"regression": float(regression.detach()), "binary": float(binary.detach()), "ranking": float(ranking.detach()), "total": float(total.detach())}


def collect(model, loader, device):
    scores, labels, uncertainties = [], [], []
    with torch.no_grad():
        for batch in loader:
            output = model(batch["image"].to(device))
            scores.append(output["logits"].softmax(1)[:, 1].cpu().numpy())
            labels.append(batch["label"].numpy())
            uncertainties.append(uncertainty_score(output["uncertainty"]).cpu().numpy())
    return np.concatenate(scores), np.concatenate(labels), np.concatenate(uncertainties)


def uncertainty_metrics(scores: np.ndarray, labels: np.ndarray, uncertainty: np.ndarray) -> dict[str, float]:
    predictions = (scores >= 0.5).astype(int)
    errors = (predictions != labels).astype(int)
    result = {"mean_uncertainty": float(uncertainty.mean()), "error_rate": float(errors.mean())}
    if len(np.unique(errors)) > 1:
        result["error_detection_auroc"] = float(roc_auc_score(errors, uncertainty))
        result["error_detection_auprc"] = float(average_precision_score(errors, uncertainty))
    order = np.argsort(uncertainty)
    sorted_errors = errors[order]
    coverage = np.arange(1, len(errors) + 1) / len(errors)
    risk = np.cumsum(sorted_errors) / np.arange(1, len(errors) + 1)
    result["aurc"] = float(np.trapezoid(risk, coverage))
    return result


def run(config: dict, checkpoint: Path, output_dir: Path, seed: int = 42) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and config["device"]["preferred"] == "cuda" else "cpu")
    if device.type != "cuda" and not config["device"].get("allow_cpu_fallback", False):
        raise RuntimeError("CUDA is required for calibrated uncertainty training.")
    protocol = Path(config["paths"].get("protocol_dir", config["paths"]["processed_dir"]))
    image_size = int(config["training"]["image_size"])
    train_set = QualityAwareDataset(protocol / "train_subject_disjoint.parquet", config["paths"]["dataset_root"], image_size, True, seed)
    val_set = QualityAwareDataset(protocol / "val_subject_disjoint.parquet", config["paths"]["dataset_root"], image_size, False, seed)
    batch_size = int(config["training"]["batch_size"])
    workers = int(config["training"].get("num_workers", 0))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    model = QualityAwareFAS(bool(config["model"].get("pretrained", True))).to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"])
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.uncertainty_head.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.AdamW(model.uncertainty_head.parameters(), lr=float(config["training"].get("calibrated_uncertainty_learning_rate", 5e-4)), weight_decay=float(config["training"]["weight_decay"]))
    epochs = int(config["training"].get("calibrated_uncertainty_epochs", 8))
    history = []
    best_key = (-float("inf"), float("inf"))
    best_state = None
    output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(epochs):
        model.train(); losses = []
        for batch in train_loader:
            images = batch["image"].to(device); labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(images)
            probabilities = output["logits"].softmax(1)[:, 1]
            loss, details = calibration_loss(output["uncertainty"], probabilities, labels)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.uncertainty_head.parameters(), 1.0); optimizer.step()
            losses.append(details)
        model.eval()
        val_scores, val_labels, val_uncertainty = collect(model, val_loader, device)
        validation = uncertainty_metrics(val_scores, val_labels, val_uncertainty)
        epoch_result = {"epoch": epoch, "train_loss": float(np.mean([item["total"] for item in losses])), "validation": validation}
        history.append(epoch_result)
        key = (validation.get("error_detection_auroc", -float("inf")), -validation["aurc"])
        if key > best_key:
            best_key = key
            best_state = {"model_state": model.state_dict(), "seed": seed, "history": history, "method": "continuous_binary_pairwise_uncertainty_calibration"}
    if best_state is None:
        raise RuntimeError("No calibrated uncertainty checkpoint was produced.")
    target = output_dir / "calibrated_uncertainty_model.pt"
    torch.save(best_state, target)
    result = {"stage": "improved_uncertainty_calibration", "seed": seed, "checkpoint": str(target), "selection": "validation_error_detection_auroc_then_aurc", "history": history, "best_validation": history[int(np.argmax([item["validation"].get("error_detection_auroc", -1.0) for item in history]))]["validation"]}
    (output_dir / "stage2_calibrated_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    print(json.dumps(run(config, args.checkpoint, args.output_dir, args.seed), indent=2))


if __name__ == "__main__":
    main()
