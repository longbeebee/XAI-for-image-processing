"""Stage-specific test evaluation, threshold selection, and calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

from .calibration import calibration_summary, fit_temperature, probabilities
from .dataset import QualityAwareDataset
from .model import QualityAwareFAS


def collect(model, loader, device):
    model.eval(); logits=[]; labels=[]; quality=[]; uncertainty=[]
    with torch.no_grad():
        for batch in loader:
            out=model(batch["image"].to(device)); logits.append(out["logits"].cpu().numpy()); labels.append(batch["label"].numpy()); quality.append(out["quality"].cpu().numpy()); uncertainty.append(out["uncertainty"].cpu().numpy())
    return np.concatenate(logits), np.concatenate(labels), np.concatenate(quality), np.concatenate(uncertainty)


def threshold_min_acer(labels: np.ndarray, scores: np.ndarray) -> float:
    candidates = np.unique(scores)
    best = (float("inf"), 0.5)
    for threshold in candidates:
        predicted = scores >= threshold
        tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
        apcer = fn / max(fn + tp, 1)
        bpcer = fp / max(fp + tn, 1)
        acer = (apcer + bpcer) / 2
        if acer < best[0]: best = (acer, float(threshold))
    return best[1]


def classification_summary(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    predicted = (scores >= threshold).astype(int); tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {"threshold": float(threshold), "accuracy": float(accuracy_score(labels, predicted)), "precision": float(precision_score(labels, predicted, zero_division=0)), "recall": float(recall_score(labels, predicted, zero_division=0)), "f1": float(f1_score(labels, predicted, zero_division=0)), "roc_auc": float(roc_auc_score(labels, scores)), "average_precision": float(average_precision_score(labels, scores)), "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]], "apcer": float(fn / max(fn + tp, 1)), "bpcer": float(fp / max(fp + tn, 1)), "acer": float((fn / max(fn + tp, 1) + fp / max(fp + tn, 1)) / 2), "spoof_detection_rate": float(tp / max(fn + tp, 1))}


def evaluate(config: dict, checkpoint: Path, protocol_dir: Path, output_dir: Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() and config["device"]["preferred"] == "cuda" else "cpu")
    if device.type != "cuda" and not config["device"].get("allow_cpu_fallback", False): raise RuntimeError("CUDA is required by this experiment.")
    model = QualityAwareFAS(bool(config["model"].get("pretrained", True))).to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False); model.load_state_dict(state["model_state"])
    val = QualityAwareDataset(protocol_dir / "val_subject_disjoint.parquet", config["paths"]["dataset_root"], int(config["training"]["image_size"]), False)
    test = QualityAwareDataset(protocol_dir / "test_subject_disjoint.parquet", config["paths"]["dataset_root"], int(config["training"]["image_size"]), False)
    val_logits, val_labels, _, _ = collect(model, DataLoader(val, batch_size=int(config["training"]["batch_size"])), device)
    test_logits, test_labels, quality, uncertainty = collect(model, DataLoader(test, batch_size=int(config["training"]["batch_size"])), device)
    temperature = fit_temperature(val_logits, val_labels) if config.get("calibration", {}).get("enabled", False) else 1.0
    val_probs = probabilities(val_logits, temperature)[:, 1]; test_probs = probabilities(test_logits, temperature)[:, 1]
    threshold = threshold_min_acer(val_labels, val_probs)
    result = {"validation": {"classification": classification_summary(val_labels, val_probs, threshold), "calibration": calibration_summary(val_logits, val_labels, temperature)}, "test": {"classification": classification_summary(test_labels, test_probs, threshold), "calibration": calibration_summary(test_logits, test_labels, temperature), "mean_quality": quality.mean(axis=0).tolist(), "mean_uncertainty": float(uncertainty.mean()), "uncertainty_error_correlation": float(np.corrcoef(uncertainty, (test_probs >= threshold).astype(int) != test_labels)[0, 1]) if len(test_labels) > 1 else 0.0}}
    output_dir.mkdir(parents=True, exist_ok=True); (output_dir / "stage_evaluation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config", required=True, type=Path); parser.add_argument("--checkpoint", required=True, type=Path); parser.add_argument("--protocol-dir", required=True, type=Path); parser.add_argument("--output-dir", required=True, type=Path); args=parser.parse_args()
    config=yaml.safe_load(args.config.read_text(encoding="utf-8")); print(json.dumps(evaluate(config,args.checkpoint,args.protocol_dir,args.output_dir),indent=2))


if __name__ == "__main__": main()
