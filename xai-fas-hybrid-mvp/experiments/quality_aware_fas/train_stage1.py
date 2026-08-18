"""Train Stage 1 quality-aware model in its isolated output directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from .dataset import QualityAwareDataset
from .losses import quality_aware_loss
from .model import QualityAwareFAS


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def run(config: dict[str, Any], seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    paths = config["paths"]
    output = Path(paths["output_dir"]) / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    data_dir = Path(paths.get("protocol_dir", paths["processed_dir"]))
    train_file = data_dir / "train_subject_disjoint.parquet"
    val_file = data_dir / "val_subject_disjoint.parquet"
    if not train_file.is_file() or not val_file.is_file():
        raise FileNotFoundError("Run the subject-disjoint protocol before Stage 1 training.")
    device = torch.device("cuda" if torch.cuda.is_available() and config["device"]["preferred"] == "cuda" else "cpu")
    if device.type != "cuda" and not config["device"].get("allow_cpu_fallback", False):
        raise RuntimeError("Stage 1 requires CUDA; CPU fallback is disabled.")
    train_set = QualityAwareDataset(train_file, paths["dataset_root"], int(config["training"]["image_size"]), True, seed)
    val_set = QualityAwareDataset(val_file, paths["dataset_root"], int(config["training"]["image_size"]), False, seed)
    train_loader = DataLoader(train_set, batch_size=int(config["training"]["batch_size"]), shuffle=True, num_workers=int(config["training"].get("num_workers", 0)), pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=int(config["training"]["batch_size"]), shuffle=False, num_workers=int(config["training"].get("num_workers", 0)), pin_memory=True)
    model = QualityAwareFAS(bool(config["model"].get("pretrained", True))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["fine_tune_learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    best_auc = -float("inf")
    for epoch in range(int(config["training"]["total_epochs"])):
        model.train()
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            targets = batch["quality"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss, _ = quality_aware_loss(outputs, labels, targets, float(config["training"]["quality_loss_weight"]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip_norm"]))
            optimizer.step()
        model.eval(); labels_all=[]; scores_all=[]
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(batch["image"].to(device, non_blocking=True))
                labels_all.extend(batch["label"].numpy().tolist()); scores_all.extend(outputs["logits"].softmax(1)[:, 1].cpu().numpy().tolist())
        auc = float(roc_auc_score(labels_all, scores_all)) if len(set(labels_all)) > 1 else 0.0
        state = {"epoch": epoch, "seed": seed, "validation_roc_auc": auc, "model_state": model.state_dict(), "config": config}
        torch.save(state, output / "last_model.pt")
        if auc > best_auc:
            best_auc = auc; torch.save(state, output / "best_model.pt")
    result = {"seed": seed, "validation_roc_auc": best_auc, "output_dir": str(output), "device": str(device)}
    (output / "stage1_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True, type=Path); parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(); config = load_config(args.config)
    seeds = [args.seed] if args.seed is not None else config.get("seeds", [config.get("seed", 42)])
    print(json.dumps([run(config, int(seed)) for seed in seeds], indent=2))


if __name__ == "__main__":
    main()
