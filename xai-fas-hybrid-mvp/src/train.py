"""Two-stage MobileNetV3 training with portable epoch checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import config_hash, ensure_output_layout, load_config
from .datasets.celeba_spoof_dataset import CelebASpoofDataset
from .device import DeviceManager
from .metrics.classification import select_threshold
from .models import build_model, set_training_stage
from .job_fallback import run_with_device_fallback
from .reproducibility import make_generator, seed_everything, seed_worker
from .utils import save_checkpoint, write_status


def _loader(
    dataset: CelebASpoofDataset,
    config: dict[str, Any],
    shuffle: bool,
) -> DataLoader[Any]:
    workers = int(config["training"].get("num_workers", 0))
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=bool(config["training"].get("pin_memory", False)),
        persistent_workers=bool(config["training"].get("persistent_workers", False)) and workers > 0,
        worker_init_fn=seed_worker,
        generator=make_generator(int(config.get("seed", 42))),
    )


@torch.no_grad()
def _validation_predictions(
    model: torch.nn.Module, loader: DataLoader[Any], manager: DeviceManager
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels, probabilities = [], []
    for batch in loader:
        images = manager.move_tensor(batch["image"])
        logits = model(images)
        labels.extend(batch["label"].numpy().tolist())
        probabilities.extend(logits.softmax(dim=1)[:, 1].detach().cpu().numpy().tolist())
    return np.asarray(labels), np.asarray(probabilities)


def train(config: dict[str, Any], resume: bool = False) -> Path:
    """Train, select on validation ROC-AUC, and return the best checkpoint."""
    seed_everything(int(config.get("seed", 42)))
    layout = ensure_output_layout(config)
    manager = DeviceManager(
        config["device"].get("preferred", "auto"),
        bool(config["device"].get("allow_cpu_fallback", False)),
    )
    processed = Path(config["paths"]["processed_dir"])
    train_dataset = CelebASpoofDataset(
        processed / "train_subset.parquet",
        config["paths"]["dataset_root"],
        int(config["training"].get("image_size", 224)),
        training=True,
    )
    val_dataset = CelebASpoofDataset(
        processed / "val_subset.parquet",
        config["paths"]["dataset_root"],
        int(config["training"].get("image_size", 224)),
    )
    train_loader = _loader(train_dataset, config, True)
    val_loader = _loader(val_dataset, config, False)
    model = manager.move_model(
        build_model(
            bool(config["model"].get("pretrained", True)),
            int(config["model"].get("num_classes", 2)),
        )
    )
    start_epoch = 0
    last_path = layout["checkpoints"] / "last_model.pt"
    if resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        model.to(manager.get_torch_device())
        start_epoch = int(checkpoint["epoch"]) + 1

    criterion = torch.nn.CrossEntropyLoss()
    best_auc = -float("inf")
    best_path = layout["checkpoints"] / "best_model.pt"
    total_epochs = int(config["training"]["total_epochs"])
    head_epochs = int(config["training"].get("head_only_epochs", 2))
    for epoch in range(start_epoch, total_epochs):
        head_only = epoch < head_epochs
        set_training_stage(
            model,
            head_only,
            int(config["training"].get("unfreeze_last_blocks", 3)),
        )
        learning_rate = float(
            config["training"].get(
                "head_learning_rate" if head_only else "fine_tune_learning_rate",
                1e-3 if head_only else 1e-4,
            )
        )
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=learning_rate,
            weight_decay=float(config["training"].get("weight_decay", 1e-4)),
        )
        use_amp = (
            bool(config["training"].get("mixed_precision", False))
            and manager.supports_mixed_precision()
        )
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        model.train()
        for batch in train_loader:
            images = manager.move_tensor(batch["image"])
            labels = manager.move_tensor(batch["label"])
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["training"].get("gradient_clip_norm", 1.0))
            )
            scaler.step(optimizer)
            scaler.update()
        y_true, p_spoof = _validation_predictions(model, val_loader, manager)
        threshold = select_threshold(
            y_true, p_spoof, config.get("evaluation", {}).get("threshold_strategy", "min_acer")
        )
        from sklearn.metrics import roc_auc_score

        validation_auc = (
            float(roc_auc_score(y_true, p_spoof)) if len(np.unique(y_true)) > 1 else 0.0
        )
        state = {
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": None,
            "scaler_state": scaler.state_dict() if use_amp else None,
            "epoch": epoch,
            "training_stage": "head" if head_only else "final_feature_blocks",
            "validation_metrics": {"roc_auc": validation_auc},
            "selected_threshold": threshold,
            "config_snapshot": config,
            "label_mapping": {0: "real", 1: "spoof"},
            "dataset_manifest_hash": None,
            "subset_manifest_hash": None,
            "actual_training_backend": manager.get_logical_device_name(),
            "pytorch_version": torch.__version__,
        }
        save_checkpoint(last_path, model, **state)
        if validation_auc > best_auc:
            best_auc = validation_auc
            save_checkpoint(best_path, model, **state)
    return best_path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    try:
        run_with_device_fallback(
            config, "training", lambda selected: train(selected, args.resume)
        )
        write_status(
            config["paths"]["output_dir"],
            "training",
            "completed",
            config_hash=config_hash(config),
        )
    except Exception as exc:
        write_status(
            config["paths"]["output_dir"],
            "training",
            "failed",
            config_hash=config_hash(config),
            error_message=f"{type(exc).__name__}: {exc}",
        )
        raise


if __name__ == "__main__":
    main()
