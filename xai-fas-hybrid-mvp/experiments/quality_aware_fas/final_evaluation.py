"""Complete evaluation for the final uncertainty and Grad-CAM-trained model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from .dataset import QualityAwareDataset
from .evaluation import collect, classification_summary, threshold_min_acer
from .xai_evaluation import evaluate as evaluate_xai
from .faithfulness_sanity import evaluate_faithfulness, evaluate_sanity


def evaluate_final(config: dict, checkpoint: Path, protocol_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and config["device"]["preferred"] == "cuda" else "cpu")
    if device.type != "cuda" and not config["device"].get("allow_cpu_fallback", False): raise RuntimeError("CUDA is required by this experiment.")
    from .model import QualityAwareFAS
    model = QualityAwareFAS(bool(config["model"].get("pretrained", True))).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state"]); model.eval()
    val = QualityAwareDataset(protocol_dir / "val_subject_disjoint.parquet", config["paths"]["dataset_root"], int(config["training"]["image_size"]), False)
    test = QualityAwareDataset(protocol_dir / "test_subject_disjoint.parquet", config["paths"]["dataset_root"], int(config["training"]["image_size"]), False)
    val_logits, val_labels, _, _ = collect(model, DataLoader(val, batch_size=32), device); test_logits, test_labels, quality, uncertainty = collect(model, DataLoader(test, batch_size=32), device)
    val_scores = torch.softmax(torch.tensor(val_logits), 1)[:, 1].numpy(); test_scores = torch.softmax(torch.tensor(test_logits), 1)[:, 1].numpy(); threshold = threshold_min_acer(val_labels, val_scores)
    predictions = (test_scores >= threshold).astype(int)
    test_frame = pd.read_parquet(protocol_dir / "test_subject_disjoint.parquet").reset_index(drop=True)
    test_frame["score"] = test_scores; test_frame["prediction"] = predictions
    test_frame["uncertainty"] = uncertainty
    test_frame["quality_brightness"] = quality[:, 0]
    test_frame["quality_blur"] = quality[:, 1]
    test_frame["quality_jpeg"] = quality[:, 2]
    test_frame.to_parquet(output_dir / "test_predictions.parquet", index=False)
    spoof_metrics = []
    if "spoof_type" in test_frame:
        for spoof_type, group in test_frame[test_frame["label_id"] == 1].groupby("spoof_type"):
            spoof_metrics.append({"spoof_type": str(spoof_type), "sample_count": int(len(group)), "false_acceptance_rate": float((group["prediction"] == 0).mean()), "spoof_detection_rate": float((group["prediction"] == 1).mean()), "mean_spoof_score": float(group["score"].mean())})
    paired = pd.read_parquet(protocol_dir / "test_subject_disjoint.parquet")
    paired_set = QualityAwareDataset(protocol_dir / "test_subject_disjoint.parquet", config["paths"]["dataset_root"], int(config["training"]["image_size"]), False)
    unchanged=[]
    for index in range(min(len(paired_set), 1000)):
        item=paired_set[index]; image=item["image"].unsqueeze(0).to(device)
        with torch.no_grad(): original=int(model(image)["logits"].argmax(1).item())
        unchanged.append(original == int(item["label"]))
    start=time.perf_counter()
    with torch.no_grad():
        for index in range(min(50,len(test))): model(test[index]["image"].unsqueeze(0).to(device))
    prediction_ms=(time.perf_counter()-start)/max(min(50,len(test)),1)*1000
    xai_summary=evaluate_xai(config,checkpoint,protocol_dir,output_dir/"xai",samples=300)
    result={"checkpoint":str(checkpoint),"device":str(device),"classification":classification_summary(test_labels,test_scores,threshold),"prediction_accuracy":float(np.mean(unchanged)),"quality_mean":quality.mean(0).tolist(),"uncertainty_mean":float(uncertainty.mean()),"spoof_type_metrics":spoof_metrics,"runtime":{"classifier_prediction_mean_ms":float(prediction_ms)},"xai_consistency":xai_summary,"faithfulness":evaluate_faithfulness(model,test,device,samples=20),"sanity":evaluate_sanity(model,test,device,samples=20)}
    (output_dir/"final_evaluation.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8"); return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True,type=Path); parser.add_argument("--checkpoint",required=True,type=Path); parser.add_argument("--protocol-dir",required=True,type=Path); parser.add_argument("--output-dir",required=True,type=Path); args=parser.parse_args(); config=yaml.safe_load(args.config.read_text(encoding="utf-8")); print(json.dumps(evaluate_final(config,args.checkpoint,args.protocol_dir,args.output_dir),indent=2))


if __name__ == "__main__": main()
