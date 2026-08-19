"""Train the explanation-consistency stage in an isolated output directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from .model import QualityAwareFAS
from .paired_dataset import PairedQualityDataset
from .gradcam_training import consistency_loss, differentiable_gradcam
from .train_stage2_calibrated import calibration_loss


def saliency(model, images):
    images = images.detach().requires_grad_(True)
    logits = model(images)["logits"]
    target = logits.max(dim=1).values.sum()
    gradients = torch.autograd.grad(target, images, create_graph=True, retain_graph=True)[0]
    maps = gradients.abs().mean(dim=1).flatten(1)
    return maps / (maps.norm(dim=1, keepdim=True) + 1e-8)


def run(config: dict, checkpoint: Path, output_dir: Path, seed: int = 42) -> dict:
    torch.manual_seed(seed); device=torch.device("cuda" if torch.cuda.is_available() and config["device"]["preferred"] == "cuda" else "cpu")
    if device.type != "cuda" and not config["device"].get("allow_cpu_fallback", False): raise RuntimeError("CUDA is required by this experiment.")
    protocol=Path(config["paths"].get("protocol_dir", config["paths"]["processed_dir"]))
    dataset=PairedQualityDataset(protocol/"train_subject_disjoint.parquet",config["paths"]["dataset_root"],int(config["training"]["image_size"]),seed)
    loader=DataLoader(dataset,batch_size=int(config["training"]["batch_size"]),shuffle=True,num_workers=int(config["training"].get("num_workers",0)),pin_memory=True)
    model=QualityAwareFAS(bool(config["model"].get("pretrained",True))).to(device)
    if checkpoint.is_file(): model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=False)["model_state"])
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(config["training"]["fine_tune_learning_rate"]),weight_decay=float(config["training"]["weight_decay"]))
    history=[]
    for epoch in range(int(config["training"].get("consistency_epochs",4))):
        model.train(); total=0.0
        for batch in loader:
            original=batch["original"].to(device); degraded=batch["degraded"].to(device); labels=batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True); original_out=model(original); degraded_out=model(degraded)
            classification=(torch.nn.functional.cross_entropy(original_out["logits"],labels)+torch.nn.functional.cross_entropy(degraded_out["logits"],labels))/2
            representation=torch.nn.functional.mse_loss(original_out["representation"],degraded_out["representation"].detach())
            original_map=differentiable_gradcam(model,original); degraded_map=differentiable_gradcam(model,degraded)
            explanation=consistency_loss(original_map, degraded_map)
            original_uncertainty, _ = calibration_loss(original_out["uncertainty"], original_out["logits"].softmax(1)[:, 1], labels)
            degraded_uncertainty, _ = calibration_loss(degraded_out["uncertainty"], degraded_out["logits"].softmax(1)[:, 1], labels)
            uncertainty_preservation=(original_uncertainty+degraded_uncertainty)/2
            loss=classification+float(config["training"].get("representation_consistency_weight",0.1))*representation+float(config["training"].get("explanation_loss_weight",0.05))*explanation+float(config["training"].get("uncertainty_preservation_weight",0.20))*uncertainty_preservation
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),float(config["training"]["gradient_clip_norm"])); optimizer.step(); total += float(loss.detach())
        history.append({"epoch":epoch,"loss":total/max(len(loader),1),"uncertainty_preservation_weight":float(config["training"].get("uncertainty_preservation_weight",0.20))})
    output_dir.mkdir(parents=True,exist_ok=True); torch.save({"model_state":model.state_dict(),"seed":seed,"history":history},output_dir/"best_explanation_consistent_model.pt")
    result={"stage":"explanation_consistency","seed":seed,"device":str(device),"history":history}; (output_dir/"stage3_result.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8"); return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True,type=Path); parser.add_argument("--checkpoint",required=True,type=Path); parser.add_argument("--output-dir",required=True,type=Path); parser.add_argument("--seed",type=int,default=42); args=parser.parse_args(); config=yaml.safe_load(args.config.read_text(encoding="utf-8")); print(json.dumps(run(config,args.checkpoint,args.output_dir,args.seed),indent=2))


if __name__ == "__main__": main()
