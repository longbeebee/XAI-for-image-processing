"""Evaluate Grad-CAM and Integrated Gradients consistency for the new model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .model import QualityAwareFAS
from .paired_dataset import PairedQualityDataset


def normalize(values):
    values = values.flatten(1); return values / (values.norm(dim=1, keepdim=True) + 1e-8)


def gradcam(model, images):
    activations=[]; gradients=[]
    layer = [m for m in model.features.modules() if isinstance(m, torch.nn.Conv2d)][-1]
    handle_a = layer.register_forward_hook(lambda _, __, output: activations.append(output))
    handle_g = layer.register_full_backward_hook(lambda _, __, output: gradients.append(output[0]))
    model.zero_grad(set_to_none=True); score=model(images)["logits"].max(dim=1).values.sum(); score.backward(retain_graph=True)
    handle_a.remove(); handle_g.remove()
    weights=gradients[-1].mean(dim=(2,3),keepdim=True); maps=torch.relu((weights*activations[-1]).sum(dim=1,keepdim=True)); maps=torch.nn.functional.interpolate(maps,size=images.shape[-2:],mode="bilinear",align_corners=False).squeeze(1); return normalize(maps)


def integrated_gradients(model, images, steps=24):
    baseline=torch.zeros_like(images); total=torch.zeros_like(images)
    for alpha in torch.linspace(0.0,1.0,steps,device=images.device):
        point=(baseline+alpha*(images-baseline)).detach().requires_grad_(True); model.zero_grad(set_to_none=True); score=model(point)["logits"].max(dim=1).values.sum(); grad=torch.autograd.grad(score,point)[0]; total += grad
    maps=((images-baseline)*total/steps).abs().mean(dim=1); return normalize(maps)


def summarize(original, degraded):
    cosine=(original*degraded).sum(dim=1); k=max(1,int(original.shape[1]*0.1)); a=original.topk(k,dim=1).indices; b=degraded.topk(k,dim=1).indices; iou=[]
    for x,y in zip(a,b): iou.append(float(len(set(x.tolist())&set(y.tolist()))/max(len(set(x.tolist())|set(y.tolist())),1)))
    return {"cosine_similarity": float(cosine.mean()), "top10_iou": float(np.mean(iou))}


def evaluate(config, checkpoint: Path, protocol_dir: Path, output_dir: Path, samples: int = 300):
    device=torch.device("cuda" if torch.cuda.is_available() and config["device"]["preferred"]=="cuda" else "cpu")
    if device.type != "cuda" and not config["device"].get("allow_cpu_fallback",False): raise RuntimeError("CUDA is required by this experiment.")
    model=QualityAwareFAS(bool(config["model"].get("pretrained",True))).to(device); model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=False)["model_state"]); model.eval()
    data=PairedQualityDataset(protocol_dir/"test_subject_disjoint.parquet",config["paths"]["dataset_root"],int(config["training"]["image_size"])); loader=DataLoader(data,batch_size=1,shuffle=False)
    records={"gradcam":[],"integrated_gradients":[]}
    for index,batch in enumerate(loader):
        if index>=samples: break
        original=batch["original"].to(device); degraded=batch["degraded"].to(device)
        for name,fn in [("gradcam",gradcam),("integrated_gradients",integrated_gradients)]:
            first=fn(model,original); second=fn(model,degraded); records[name].append(summarize(first,second))
    summary={name:{key:float(np.mean([row[key] for row in rows])) for key in rows[0]} for name,rows in records.items() if rows}
    output_dir.mkdir(parents=True,exist_ok=True); (output_dir/"xai_consistency.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8"); return summary


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True,type=Path); parser.add_argument("--checkpoint",required=True,type=Path); parser.add_argument("--protocol-dir",required=True,type=Path); parser.add_argument("--output-dir",required=True,type=Path); args=parser.parse_args(); config=yaml.safe_load(args.config.read_text(encoding="utf-8")); print(json.dumps(evaluate(config,args.checkpoint,args.protocol_dir,args.output_dir),indent=2))


if __name__ == "__main__": main()
