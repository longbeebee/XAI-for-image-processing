"""Faithfulness and sanity evaluation for the final model."""

from __future__ import annotations

import copy
import numpy as np
import torch

from .xai_evaluation import gradcam


def probability(model, image):
    return float(model(image)["logits"].softmax(1)[0, 1].detach().cpu())


def curves(model, image, steps=10):
    original=image.clone(); saliency=gradcam(model,image).detach(); order=saliency[0].argsort(descending=True); total=image.shape[-1]*image.shape[-2]; deletion=[]; insertion=[]
    for fraction in np.linspace(0,1,steps+1):
        count=int(fraction*total); deleted=original.clone().flatten(2); inserted=torch.zeros_like(deleted); source=original.flatten(2); deleted[:,:,order[:count]]=0.0; inserted[:,:,order[:count]]=source[:,:,order[:count]]
        with torch.no_grad(): deletion.append(probability(model,deleted.view_as(image))); insertion.append(probability(model,inserted.view_as(image)))
    x=np.linspace(0,1,len(deletion)); return {"deletion_auc":float(np.trapezoid(deletion,x)),"insertion_auc":float(np.trapezoid(insertion,x)),"original_probability":probability(model,image)}


def evaluate_faithfulness(model, dataset, device, samples=20):
    rows=[curves(model,dataset[index]["image"].unsqueeze(0).to(device)) for index in range(min(samples,len(dataset)))]
    return {key:float(np.mean([row[key] for row in rows])) for key in rows[0]} if rows else {}


def evaluate_sanity(model, dataset, device, samples=20):
    original_state=copy.deepcopy(model.state_dict()); references=[]
    for index in range(min(samples,len(dataset))): references.append((index,gradcam(model,dataset[index]["image"].unsqueeze(0).to(device)).detach()))
    results=[]
    for level in (1,2,3):
        model.load_state_dict(original_state); modules=list(model.features); targets=[model.classifier] if level==1 else modules[-3:] if level==2 else list(model.children())
        for target in targets:
            for parameter in target.parameters(): parameter.data.normal_(0.0,0.02)
        similarities=[]
        for index,reference in references:
            current=gradcam(model,dataset[index]["image"].unsqueeze(0).to(device)).detach(); similarities.append(float((reference*current).sum()))
        results.append({"randomization_level":level,"cosine_similarity":float(np.mean(similarities))})
    model.load_state_dict(original_state); return results
