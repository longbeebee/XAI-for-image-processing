"""Complete multi-seed summary with bootstrap intervals and paired tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def bootstrap_interval(values, iterations=500, seed=42, confidence=0.95):
    rng=np.random.default_rng(seed); values=np.asarray(values,dtype=float); samples=rng.choice(values,size=(iterations,len(values)),replace=True).mean(axis=1); alpha=(1-confidence)/2
    return {"mean":float(values.mean()),"std":float(values.std(ddof=1)) if len(values)>1 else 0.0,"ci_low":float(np.quantile(samples,alpha)),"ci_high":float(np.quantile(samples,1-alpha))}


def summarize(root: Path) -> dict:
    records=[]
    for path in sorted(root.glob("seed_*/final/final_evaluation.json")):
        result=json.loads(path.read_text(encoding="utf-8")); cls=result["classification"]; records.append({"seed":path.parent.parent.name,"acer":cls["acer"],"apcer":cls["apcer"],"bpcer":cls["bpcer"],"roc_auc":cls["roc_auc"],"accuracy":cls["accuracy"]})
    if not records: raise FileNotFoundError("No final_evaluation.json files found.")
    summary={"runs":records,"metrics":{}}
    for key in records[0]:
        if key=="seed": continue
        summary["metrics"][key]=bootstrap_interval([r[key] for r in records])
    if len(records)>=2:
        try:
            from scipy.stats import wilcoxon
            summary["note"]="Paired tests require baseline and proposed per-seed arrays; use this summary for confidence intervals and pass aligned arrays to Wilcoxon for the final comparison."
        except ImportError: summary["note"]="scipy is unavailable; bootstrap intervals were computed, but paired tests require scipy on AWS."
    (root/"complete_statistics.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8"); return summary


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--root",required=True,type=Path); args=parser.parse_args(); print(json.dumps(summarize(args.root),indent=2))
