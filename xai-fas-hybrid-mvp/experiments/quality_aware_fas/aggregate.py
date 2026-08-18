"""Aggregate independent seed results for a reproducible experiment table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


def aggregate(output_dir: Path) -> dict:
    rows=[]
    for path in sorted(output_dir.glob("seed_*/stage_evaluation.json")):
        data=json.loads(path.read_text(encoding="utf-8")); test=data["test"]; rows.append({"seed":path.parent.name,"accuracy":test["classification"]["accuracy"],"acer":test["classification"]["acer"],"apcer":test["classification"]["apcer"],"bpcer":test["classification"]["bpcer"],"roc_auc":test["classification"]["roc_auc"],"ece":test["calibration"]["ece"],"brier":test["calibration"]["brier"]})
    if not rows: raise FileNotFoundError(f"No stage_evaluation.json files found in {output_dir}")
    summary={"seed_count":len(rows),"runs":rows,"mean":{},"std":{}}
    for key in rows[0]:
        if key=="seed": continue
        values=np.array([row[key] for row in rows],dtype=float); summary["mean"][key]=float(values.mean()); summary["std"][key]=float(values.std(ddof=1)) if len(values)>1 else 0.0
    (output_dir/"multi_seed_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8"); return summary


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--output-dir",required=True,type=Path); args=parser.parse_args(); print(json.dumps(aggregate(args.output_dir),indent=2))
