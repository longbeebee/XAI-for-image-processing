"""Evaluate one ablation checkpoint on the frozen subject-disjoint protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .final_evaluation import evaluate_final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--protocol-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--variant", required=True, choices=["stage1_only", "stage1_stage2", "full_model"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--xai-samples", default=300, type=int)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    result = evaluate_final(config, args.checkpoint, args.protocol_dir, args.output_dir, args.xai_samples)
    result["ablation_variant"] = args.variant
    result["seed"] = args.seed
    result["uncertainty_status"] = "not_calibrated" if args.variant == "stage1_only" else "learned_error_aware"
    (args.output_dir / "ablation_metadata.json").write_text(json.dumps({
        "variant": args.variant,
        "seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "uncertainty_status": result["uncertainty_status"],
    }, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "final_evaluation.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
