"""Stage 1 runner: audit baseline, build isolated protocol, then train quality-aware model."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from .audit_stage0 import audit
from .evaluation import evaluate
from .protocol import make_protocol
from .train_stage1 import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", action="store_true", help="Reserved for compatible checkpoint resume.")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = Path(config["paths"]["output_dir"])
    baseline = Path(config["paths"]["baseline_results_dir"])
    audit_result = audit(baseline, output)
    if not audit_result["ready_for_stage1"]:
        raise RuntimeError("Frozen baseline audit did not pass.")
    metadata = Path(config["paths"]["processed_dir"]) / "celeba_spoof_metadata.parquet"
    protocol_dir = Path(config["paths"].get("protocol_dir", output / "protocol"))
    protocol = make_protocol(metadata, protocol_dir, int(config.get("seed", 42)))
    if not protocol["subject_disjoint"]:
        raise RuntimeError("Subject-disjoint protocol failed: subject overlap remains.")
    stage_config = copy.deepcopy(config)
    stage_config["paths"]["protocol_dir"] = str(protocol_dir)
    results = []
    for seed in config.get("seeds", [config.get("seed", 42)]):
        training_result = run(stage_config, int(seed))
        evaluation_result = evaluate(stage_config, Path(training_result["output_dir"]) / "best_model.pt", protocol_dir, Path(training_result["output_dir"]))
        results.append({"training": training_result, "evaluation": evaluation_result})
    (output / "stage1_manifest.json").write_text(json.dumps({"stage": "quality_aware_learning", "baseline_audit": audit_result, "protocol": protocol, "results": results}, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
