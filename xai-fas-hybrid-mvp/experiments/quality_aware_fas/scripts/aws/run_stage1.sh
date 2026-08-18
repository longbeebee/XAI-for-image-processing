#!/usr/bin/env bash
set -euo pipefail
python -m experiments.quality_aware_fas.run_stage1 \
  --config experiments/quality_aware_fas/configs/aws_quality_aware_stage1.yaml \
  --resume
