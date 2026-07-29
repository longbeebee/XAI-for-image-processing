#!/usr/bin/env bash
set -euo pipefail

python -m src.run_smoke_test --config configs/aws_smoke_test.yaml

