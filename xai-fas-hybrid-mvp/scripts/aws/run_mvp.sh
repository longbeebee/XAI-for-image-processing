#!/usr/bin/env bash
set -euo pipefail

python -m src.run_mvp --config configs/aws_mvp.yaml --resume

