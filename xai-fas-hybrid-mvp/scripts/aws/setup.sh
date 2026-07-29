#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/aws_mvp.yaml}"
nvidia-smi
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
python -m pip install -r requirements-aws-extra.txt
python -m pip freeze > requirements-aws.lock.txt
python -m src.check_environment --config "${CONFIG_PATH}"

