#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 s3://PRIVATE_BUCKET/xai-fas/aws_bundle /home/ubuntu/xai-fas/aws_bundle" >&2
  exit 2
fi
aws s3 sync "$1" "$2" --only-show-errors
python -m src.verify_aws_bundle --bundle-dir "$2"
