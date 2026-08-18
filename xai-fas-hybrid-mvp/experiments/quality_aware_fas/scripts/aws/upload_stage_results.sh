#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 LOCAL_STAGE_OUTPUT s3://PRIVATE_BUCKET/xai-fas/quality-aware/stage1" >&2
  exit 2
fi
aws s3 sync "$1" "$2" --only-show-errors
