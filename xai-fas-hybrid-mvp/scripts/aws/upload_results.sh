#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 /home/ubuntu/xai-fas/results s3://private-bucket/prefix" >&2
  exit 2
fi
for directory in checkpoints metrics figures status logs predictions explanations curves; do
  if [[ -d "$1/${directory}" ]]; then
    aws s3 sync "$1/${directory}" "$2/${directory}" --only-show-errors
  fi
done
if [[ -f "$1/report.md" ]]; then
  aws s3 cp "$1/report.md" "$2/report.md" --only-show-errors
fi

