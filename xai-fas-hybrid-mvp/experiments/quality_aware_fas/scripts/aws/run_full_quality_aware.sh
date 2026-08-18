#!/usr/bin/env bash
set -euo pipefail

# Full new-experiment runner. It never calls the frozen MVP runner.
CONFIG="${1:-experiments/quality_aware_fas/configs/aws_quality_aware_stage1.yaml}"
python -m experiments.quality_aware_fas.check_environment \
  --config "$CONFIG" \
  --output results_quality_aware_stage1/new_environment_report.json
python -m experiments.quality_aware_fas.run_stage1 --config "$CONFIG" --resume

for seed in 42 123 2024; do
  SEED_DIR="results_quality_aware_stage1/seed_${seed}"
  python -m experiments.quality_aware_fas.train_stage2 \
    --config "$CONFIG" \
    --checkpoint "$SEED_DIR/best_model.pt" \
    --output-dir "$SEED_DIR/uncertainty" \
    --seed "$seed"
  python -m experiments.quality_aware_fas.train_stage3 \
    --config "$CONFIG" \
    --checkpoint "$SEED_DIR/uncertainty/uncertainty_calibrated_model.pt" \
    --output-dir "$SEED_DIR/explanation_consistency" \
    --seed "$seed"
  python -m experiments.quality_aware_fas.xai_evaluation \
    --config "$CONFIG" \
    --checkpoint "$SEED_DIR/explanation_consistency/best_explanation_consistent_model.pt" \
    --protocol-dir "${PROTOCOL_DIR:-results_quality_aware_stage1/protocol}" \
    --output-dir "$SEED_DIR/explanation_consistency"
  python -m experiments.quality_aware_fas.final_evaluation \
    --config "$CONFIG" \
    --checkpoint "$SEED_DIR/explanation_consistency/best_explanation_consistent_model.pt" \
    --protocol-dir "${PROTOCOL_DIR:-results_quality_aware_stage1/protocol}" \
    --output-dir "$SEED_DIR/final"
done
python -m experiments.quality_aware_fas.aggregate --output-dir results_quality_aware_stage1
python -m experiments.quality_aware_fas.statistics --root results_quality_aware_stage1
