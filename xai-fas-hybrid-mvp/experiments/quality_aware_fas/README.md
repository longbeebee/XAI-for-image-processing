# Quality-Aware FAS Experiments

This directory contains new experiments for the quality-aware, uncertainty-calibrated, and explanation-consistent face anti-spoofing study.

The existing MVP baseline is frozen. No file under `src/`, `configs/`, `scripts/aws/`, or `results_aws_mvp/` is modified by this experiment namespace. New runs use their own output directory and S3 prefix.

## Research sequence

1. **Stage 0 - frozen baseline audit:** reproduce the existing MobileNetV3-Small result without changing its code or artifacts.
2. **Stage 1 - quality-aware learning:** add image-quality supervision for brightness, blur, and JPEG degradation.
3. **Stage 2 - uncertainty calibration:** estimate when a prediction is unreliable and evaluate calibration.
4. **Stage 3 - explanation consistency:** add an XAI consistency objective between original and degraded views.
5. **Stage 4 - full ablation:** compare all variants over multiple seeds.

## Data and AWS contract

The experiment uses the existing CelebA-Spoof bundle and portable relative-path metadata. AWS jobs run on CUDA without CPU fallback. Results are written to `results_quality_aware_stage1/` and uploaded to a private prefix such as `s3://PRIVATE_BUCKET/xai-fas/quality-aware/stage1/`.

The old baseline prefix and `results_aws_mvp/` remain read-only references.

## Stage 1 hypothesis

An auxiliary image-quality objective will improve robustness analysis and confidence behavior without changing the binary task definition. Stage 1 is successful only if it is evaluated on the same fixed test set and compared with the frozen baseline using APCER, BPCER, ACER, ROC-AUC, prediction stability, calibration, and runtime.

## Acceptance criteria

- Old baseline artifacts are unchanged.
- The experiment uses a separate output directory and S3 prefix.
- The quality-aware model is trained with at least three seeds.
- The test threshold is selected from validation only.
- Results include mean, standard deviation, and 95% confidence intervals.
- No improvement is claimed unless ACER or target-FAR performance improves without unacceptable BPCER degradation.

## Planned new modules

Implementation will be added only inside this experiment area or in new modules explicitly owned by it: quality-label construction, a quality-aware model wrapper, stage-specific training, calibration and uncertainty evaluation, explanation-consistency training, and stage-specific reporting.
