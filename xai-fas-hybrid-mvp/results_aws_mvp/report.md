# XAI Face Anti-Spoofing MVP Report



## Dataset and subset

Not run.

## Data leakage analysis

Not run.

## Hardware and software environment

Environment check not run.

## Classification results

`{"threshold": 0.990067720413208, "accuracy": 0.766, "precision": 0.9889705882352942, "recall": 0.538, "f1": 0.6968911917098446, "average_precision": 0.9525583755443847, "confusion_matrix": [[497, 3], [231, 269]], "roc_auc": 0.9445819999999999, "apcer": 0.462, "bpcer": 0.006, "acer": 0.234}`

## Threshold and anti-spoofing operating points

Definitions: FAR = APCER (spoof accepted as real), FFR = BPCER (real rejected as spoof), and TAR = bona-fide acceptance rate (`1 - FFR`). The validation-selected threshold evaluated on test was approximately `0.990460`, giving FAR/APCER `0.462`, FFR/BPCER `0.006`, TAR `0.994`, spoof detection rate `0.538`, and ACER `0.234`.

Diagnostic test-only analysis found a minimum ACER of `0.114` at threshold `0.070387` (FAR `0.150`, FFR `0.078`) and a closest test EER operating point at threshold `0.020117` (FAR/FFR `0.124`). These test-derived operating points are descriptive only and must not replace validation-based threshold selection.

The complete threshold sweep is in `metrics/threshold_operating_points.csv`, with a machine-readable summary in `metrics/threshold_summary.json` and an explanation in `threshold_analysis.md`.

## Prediction and explanation stability

See `metrics/prediction_stability.csv`, `metrics/explanation_stability.csv`, and `metrics/pces_summary.csv` when present.

## Faithfulness and sanity

See `metrics/faithfulness.csv` and `metrics/sanity.csv` when present.

## Limitations

- Stability does not imply faithfulness.

- Faithfulness does not imply classifier correctness.

- Passing sanity checks does not prove that an explanation is perfect.

- Grad-CAM and Integrated Gradients are not causal explanations.

- Official results require one homogeneous CUDA run and all validation gates.

## Reproducibility

Seed: `42`. Paths stored in metadata are relative POSIX paths.
