# XAI Face Anti-Spoofing MVP Report



## Dataset and subset

Not run.

## Data leakage analysis

Not run.

## Hardware and software environment

Environment check not run.

## Classification results

`{"threshold": 0.990067720413208, "accuracy": 0.766, "precision": 0.9889705882352942, "recall": 0.538, "f1": 0.6968911917098446, "average_precision": 0.9525583755443847, "confusion_matrix": [[497, 3], [231, 269]], "roc_auc": 0.9445819999999999, "apcer": 0.462, "bpcer": 0.006, "acer": 0.234}`

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
