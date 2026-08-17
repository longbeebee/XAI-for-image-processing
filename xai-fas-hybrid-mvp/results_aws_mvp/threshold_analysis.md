# Threshold and anti-spoofing operating-point analysis

Definitions: FAR = APCER (spoof accepted as real), FFR = BPCER (real rejected as spoof), TAR = bona-fide acceptance rate (1 - FFR).

## Validation-selected threshold evaluated on test

- Threshold: `0.990460`
- FAR/APCER: `0.4620`
- FFR/BPCER: `0.0060`
- TAR (real acceptance): `0.9940`
- Spoof detection rate: `0.5380`
- ACER: `0.2340`

## Diagnostic test-only reference points

These points are descriptive only; selecting them on test labels would be optimistic.

- Minimum test ACER: threshold `0.070387`, FAR `0.1500`, FFR `0.0780`, TAR `0.9220`, ACER `0.1140`
- Closest test EER: threshold `0.020117`, FAR `0.1240`, FFR `0.1240`, TAR `0.8760`, ACER `0.1240`

Full operating points are in `metrics/threshold_operating_points.csv`; the machine-readable summary is `metrics/threshold_summary.json`.
