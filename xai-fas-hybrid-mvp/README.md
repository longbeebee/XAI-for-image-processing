# XAI Face Anti-Spoofing Hybrid MVP

This project evaluates whether Grad-CAM and Integrated Gradients remain stable and
faithful when CelebA-Spoof images undergo brightness reduction, Gaussian blur, and
JPEG compression. It trains one MobileNetV3-Small binary classifier with the fixed
convention `0 = real`, `1 = spoof`.

Stability measures whether an explanation remains similar after a quality shift.
Primary explanation stability is prediction-conditioned (PCES): it is summarized
only where the classifier prediction did not change. Faithfulness uses RGB-space
patch deletion (lower AUC is better) and insertion (higher AUC is better). Sanity
randomizes increasingly large parts of the model and checks whether explanations
respond.

Important: high stability does not imply faithfulness; high faithfulness does not
imply the classifier is correct; passing sanity does not prove that an explanation
is perfect. Grad-CAM and Integrated Gradients are not causal explanations.

## Architecture

The local Windows machine inspects the original dataset, writes resumable portable
metadata, samples a small subset, runs tests, and exports only subset images. A
private S3 bucket transfers the verified bundle to an AWS EC2 CUDA instance. The
official experiment runs on one CUDA backend and uploads results privately.

The AMD RX 5700 XT cannot run CUDA. DirectML is supported for local development
only, with job-level visible CPU fallback when enabled. Official results require
AWS CUDA (the recommended baseline is a `g4dn.xlarge` with NVIDIA T4).

## Local Windows setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-local-directml.txt
python -m pip freeze > requirements-local-directml.lock.txt
Copy-Item configs\windows_mvp.example.yaml configs\windows_mvp.yaml
Copy-Item configs\windows_smoke_test.example.yaml configs\windows_smoke_test.yaml
pytest -q
python -m src.check_environment --config configs/windows_mvp.yaml
python -m src.inspect_celeba_spoof --config configs/windows_mvp.yaml
python -m src.prepare_celeba_spoof --config configs/windows_mvp.yaml
python -m src.run_smoke_test --config configs/windows_smoke_test.yaml
python -m src.export_aws_bundle --config configs/windows_mvp.yaml --output-dir "D:/XAI-FAS/aws_bundle"
python -m src.verify_aws_bundle --bundle-dir "D:/XAI-FAS/aws_bundle"
```

Use only POSIX-style relative paths in Parquet metadata; the code resolves them
against either a Windows or Linux dataset root. Metadata parts are written through a
temporary file then atomically renamed. Existing incompatible parts are rejected.

If Windows DataLoader multiprocessing fails, set `num_workers: 0` and
`persistent_workers: false`. If memory is exhausted, reduce `batch_size`, IG
`internal_batch_size`, and XAI evaluation sample limits. Do not use the research
quality shifts as training augmentation.

## Private S3 and AWS CUDA

Create a private bucket/prefix with your organization’s IAM policy. Do not use
public ACLs. Upload the local bundle:

```powershell
aws s3 sync "D:/XAI-FAS/aws_bundle" "s3://PRIVATE_BUCKET/xai-fas/aws_bundle"
```

On a CUDA-enabled Deep Learning AMI:

```bash
git clone YOUR_PRIVATE_REPOSITORY xai-fas
cd xai-fas/xai-fas-hybrid-mvp
cp configs/aws_mvp.example.yaml configs/aws_mvp.yaml
cp configs/aws_smoke_test.example.yaml configs/aws_smoke_test.yaml
bash scripts/aws/setup.sh configs/aws_mvp.yaml
bash scripts/aws/download_bundle.sh s3://PRIVATE_BUCKET/xai-fas/aws_bundle /home/ubuntu/xai-fas/aws_bundle
bash scripts/aws/run_smoke_test.sh
bash scripts/aws/run_mvp.sh
bash scripts/aws/upload_results.sh /home/ubuntu/xai-fas/results s3://PRIVATE_BUCKET/xai-fas/results
```

The setup script intentionally does not reinstall PyTorch or CUDA from a DLAMI.
`run_mvp` refuses to run unless all seven validation gates in
`results/validation_gates.json` are `passed`. Resume uses portable CPU-tensor epoch
checkpoints; it does not treat output-file existence as proof of a completed stage.

## Outputs

Results include environment and dataset manifests, leakage reports, portable
checkpoints, original/shifted predictions, classification and stability metrics,
PCES, faithfulness and sanity tables, runtime-ready schemas, figures, per-stage
statuses, and `report.md`. Generated data, Parquet, CSV, TAR, checkpoints, caches,
credentials, and local path configs are ignored by Git.

## Dataset and limitations

The code never downloads, modifies, renames, or republishes CelebA-Spoof. Configure
an existing local copy and comply with its non-commercial research license. Cite the
official CelebA-Spoof publication and dataset page in downstream reports.

This MVP intentionally excludes RISE, ResNet18, the full ~625k-image experiment,
cross-dataset evaluation, regularization, web UI, distributed/multi-GPU training,
SageMaker, Spot interruption recovery, and ROCm. Real dataset structure, CUDA,
DirectML operator support, S3 transfer, AWS smoke execution, and scientific results
must be validated in their actual environments.

