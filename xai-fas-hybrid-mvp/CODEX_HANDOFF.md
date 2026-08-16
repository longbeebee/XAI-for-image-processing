# Codex Handoff – XAI Face Anti-Spoofing MVP

## Project

Project: `xai-fas-hybrid-mvp`

Purpose: evaluate MobileNetV3-Small face anti-spoofing with Grad-CAM and Integrated Gradients under brightness, blur, and JPEG perturbations.

Label convention:

- `0 = real`
- `1 = spoof`

## Local Windows environment

- Python: 3.11 required (`>=3.10,<3.12`)
- Virtual environment: `.venv`
- Local backend: AMD DirectML with CPU fallback
- CUDA is not available on the AMD RX 5700 XT
- DirectML may show this warning; it is expected and not fatal:

```text
aten::lerp.Scalar_out is not currently supported on the DML backend and will fall back to run on the CPU
```

Activate the environment in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks scripts:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Alternatively call `.\.venv\Scripts\python.exe` directly.

## Dataset

Dataset root:

```text
F:\Celeb-spoof\CelebA_Spoof
```

Portable YAML path:

```yaml
dataset_root: "F:/Celeb-spoof/CelebA_Spoof"
```

The dataset inspection found:

- 561,575 images
- 494,405 JPG files
- 67,170 PNG files
- valid 44-value annotations
- readable sample images

The project uses `metas/intra_test` as the primary official train/test annotation set. `protocol1` and `protocol2` must not be merged into the same metadata because they duplicate image paths.

## Important code fixes already applied

The following changes are already present in the working tree:

1. `src/metrics/faithfulness.py`
   - Supports NumPy 2.x by using `np.trapezoid` when available and falling back to `np.trapz` for older NumPy.

2. `src/prepare_celeba_spoof.py`
   - Prefers annotation files under `metas/intra_test` to avoid protocol duplication.
   - Corrects subject ID extraction from paths such as `Data/train/2623/live/image.jpg`.

3. `src/metrics/stability.py`
   - Explicit constant-map semantics:
     - identical constant maps → `1.0`
     - different maps where either map is constant → `0.0`
     - normal maps → Spearman correlation
   - Avoids SciPy `ConstantInputWarning`.

4. Tests
   - Windows path test is separator-independent.
   - Device test accepts CPU or DirectML on Windows.
   - Constant-map metric behavior is covered.

Current test result:

```text
25 passed
```

## Local configurations

Files:

```text
configs/windows_mvp.yaml
configs/windows_smoke_test.yaml
```

Both configs use:

```yaml
processed_dir: "D:/XAI-FAS/data/processed"
output_dir: "D:/XAI-FAS/results"
cache_dir: "D:/XAI-FAS/cache"
```

The smoke config was corrected to use:

```yaml
dataset_root: "F:/Celeb-spoof/CelebA_Spoof"
```

## Successful local commands

From the project directory:

```powershell
pytest -q
python -m src.check_environment --config configs/windows_mvp.yaml
python -m src.inspect_celeba_spoof --config configs/windows_mvp.yaml
python -m src.prepare_celeba_spoof --config configs/windows_mvp.yaml --force
python -m src.run_smoke_test --config configs/windows_smoke_test.yaml
```

The current smoke report is:

```text
D:/XAI-FAS/results/smoke_test_report.json
```

Status: `passed`. All stages passed:

- environment
- training
- classifier
- prediction_stability
- explanation_stability
- faithfulness
- sanity
- runtime
- report

Dataset leakage report:

```text
D:/XAI-FAS/results/subject_leakage_report.json
```

Expected/current values:

```json
{
  "duplicate_relative_paths": 0,
  "duplicate_image_ids": 0,
  "split_overlap": false,
  "subject_id_available": true
}
```

## AWS bundle

Local bundle:

```text
D:/XAI-FAS/aws_bundle
```

Bundle verification passed:

```json
{
  "status": "passed",
  "number_of_shards": 1,
  "number_of_images": 12000
}
```

S3 location:

```text
s3://xai-private-bucket/xai-fas/aws_bundle
```

The S3 upload completed successfully. Do not put AWS Access Keys or Secret Keys in this file or in the repository.

## AWS EC2 GPU plan

Recommended starting instance: `g4dn.xlarge` with NVIDIA T4 GPU, subject to regional availability and account quota.

Use an AWS Deep Learning GPU AMI, attach an EC2 IAM role with least-privilege access to:

- read `s3://xai-private-bucket/xai-fas/aws_bundle/*`
- write `s3://xai-private-bucket/xai-fas/results/*`

On EC2:

```bash
git clone YOUR_PRIVATE_REPOSITORY xai-fas
cd xai-fas/xai-fas-hybrid-mvp

cp configs/aws_mvp.example.yaml configs/aws_mvp.yaml
cp configs/aws_smoke_test.example.yaml configs/aws_smoke_test.yaml

bash scripts/aws/setup.sh configs/aws_mvp.yaml

mkdir -p /home/ubuntu/xai-fas/aws_bundle
bash scripts/aws/download_bundle.sh \
  s3://xai-private-bucket/xai-fas/aws_bundle \
  /home/ubuntu/xai-fas/aws_bundle
```

The current download script verifies the TAR bundle but does not extract it. Extract the shard and copy manifests before running training:

```bash
mkdir -p data/celeba_spoof_mvp data/processed

for shard in aws_bundle/shards/*.tar; do
  tar -xf "$shard" -C data/celeba_spoof_mvp
done

cp aws_bundle/manifests/*_subset.parquet data/processed/
cp aws_bundle/manifests/subset_manifest.json data/processed/
```

Then verify CUDA and run AWS smoke test:

```bash
nvidia-smi
bash scripts/aws/run_smoke_test.sh
```

Only after AWS smoke test passes:

```bash
bash scripts/aws/run_mvp.sh
bash scripts/aws/upload_results.sh \
  /home/ubuntu/xai-fas/results \
  s3://xai-private-bucket/xai-fas/results
```

Stop or terminate the EC2 instance after the experiment to avoid unnecessary charges.

## Current status

The Windows local pipeline and S3 bundle preparation are complete. The remaining step is to provision an NVIDIA CUDA EC2 instance and run the AWS smoke test followed by the official MVP.
