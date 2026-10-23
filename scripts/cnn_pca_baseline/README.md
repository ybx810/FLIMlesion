# Frozen CNN + fold-local PCA64 baseline

Run from `D:\FLIMlesion` using `.venv-cnn\Scripts\python.exe` (the system has no `python` on PATH).

This environment inherits read-only scientific packages from `.conda-radiomics` via `--system-site-packages`; new Torch packages are installed only inside `.venv-cnn`. It depends on the original environment remaining available. No CUDA or CNN fine-tuning is used.

## Commands

```powershell
Set-Location D:\FLIMlesion
$py = '.\.venv-cnn\Scripts\python.exe'
& $py -m scripts.cnn_pca_baseline.run --stage inspect
& $py -m scripts.cnn_pca_baseline.run --stage all --cnn all --smoke-test
& $py -m scripts.cnn_pca_baseline.run --stage all --method area --resolution 1024 --cnn all --classifier all
# Extraction and training are independent; this does not instantiate a CNN:
& $py -m scripts.cnn_pca_baseline.run --stage train --method area --resolution 1024
& $py -m scripts.cnn_pca_baseline.run --stage compare
& $py -m unittest scripts.cnn_pca_baseline.test_pipeline
```

`--dry-run` validates data/splits without loading CNNs or training. `--force` explicitly rebuilds selected caches/results. Default batch size 4, workers 0, CPU threads 4; no GPU option. `--smoke-test` uses two images and saves only QC, never a partial official embedding cache. `--pca-components` accepts only 64. No all-configurations default exists.

Remaining eight configurations (run only when wanted):

```powershell
Set-Location D:\FLIMlesion
$py = '.\.venv-cnn\Scripts\python.exe'
foreach ($method in @('area','lanczos','gaussain_pyramid')) {
  foreach ($resolution in @(1024,2048,4096)) {
    if ($method -eq 'area' -and $resolution -eq 1024) { continue }
    & $py -m scripts.cnn_pca_baseline.run --stage all --method $method --resolution $resolution --cnn all --classifier all --device cpu --batch-size 4 --num-workers 0 --pca-components 64
    if ($LASTEXITCODE -ne 0) { throw "Failed: $method/$resolution" }
  }
}
```

## Data and fair comparisons

`metadata.xlsx` is read through the original exact metadata matching code. `processing_log.csv` supplies original sample IDs; patient IDs are never inferred. The current cohort contains 187 images / 115 patients. All current runs use the existing patient-level five-fold assignments from `outputs/radiomics_hsv_baseline_5fold/cv_splits/cv_splits.csv` and `config_5fold.yaml`. Missing or incompatible splits stop training; assignments are never regenerated or recovered from historical OOF predictions. The verified copy is `cv_splits/reused_5fold.csv`, with source hash in `qc/split_provenance.json`. Current results are saved to `outputs/cnn_pca_baseline_5fold`; the historical three-fold directory has been removed. The configured pretrained weight cache is currently missing and must be restored before fresh embedding extraction or cache validation.

Actual pyramid directory on this machine is `gaussian_pyramid`, although the requested CLI spelling is `gaussain_pyramid`. Existing discovery resolves the actual spelling and logs it; no directory is renamed. Output configuration names reflect the actual directory.

Radiomics currently uses full image ROI (no background filtering or mask). CNN uses RGB conversion and only official weight transforms, with no extra resize. The center crop can discard peripheral tissue. `qc/<config>/<cnn>/crop_geometry.csv` records retained area for every input; previews show the crop rectangle on the full image. This is a geometric risk check, not a validated tissue segmentation or a guarantee that tissue is retained. Formal CNN tensors have the official crop size, not the source 1024/2048/4096 dimensions.

## Reuse and leakage protections

Models are sequential, pretrained, frozen, eval-mode and inference-mode. EfficientNet replaces its entire classifier; ResNet replaces fc; DenseNet replaces classifier while retaining its native ReLU and global pooling. Dimensions are checked against 1280/2048/1024 and repeated inference is checked numerically.

Each cache includes sample_id, patient_id, label and EMB columns. SHA256 signatures cover every input file, metadata, checkpoint bytes, enum, versions, transform and extraction source; CSV bytes and keys are validated on reuse. Mismatch stops with a rebuild instruction. Do not edit a cached CSV or its sidecar. Keep checkpoints for validation. Classification imports Torch/TorchVision to validate the cache contract, but never calls a CNN or downloads weights.

Shared Radiomics `train()` now accepts optional preprocessor factory and fold callback; its original default behavior remains unchanged. The CNN preprocessor contains StandardScaler and exact full-SVD PCA(64), fitted separately within every fold by the existing trainer. Rank/sample/finite checks fail rather than reducing dimension. Complete pipelines and separate scaler/PCA files are saved for each fold. The shared RF/XGBoost constructors, historical parameters, seed, OOF checks, metrics and figures are reused.

Importance describes fold-specific PCA coordinates, not image texture, color or biomarkers. Averaging same-index PCA importance across folds is exploratory because the bases differ. Explained variance is not predictive performance.

Comparison requires matching samples, patients, labels, class encoding and folds. Missing/incompatible Radiomics configurations are recorded as non-comparable, never silently added to rankings. Heatmaps are generated once nine configurations have completed.

## Dependencies and offline weights

Recreate the overlay without changing the Radiomics environment:

```powershell
& .\.conda-radiomics\python.exe -m venv --system-site-packages .venv-cnn
& .\.venv-cnn\Scripts\python.exe -m pip install --upgrade pip
& .\.venv-cnn\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

If network access is unavailable, download official checkpoints on a connected machine and copy the unchanged files to `outputs/cnn_pca_baseline/weights/checkpoints/`:

- https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth
- https://download.pytorch.org/models/resnet50-11ad3fa6.pth
- https://download.pytorch.org/models/densenet121-a639ec97.pth

These correspond to the DEFAULT enums in pinned TorchVision 0.20.1. TorchVision uses the official local filenames automatically. No random-initialization fallback exists. For offline packages, use matching cp39 Windows x64 CPU wheels and `pip install --no-index --find-links <wheel_directory> torch==2.5.1 torchvision==0.20.1`.
