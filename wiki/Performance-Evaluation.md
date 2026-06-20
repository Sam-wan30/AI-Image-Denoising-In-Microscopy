# Performance Evaluation

## Status

FluoClean AI is an image-restoration research demo. The deployed ONNX
checkpoint has no preserved train/validation/test manifest, so it cannot be
claimed as independently validated. Previous benchmark tables were removed
because no matching evaluation artifacts exist.

## Applicable Metrics

This is paired image-to-image regression. Evaluation uses:

- PSNR for pixel reconstruction fidelity.
- SSIM for local structural similarity.
- MAE and MSE for absolute and squared reconstruction error.
- Improvement over the noisy-input baseline.
- Per-specimen results and worst-case error maps.

Confusion matrices, classification reports, ROC-AUC, PR-AUC, class imbalance,
and classifier feature importance do not apply because the model has no classes
or classification threshold.

## Leakage Controls

Files are grouped by specimen/session before splitting. Laser-power variants
from the same session remain together because many share byte-identical clean
targets. Training augmentation is synchronized between noisy and clean images
and disabled for validation and test data.

Every new training run records the random seed, split filenames and groups,
validation history, held-out test metrics, and checkpoint provenance.

## Current Retrospective Audit

The deployed checkpoint was evaluated over all 105 repository pairs at the
128x128 inference size used by the free-tier deployment.

| Measure | Model | Noisy input |
|---|---:|---:|
| Mean PSNR | 19.13 dB | 11.17 dB |
| Mean SSIM | 0.623 | 0.452 |
| Mean MAE | 0.153 | 0.302 |
| PSNR improved | 84 / 105 | - |
| SSIM improved | 76 / 105 | - |

These are retrospective results, not a held-out test. The July 21 PVD specimen
group lost 2.60 dB mean PSNR relative to the noisy input, and 21 images were
worse by PSNR. The checkpoint is not reliable enough for scientific
measurement or diagnostic use.

## Reproducing Evaluation

```bash
python evaluate_model.py \
  --model models/deploy/model.onnx \
  --data-dir data \
  --manifest models/split_manifest.json \
  --split test \
  --output-dir reports/model_validation
```

The output contains `evaluation_report.json`, `per_image_metrics.csv`, saved
predictions, and panels showing noisy input, prediction, clean target, and
absolute error for the worst failures.

## Release Gate

A future checkpoint should be promoted only when:

1. It was trained with a saved leakage-safe split manifest.
2. Held-out metrics improve over the noisy baseline across specimen groups.
3. Worst-case outputs pass review for hallucination and oversmoothing.
4. The export preserves seed, split, test metrics, and source hash.
5. PyTorch and deployed ONNX inference parity is verified.
6. The intended acquisition domain and limitations are documented.

Independent external validation remains required for quantitative microscopy.
