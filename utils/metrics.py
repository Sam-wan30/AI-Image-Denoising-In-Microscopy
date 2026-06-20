#!/usr/bin/env python3
"""
Metrics module for microscopy image quality assessment.

This module provides lightweight, numpy-based PSNR and SSIM
implementations so the Streamlit app can run with ONNX-only inference.
"""

import cv2
import numpy as np
from typing import Union, Tuple, Optional


def _ensure_numpy(image: Union[np.ndarray, object]) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        image = np.asarray(image)
    return image.astype(np.float32)


def _normalize_image(image: np.ndarray) -> np.ndarray:
    if image.max() <= 1.0:
        return np.clip(image, 0.0, 1.0)
    if image.max() > 255.0:
        image = image / 65535.0
    else:
        image = image / 255.0
    return np.clip(image, 0.0, 1.0)


def _prepare_image_array(
    image: Union[np.ndarray, object],
    max_val: Optional[float],
) -> tuple[np.ndarray, float]:
    image = _ensure_numpy(image)

    if image.ndim == 2:
        image = image[np.newaxis, np.newaxis, :, :]
    elif image.ndim == 3:
        if image.shape[0] in (1, 3):
            image = image.mean(axis=0, keepdims=True)[np.newaxis, ...]
        elif image.shape[-1] in (1, 3, 4):
            image = image[..., :3].mean(axis=-1, keepdims=False)[
                np.newaxis, np.newaxis, ...
            ]
        else:
            image = image[:, np.newaxis, :, :]
    elif image.ndim == 4:
        image = image.mean(axis=1, keepdims=True)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    if max_val is None:
        max_val = 1.0 if image.max() <= 1.0 else 255.0
    max_val = float(max_val)

    if max_val == 255.0 and image.max() <= 1.0:
        image = image * 255.0
    elif max_val == 1.0 and image.max() > 1.0:
        image = image / 255.0

    return image, max_val


def calculate_psnr(
    pred: Union[np.ndarray, object],
    target: Union[np.ndarray, object],
    max_val: Optional[float] = None,
    reduction: str = 'mean',
) -> float:
    pred_arr, max_val = _prepare_image_array(pred, max_val)
    target_arr, _ = _prepare_image_array(target, max_val)

    if pred_arr.shape != target_arr.shape:
        raise ValueError("Prediction and target must have the same shape.")

    mse = np.mean((pred_arr - target_arr) ** 2, axis=(1, 2, 3))
    mse = np.maximum(mse, 1e-12)
    psnr = 20.0 * np.log10(max_val) - 10.0 * np.log10(mse)
    psnr = np.minimum(psnr, 200.0)

    if reduction == 'mean':
        return float(np.mean(psnr))
    if reduction == 'sum':
        return float(np.sum(psnr))
    if reduction == 'none':
        return psnr
    raise ValueError(f"Unsupported reduction: {reduction}")


def calculate_ssim(
    pred: Union[np.ndarray, object],
    target: Union[np.ndarray, object],
    window_size: int = 11,
    max_val: Optional[float] = None,
    reduction: str = 'mean',
) -> float:
    pred_arr, max_val = _prepare_image_array(pred, max_val)
    target_arr, _ = _prepare_image_array(target, max_val)

    if pred_arr.shape != target_arr.shape:
        raise ValueError("Prediction and target must have the same shape.")

    values = []
    for i in range(pred_arr.shape[0]):
        pred_img = pred_arr[i, 0]
        target_img = target_arr[i, 0]
        values.append(_calculate_single_ssim(pred_img, target_img, window_size, max_val))

    values = np.asarray(values, dtype=np.float32)
    if reduction == 'mean':
        return float(np.mean(values))
    if reduction == 'sum':
        return float(np.sum(values))
    if reduction == 'none':
        return values
    raise ValueError(f"Unsupported reduction: {reduction}")


def _calculate_single_ssim(
    pred: np.ndarray,
    target: np.ndarray,
    window_size: int = 11,
    max_val: float = 1.0,
) -> float:
    if pred.shape != target.shape:
        raise ValueError("Predicted and target images must have the same shape.")

    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    kernel = np.ones((window_size, window_size), dtype=np.float32) / (window_size * window_size)

    mu1 = cv2.filter2D(pred, -1, kernel, borderType=cv2.BORDER_REFLECT)
    mu2 = cv2.filter2D(target, -1, kernel, borderType=cv2.BORDER_REFLECT)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(pred * pred, -1, kernel, borderType=cv2.BORDER_REFLECT) - mu1_sq
    sigma2_sq = cv2.filter2D(target * target, -1, kernel, borderType=cv2.BORDER_REFLECT) - mu2_sq
    sigma12 = cv2.filter2D(pred * target, -1, kernel, borderType=cv2.BORDER_REFLECT) - mu1_mu2

    sigma1_sq = np.maximum(sigma1_sq, 1e-10)
    sigma2_sq = np.maximum(sigma2_sq, 1e-10)

    numerator = (2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / denominator

    return float(np.mean(ssim_map))


def calculate_mae(
    pred: Union[np.ndarray, object],
    target: Union[np.ndarray, object],
    reduction: str = 'mean',
) -> float:
    pred_arr, _ = _prepare_image_array(pred, None)
    target_arr, _ = _prepare_image_array(target, None)

    if pred_arr.shape != target_arr.shape:
        raise ValueError("Prediction and target must have the same shape.")

    mae = np.mean(np.abs(pred_arr - target_arr), axis=(1, 2, 3))
    if reduction == 'mean':
        return float(np.mean(mae))
    if reduction == 'sum':
        return float(np.sum(mae))
    if reduction == 'none':
        return mae
    raise ValueError(f"Unsupported reduction: {reduction}")


def calculate_mse(
    pred: Union[np.ndarray, object],
    target: Union[np.ndarray, object],
    reduction: str = 'mean',
) -> float:
    pred_arr, _ = _prepare_image_array(pred, None)
    target_arr, _ = _prepare_image_array(target, None)

    if pred_arr.shape != target_arr.shape:
        raise ValueError("Prediction and target must have the same shape.")

    mse = np.mean((pred_arr - target_arr) ** 2, axis=(1, 2, 3))
    if reduction == 'mean':
        return float(np.mean(mse))
    if reduction == 'sum':
        return float(np.sum(mse))
    if reduction == 'none':
        return mse
    raise ValueError(f"Unsupported reduction: {reduction}")


class MetricsCalculator:
    """Utility class for calculating multiple image quality metrics."""

    def __init__(self, max_val: Optional[float] = None):
        self.max_val = max_val

    def calculate_all_metrics(
        self,
        pred: Union[np.ndarray, object],
        target: Union[np.ndarray, object],
        reduction: str = 'mean',
    ) -> dict:
        return {
            'psnr': calculate_psnr(pred, target, self.max_val, reduction),
            'ssim': calculate_ssim(pred, target, max_val=self.max_val, reduction=reduction),
            'mae': calculate_mae(pred, target, reduction),
            'mse': calculate_mse(pred, target, reduction),
        }

    def calculate_batch_metrics(
        self,
        pred_batch: Union[np.ndarray, object],
        target_batch: Union[np.ndarray, object],
    ) -> dict:
        return self.calculate_all_metrics(pred_batch, target_batch, reduction='mean')


if __name__ == '__main__':
    import numpy as np

    print('Running basic metric checks...')
    test_image = np.ones((1, 1, 256, 256), dtype=np.float32) * 0.5
    assert abs(calculate_psnr(test_image, test_image) - 120.0) < 1e-6
    assert abs(calculate_ssim(test_image, test_image) - 1.0) < 1e-6
    print('✓ Metrics module loaded successfully.')
