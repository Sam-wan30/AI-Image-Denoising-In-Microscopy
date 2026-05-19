#!/usr/bin/env python3
"""Salt-and-pepper noise detection and denoising helpers."""

import cv2
import numpy as np


def to_uint8_image(image: np.ndarray) -> np.ndarray:
    """Convert image array to uint8 without changing shape."""
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    if arr.max() <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def as_grayscale_uint8(image: np.ndarray) -> np.ndarray:
    """Convert RGB/RGBA/grayscale image to grayscale uint8."""
    arr = to_uint8_image(image)
    if arr.ndim == 2:
        return arr
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def estimate_salt_pepper_ratio(image: np.ndarray, low: int = 5, high: int = 250) -> float:
    """Estimate percentage of near-black/near-white impulse pixels."""
    gray = as_grayscale_uint8(image)
    impulse = (gray <= low) | (gray >= high)
    return float(np.mean(impulse))


def median_kernel_for_ratio(ratio: float) -> int:
    """Pick an odd median kernel from estimated impulse-noise density."""
    if ratio >= 0.45:
        return 9
    if ratio >= 0.25:
        return 7
    if ratio >= 0.08:
        return 5
    return 3


def denoise_salt_pepper(image: np.ndarray, kernel_size: int | None = None) -> np.ndarray:
    """Denoise salt-and-pepper noise with median filtering."""
    arr = to_uint8_image(image)
    ratio = estimate_salt_pepper_ratio(arr)
    ksize = kernel_size or median_kernel_for_ratio(ratio)

    if arr.ndim == 2:
        return cv2.medianBlur(arr, ksize)

    if arr.shape[2] == 4:
        rgb = arr[:, :, :3]
        alpha = arr[:, :, 3]
        denoised_rgb = cv2.medianBlur(rgb, ksize)
        return np.dstack([denoised_rgb, alpha])

    return cv2.medianBlur(arr, ksize)
