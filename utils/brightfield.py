#!/usr/bin/env python3
"""Brightfield-style microscopy cleanup helpers."""

import cv2
import numpy as np

from utils.salt_pepper import as_grayscale_uint8


def brightfield_object_mask(image: np.ndarray) -> np.ndarray:
    """
    Convert a noisy brightfield/SEM-like image into a clean black-on-white mask.

    This is for images where the foreground object is darker than a gray/noisy
    background. It is intentionally not the fluorescence U-Net denoising task.
    """
    gray = as_grayscale_uint8(image)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # Keep dark foreground black and clean small background speckles.
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask
