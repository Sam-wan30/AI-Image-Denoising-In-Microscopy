#!/usr/bin/env python3
"""
Shared image preprocessing for training, inference, and the web app.

All pipelines MUST use these functions so grayscale conversion, resize,
normalization, and tensor layout stay identical.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from typing import Tuple, Union

IMAGE_SIZE = (256, 256)


def load_grayscale(image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
    """Load an image path, array, or PIL image as a 2D grayscale float32 array (H, W)."""
    pil_source = False

    if isinstance(image, Image.Image):
        pil_source = True
        image = image.convert("RGB")
        img = np.asarray(image)
    elif isinstance(image, (str, Path)):
        img = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not load image: {image}")
    else:
        img = np.asarray(image)

    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[:, :, :3]

        if img.shape[2] == 1:
            img = img[:, :, 0]
        elif np.array_equal(img[:, :, 0], img[:, :, 1]) and np.array_equal(
            img[:, :, 0], img[:, :, 2]
        ):
            img = img[:, :, 0]
        else:
            if pil_source:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img.astype(np.float32)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize pixel values to [0, 1]. Handles uint8 and uint16."""
    image = image.astype(np.float32)
    if image.max() <= 1.0:
        return np.clip(image, 0.0, 1.0)

    # uint8 or uint16 microscopy data
    if image.max() > 255.0:
        image = image / 65535.0
    else:
        image = image / 255.0
    return np.clip(image, 0.0, 1.0)


def resize_image(image: np.ndarray, size: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Resize to (width, height) = size."""
    if image.shape[:2] == (size[1], size[0]):
        return image
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def preprocess_numpy(
    image: Union[str, np.ndarray],
    image_size: Tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    """Full pipeline: grayscale -> resize -> normalize -> [0,1] float32 (H,W)."""
    img = load_grayscale(image)
    img = resize_image(img, image_size)
    return normalize_image(img)


def to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert (H,W) float image to tensor (1, 1, H, W)."""
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")
    return torch.from_numpy(image.copy()).float().unsqueeze(0).unsqueeze(0)


def preprocess_tensor(
    image: Union[str, np.ndarray],
    image_size: Tuple[int, int] = IMAGE_SIZE,
) -> torch.Tensor:
    """Preprocess and return model input tensor (1, 1, H, W)."""
    return to_tensor(preprocess_numpy(image, image_size))


def postprocess_tensor(
    output: torch.Tensor,
    original_shape: Tuple[int, int],
    model_size: Tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    """Convert model output to uint8 grayscale (H, W), resizing to original_shape."""
    out = output.squeeze().detach().cpu().numpy()
    out = np.clip(out, 0.0, 1.0)

    if original_shape != (model_size[1], model_size[0]):
        out = cv2.resize(
            out,
            (original_shape[1], original_shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )

    return (out * 255.0).astype(np.uint8)


def get_original_shape(image: Union[str, np.ndarray]) -> Tuple[int, int]:
    """Return (H, W) of image before model resize."""
    img = load_grayscale(image)
    return img.shape[:2]
