#!/usr/bin/env python3
"""
Metrics module for microscopy image quality assessment.

This module provides numerically stable implementations of PSNR and SSIM
specifically optimized for grayscale microscopy images.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, Tuple, Optional
import math


def calculate_psnr(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    max_val: Optional[float] = None,
    reduction: str = 'mean'
) -> Union[float, torch.Tensor]:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) with numerical stability.
    
    Args:
        pred: Predicted image(s) tensor or numpy array
        target: Ground truth image(s) tensor or numpy array
        max_val: Maximum possible pixel value. If None, inferred from data type
        reduction: Reduction method ('mean', 'none', 'sum')
        
    Returns:
        PSNR value(s) in dB
    """
    # Convert to tensors if numpy arrays
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred).float()
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).float()
    
    # Ensure same device
    if pred.device != target.device:
        target = target.to(pred.device)
    
    # Handle different input shapes
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)  # Add channel dimension
    if target.dim() == 2:
        target = target.unsqueeze(0)
    
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)  # Add batch dimension
    if target.dim() == 3:
        target = target.unsqueeze(0)
    
    # Determine max_val if not provided
    if max_val is None:
        if pred.dtype == torch.uint8:
            max_val = 255.0
        elif pred.dtype == torch.float16 or pred.dtype == torch.bfloat16:
            max_val = 1.0
        else:  # float32, float64
            max_val = 1.0 if pred.max() <= 1.0 else 255.0
    
    max_val = torch.tensor(max_val, dtype=pred.dtype, device=pred.device)
    
    # Calculate MSE with numerical stability
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])  # Reduce over H, W, C
    
    # Handle edge cases for numerical stability
    mse = torch.clamp(mse, min=1e-12)  # Prevent division by zero
    
    # Calculate PSNR
    psnr = 20 * torch.log10(max_val) - 10 * torch.log10(mse)
    
    # Cap maximum PSNR to prevent infinite values for identical images
    psnr = torch.clamp(psnr, max=200)
    
    # Apply reduction
    if reduction == 'mean':
        return torch.mean(psnr).item()
    elif reduction == 'sum':
        return torch.sum(psnr).item()
    elif reduction == 'none':
        return psnr
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")


def calculate_ssim(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    window_size: int = 11,
    max_val: Optional[float] = None,
    reduction: str = 'mean'
) -> Union[float, torch.Tensor]:
    """
    Calculate Structural Similarity Index (SSIM) with numerical stability.
    
    This implementation uses the original SSIM formula with proper handling
    of edge cases and numerical stability for microscopy images.
    
    Args:
        pred: Predicted image(s) tensor or numpy array
        target: Ground truth image(s) tensor or numpy array
        window_size: Size of the sliding window for local statistics
        max_val: Maximum possible pixel value. If None, inferred from data type
        reduction: Reduction method ('mean', 'none', 'sum')
        
    Returns:
        SSIM value(s) in range [0, 1]
    """
    # Convert to tensors if numpy arrays
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred).float()
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).float()
    
    # Ensure same device
    if pred.device != target.device:
        target = target.to(pred.device)
    
    # Handle different input shapes
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)  # Add channel dimension
    if target.dim() == 2:
        target = target.unsqueeze(0)
    
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)  # Add batch dimension
    if target.dim() == 3:
        target = target.unsqueeze(0)
    
    # Determine max_val if not provided
    if max_val is None:
        if pred.dtype == torch.uint8:
            max_val = 255.0
        elif pred.dtype == torch.float16 or pred.dtype == torch.bfloat16:
            max_val = 1.0
        else:  # float32, float64
            max_val = 1.0 if pred.max() <= 1.0 else 255.0
    
    # Calculate SSIM for each image in the batch
    ssim_values = []
    
    for i in range(pred.shape[0]):
        pred_img = pred[i, 0]  # Remove channel dimension (grayscale)
        target_img = target[i, 0]
        
        ssim_val = _calculate_single_ssim(
            pred_img, target_img, window_size, max_val
        )
        ssim_values.append(ssim_val)
    
    ssim_tensor = torch.stack(ssim_values)
    
    # Apply reduction
    if reduction == 'mean':
        return torch.mean(ssim_tensor).item()
    elif reduction == 'sum':
        return torch.sum(ssim_tensor).item()
    elif reduction == 'none':
        return ssim_tensor
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")


def _calculate_single_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int,
    max_val: float
) -> torch.Tensor:
    """
    Calculate SSIM for a single image pair.
    
    Args:
        pred: Single predicted image tensor
        target: Single target image tensor
        window_size: Size of the sliding window
        max_val: Maximum pixel value
        
    Returns:
        SSIM value tensor
    """
    # Ensure 2D tensors
    if pred.dim() != 2 or target.dim() != 2:
        raise ValueError("Both pred and target must be 2D tensors")
    
    # SSIM constants (C1, C2) for numerical stability
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    
    # Calculate local means using convolution
    pad = window_size // 2
    pred_pad = F.pad(pred.unsqueeze(0).unsqueeze(0), (pad, pad, pad, pad), mode='reflect')
    target_pad = F.pad(target.unsqueeze(0).unsqueeze(0), (pad, pad, pad, pad), mode='reflect')
    
    # Create uniform filter kernel
    kernel = torch.ones(1, 1, window_size, window_size, dtype=pred.dtype, device=pred.device)
    kernel = kernel / (window_size * window_size)
    
    # Calculate local means
    mu1 = F.conv2d(pred_pad, kernel)
    mu2 = F.conv2d(target_pad, kernel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # Calculate local variances and covariance
    sigma1_sq = F.conv2d(pred_pad.pow(2), kernel) - mu1_sq
    sigma2_sq = F.conv2d(target_pad.pow(2), kernel) - mu2_sq
    sigma12 = F.conv2d(pred_pad * target_pad, kernel) - mu1_mu2
    
    # Ensure numerical stability
    sigma1_sq = torch.clamp(sigma1_sq, min=1e-10)
    sigma2_sq = torch.clamp(sigma2_sq, min=1e-10)
    
    # Calculate SSIM
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim_map = numerator / denominator
    
    # Return mean SSIM
    return torch.mean(ssim_map)


def calculate_mae(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    reduction: str = 'mean'
) -> Union[float, torch.Tensor]:
    """
    Calculate Mean Absolute Error (MAE).
    
    Args:
        pred: Predicted image(s) tensor or numpy array
        target: Ground truth image(s) tensor or numpy array
        reduction: Reduction method ('mean', 'none', 'sum')
        
    Returns:
        MAE value(s)
    """
    # Convert to tensors if numpy arrays
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred).float()
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).float()
    
    # Ensure same device
    if pred.device != target.device:
        target = target.to(pred.device)
    
    # Calculate MAE
    mae = torch.mean(torch.abs(pred - target), dim=[1, 2, 3])  # Reduce over H, W, C
    
    # Apply reduction
    if reduction == 'mean':
        return torch.mean(mae).item()
    elif reduction == 'sum':
        return torch.sum(mae).item()
    elif reduction == 'none':
        return mae
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")


def calculate_mse(
    pred: Union[torch.Tensor, np.ndarray],
    target: Union[torch.Tensor, np.ndarray],
    reduction: str = 'mean'
) -> Union[float, torch.Tensor]:
    """
    Calculate Mean Squared Error (MSE).
    
    Args:
        pred: Predicted image(s) tensor or numpy array
        target: Ground truth image(s) tensor or numpy array
        reduction: Reduction method ('mean', 'none', 'sum')
        
    Returns:
        MSE value(s)
    """
    # Convert to tensors if numpy arrays
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred).float()
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).float()
    
    # Ensure same device
    if pred.device != target.device:
        target = target.to(pred.device)
    
    # Calculate MSE
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])  # Reduce over H, W, C
    
    # Apply reduction
    if reduction == 'mean':
        return torch.mean(mse).item()
    elif reduction == 'sum':
        return torch.sum(mse).item()
    elif reduction == 'none':
        return mse
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")


class MetricsCalculator:
    """
    Utility class for calculating multiple metrics efficiently.
    """
    
    def __init__(self, max_val: Optional[float] = None):
        """
        Initialize metrics calculator.
        
        Args:
            max_val: Maximum pixel value for normalization
        """
        self.max_val = max_val
    
    def calculate_all_metrics(
        self,
        pred: Union[torch.Tensor, np.ndarray],
        target: Union[torch.Tensor, np.ndarray],
        reduction: str = 'mean'
    ) -> dict:
        """
        Calculate all available metrics.
        
        Args:
            pred: Predicted image(s)
            target: Ground truth image(s)
            reduction: Reduction method
            
        Returns:
            Dictionary containing all metric values
        """
        metrics = {
            'psnr': calculate_psnr(pred, target, self.max_val, reduction),
            'ssim': calculate_ssim(pred, target, max_val=self.max_val, reduction=reduction),
            'mae': calculate_mae(pred, target, reduction),
            'mse': calculate_mse(pred, target, reduction)
        }
        
        return metrics
    
    def calculate_batch_metrics(
        self,
        pred_batch: torch.Tensor,
        target_batch: torch.Tensor
    ) -> dict:
        """
        Calculate metrics for a batch of images.
        
        Args:
            pred_batch: Batch of predicted images [B, C, H, W]
            target_batch: Batch of target images [B, C, H, W]
            
        Returns:
            Dictionary of metric values for the batch
        """
        return self.calculate_all_metrics(pred_batch, target_batch, reduction='mean')


# Test and validation functions
def test_metrics_numerical_stability():
    """
    Test metrics for numerical stability with edge cases.
    """
    print("Testing metrics numerical stability...")
    
    # Test case 1: Identical images (should give perfect scores)
    perfect_image = torch.randn(1, 1, 256, 256)
    psnr_perfect = calculate_psnr(perfect_image, perfect_image)
    ssim_perfect = calculate_ssim(perfect_image, perfect_image)
    
    print(f"Identical images - PSNR: {psnr_perfect:.2f}, SSIM: {ssim_perfect:.6f}")
    assert psnr_perfect > 100, "PSNR should be very high for identical images"
    assert abs(ssim_perfect - 1.0) < 1e-6, "SSIM should be 1.0 for identical images"
    
    # Test case 2: Zero images (edge case)
    zero_pred = torch.zeros(1, 1, 256, 256)
    zero_target = torch.zeros(1, 1, 256, 256)
    psnr_zero = calculate_psnr(zero_pred, zero_target)
    ssim_zero = calculate_ssim(zero_pred, zero_target)
    
    print(f"Zero images - PSNR: {psnr_zero:.2f}, SSIM: {ssim_zero:.6f}")
    assert psnr_zero >= 100, "PSNR should be very high for zero images"
    assert abs(ssim_zero - 1.0) < 1e-6, "SSIM should be 1.0 for zero images"
    
    # Test case 3: Very small differences
    small_diff_pred = torch.ones(1, 1, 256, 256) * 0.5
    small_diff_target = torch.ones(1, 1, 256, 256) * 0.5001
    psnr_small = calculate_psnr(small_diff_pred, small_diff_target)
    ssim_small = calculate_ssim(small_diff_pred, small_diff_target)
    
    print(f"Small difference - PSNR: {psnr_small:.2f}, SSIM: {ssim_small:.6f}")
    assert psnr_small > 40, "PSNR should be high for small differences"
    assert ssim_small > 0.99, "SSIM should be very high for small differences"
    
    print("✓ All numerical stability tests passed!")


def test_metrics_batch_processing():
    """
    Test metrics with batch processing.
    """
    print("Testing batch processing...")
    
    # Create batch of images
    batch_size = 4
    pred_batch = torch.randn(batch_size, 1, 128, 128)
    target_batch = torch.randn(batch_size, 1, 128, 128)
    
    # Test batch metrics
    calculator = MetricsCalculator()
    batch_metrics = calculator.calculate_batch_metrics(pred_batch, target_batch)
    
    print(f"Batch metrics - PSNR: {batch_metrics['psnr']:.2f}, "
          f"SSIM: {batch_metrics['ssim']:.4f}, "
          f"MAE: {batch_metrics['mae']:.4f}, "
          f"MSE: {batch_metrics['mse']:.4f}")
    
    # Test individual reduction methods
    psnr_mean = calculate_psnr(pred_batch, target_batch, reduction='mean')
    psnr_none = calculate_psnr(pred_batch, target_batch, reduction='none')
    psnr_sum = calculate_psnr(pred_batch, target_batch, reduction='sum')
    
    print(f"PSNR reductions - Mean: {psnr_mean:.2f}, "
          f"Sum: {psnr_sum:.2f}, "
          f"Shape (none): {psnr_none.shape}")
    
    assert len(psnr_none) == batch_size, "Should return one PSNR per image"
    assert abs(psnr_mean - torch.mean(psnr_none).item()) < 1e-6, "Mean reduction should match"
    assert abs(psnr_sum - torch.sum(psnr_none).item()) < 1e-6, "Sum reduction should match"
    
    print("✓ Batch processing tests passed!")


if __name__ == "__main__":
    # Run tests
    test_metrics_numerical_stability()
    test_metrics_batch_processing()
    
    print("\n✓ All metrics tests completed successfully!")
    print("Metrics module is ready for use in microscopy image denoising.")
