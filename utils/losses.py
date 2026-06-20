#!/usr/bin/env python3
"""Training loss functions for microscopy denoising."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def differentiable_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    max_val: float = 1.0,
) -> torch.Tensor:
    """Return mean SSIM without leaving PyTorch's autograd graph."""
    if pred.shape != target.shape:
        raise ValueError(f"SSIM shape mismatch: {pred.shape} != {target.shape}")
    if pred.ndim != 4:
        raise ValueError(f"SSIM expects NCHW tensors, got {pred.ndim} dimensions")

    padding = window_size // 2
    mu_pred = F.avg_pool2d(pred, window_size, stride=1, padding=padding)
    mu_target = F.avg_pool2d(target, window_size, stride=1, padding=padding)
    mu_pred_sq = mu_pred.square()
    mu_target_sq = mu_target.square()
    mu_cross = mu_pred * mu_target

    sigma_pred = F.avg_pool2d(pred.square(), window_size, 1, padding) - mu_pred_sq
    sigma_target = F.avg_pool2d(target.square(), window_size, 1, padding) - mu_target_sq
    sigma_cross = F.avg_pool2d(pred * target, window_size, 1, padding) - mu_cross

    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    numerator = (2.0 * mu_cross + c1) * (2.0 * sigma_cross + c2)
    denominator = (mu_pred_sq + mu_target_sq + c1) * (
        sigma_pred + sigma_target + c2
    )
    return (numerator / denominator.clamp_min(1e-12)).mean()


class DenoisingLoss(nn.Module):
    """Combined loss: 0.7 * L1 + 0.3 * (1 - SSIM)."""

    def __init__(self, l1_weight: float = 0.7, ssim_weight: float = 0.3, use_ssim: bool = True):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.use_ssim = use_ssim
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1_loss = self.l1(pred, target)
        if self.use_ssim:
            ssim_loss = 1.0 - differentiable_ssim(pred, target, max_val=1.0)
            return self.l1_weight * l1_loss + self.ssim_weight * ssim_loss
        return l1_loss
