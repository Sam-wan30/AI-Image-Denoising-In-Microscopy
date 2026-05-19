#!/usr/bin/env python3
"""Training loss functions for microscopy denoising."""

import torch
import torch.nn as nn

from utils.metrics import calculate_ssim


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
            # Use reduction='none' so SSIM stays differentiable for backprop
            ssim_val = calculate_ssim(pred, target, max_val=1.0, reduction="none")
            ssim_loss = 1.0 - torch.mean(ssim_val)
            return self.l1_weight * l1_loss + self.ssim_weight * ssim_loss
        else:
            return l1_loss
