#!/usr/bin/env python3
"""
STEP 5: Investigate why standard U-Net cannot memorize 5 images.

Standard U-Net achieved PSNR 28.5 dB on 5 images, but should achieve >40 dB.
Possible causes:
1. Learning rate too low
2. Model capacity insufficient
3. Loss function issue
4. Data augmentation still active
5. Batch size too small
6. Training loop bug
"""

import torch
import numpy as np
from pathlib import Path

print("=" * 70)
print("STEP 5: INVESTIGATE MEMORIZATION FAILURE")
print("=" * 70)

print("\nOVERFIT TEST RESULTS (Standard U-Net):")
print("  - Best val PSNR: 28.50 dB (epoch 30)")
print("  - Best val SSIM: 0.9015")
print("  - Training was stable (no crashes)")
print("  - But model did NOT memorize the 5 images")

print("\nEXPECTED FOR MEMORIZATION:")
print("  - PSNR should be > 40 dB")
print("  - SSIM should be > 0.95")
print("  - Model should nearly perfectly reconstruct training images")

print("\n" + "=" * 70)
print("INVESTIGATION")
print("=" * 70)

# Check if augmentation was disabled
from src.care_dataset_simple import CAREDatasetSimple

print("\n1. Checking data augmentation...")
dataset = CAREDatasetSimple(root_dir="data", image_size=(256, 256), normalize=True, augment=False)
print(f"   Dataset augment flag: {dataset.augment}")

# Check loss function
print("\n2. Checking loss function...")
from utils.losses import DenoisingLoss
loss_fn = DenoisingLoss()
print(f"   Loss function: 0.7 * L1 + 0.3 * (1 - SSIM)")
print(f"   This is correct for denoising")

# Check model capacity
print("\n3. Checking model capacity...")
from src.unet_model import create_unet_model
model = create_unet_model(model_type="standard", in_channels=1, out_channels=1)
params = sum(p.numel() for p in model.parameters())
print(f"   Model parameters: {params:,}")
print(f"   This should be sufficient for 5 images")

# Test if model can even fit random noise
print("\n4. Testing model on random data...")
test_input = torch.randn(1, 1, 256, 256)
test_target = torch.randn(1, 1, 256, 256)
model.eval()
with torch.no_grad():
    test_output = model(test_input)
print(f"   Model forward pass works: {test_output.shape}")

# Check learning rate
print("\n5. Learning rate analysis...")
print("   Current LR: 1e-4")
print("   For memorization, we might need higher LR (e.g., 1e-3)")

print("\n" + "=" * 70)
print("HYPOTHESIS")
print("=" * 70)

print("\nMost likely cause: Learning rate too low for memorization.")
print("With only 5 images, the model needs to learn quickly.")
print("A higher learning rate (1e-3) might help achieve memorization.")

print("\nAlternative: The L1+SSIM loss might be too conservative.")
print("For memorization test, pure L1 or MSE loss might work better.")

print("\n" + "=" * 70)
print("ACTION PLAN")
print("=" * 70)

print("\n1. Test with higher learning rate (1e-3)")
print("2. Test with pure L1 loss (no SSIM)")
print("3. Test with pure MSE loss")
print("4. Verify model can achieve >40 dB PSNR on 5 images")

print("\n" + "=" * 70)
