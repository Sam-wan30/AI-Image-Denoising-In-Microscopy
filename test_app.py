#!/usr/bin/env python3
"""
Test script for the Streamlit app functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from PIL import Image
import io

# Test imports
try:
    from src.unet_model import create_unet_model
    from utils.metrics import calculate_psnr, calculate_ssim
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test model creation
try:
    model = create_unet_model(model_type='standard', in_channels=1, out_channels=1)
    print(f"✅ Model created successfully: {sum(p.numel() for p in model.parameters()):,} parameters")
except Exception as e:
    print(f"❌ Model creation error: {e}")
    sys.exit(1)

# Test metrics calculation
try:
    # Create test images
    img1 = torch.randn(1, 256, 256)
    img2 = img1 + 0.1 * torch.randn(1, 256, 256)
    
    psnr = calculate_psnr(img1, img2)
    ssim = calculate_ssim(img1, img2)
    
    print(f"✅ Metrics calculated successfully: PSNR={psnr:.2f}, SSIM={ssim:.4f}")
except Exception as e:
    print(f"❌ Metrics calculation error: {e}")
    sys.exit(1)

# Test image processing
try:
    # Create a test image
    test_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    pil_image = Image.fromarray(test_image)
    
    # Convert to bytes
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    print(f"✅ Image processing test successful")
except Exception as e:
    print(f"❌ Image processing error: {e}")
    sys.exit(1)

print("\n🎉 All tests passed! The Streamlit app should work correctly.")
print("\nTo run the app, use:")
print("streamlit run app.py")
