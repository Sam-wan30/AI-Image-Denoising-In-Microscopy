#!/usr/bin/env python3
"""
STEP 2: Verify Image Values

Check shape, dtype, min/max, unique values for noisy and clean images.
Detect blank images, corrupted images, incorrect normalization, uint16 issues.
"""

import os
import glob
import cv2
import numpy as np
from pathlib import Path

# Paths
noisy_dir = "data/train/noisy"
clean_dir = "data/train/clean"

print("=" * 70)
print("STEP 2: VERIFY IMAGE VALUES")
print("=" * 70)

# Get all image files
noisy_files = sorted(glob.glob(os.path.join(noisy_dir, "*.png")))
clean_files = sorted(glob.glob(os.path.join(clean_dir, "*.png")))

def analyze_image(image_path, image_type):
    """Analyze a single image."""
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        return {
            'error': 'Could not read image',
            'blank': True,
            'corrupted': True
        }
    
    # Check if blank
    is_blank = np.all(img == 0) or np.all(img == img.flat[0])
    
    # Check if corrupted (all same value)
    is_corrupted = len(np.unique(img)) == 1
    
    return {
        'shape': img.shape,
        'dtype': str(img.dtype),
        'min': float(img.min()),
        'max': float(img.max()),
        'mean': float(img.mean()),
        'std': float(img.std()),
        'unique_values': len(np.unique(img)),
        'blank': is_blank,
        'corrupted': is_corrupted,
        'error': None
    }

# Analyze first 10 noisy images
print("\n" + "=" * 70)
print("ANALYZING FIRST 10 NOISY IMAGES:")
print("=" * 70)

noisy_stats = []
for i in range(min(10, len(noisy_files))):
    stats = analyze_image(noisy_files[i], 'noisy')
    noisy_stats.append(stats)
    
    print(f"\n{i+1}. {Path(noisy_files[i]).name}")
    if stats['error']:
        print(f"   ERROR: {stats['error']}")
    else:
        print(f"   Shape: {stats['shape']}")
        print(f"   Dtype: {stats['dtype']}")
        print(f"   Min/Max: {stats['min']:.2f} / {stats['max']:.2f}")
        print(f"   Mean/Std: {stats['mean']:.2f} / {stats['std']:.2f}")
        print(f"   Unique values: {stats['unique_values']}")
        if stats['blank']:
            print(f"   ⚠️  BLANK IMAGE DETECTED")
        if stats['corrupted']:
            print(f"   ⚠️  CORRUPTED IMAGE DETECTED (all same value)")

# Analyze first 10 clean images
print("\n" + "=" * 70)
print("ANALYZING FIRST 10 CLEAN IMAGES:")
print("=" * 70)

clean_stats = []
for i in range(min(10, len(clean_files))):
    stats = analyze_image(clean_files[i], 'clean')
    clean_stats.append(stats)
    
    print(f"\n{i+1}. {Path(clean_files[i]).name}")
    if stats['error']:
        print(f"   ERROR: {stats['error']}")
    else:
        print(f"   Shape: {stats['shape']}")
        print(f"   Dtype: {stats['dtype']}")
        print(f"   Min/Max: {stats['min']:.2f} / {stats['max']:.2f}")
        print(f"   Mean/Std: {stats['mean']:.2f} / {stats['std']:.2f}")
        print(f"   Unique values: {stats['unique_values']}")
        if stats['blank']:
            print(f"   ⚠️  BLANK IMAGE DETECTED")
        if stats['corrupted']:
            print(f"   ⚠️  CORRUPTED IMAGE DETECTED (all same value)")

# Analyze all images for statistics
print("\n" + "=" * 70)
print("FULL DATASET STATISTICS:")
print("=" * 70)

def get_full_stats(files, image_type):
    """Get statistics for all images."""
    shapes = []
    dtypes = set()
    mins = []
    maxs = []
    means = []
    stds = []
    unique_counts = []
    blank_count = 0
    corrupted_count = 0
    error_count = 0
    
    for f in files:
        stats = analyze_image(f, image_type)
        if stats['error']:
            error_count += 1
            continue
        
        shapes.append(stats['shape'])
        dtypes.add(stats['dtype'])
        mins.append(stats['min'])
        maxs.append(stats['max'])
        means.append(stats['mean'])
        stds.append(stats['std'])
        unique_counts.append(stats['unique_values'])
        
        if stats['blank']:
            blank_count += 1
        if stats['corrupted']:
            corrupted_count += 1
    
    return {
        'count': len(files),
        'error_count': error_count,
        'shapes': shapes,
        'dtypes': dtypes,
        'min_range': (min(mins), max(maxs)) if mins else (0, 0),
        'max_range': (min(maxs), max(maxs)) if maxs else (0, 0),
        'mean_range': (min(means), max(means)) if means else (0, 0),
        'std_range': (min(stds), max(stds)) if stds else (0, 0),
        'unique_range': (min(unique_counts), max(unique_counts)) if unique_counts else (0, 0),
        'blank_count': blank_count,
        'corrupted_count': corrupted_count
    }

noisy_full = get_full_stats(noisy_files, 'noisy')
clean_full = get_full_stats(clean_files, 'clean')

print(f"\nNOISY IMAGES ({noisy_full['count']} total):")
print(f"  Errors: {noisy_full['error_count']}")
print(f"  Data types: {noisy_full['dtypes']}")
print(f"  Shapes: {set(noisy_full['shapes'])}")
print(f"  Min value range: {noisy_full['min_range']}")
print(f"  Max value range: {noisy_full['max_range']}")
print(f"  Mean range: {noisy_full['mean_range']}")
print(f"  Std range: {noisy_full['std_range']}")
print(f"  Unique values range: {noisy_full['unique_range']}")
print(f"  Blank images: {noisy_full['blank_count']}")
print(f"  Corrupted images: {noisy_full['corrupted_count']}")

print(f"\nCLEAN IMAGES ({clean_full['count']} total):")
print(f"  Errors: {clean_full['error_count']}")
print(f"  Data types: {clean_full['dtypes']}")
print(f"  Shapes: {set(clean_full['shapes'])}")
print(f"  Min value range: {clean_full['min_range']}")
print(f"  Max value range: {clean_full['max_range']}")
print(f"  Mean range: {clean_full['mean_range']}")
print(f"  Std range: {clean_full['std_range']}")
print(f"  Unique values range: {clean_full['unique_range']}")
print(f"  Blank images: {clean_full['blank_count']}")
print(f"  Corrupted images: {clean_full['corrupted_count']}")

# Check for uint16 issues
print("\n" + "=" * 70)
print("CHECKING FOR UINT16 ISSUES:")
print("=" * 70)

if 'uint16' in noisy_full['dtypes'] or 'uint16' in clean_full['dtypes']:
    print("⚠️  UINT16 DETECTED - This can cause normalization issues!")
    print("   Images should be uint8 (0-255) or float32 (0.0-1.0)")
else:
    print("✓ No uint16 issues detected")

# Check for normalization issues
print("\n" + "=" * 70)
print("CHECKING NORMALIZATION:")
print("=" * 70)

if noisy_full['max_range'][1] > 255:
    print("⚠️  Values exceed 255 - possible normalization issue")
elif noisy_full['max_range'][1] == 255 and noisy_full['min_range'][0] == 0:
    print("✓ Images appear to be in correct uint8 range [0, 255]")
elif noisy_full['max_range'][1] <= 1.0 and noisy_full['min_range'][0] >= 0.0:
    print("✓ Images appear to be normalized to [0, 1]")
else:
    print(f"⚠️  Unexpected value range: [{noisy_full['min_range'][0]:.2f}, {noisy_full['max_range'][1]:.2f}]")

# Summary
print("\n" + "=" * 70)
print("STEP 2 SUMMARY:")
print("=" * 70)

issues = []

if noisy_full['error_count'] > 0 or clean_full['error_count'] > 0:
    issues.append(f"Error reading images: noisy={noisy_full['error_count']}, clean={clean_full['error_count']}")

if noisy_full['blank_count'] > 0 or clean_full['blank_count'] > 0:
    issues.append(f"Blank images: noisy={noisy_full['blank_count']}, clean={clean_full['blank_count']}")

if noisy_full['corrupted_count'] > 0 or clean_full['corrupted_count'] > 0:
    issues.append(f"Corrupted images: noisy={noisy_full['corrupted_count']}, clean={clean_full['corrupted_count']}")

if 'uint16' in noisy_full['dtypes'] or 'uint16' in clean_full['dtypes']:
    issues.append("Uint16 dtype detected")

if noisy_full['max_range'][1] > 255 or clean_full['max_range'][1] > 255:
    issues.append("Values exceed 255")

if len(set(noisy_full['shapes'])) > 1 or len(set(clean_full['shapes'])) > 1:
    issues.append(f"Inconsistent image shapes: noisy={set(noisy_full['shapes'])}, clean={set(clean_full['shapes'])}")

if issues:
    print("⚠️  ISSUES DETECTED:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("✓ All image values are CORRECT")
    print("✓ No blank or corrupted images")
    print("✓ Normalization appears correct")
    print("✓ Data types are appropriate")

print("=" * 70)
