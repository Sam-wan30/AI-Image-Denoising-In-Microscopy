#!/usr/bin/env python3
"""
STEP 1: Verify Dataset Pairing

Check if noisy and clean images are correctly paired.
"""

import os
import glob
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Paths
noisy_dir = "data/train/noisy"
clean_dir = "data/train/clean"

print("=" * 70)
print("STEP 1: VERIFY DATASET PAIRING")
print("=" * 70)

# Get all image files
noisy_files = sorted(glob.glob(os.path.join(noisy_dir, "*.png")))
clean_files = sorted(glob.glob(os.path.join(clean_dir, "*.png")))

print(f"\nTotal noisy images: {len(noisy_files)}")
print(f"Total clean images: {len(clean_files)}")

# Build stem-based pairs (same logic as CAREDatasetSimple)
clean_map = {Path(f).stem: f for f in clean_files}
pairs = []
unmatched_noisy = []
for nf in noisy_files:
    stem = Path(nf).stem
    if stem in clean_map:
        pairs.append((nf, clean_map[stem]))
    else:
        unmatched_noisy.append(Path(nf).name)

print("\n" + "=" * 70)
print("FIRST 20 NOISY-CLEAN FILENAME PAIRS (stem-matched):")
print("=" * 70)

for i, (noisy_path, clean_path) in enumerate(pairs[:20]):
    n_name = Path(noisy_path).name
    c_name = Path(clean_path).name
    print(f"{i+1:3d}. Noisy: {n_name}")
    print(f"     Clean: {c_name}")
    print(f"     Match: {'✓' if Path(noisy_path).stem == Path(clean_path).stem else '✗ MISMATCH'}")
    print()

# Check for mismatches (should be none with stem matching)
print("=" * 70)
print("CHECKING FOR MISMATCHES:")
print("=" * 70)

mismatches = unmatched_noisy

if mismatches:
    print(f"\n⚠️  FOUND {len(mismatches)} UNMATCHED NOISY FILES:")
    for name in mismatches[:10]:
        print(f"  {name}")
else:
    print("✓ All noisy files have clean counterparts (stem-matched)!")

# Check for missing pairs
print("=" * 70)
print("CHECKING FOR MISSING PAIRS:")
print("=" * 70)

noisy_set = {Path(f).name for f in noisy_files}
clean_set = {Path(f).name for f in clean_files}

missing_in_clean = noisy_set - clean_set
missing_in_noisy = clean_set - noisy_set

if missing_in_clean:
    print(f"\n⚠️  {len(missing_in_clean)} noisy images missing clean counterparts:")
    for name in list(missing_in_clean)[:10]:
        print(f"  {name}")

if missing_in_noisy:
    print(f"\n⚠️  {len(missing_in_noisy)} clean images missing noisy counterparts:")
    for name in list(missing_in_noisy)[:10]:
        print(f"  {name}")

if not missing_in_clean and not missing_in_noisy:
    print("✓ All images have corresponding pairs!")

# Display 10 random image pairs
print("\n" + "=" * 70)
print("DISPLAYING 10 RANDOM NOISY-CLEAN IMAGE PAIRS:")
print("=" * 70)

import random
random.seed(42)
sample_indices = random.sample(range(len(pairs)), min(10, len(pairs)))

fig, axes = plt.subplots(len(sample_indices), 2, figsize=(12, 5 * len(sample_indices)))
fig.suptitle('Random Noisy-Clean Image Pairs (stem-matched)', fontsize=16, y=1.005)

for idx, i in enumerate(sample_indices):
    noisy_path, clean_path = pairs[i]
    noisy_img = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE)
    clean_img = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)

    axes[idx, 0].imshow(noisy_img, cmap='gray')
    axes[idx, 0].set_title(f'Noisy: {Path(noisy_path).name[:40]}', fontsize=8)
    axes[idx, 0].axis('off')

    axes[idx, 1].imshow(clean_img, cmap='gray')
    axes[idx, 1].set_title(f'Clean: {Path(clean_path).name[:40]}', fontsize=8)
    axes[idx, 1].axis('off')

plt.tight_layout()
plt.savefig('debug_step1_image_pairs.png', dpi=100, bbox_inches='tight')
print("\n✓ Saved visualization to: debug_step1_image_pairs.png")

# Summary
print("\n" + "=" * 70)
print("STEP 1 SUMMARY:")
print("=" * 70)
if not mismatches and not missing_in_clean and not missing_in_noisy:
    print("✓ Dataset pairing is CORRECT")
    print("✓ All noisy-clean pairs match by filename")
    print("✓ No missing pairs detected")
else:
    print("⚠️  Dataset pairing has ISSUES")
    if mismatches:
        print(f"  - {len(mismatches)} filename mismatches")
    if missing_in_clean:
        print(f"  - {len(missing_in_clean)} noisy images without clean counterparts")
    if missing_in_noisy:
        print(f"  - {len(missing_in_noisy)} clean images without noisy counterparts")
print("=" * 70)
