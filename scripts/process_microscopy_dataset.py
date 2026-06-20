#!/usr/bin/env python3
"""
Fluorescence Microscopy Image Denoising Dataset Processor

This script processes the Fluorescence Microscopy Image Denoising Dataset from Kaggle.
It handles HDF5 files containing paired noisy and clean images, resizes them to 256x256,
normalizes pixel values to [0,1], and saves them in the required directory structure.
"""

import os
import h5py
import numpy as np
import cv2
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.preprocessing import normalize_image, resize_image

def load_dataset_info(csv_path):
    """Load dataset information from CSV file."""
    df = pd.read_csv(csv_path)
    return df

def create_output_directories(base_path="data"):
    """Create the required directory structure."""
    noisy_dir = os.path.join(base_path, "train", "noisy")
    clean_dir = os.path.join(base_path, "train", "clean")
    
    os.makedirs(noisy_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)
    
    return noisy_dir, clean_dir

def resize_and_normalize_image(image, target_size=(256, 256)):
    """
    Resize image to target size and normalize pixel values to [0,1].
    
    Args:
        image: Input image as numpy array
        target_size: Target size as (height, width)
    
    Returns:
        Processed image as numpy array with values in [0,1]
    """
    # Handle different image dimensions
    if len(image.shape) == 3:
        # For 3D images, take the middle slice or max projection
        if image.shape[0] > 1:
            # Use maximum intensity projection for 3D data
            image = np.max(image, axis=0)
        else:
            image = image[0]
    
    # Apply the same fixed-range normalization used by training and inference.
    # Per-image max normalization would destroy relative intensity calibration.
    return normalize_image(resize_image(image, target_size))

def process_h5_file(h5_path, dataset_info, noisy_dir, clean_dir, max_samples=None):
    """
    Process a single HDF5 file and save paired images.
    
    Args:
        h5_path: Path to HDF5 file
        dataset_info: Dataset information row from CSV
        noisy_dir: Directory to save noisy images
        clean_dir: Directory to save clean images
        max_samples: Maximum number of samples to process (None for all)
    """
    try:
        with h5py.File(h5_path, 'r') as f:
            # Get the keys for noisy and clean images
            noisy_key = dataset_info['Noisy Key']
            clean_key = dataset_info['Clean Key']
            
            # Get image names from clean group (they should match noisy group)
            clean_group = f[clean_key]
            image_names = list(clean_group.keys())
            
            # Limit samples if specified
            if max_samples is not None:
                image_names = image_names[:max_samples]
            
            # Process each sample
            base_filename = os.path.splitext(os.path.basename(h5_path))[0]
            condition = dataset_info['Condition (noisy - clean)'].replace(' - ', '_').replace(' ', '_')
            
            for img_name in tqdm(image_names, desc=f"Processing {base_filename}"):
                # Extract images
                clean_img_data = f[clean_key][img_name][:]
                noisy_img_data = f[noisy_key][img_name][:]
                
                # Process images
                noisy_img = resize_and_normalize_image(noisy_img_data)
                clean_img = resize_and_normalize_image(clean_img_data)
                
                # Generate filename with condition info
                filename = f"{base_filename}_{condition}_{img_name}.png"
                
                # Save images
                noisy_path = os.path.join(noisy_dir, filename)
                clean_path = os.path.join(clean_dir, filename)
                
                # Convert back to uint8 for saving as PNG
                noisy_uint8 = (noisy_img * 255).astype(np.uint8)
                clean_uint8 = (clean_img * 255).astype(np.uint8)
                
                cv2.imwrite(noisy_path, noisy_uint8)
                cv2.imwrite(clean_path, clean_uint8)
                
            print(f"Processed {len(image_names)} samples from {base_filename} ({condition})")
            
    except Exception as e:
        print(f"Error processing {h5_path}: {str(e)}")

def find_h5_files(dataset_dir):
    """Find all HDF5 files in the dataset directory and subdirectories."""
    h5_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.h5'):
                h5_files.append(os.path.join(root, file))
    return h5_files

def main():
    parser = argparse.ArgumentParser(description='Process Fluorescence Microscopy Denoising Dataset')
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default='.',
        help='Path to dataset directory containing HDF5 files',
    )
    parser.add_argument(
        '--csv_path',
        type=str,
        default='dataset_description.csv',
        help='Path to dataset description CSV file',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data',
        help='Output directory for processed images',
    )
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to process per file')
    
    args = parser.parse_args()
    
    print("Starting dataset processing...")
    
    # Load dataset information
    print("Loading dataset information...")
    dataset_info = load_dataset_info(args.csv_path)
    
    # Create output directories
    print("Creating output directories...")
    noisy_dir, clean_dir = create_output_directories(args.output_dir)
    
    # Find all HDF5 files
    print("Finding HDF5 files...")
    h5_files = find_h5_files(args.dataset_dir)
    print(f"Found {len(h5_files)} HDF5 files")
    
    # Process each file
    for h5_file in h5_files:
        filename = os.path.basename(h5_file)
        
        # Find corresponding dataset info
        matching_rows = dataset_info[dataset_info.iloc[:, 0].str.contains(filename)]
        
        if matching_rows.empty:
            print(f"No dataset info found for {filename}")
            continue
        
        # Process each matching row (some files have multiple conditions)
        for _, row in matching_rows.iterrows():
            print(f"\nProcessing {filename} with condition: {row['Condition (noisy - clean)']}")
            process_h5_file(h5_file, row, noisy_dir, clean_dir, args.max_samples)
    
    print(f"\nDataset processing complete!")
    print(f"Noisy images saved to: {noisy_dir}")
    print(f"Clean images saved to: {clean_dir}")
    
    # Count processed images
    noisy_count = len([f for f in os.listdir(noisy_dir) if f.endswith('.png')])
    clean_count = len([f for f in os.listdir(clean_dir) if f.endswith('.png')])
    
    print(f"Total noisy images: {noisy_count}")
    print(f"Total clean images: {clean_count}")

if __name__ == "__main__":
    main()
