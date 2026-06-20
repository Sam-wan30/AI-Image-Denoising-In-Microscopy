#!/usr/bin/env python3
"""
PyTorch Dataset class for CARE microscopy denoising (Simple version).

This dataset class handles paired noisy and clean images for training
denoising models on microscopy data without external dependencies.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from pathlib import Path
import glob
from typing import Tuple, List, Optional

from utils.preprocessing import (
    IMAGE_SIZE,
    load_grayscale,
    normalize_image,
    resize_image,
)
from utils.data_splitting import specimen_group_id


class CAREDatasetSimple(Dataset):
    """
    PyTorch Dataset for CARE microscopy denoising.
    
    Args:
        root_dir (str): Root directory containing 'train/noisy' and 'train/clean' folders
        image_size (tuple): Target image size (height, width). Default: (256, 256)
        normalize (bool): Whether to normalize images to [0,1]. Default: True
        augment (bool): Whether to apply data augmentation. Default: False
    """
    
    def __init__(
        self, 
        root_dir: str, 
        image_size: Tuple[int, int] = IMAGE_SIZE,
        normalize: bool = True,
        augment: bool = False
    ):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.normalize = normalize
        self.augment = augment
        
        # Define noisy and clean directories
        self.noisy_dir = self.root_dir / "train" / "noisy"
        self.clean_dir = self.root_dir / "train" / "clean"
        
        # Check if directories exist
        if not self.noisy_dir.exists():
            raise FileNotFoundError(f"Noisy directory not found: {self.noisy_dir}")
        if not self.clean_dir.exists():
            raise FileNotFoundError(f"Clean directory not found: {self.clean_dir}")
        
        # Get matched image pairs
        self.image_pairs = self._get_image_pairs()
        
        if len(self.image_pairs) == 0:
            raise ValueError(f"No matching image pairs found in {self.root_dir}")
        
        print(f"Found {len(self.image_pairs)} image pairs")
    
    def _get_image_pairs(self) -> List[Tuple[str, str]]:
        """
        Get matched pairs of noisy and clean image files based on filenames.
        
        Returns:
            List of tuples containing (noisy_path, clean_path)
        """
        # Get all image files
        noisy_files = self._get_image_files(self.noisy_dir)
        clean_files = self._get_image_files(self.clean_dir)
        
        # Create filename mapping for clean files
        clean_map = {}
        for clean_file in clean_files:
            clean_name = Path(clean_file).stem  # Get filename without extension
            if clean_name in clean_map:
                raise ValueError(f"Duplicate clean-image stem: {clean_name}")
            clean_map[clean_name] = clean_file
        
        # Match noisy files with clean files
        pairs = []
        unmatched_noisy = []
        
        for noisy_file in noisy_files:
            noisy_name = Path(noisy_file).stem
            
            if noisy_name in clean_map:
                pairs.append((noisy_file, clean_map[noisy_name]))
            else:
                unmatched_noisy.append(noisy_name)
        
        # Report unmatched files
        if unmatched_noisy:
            print(f"Warning: {len(unmatched_noisy)} noisy files without clean matches")
            print(f"Sample unmatched: {unmatched_noisy[:5]}")
        matched_clean = {Path(clean_path).stem for _noisy_path, clean_path in pairs}
        unmatched_clean = sorted(set(clean_map).difference(matched_clean))
        if unmatched_clean:
            print(f"Warning: {len(unmatched_clean)} clean files without noisy matches")
        
        return pairs
    
    def _get_image_files(self, directory: Path) -> List[str]:
        """Get all image files from a directory."""
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp']
        files = []
        
        for ext in extensions:
            files.extend(glob.glob(str(directory / ext)))
            files.extend(glob.glob(str(directory / ext.upper())))
        
        return sorted(files)
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load grayscale image using shared preprocessing."""
        try:
            return load_grayscale(image_path)
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            raise

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Resize and normalize using shared preprocessing."""
        image = resize_image(image, self.image_size)
        if self.normalize:
            image = normalize_image(image)
        return image
    
    def _augment_image(self, image: np.ndarray, ops: Tuple[bool, bool, int]) -> np.ndarray:
        """
        Apply simple data augmentation to image.

        The same operation tuple must be used for the noisy and clean image.
        Using one RNG sequentially for both images desynchronizes the pair.
        
        Args:
            image: Input image as numpy array
            ops: Tuple of (horizontal_flip, vertical_flip, rot90_k)
            
        Returns:
            Augmented image as numpy array
        """
        if not self.augment:
            return image

        import cv2

        horizontal_flip, vertical_flip, rot90_k = ops

        if horizontal_flip:
            image = cv2.flip(image, 1)
        if vertical_flip:
            image = cv2.flip(image, 0)
        if rot90_k:
            image = np.rot90(image, rot90_k)

        return np.ascontiguousarray(image)
    
    def __len__(self) -> int:
        """Return the number of image pairs in the dataset."""
        return len(self.image_pairs)

    def group_id(self, idx: int) -> str:
        """Return the specimen/session group used for leakage-safe splitting."""
        return specimen_group_id(self.image_pairs[idx][0])
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (noisy_tensor, clean_tensor)
        """
        # Get file paths
        noisy_path, clean_path = self.image_pairs[idx]
        
        # Load images
        noisy_img = self._load_image(noisy_path)
        clean_img = self._load_image(clean_path)
        if noisy_img.shape != clean_img.shape:
            raise ValueError(
                f"Paired image shape mismatch: {Path(noisy_path).name} "
                f"{noisy_img.shape} != {clean_img.shape}"
            )
        
        # Preprocess images
        noisy_img = self._preprocess_image(noisy_img)
        clean_img = self._preprocess_image(clean_img)
        
        # Apply identical augmentation to noisy and clean
        if self.augment:
            ops = (
                np.random.random() > 0.5,
                np.random.random() > 0.5,
                np.random.randint(1, 4) if np.random.random() > 0.5 else 0,
            )
            noisy_img = self._augment_image(noisy_img, ops)
            clean_img = self._augment_image(clean_img, ops)
        
        # Convert to tensors (make copies to avoid negative stride issues)
        noisy_tensor = torch.from_numpy(noisy_img.copy()).float()
        clean_tensor = torch.from_numpy(clean_img.copy()).float()
        
        # Add channel dimension if not present (C, H, W format)
        if len(noisy_tensor.shape) == 2:
            noisy_tensor = noisy_tensor.unsqueeze(0)
            clean_tensor = clean_tensor.unsqueeze(0)
        
        return noisy_tensor, clean_tensor


def create_dataloader(
    root_dir: str,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    augment: bool = False,
    image_size: Tuple[int, int] = (256, 256)
) -> DataLoader:
    """
    Create a DataLoader for the CARE dataset.
    
    Args:
        root_dir: Root directory containing the dataset
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        augment: Whether to apply data augmentation
        image_size: Target image size
        
    Returns:
        DataLoader instance
    """
    # Create dataset
    dataset = CAREDatasetSimple(
        root_dir=root_dir,
        image_size=image_size,
        normalize=True,
        augment=augment
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    
    return dataloader


# Example usage and testing
if __name__ == "__main__":
    # Test the dataset
    try:
        print("Testing CAREDatasetSimple...")
        
        # Create dataset
        dataset = CAREDatasetSimple(root_dir="data")
        
        print(f"Dataset size: {len(dataset)}")
        
        # Test loading a sample
        noisy, clean = dataset[0]
        print(f"Sample shapes - Noisy: {noisy.shape}, Clean: {clean.shape}")
        print(f"Sample ranges - Noisy: [{noisy.min():.3f}, {noisy.max():.3f}], Clean: [{clean.min():.3f}, {clean.max():.3f}]")
        
        # Test DataLoader
        dataloader = create_dataloader(
            root_dir="data",
            batch_size=4,
            shuffle=True,
            augment=True
        )
        
        print(f"DataLoader created with {len(dataloader)} batches")
        
        # Test a batch
        for batch_noisy, batch_clean in dataloader:
            print(f"Batch shapes - Noisy: {batch_noisy.shape}, Clean: {batch_clean.shape}")
            print(f"Batch ranges - Noisy: [{batch_noisy.min():.3f}, {batch_noisy.max():.3f}], Clean: [{batch_clean.min():.3f}, {batch_clean.max():.3f}]")
            break
        
        print("Dataset test completed successfully!")
        
    except Exception as e:
        print(f"Error testing dataset: {str(e)}")
        print("Make sure you have processed data in the 'data' directory first.")
