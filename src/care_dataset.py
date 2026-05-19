#!/usr/bin/env python3
"""
PyTorch Dataset class for CARE microscopy denoising.

This dataset class handles paired noisy and clean images for training
denoising models on microscopy data.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import glob
from typing import Tuple, List, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CAREDataset(Dataset):
    """
    PyTorch Dataset for CARE microscopy denoising.
    
    Args:
        root_dir (str): Root directory containing 'train/noisy' and 'train/clean' folders
        transform (callable, optional): Optional transform to be applied to images
        image_size (tuple): Target image size (height, width). Default: (256, 256)
        normalize (bool): Whether to normalize images to [0,1]. Default: True
    """
    
    def __init__(
        self, 
        root_dir: str, 
        transform: Optional[callable] = None,
        image_size: Tuple[int, int] = (256, 256),
        normalize: bool = True
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_size = image_size
        self.normalize = normalize
        
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
        """
        Load an image from file path.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image as numpy array (H, W) or (H, W, C)
        """
        try:
            # Load image using OpenCV
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                # Check if already grayscale (all channels equal)
                if np.array_equal(image[:, :, 0], image[:, :, 1]) and np.array_equal(image[:, :, 0], image[:, :, 2]):
                    image = image[:, :, 0]
                else:
                    # Convert to grayscale
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            return image
            
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            raise
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image: resize and normalize.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image as numpy array
        """
        # Resize image
        if image.shape[:2] != self.image_size:
            image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)
        
        # Normalize to [0,1] if requested
        if self.normalize:
            if image.dtype != np.float32:
                image = image.astype(np.float32)
            
            if image.max() > 1.0:
                image = image / 255.0
            
            image = np.clip(image, 0.0, 1.0)
        
        return image
    
    def __len__(self) -> int:
        """Return the number of image pairs in the dataset."""
        return len(self.image_pairs)
    
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
        
        # Preprocess images
        noisy_img = self._preprocess_image(noisy_img)
        clean_img = self._preprocess_image(clean_img)
        
        # Apply transforms if specified
        if self.transform:
            # Add channel dimension if needed for albumentations
            if len(noisy_img.shape) == 2:
                noisy_img = np.expand_dims(noisy_img, axis=-1)
                clean_img = np.expand_dims(clean_img, axis=-1)
            
            # Apply same random transform to both images
            transformed = self.transform(image=noisy_img, mask=clean_img)
            noisy_img = transformed['image']
            clean_img = transformed['mask']
            
            # Remove channel dimension if it was added
            if noisy_img.shape[-1] == 1:
                noisy_img = noisy_img.squeeze(-1)
                clean_img = clean_img.squeeze(-1)
        
        # Convert to tensors
        if isinstance(noisy_img, np.ndarray):
            noisy_tensor = torch.from_numpy(noisy_img).float()
            clean_tensor = torch.from_numpy(clean_img).float()
        else:
            noisy_tensor = noisy_img.float()
            clean_tensor = clean_img.float()
        
        # Add channel dimension if not present (C, H, W format)
        if len(noisy_tensor.shape) == 2:
            noisy_tensor = noisy_tensor.unsqueeze(0)
            clean_tensor = clean_tensor.unsqueeze(0)
        
        return noisy_tensor, clean_tensor


def get_transforms(image_size: Tuple[int, int] = (256, 256), augment: bool = False) -> A.Compose:
    """
    Get transforms for training and validation.
    
    Args:
        image_size: Target image size
        augment: Whether to apply data augmentation
        
    Returns:
        Albumentations transform composition
    """
    if augment:
        transform = A.Compose([
            A.RandomCrop(height=image_size[0], width=image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            A.Normalize(mean=0.0, std=1.0),  # Keep [0,1] range
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=0.0, std=1.0),  # Keep [0,1] range
            ToTensorV2()
        ])
    
    return transform


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
    # Get transforms
    transform = get_transforms(image_size=image_size, augment=augment)
    
    # Create dataset
    dataset = CAREDataset(
        root_dir=root_dir,
        transform=transform,
        image_size=image_size,
        normalize=True
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader


# Example usage and testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Test the dataset
    try:
        print("Testing CAREDataset...")
        
        # Create dataset
        dataset = CAREDataset(root_dir="data")
        
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
