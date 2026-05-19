#!/usr/bin/env python3
"""
Inference script for microscopy image denoising.

This script loads a trained U-Net model and performs denoising on
noisy microscopy images with visualization and saving capabilities.
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
import argparse
import time
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Union, Tuple, Optional
import glob

# Import our custom modules
from src.unet_model import create_unet_model
from utils.metrics import calculate_psnr, calculate_ssim, MetricsCalculator
from utils.preprocessing import (
    IMAGE_SIZE,
    get_original_shape,
    postprocess_tensor,
    preprocess_tensor,
)


def detect_model_type(state_dict_keys):
    """Infer architecture from checkpoint keys."""
    if any(k.startswith("unet.") for k in state_dict_keys):
        return "residual"
    if any("residual_blocks" in k for k in state_dict_keys):
        return "enhanced"
    return "standard"


class DenoisingInference:
    """
    Class for performing microscopy image denoising inference.
    """
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize the inference class.
        
        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run inference on (cuda/cpu). If None, auto-detect
        """
        self.model_path = model_path
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_info = {}
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the trained model from checkpoint."""
        print(f"Loading model from: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Extract model configuration
        if 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
            self.model_info = {
                'epoch': checkpoint.get('epoch', 0),
                'val_loss': checkpoint.get('val_loss', 0.0),
                'val_psnr': checkpoint.get('val_psnr', 0.0),
                'val_ssim': checkpoint.get('val_ssim', 0.0),
            }
        else:
            # Handle case where only state dict is saved
            model_state_dict = checkpoint
            self.model_info = {'epoch': 0, 'val_loss': 0.0}
        
        state_dict_keys = list(model_state_dict.keys())
        model_type = checkpoint.get("model_type") or detect_model_type(state_dict_keys)

        # Get input/output channels from first/last conv layers
        if model_type == "residual":
            first_conv_key = [k for k in state_dict_keys if "unet.inc.double_conv.0.weight" in k][0]
            last_conv_key = [k for k in state_dict_keys if "unet.outc.conv.weight" in k][0]
        else:
            first_conv_key = [k for k in state_dict_keys if "inc.double_conv.0.weight" in k][0]
            last_conv_key = [k for k in state_dict_keys if "outc.conv.weight" in k][0]

        in_channels = model_state_dict[first_conv_key].shape[1]
        out_channels = model_state_dict[last_conv_key].shape[0]
        self.model = create_unet_model(
            model_type=model_type,
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=64
        )
        
        # Load state dict. strict=False keeps older BatchNorm checkpoints loadable
        # after the training model moved to GroupNorm for small-batch stability.
        missing, unexpected = self.model.load_state_dict(model_state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "Checkpoint is not compatible with the current model code: "
                f"missing={len(missing)}, unexpected={len(unexpected)}. "
                "Use models/overfit_residual_blocks/best_model.pth or retrain with train.py."
            )
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded successfully!")
        print(f"  - Type: {model_type} U-Net")
        print(f"  - Input channels: {in_channels}")
        print(f"  - Output channels: {out_channels}")
        print(f"  - Device: {self.device}")
        print(f"  - Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        if missing or unexpected:
            print(f"  - Checkpoint compatibility: missing={len(missing)}, unexpected={len(unexpected)}")
        if self.model_info['epoch'] > 0:
            print(f"  - Training epoch: {self.model_info['epoch']}")
            print(f"  - Validation PSNR: {self.model_info['val_psnr']:.2f}")
            print(f"  - Validation SSIM: {self.model_info['val_ssim']:.4f}")
    
    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, np.ndarray, Tuple[int, int]]:
        """Preprocess using shared pipeline (identical to training)."""
        original_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if original_image is None:
            raise ValueError(f"Could not load image: {image_path}")

        original_shape = get_original_shape(image_path)
        image_tensor = preprocess_tensor(image_path, IMAGE_SIZE)
        return image_tensor, original_image, original_shape

    def postprocess_output(
        self, output_tensor: torch.Tensor, original_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Postprocess using shared pipeline."""
        return postprocess_tensor(output_tensor, original_shape, IMAGE_SIZE)
    
    def denoise_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Denoise a single image.
        
        Args:
            image_path: Path to the noisy image
            
        Returns:
            Tuple of (denoised_image, original_image)
        """
        input_tensor, original_image, original_shape = self.preprocess_image(image_path)
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            if self.device == "cuda":
                with torch.cuda.amp.autocast():
                    output_tensor = self.model(input_tensor)
            else:
                output_tensor = self.model(input_tensor)

        denoised_image = self.postprocess_output(output_tensor, original_shape)
        
        return denoised_image, original_image
    
    def calculate_metrics(self, original: np.ndarray, denoised: np.ndarray, ground_truth: Optional[np.ndarray] = None) -> dict:
        """
        Calculate quality metrics.
        
        Args:
            original: Original noisy image
            denoised: Denoised image
            ground_truth: Optional ground truth clean image
            
        Returns:
            Dictionary of calculated metrics
        """
        metrics = {}
        
        if ground_truth is not None:
            # Convert to tensors for metric calculation
            denoised_tensor = torch.from_numpy(denoised).float() / 255.0
            gt_tensor = torch.from_numpy(ground_truth).float() / 255.0
            
            # Calculate PSNR and SSIM
            psnr = calculate_psnr(denoised_tensor, gt_tensor, max_val=1.0)
            ssim = calculate_ssim(denoised_tensor, gt_tensor, max_val=1.0)
            
            metrics['psnr_vs_gt'] = psnr
            metrics['ssim_vs_gt'] = ssim
        
        # Calculate noise reduction metrics
        original_tensor = torch.from_numpy(original).float() / 255.0
        denoised_tensor = torch.from_numpy(denoised).float() / 255.0
        
        # Simple noise estimation (standard deviation)
        noise_original = torch.std(original_tensor)
        noise_denoised = torch.std(denoised_tensor)
        noise_reduction = (noise_original - noise_denoised) / noise_original * 100
        
        metrics['noise_original'] = noise_original.item()
        metrics['noise_denoised'] = noise_denoised.item()
        metrics['noise_reduction_percent'] = noise_reduction.item()
        
        return metrics
    
    def create_comparison_image(self, original: np.ndarray, denoised: np.ndarray, 
                              ground_truth: Optional[np.ndarray] = None, 
                              metrics: Optional[dict] = None) -> np.ndarray:
        """
        Create side-by-side comparison image.
        
        Args:
            original: Original noisy image
            denoised: Denoised image
            ground_truth: Optional ground truth image
            metrics: Optional metrics dictionary
            
        Returns:
            Comparison image as numpy array
        """
        # Ensure all images have the same shape
        h, w = original.shape[:2]
        
        # Prepare images
        if len(original.shape) == 3:
            original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) if original.shape[2] == 3 else original[:, :, 0]
        else:
            original_gray = original
        
        if len(denoised.shape) == 3:
            denoised_gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY) if denoised.shape[2] == 3 else denoised[:, :, 0]
        else:
            denoised_gray = denoised
        
        # Create comparison layout
        if ground_truth is not None:
            if len(ground_truth.shape) == 3:
                gt_gray = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY) if ground_truth.shape[2] == 3 else ground_truth[:, :, 0]
            else:
                gt_gray = ground_truth
            
            # Three image layout: Original | Denoised | Ground Truth
            comparison = np.zeros((h, w * 3), dtype=np.uint8)
            comparison[:, :w] = original_gray
            comparison[:, w:w*2] = denoised_gray
            comparison[:, w*2:] = gt_gray
            
            # Add labels
            labels = ['Noisy Input', 'Denoised Output', 'Ground Truth']
        else:
            # Two image layout: Original | Denoised
            comparison = np.zeros((h, w * 2), dtype=np.uint8)
            comparison[:, :w] = original_gray
            comparison[:, w:] = denoised_gray
            
            # Add labels
            labels = ['Noisy Input', 'Denoised Output']
        
        # Add title and metrics
        comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_GRAY2BGR)
        
        # Add title
        title = "Microscopy Image Denoising Results"
        cv2.putText(comparison_bgr, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add column labels
        for i, label in enumerate(labels):
            x_pos = (i * w) + (w // 2) - (len(label) * 8)
            cv2.putText(comparison_bgr, label, (x_pos, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Add metrics if available
        if metrics:
            metrics_text = []
            if 'psnr_vs_gt' in metrics:
                metrics_text.append(f"PSNR: {metrics['psnr_vs_gt']:.2f} dB")
            if 'ssim_vs_gt' in metrics:
                metrics_text.append(f"SSIM: {metrics['ssim_vs_gt']:.4f}")
            if 'noise_reduction_percent' in metrics:
                metrics_text.append(f"Noise Reduction: {metrics['noise_reduction_percent']:.1f}%")
            
            for i, text in enumerate(metrics_text):
                cv2.putText(comparison_bgr, text, (10, 70 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return comparison_bgr
    
    def process_single_image(self, input_path: str, output_dir: str, 
                            ground_truth_path: Optional[str] = None,
                            save_comparison: bool = True,
                            show_result: bool = False) -> dict:
        """
        Process a single image and save results.
        
        Args:
            input_path: Path to input noisy image
            output_dir: Directory to save results
            ground_truth_path: Optional path to ground truth image
            save_comparison: Whether to save comparison image
            show_result: Whether to display result
            
        Returns:
            Dictionary of results and metrics
        """
        print(f"Processing image: {input_path}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Denoise image
        start_time = time.time()
        denoised_image, original_image = self.denoise_image(input_path)
        inference_time = time.time() - start_time
        
        # Load ground truth if provided
        ground_truth = None
        if ground_truth_path and os.path.exists(ground_truth_path):
            ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
            if ground_truth is None:
                print(f"Warning: Could not load ground truth image: {ground_truth_path}")
        
        # Calculate metrics
        metrics = self.calculate_metrics(original_image, denoised_image, ground_truth)
        metrics['inference_time'] = inference_time
        
        # Save denoised image
        input_name = Path(input_path).stem
        denoised_path = os.path.join(output_dir, f"{input_name}_denoised.png")
        cv2.imwrite(denoised_path, denoised_image)
        
        # Save comparison image
        if save_comparison:
            comparison = self.create_comparison_image(original_image, denoised_image, ground_truth, metrics)
            comparison_path = os.path.join(output_dir, f"{input_name}_comparison.png")
            cv2.imwrite(comparison_path, comparison)
        
        # Show result if requested
        if show_result:
            comparison = self.create_comparison_image(original_image, denoised_image, ground_truth, metrics)
            cv2.imshow('Denoising Result', comparison)
            print("Press any key to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Print results
        print(f"✓ Processing completed!")
        print(f"  - Inference time: {inference_time:.3f}s")
        print(f"  - Denoised image saved to: {denoised_path}")
        if save_comparison:
            print(f"  - Comparison image saved to: {comparison_path}")
        
        if 'psnr_vs_gt' in metrics:
            print(f"  - PSNR (vs ground truth): {metrics['psnr_vs_gt']:.2f} dB")
        if 'ssim_vs_gt' in metrics:
            print(f"  - SSIM (vs ground truth): {metrics['ssim_vs_gt']:.4f}")
        print(f"  - Noise reduction: {metrics['noise_reduction_percent']:.1f}%")
        
        return {
            'input_path': input_path,
            'denoised_path': denoised_path,
            'metrics': metrics
        }
    
    def process_directory(self, input_dir: str, output_dir: str,
                          ground_truth_dir: Optional[str] = None,
                          save_comparisons: bool = True) -> list:
        """
        Process all images in a directory.
        
        Args:
            input_dir: Directory containing noisy images
            output_dir: Directory to save results
            ground_truth_dir: Optional directory containing ground truth images
            save_comparisons: Whether to save comparison images
            
        Returns:
            List of processing results
        """
        print(f"Processing directory: {input_dir}")
        
        # Find all image files
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_dir, ext)))
            image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
        
        if not image_files:
            print(f"No image files found in {input_dir}")
            return []
        
        print(f"Found {len(image_files)} images to process")
        
        results = []
        for i, image_path in enumerate(image_files):
            print(f"\n[{i+1}/{len(image_files)}] Processing: {Path(image_path).name}")
            
            # Find corresponding ground truth if available
            gt_path = None
            if ground_truth_dir:
                image_name = Path(image_path).stem
                gt_candidates = glob.glob(os.path.join(ground_truth_dir, f"{image_name}.*"))
                if gt_candidates:
                    gt_path = gt_candidates[0]
            
            try:
                result = self.process_single_image(
                    image_path, 
                    output_dir, 
                    gt_path, 
                    save_comparison=save_comparisons,
                    show_result=False
                )
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        # Print summary
        print(f"\n✓ Directory processing completed!")
        print(f"  - Successfully processed: {len(results)}/{len(image_files)} images")
        
        if results:
            avg_psnr = np.mean([r['metrics'].get('psnr_vs_gt', 0) for r in results if 'psnr_vs_gt' in r['metrics']])
            avg_ssim = np.mean([r['metrics'].get('ssim_vs_gt', 0) for r in results if 'ssim_vs_gt' in r['metrics']])
            avg_noise_reduction = np.mean([r['metrics']['noise_reduction_percent'] for r in results])
            
            print(f"  - Average PSNR: {avg_psnr:.2f} dB" if avg_psnr > 0 else "")
            print(f"  - Average SSIM: {avg_ssim:.4f}" if avg_ssim > 0 else "")
            print(f"  - Average noise reduction: {avg_noise_reduction:.1f}%")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Microscopy Image Denoising Inference')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input image or directory')
    parser.add_argument('--output', type=str, default='inference_results',
                       help='Output directory for results')
    parser.add_argument('--ground_truth', type=str, default=None,
                       help='Path to ground truth image or directory (optional)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu). Auto-detect if not specified')
    parser.add_argument('--no_comparison', action='store_true',
                       help='Skip saving comparison images')
    parser.add_argument('--show', action='store_true',
                       help='Show results in a window')
    
    args = parser.parse_args()
    
    print("Starting microscopy image denoising inference...")
    
    try:
        # Initialize inference
        inference = DenoisingInference(args.model, args.device)
        
        # Process input
        if os.path.isfile(args.input):
            # Single image processing
            result = inference.process_single_image(
                args.input,
                args.output,
                args.ground_truth,
                save_comparison=not args.no_comparison,
                show_result=args.show
            )
        elif os.path.isdir(args.input):
            # Directory processing
            gt_dir = args.ground_truth if args.ground_truth and os.path.isdir(args.ground_truth) else None
            results = inference.process_directory(
                args.input,
                args.output,
                gt_dir,
                save_comparisons=not args.no_comparison
            )
        else:
            print(f"Error: Input path does not exist: {args.input}")
            return
        
        print("\n✓ Inference completed successfully!")
        print(f"Results saved to: {args.output}")
        
    except Exception as e:
        print(f"\nError during inference: {e}")
        return


if __name__ == "__main__":
    main()
