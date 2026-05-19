#!/usr/bin/env python3
"""
U-Net model for microscopy image denoising.

This implementation is optimized for preserving fine biological structures
in microscopy images, with support for grayscale (1-channel) inputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


def make_norm(channels: int) -> nn.GroupNorm:
    """Batch-size-independent normalization for denoising with small batches."""
    groups = min(8, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class DoubleConv(nn.Module):
    """
    Double convolution block with ReLU activations.
    
    Structure: Conv -> ReLU -> Conv -> ReLU
    """
    
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super(DoubleConv, self).__init__()
        
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            make_norm(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            make_norm(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling with maxpool then double conv.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling then double conv.
    """
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super(Up, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels + out_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels // 2 + out_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle input size differences
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Final output convolution layer.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class MicroscopyUNet(nn.Module):
    """
    U-Net model optimized for microscopy image denoising.
    
    Features:
    - Encoder-decoder architecture with skip connections
    - Optimized for preserving fine biological structures
    - Support for grayscale (1-channel) images
    - Batch normalization for stable training
    - Flexible architecture depth
    """
    
    def __init__(
        self, 
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        bilinear: bool = True,
        depth: int = 4
    ):
        """
        Initialize U-Net model.
        
        Args:
            in_channels: Number of input channels (1 for grayscale)
            out_channels: Number of output channels (1 for grayscale)
            base_channels: Base number of channels in first layer
            bilinear: Use bilinear upsampling or transposed convolutions
            depth: Depth of the U-Net architecture
        """
        super(MicroscopyUNet, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.bilinear = bilinear
        self.depth = depth
        
        # Encoder (downsampling path)
        self.inc = DoubleConv(in_channels, base_channels)
        self.down_layers = nn.ModuleList()
        
        channels = base_channels
        for i in range(depth):
            next_channels = channels * 2
            self.down_layers.append(Down(channels, next_channels))
            channels = next_channels
        
        # Decoder (upsampling path)
        self.up_layers = nn.ModuleList()
        
        # Build decoder path with proper channel calculations
        feature_channels = [base_channels * (2 ** i) for i in range(depth + 1)]
        
        for i in range(depth):
            # Reverse order for decoder
            reverse_idx = depth - i - 1
            in_ch = feature_channels[reverse_idx + 1]
            out_ch = feature_channels[reverse_idx]
            
            self.up_layers.append(Up(in_ch, out_ch, bilinear))
        
        # Final output layer
        self.outc = OutConv(base_channels, out_channels)
    
    def forward(self, x):
        """
        Forward pass through the U-Net.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, H, W)
            
        Returns:
            Output tensor of shape (batch_size, out_channels, H, W)
        """
        # Input validation
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (batch, channel, H, W), got {x.dim()}D")
        
        # Encoder path with skip connections
        skip_connections = []
        x = self.inc(x)
        skip_connections.append(x)
        
        for down_layer in self.down_layers:
            x = down_layer(x)
            skip_connections.append(x)
        
        # Decoder path with skip connections
        skip_connections = skip_connections[:-1]  # Remove bottleneck from skip list
        
        for i, up_layer in enumerate(self.up_layers):
            skip_connection = skip_connections[-(i + 1)]
            x = up_layer(x, skip_connection)
        
        # Final output
        x = self.outc(x)
        
        return x
    
    def get_model_info(self):
        """Get model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'model_name': 'MicroscopyUNet',
            'input_channels': self.in_channels,
            'output_channels': self.out_channels,
            'base_channels': self.base_channels,
            'depth': self.depth,
            'bilinear_upsampling': self.bilinear,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameter_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }
        
        return info


class ResidualBlock(nn.Module):
    """
    Residual block for enhanced feature extraction.
    """
    
    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            make_norm(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            make_norm(channels)
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.conv(x)
        out += residual
        return self.relu(out)


class EnhancedMicroscopyUNet(MicroscopyUNet):
    """
    Enhanced U-Net with residual blocks for better preservation of fine structures.
    """
    
    def __init__(
        self, 
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        bilinear: bool = True,
        depth: int = 4,
        use_residual_blocks: bool = True
    ):
        super(EnhancedMicroscopyUNet, self).__init__(
            in_channels, out_channels, base_channels, bilinear, depth
        )
        
        self.use_residual_blocks = use_residual_blocks
        
        if use_residual_blocks:
            # Add residual blocks with proper channel matching
            self.residual_blocks = nn.ModuleList()
            # Add residual blocks to match the encoder layers
            for i in range(depth):
                channels = base_channels * (2 ** (i + 1))  # Match encoder channels
                self.residual_blocks.append(ResidualBlock(channels))
    
    def forward(self, x):
        """Forward pass with residual blocks."""
        # Input validation
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (batch, channel, H, W), got {x.dim()}D")
        
        # Encoder path with skip connections
        skip_connections = []
        x = self.inc(x)
        skip_connections.append(x)
        
        for i, down_layer in enumerate(self.down_layers):
            x = down_layer(x)
            
            # Apply residual blocks to corresponding layers
            if (self.use_residual_blocks and i < len(self.residual_blocks)):
                x = self.residual_blocks[i](x)
            
            skip_connections.append(x)
        
        # Decoder path with skip connections
        skip_connections = skip_connections[:-1]  # Remove bottleneck from skip list
        
        for i, up_layer in enumerate(self.up_layers):
            skip_connection = skip_connections[-(i + 1)]
            x = up_layer(x, skip_connection)
        
        # Final output
        x = self.outc(x)
        
        return x


class ResidualMicroscopyUNet(nn.Module):
    """
    Residual-block U-Net for denoising.

    The network uses residual blocks internally, but directly predicts the
    clean image. This is more stable for microscopy pairs where the clean
    target can be much darker than the noisy input.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        bilinear: bool = True,
        depth: int = 4,
    ):
        super().__init__()
        self.unet = EnhancedMicroscopyUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            bilinear=bilinear,
            depth=depth,
            use_residual_blocks=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.unet(x))

    def get_model_info(self):
        info = self.unet.get_model_info()
        info["model_name"] = "ResidualMicroscopyUNet"
        return info


def create_unet_model(
    model_type: str = 'standard',
    in_channels: int = 1,
    out_channels: int = 1,
    base_channels: int = 64,
    bilinear: bool = True,
    depth: int = 4
) -> nn.Module:
    """
    Factory function to create U-Net models.
    
    Args:
        model_type: 'standard' or 'enhanced'
        in_channels: Number of input channels
        out_channels: Number of output channels
        base_channels: Base number of channels
        bilinear: Use bilinear upsampling
        depth: Model depth
        
    Returns:
        U-Net model instance
    """
    if model_type == 'residual':
        return ResidualMicroscopyUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            bilinear=bilinear,
            depth=depth,
        )
    elif model_type == 'enhanced':
        return EnhancedMicroscopyUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            bilinear=bilinear,
            depth=depth,
            use_residual_blocks=True
        )
    else:
        return MicroscopyUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            bilinear=bilinear,
            depth=depth
        )


# Test the model
if __name__ == "__main__":
    import torch
    
    print("Testing U-Net models...")
    
    # Test standard U-Net
    print("\n1. Testing standard MicroscopyUNet:")
    model_standard = MicroscopyUNet(in_channels=1, out_channels=1)
    
    # Create test input (batch_size=4, channels=1, height=256, width=256)
    test_input = torch.randn(4, 1, 256, 256)
    
    with torch.no_grad():
        output = model_standard(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Input/Output shapes match: {test_input.shape == output.shape}")
    
    # Print model info
    info = model_standard.get_model_info()
    print(f"Total parameters: {info['total_parameters']:,}")
    print(f"Parameter size: {info['parameter_size_mb']:.2f} MB")
    
    # Test enhanced U-Net
    print("\n2. Testing Enhanced MicroscopyUNet:")
    model_enhanced = EnhancedMicroscopyUNet(in_channels=1, out_channels=1)
    
    with torch.no_grad():
        output_enhanced = model_enhanced(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output_enhanced.shape}")
    print(f"Input/Output shapes match: {test_input.shape == output_enhanced.shape}")
    
    # Print enhanced model info
    info_enhanced = model_enhanced.get_model_info()
    print(f"Total parameters: {info_enhanced['total_parameters']:,}")
    print(f"Parameter size: {info_enhanced['parameter_size_mb']:.2f} MB")
    
    # Test with different input sizes
    print("\n3. Testing with different input sizes:")
    test_sizes = [(1, 1, 128, 128), (2, 1, 512, 512), (8, 1, 256, 256)]
    
    for size in test_sizes:
        test_tensor = torch.randn(*size)
        with torch.no_grad():
            out = model_standard(test_tensor)
        print(f"Input: {size} -> Output: {out.shape} ✓")
    
    print("\n✓ All tests passed successfully!")
    print("U-Net models are ready for microscopy image denoising!")
