# Installation Guide

**System Requirements, Setup Instructions, and Troubleshooting**

---

## Overview

This guide provides comprehensive installation instructions for FluoClean AI across different operating systems and use cases. The installation process is designed to be straightforward while accommodating various hardware configurations and deployment scenarios.

### Installation Scenarios
- **Local Development**: Full setup with training capabilities
- **Production Deployment**: Minimal setup for inference only
- **Docker Deployment**: Containerized installation
- **Cloud Deployment**: Cloud platform deployment

---

## System Requirements

### Minimum Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Operating System** | Ubuntu 18.04+, macOS 10.15+, Windows 10+ | Latest OS version |
| **Python** | 3.11+ | 3.11 or 3.12 |
| **RAM** | 8GB | 16GB+ |
| **Storage** | 5GB | 10GB+ |
| **GPU** | None (CPU inference) | NVIDIA GPU with 8GB+ VRAM |

### GPU Requirements (Optional)

| GPU Model | VRAM | Training Speed | Inference Speed |
|----------|------|---------------|----------------|
| **NVIDIA GTX 1660** | 6GB | 2-3× faster | 3-4× faster |
| **NVIDIA RTX 3060** | 12GB | 4-5× faster | 5-6× faster |
| **NVIDIA RTX 3080** | 10GB | 6-8× faster | 8-10× faster |
| **NVIDIA V100** | 16GB | 10-15× faster | 15-20× faster |

### Software Dependencies

#### Core Dependencies
- **Python 3.11+**: Interpreter and standard library
- **pip**: Python package manager
- **git**: Version control system

#### Optional Dependencies
- **CUDA Toolkit 11.8+**: For GPU acceleration
- **cuDNN 8.6+**: GPU-accelerated deep learning
- **Docker**: Containerized deployment

---

## Installation Methods

### Method 1: Standard Installation (Recommended)

#### Step 1: Clone Repository

```bash
git clone https://github.com/Sam-wan30/AI-Image-Denoising-In-Microscopy.git
cd AI-Image-Denoising-In-Microscopy
```

#### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate on Linux/macOS
source .venv/bin/activate

# Activate on Windows
.venv\Scripts\activate
```

#### Step 3: Upgrade pip

```bash
pip install --upgrade pip
```

#### Step 4: Install Core Dependencies

```bash
# Core dependencies (web app + inference)
pip install -r requirements.txt
```

**Core Dependencies Include:**
- numpy>=1.24.0,<2.0.0
- opencv-python-headless>=4.8.1,<5.0.0
- pillow>=10.0.0,<12.0.0
- onnxruntime>=1.15.0,<2.0.0
- Flask>=2.2.0

#### Step 5: Install Training Dependencies (Optional)

```bash
# Extended dependencies (training + development)
pip install -r requirements_torch.txt
```

**Training Dependencies Include:**
- torch==2.2.2
- torchvision==0.17.2
- Additional ML libraries (albumentations, h5py, pandas, matplotlib, tqdm, streamlit)

#### Step 6: Verify Installation

```bash
# Test Python imports
python -c "import torch; import flask; import cv2; print('✓ Dependencies installed successfully')"
```

---

### Method 2: GPU Installation (CUDA)

#### Prerequisites
- NVIDIA GPU with compute capability 3.5+
- CUDA Toolkit 11.8 or later
- cuDNN 8.6 or later

#### Step 1: Install CUDA Toolkit

```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# Verify CUDA installation
nvcc --version
```

#### Step 2: Install cuDNN

```bash
# Download cuDNN from NVIDIA website
# Extract and copy to CUDA directory
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
```

#### Step 3: Install PyTorch with CUDA

```bash
# Install PyTorch with CUDA 11.8 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### Step 4: Verify GPU Installation

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

---

### Method 3: Docker Installation

#### Step 1: Create Dockerfile

```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Run application
CMD ["python", "application.py"]
```

#### Step 2: Build Docker Image

```bash
docker build -t neuroscope:latest .
```

#### Step 3: Run Container

```bash
# Basic run
docker run -p 5000:5000 neuroscope:latest

# With volume mount for models
docker run -p 5000:5000 \
  -v $(pwd)/models:/app/models \
  neuroscope:latest

# With GPU support
docker run --gpus all -p 5000:5000 neuroscope:latest
```

---

## Environment Configuration

### Environment Variables

#### Required Variables

```bash
# Copy example configuration
cp .env.example .env

# Edit .env file
nano .env
```

**Key Environment Variables:**

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `MODEL_PATH` | Path to model file | `models/deploy/model.pt` | Yes |
| `SECRET_KEY` | Flask secret key | `dev-change-me-in-production` | Yes (production) |
| `PORT` | Server port | `5000` | No |
| `FLASK_DEBUG` | Debug mode | `0` | No |
| `DEVICE` | Computation device | `cpu` | No |
| `MAX_UPLOAD_MB` | Max upload size | `50` | No |

#### Configuration File (.env.example)

```bash
# Flask Configuration
SECRET_KEY=change-me-local-dev
FLASK_DEBUG=1
PORT=5000

# Model Configuration
MODEL_PATH=models/deploy/model.pt

# Inference Configuration
DEVICE=cpu
MAX_UPLOAD_MB=50

# Optional: Directory configurations
UPLOAD_DIR=uploads
OUTPUT_DIR=outputs
```

---

## Model Setup

### Download Pre-trained Models

#### Option 1: Use Provided Models
```bash
# Ensure models directory exists
mkdir -p models/deploy

# Place model.pt in models/deploy/
# (Obtain from project releases or train your own)
```

#### Option 2: Export from Training Checkpoint

```bash
python scripts/export_inference_checkpoint.py \
  --input models/overfit_residual_blocks/best_model.pth \
  --output models/deploy/model.pt
```

#### Option 3: Train New Model

```bash
python train.py \
  --data_dir data \
  --epochs 50 \
  --batch_size 8 \
  --save_dir models
```

---

## Verification

### Installation Verification

#### Test 1: Python Imports

```bash
python -c "
import torch
import flask
import cv2
import numpy as np
import onnxruntime as ort
print('✓ All core dependencies imported successfully')
"
```

#### Test 2: Flask Application

```bash
python application.py
# Expected output:
# * Running on all addresses (0.0.0.0)
# * Running on http://127.0.0.1:5000
```

#### Test 3: Streamlit UI

```bash
streamlit run app.py
# Expected output:
# You can now view your Streamlit app in your browser.
# Local URL: http://localhost:8501
```

#### Test 4: CLI Inference

```bash
python inference.py --help
# Expected: Help message with all available options
```

### Functionality Verification

#### Test Web Interface
1. Start Flask application: `python application.py`
2. Open browser: `http://localhost:5000`
3. Verify:
   - Page loads correctly
   - Model status shows "ready"
   - Upload interface is functional

#### Test Inference
```bash
# Test with a sample image
python inference.py \
  --model models/deploy/model.pt \
  --input test_image.png \
  --output test_output/
```

---

## Troubleshooting

### Common Installation Issues

#### Issue 1: Python Version Incompatibility

**Symptoms**: Import errors, version conflicts

**Solution**:
```bash
# Check Python version
python --version

# Install correct Python version
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3.11 python3.11-venv

# macOS (using Homebrew)
brew install python@3.11

# Windows: Download from python.org
```

#### Issue 2: pip Installation Fails

**Symptoms**: Permission errors, network issues

**Solution**:
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Use user directory if permissions issue
pip install --user -r requirements.txt

# Try with different index
pip install -r requirements.txt -i https://pypi.org/simple
```

#### Issue 3: GPU Not Detected

**Symptoms**: CUDA not available despite GPU installation

**Solution**:
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA installation
nvcc --version

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### Issue 4: OpenCV Import Error

**Symptoms**: `ImportError: libGL.so.1: cannot open shared object file`

**Solution**:
```bash
# Ubuntu/Debian
sudo apt-get install libgl1-mesa-glx libglib2.0-0

# For headless environments
pip install opencv-python-headless
```

#### Issue 5: Port Already in Use

**Symptoms**: `Address already in use` error

**Solution**:
```bash
# Find process using port 5000
lsof -i :5000  # Linux/macOS
netstat -ano | findstr :5000  # Windows

# Kill process or use different port
PORT=5001 python application.py
```

#### Issue 6: Memory Errors During Training

**Symptoms**: CUDA out of memory, system out of memory

**Solution**:
```bash
# Reduce batch size
python train.py --batch_size 4 --data_dir data

# Use CPU instead
python train.py --device cpu --data_dir data

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

### Platform-Specific Issues

#### Linux Issues

**Issue**: Missing system libraries

**Solution**:
```bash
sudo apt-get update
sudo apt-get install -y \
    python3-dev \
    python3-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1
```

#### macOS Issues

**Issue**: SSL certificate errors

**Solution**:
```bash
# Install certificates
/Applications/Python\ 3.11/Install\ Certificates.command

# Or manually
pip install --upgrade certifi
```

#### Windows Issues

**Issue**: Path length limitations

**Solution**:
```bash
# Enable long path support (requires admin)
# Windows Registry: Computer\HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem
# Set LongPathsEnabled to 1

# Or use shorter path
cd C:\short\path\to\project
```

---

## Uninstallation

### Clean Uninstallation

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf .venv

# Remove repository
cd ..
rm -rf AI-Image-Denoising-In-Microscopy
```

### Partial Cleanup

```bash
# Keep code but remove virtual environment
rm -rf .venv

# Remove only cached models
rm -rf models/

# Remove generated outputs
rm -rf uploads/ outputs/ logs/
```

---

## Advanced Configuration

### Custom Installation Paths

```bash
# Install to custom directory
PYTHONUSERBASE=/custom/path pip install --user -r requirements.txt

# Set custom model path
export MODEL_PATH=/custom/path/to/model.pt
```

### Development Installation

```bash
# Install in editable mode for development
pip install -e .

# Install development dependencies
pip install pytest black flake8 mypy
```

### Production Installation

```bash
# Use production-grade packages only
pip install -r requirements.txt --no-cache-dir

# Install gunicorn for production
pip install gunicorn

# Set production environment
export FLASK_ENV=production
export FLASK_DEBUG=0
```

---

## Update Procedure

### Updating Dependencies

```bash
# Update requirements
pip install --upgrade -r requirements.txt

# Update specific package
pip install --upgrade torch

# Check for outdated packages
pip list --outdated
```

### Updating Project Code

```bash
# Pull latest changes
git pull origin main

# Reinstall if dependencies changed
pip install -r requirements.txt --force-reinstall
```

---

## Performance Optimization

### CUDA Optimization

```bash
# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=0

# Enable cuDNN benchmark
export CUDNN_BENCHMARK=1

# Set memory fraction
export CUDA_VISIBLE_DEVICES=0
export CUDA_MEMORY_FRACTION=0.9
```

### CPU Optimization

```bash
# Set number of CPU threads
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Enable memory mapping
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
```

---

## Installation Checklist

### Pre-Installation
- [ ] Verify system requirements
- [ ] Check Python version (3.11+)
- [ ] Ensure sufficient disk space (5GB+)
- [ ] Verify network connectivity

### Installation
- [ ] Clone repository
- [ ] Create virtual environment
- [ ] Install core dependencies
- [ ] Install training dependencies (if needed)
- [ ] Configure environment variables
- [ ] Download or train model

### Post-Installation
- [ ] Verify Python imports
- [ ] Test Flask application
- [ ] Test Streamlit UI
- [ ] Verify model loading
- [ ] Test inference pipeline
- [ ] Check GPU availability (if applicable)

---

<div align="center">

**Proper installation ensures reliable performance and ease of use**

[⬆ Back to Wiki Home](Home) | [← Model Documentation](Model-Documentation) | [User Guide](User-Guide) →

</div>
