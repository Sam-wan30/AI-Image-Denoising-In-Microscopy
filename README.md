<div align="center">

<img src="docs/assets/fluoclean-horizontal-logo.png" alt="FluoClean AI horizontal logo" width="680" />

# FluoClean AI
### AI-Powered Microscopy Image Denoising System

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Live Demo](https://img.shields.io/badge/Live_Demo-Render-46E3B7?style=flat-square&logo=render&logoColor=white)](https://ai-image-denoising-in-microscopy-m2u0.onrender.com/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)

**Advanced deep learning pipeline for restoring microscopy images using state-of-the-art U-Net architectures with CARE-inspired training methodology.**

[🌐 Live Demo](https://ai-image-denoising-in-microscopy-m2u0.onrender.com/) · [🚀 Features](#-features) · [🏗️ Architecture](#-system-architecture) · [📚 Documentation](#-documentation) · [🔧 Installation](#-installation--setup) · [💡 Usage](#-usage-guide)

</div>

---

## 📋 Overview

FluoClean AI is a research prototype for microscopy image denoising. Microscopy images often suffer from photon shot noise, sensor noise, and artifacts that obscure fine cellular structures. The project implements an end-to-end deep learning pipeline inspired by CARE (Content-Aware Image Restoration), with a deployable Flask demonstration. It is not validated for diagnostic or quantitative scientific use.

### 🌐 Live Deployment

Try the deployed Flask application on Render:

**[Launch FluoClean AI →](https://ai-image-denoising-in-microscopy-m2u0.onrender.com/)**

The live demo supports microscopy image upload, multiple denoising modes, side-by-side comparison, input-similarity indicators, and denoised-image download.

### 🎯 Problem Statement

Microscopy imaging is fundamental to biological research, medical diagnostics, and materials science. However, acquired images frequently contain significant noise that:
- Obscures critical cellular structures and organelles
- Reduces the accuracy of quantitative measurements
- Limits the effectiveness of automated image analysis
- Requires longer exposure times that can damage samples
- Increases computational burden for downstream processing

### 💡 Why This Project Matters

FluoClean AI demonstrates the practical application of deep learning to solve real-world scientific problems:
- **Preserves Biological Fidelity**: Maintains fine structural details while removing noise
- **Restoration Metrics**: Reports PSNR and SSIM when aligned clean targets exist
- **Multiple Architectures**: Implements Standard, Enhanced, and Residual U-Net variants for optimal performance
- **Deployment Demo**: Includes a web application, CLI tools, and inference checks
- **Research-Backed**: Based on peer-reviewed CARE methodology with custom improvements

### 🎯 Key Objectives

- Implement robust U-Net architectures optimized for grayscale microscopy images
- Develop combined L1 + SSIM loss function for balanced reconstruction
- Create a deployable inference pipeline with multiple interface options
- Provide comprehensive training tools with validation and early stopping
- Deliver intuitive web interface for researchers and clinicians
- Ensure reproducibility through structured configuration and documentation

---

## 🚀 Features

### 🧠 Deep Learning Models
- **Multiple U-Net Architectures**: Standard, Enhanced (with residual blocks), and Residual variants
- **Advanced Loss Function**: Combined L1 + SSIM loss for balanced structural preservation
- **Flexible Training**: Support for CARE-style paired datasets with optional augmentation
- **Model Optimization**: GroupNorm for small-batch stability, configurable depth and channels
- **Checkpoint Management**: Automatic best-model saving based on validation PSNR

### 🌐 Web Applications
- **Production Flask API**: RESTful endpoints for image upload, denoising, and download
- **Modern Streamlit UI**: Interactive interface for local development and prototyping
- **Responsive Design**: Mobile-friendly interface with real-time status monitoring
- **Multiple Denoising Modes**: U-Net, auto-routing, salt-and-pepper filter, brightfield mask
- **Real-time Metrics**: PSNR and SSIM calculation with visual feedback
- **One-Click Download**: Automatic timestamped download of denoised images with error handling

### 🔧 Training Pipeline
- **CARE Dataset Support**: Paired noisy/clean image loading with automatic matching
- **Data Augmentation**: Random flips and rotations for improved generalization
- **Advanced Training Features**: Learning rate scheduling, early stopping, TensorBoard integration
- **Validation Tracking**: Real-time PSNR and SSIM monitoring during training
- **Sample Visualization**: Automatic generation of training progression comparisons

### 📊 Inference Capabilities
- **CLI Tool**: Batch processing for single images or entire directories
- **Web API**: RESTful endpoints with automatic model warm-up
- **Multiple Output Formats**: Comparison images, individual results, metrics reports
- **Flexible Model Loading**: Support for PyTorch checkpoints and ONNX runtime
- **Thread-Safe Service**: Lazy-loaded model inference with concurrent request handling
- **Smart Download System**: Automatic timestamped filename generation and dual-format support (URL/Base64)

### 🛠️ Technical Features
- **ONNX Runtime Support**: Optional deployment-optimized inference
- **Environment Configuration**: Flexible environment-based settings
- **Error Handling**: Comprehensive exception handling and logging
- **Memory Optimization**: Efficient image processing with configurable batch sizes
- **Cross-Platform**: Compatible with Linux, macOS, and Windows

---

## 📸 Screenshots

### Web Application Interface
*Upload interface with drag-and-drop support and real-time model status*

### Denoising Results
*Side-by-side comparison showing original noisy image and AI-restored output*

### Training Progress
*TensorBoard-style training curves showing loss, PSNR, and SSIM improvement over epochs*

---

## 🏗️ System Architecture

### High-Level Architecture

```mermaid
graph TB
    subgraph "Frontend Layer"
        A[Flask Web App]
        B[Streamlit UI]
        C[CLI Tool]
    end
    
    subgraph "API Layer"
        D[REST API Endpoints]
        E[Image Processing]
        F[Authentication]
    end
    
    subgraph "Service Layer"
        G[Denoiser Service]
        H[Model Manager]
        I[Metrics Calculator]
    end
    
    subgraph "ML Layer"
        J[U-Net Models]
        K[Training Pipeline]
        L[Inference Engine]
    end
    
    subgraph "Data Layer"
        M[Image Storage]
        N[Model Checkpoints]
        O[Training Dataset]
    end
    
    A --> D
    B --> D
    C --> E
    D --> G
    E --> G
    F --> G
    G --> H
    H --> J
    G --> I
    J --> L
    K --> J
    L --> M
    H --> N
    K --> O
    
    style A fill:#4CAF50
    style B fill:#2196F3
    style C fill:#FF9800
    style J fill:#9C27B0
```

### Data Flow Architecture

```mermaid
sequenceDiagram
    participant User
    participant WebApp
    participant API
    participant Denoiser
    participant Model
    
    User->>WebApp: Upload Image
    WebApp->>API: POST /api/denoise
    API->>Denoiser: Process Image
    Denoiser->>Model: Load Model
    Model-->>Denoiser: Inference Result
    Denoiser->>Denoiser: Calculate Metrics
    Denoiser-->>API: Denoised Image + Metrics
    API-->>WebApp: JSON Response
    WebApp-->>User: Display Results
```

### Component Interaction

```mermaid
graph LR
    subgraph "Input Processing"
        A[Raw Image] --> B[Preprocessing]
        B --> C[Grayscale Conversion]
        C --> D[Resize & Normalize]
    end
    
    subgraph "Model Inference"
        D --> E[U-Net Encoder]
        E --> F[Bottleneck]
        F --> G[U-Net Decoder]
        G --> H[Skip Connections]
    end
    
    subgraph "Output Processing"
        H --> I[Postprocessing]
        I --> J[Resize to Original]
        J --> K[Denormalize]
        K --> L[Quality Metrics]
    end
    
    L --> M[Final Output]
    
    style B fill:#E3F2FD
    style E fill:#F3E5F5
    style I fill:#E8F5E9
```

---

## 🔧 Tech Stack

### Backend Framework
| Technology | Purpose |
|------------|---------|
| **Python 3.11+** | Core language with modern type hints and async support |
| **Flask 3.x** | Lightweight web framework for REST API |
| **PyTorch 2.2+** | Deep learning framework for model training and inference |
| **ONNX Runtime 1.15+** | Optional deployment-optimized inference engine |

### Machine Learning
| Component | Implementation |
|-----------|----------------|
| **Model Architecture** | Custom U-Net variants (Standard, Enhanced, Residual) |
| **Loss Function** | Combined L1 + SSIM (0.7:0.3 ratio) |
| **Optimizer** | Adam with learning rate scheduling |
| **Normalization** | GroupNorm for small-batch stability |
| **Activation** | ReLU with inplace operations |

### Image Processing
| Library | Usage |
|---------|-------|
| **OpenCV** | Image I/O, preprocessing, augmentation |
| **Pillow** | Image format handling and conversion |
| **NumPy** | Numerical operations and array manipulation |
| **Albumentations** | Advanced data augmentation (training mode) |

### Frontend Technologies
| Technology | Purpose |
|------------|---------|
| **HTML5/CSS3** | Responsive web interface with modern styling |
| **JavaScript (ES6+)** | Client-side interactions and API communication |
| **Streamlit** | Alternative Python-based UI for rapid prototyping |
| **CSS Grid/Flexbox** | Modern layout system for responsive design |

### Development Tools
| Tool | Purpose |
|------|---------|
| **TensorBoard** | Training visualization and metrics tracking |
| **tqdm** | Progress bars for training and inference |
| **matplotlib** | Training curve generation and visualization |
| **pytest** | Unit testing framework |

### File Formats Support
| Format | Support Level |
|--------|---------------|
| **PNG** | Full support (recommended) |
| **JPEG/JPG** | Full support |
| **TIFF/TIF** | Full support |
| **WebP** | Full support |
| **BMP** | Full support |

---

## 📁 Project Structure

```
AI Image Denoising In Microscopy/
├── application.py              # Flask production web application
├── app.py                      # Streamlit UI prototype
├── config.py                   # Environment-based configuration management
├── train.py                    # Model training script with advanced features
├── inference.py                # CLI inference tool for batch processing
├── requirements.txt            # Core dependencies (Flask, ONNX, etc.)
├── requirements_torch.txt     # Extended dependencies for training
│
├── src/                        # Core deep learning modules
│   ├── unet_model.py          # U-Net architecture implementations
│   ├── care_dataset.py       # Original CARE dataset loader
│   └── care_dataset_simple.py # Simplified dataset implementation
│
├── services/                   # Application service layer
│   ├── denoiser.py           # Thread-safe inference service
│   ├── bootstrap.py          # Startup helpers and directory management
│   └── model_utils.py        # Model type detection and utilities
│
├── utils/                      # Shared utility modules
│   ├── preprocessing.py      # Image preprocessing pipeline
│   ├── metrics.py            # PSNR, SSIM, and quality calculations
│   ├── losses.py             # Custom loss functions
│   ├── salt_pepper.py        # Salt-and-pepper noise removal
│   └── brightfield.py        # Brightfield object mask processing
│
├── templates/                  # Flask HTML templates
│   └── index.html            # Main web application interface
│
├── static/                     # Static web assets
│   ├── css/                   # Stylesheets
│   └── js/                    # Client-side JavaScript
│
├── ui/                         # Streamlit UI components
│   ├── components.py          # Reusable UI elements
│   └── run_layout.py          # Layout configuration
│
├── scripts/                    # Utility scripts
│   ├── export_inference_checkpoint.py  # Model optimization for deployment
│   └── process_microscopy_dataset.py  # Dataset preprocessing
│
├── models/                     # Trained model checkpoints (git-ignored)
│   ├── deploy/                # Deployment artifacts
│   └── overfit_residual_blocks/  # Experimental models
│
├── data/                       # Training datasets (git-ignored)
│   └── train/
│       ├── noisy/             # Noisy microscopy images
│       └── clean/             # Ground truth clean images
│
├── uploads/                    # Runtime upload directory (auto-created)
├── outputs/                    # Denoised output directory (auto-created)
├── logs/                       # TensorBoard logs and training artifacts
└── tests/                      # Unit and integration tests
```

---

## 🔧 Installation & Setup

### Prerequisites

- **Python 3.11+** with pip package manager
- **Virtual Environment** (recommended but not required)
- **CUDA** (optional, for GPU-accelerated training)
- **8GB+ RAM** recommended for training
- **2GB+ disk space** for dependencies and models

### Step 1: Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-Image-Denoising-In-Microscopy.git
cd AI-Image-Denoising-In-Microscopy
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Core dependencies (web app and inference)
pip install -r requirements.txt

# Extended dependencies (training and development)
pip install -r requirements_torch.txt
```

### Step 4: Environment Configuration

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your configuration
# Required variables:
# - MODEL_PATH=models/deploy/model.onnx
# - SECRET_KEY=your-secret-key-here
# - PORT=5000
```

### Step 5: Prepare Model Weights

```bash
# Option 1: Use the bundled validated ONNX model
# models/deploy/model.onnx is loaded by default

# Option 2: Export a checkpoint with held-out test provenance
python scripts/export_inference_checkpoint.py \
  --input models/retrained_compact/best_model.pth \
  --output models/deploy/model.pt

# Export and numerically verify ONNX parity
python scripts/export_to_onnx.py \
  --input models/retrained_compact/best_model.pth \
  --output models/deploy/model.onnx \
  --image-size 128
```

### Step 6: Verify Installation

```bash
# Test Flask application
python application.py
# Visit http://localhost:5000

# Test Streamlit interface
streamlit run app.py
# Visit http://localhost:8501
```

---

## 💡 Usage Guide

### Web Application (Recommended)

#### Starting the Flask Server

```bash
python application.py
# Server runs on http://localhost:5000
```

#### Using the Interface

1. **Upload Image**: Drag and drop or click to select a microscopy image
2. **Select Mode**: Choose denoising algorithm (Auto, U-Net, Salt-Pepper, Brightfield)
3. **Process**: Click "Start denoising" to begin processing
4. **View Results**: Compare side-by-side with input-similarity indicators. These are not quality scores because uploaded images have no clean reference.
5. **Download**: Click "⬇ Download denoised image" to save the AI-restored image with automatic timestamped filename (format: `denoised-image-YYYYMMDD-HHMMSS.png`)

#### API Endpoints

<details>
<summary>📖 API Documentation</summary>

### Health Check
```http
GET /health
```
**Response**: Service status and model readiness

### Model Status
```http
GET /api/status
```
**Response**: Detailed model information and loading state

### Denoise Image
```http
POST /api/denoise
Content-Type: multipart/form-data
```
**Parameters**:
- `image`: Image file (required)
- `mode`: Denoising mode (optional, default: "auto")

**Response**: JSON with base64-encoded images and metrics

### Download Result
```http
GET /api/download/<filename>
```
**Response**: Denoised image file
</details>

### Command Line Interface

#### Single Image Processing

```bash
python inference.py \
  --model models/deploy/model.onnx \
  --input path/to/noisy_image.png \
  --output results/
```

#### Batch Directory Processing

```bash
python inference.py \
  --model models/deploy/model.onnx \
  --input_dir path/to/noisy_images/ \
  --output_dir results/ \
  --batch
```

#### Advanced Options

```bash
python inference.py \
  --model models/deploy/model.onnx \
  --input image.png \
  --output results/ \
  --ground_truth clean_image.png \
  --save_comparison \
  --metrics_only \
  --device cuda
```

### Training Pipeline

#### Basic Training

```bash
python train.py \
  --data_dir data \
  --epochs 50 \
  --batch_size 4 \
  --lr 0.0001 \
  --seed 42 \
  --val_split 0.20 \
  --test_split 0.15 \
  --save_dir models
```

#### Advanced Training Configuration

```bash
python train.py \
  --data_dir data \
  --epochs 100 \
  --batch_size 4 \
  --lr 0.0001 \
  --val_split 0.2 \
  --test_split 0.15 \
  --early_stop 15 \
  --seed 42 \
  --save_dir models \
  --log_dir logs
```

#### Validated Compact Training Configuration

```bash
python train.py \
  --data_dir data \
  --epochs 60 \
  --batch_size 8 \
  --image_size 128 \
  --base_channels 16 \
  --depth 3 \
  --save_dir models/retrained_compact
```

---

## Model Validation & Reliability

### Validation design

- **Task type:** paired image-to-image regression, not classification.
- **Leakage control:** splitting is performed by specimen/session rather than by
  individual frame. Byte-identical clean targets are checked across splits.
- **Reproducibility:** Python, NumPy, PyTorch, CUDA, split, and DataLoader seeds
  are controlled by `--seed` (default `42`).
- **Clean evaluation:** augmentation is enabled only for training. Validation
  and test datasets use deterministic preprocessing.
- **Selection and test:** validation PSNR selects the checkpoint; a separate
  held-out test split is evaluated once after selection.
- **Metrics:** PSNR, SSIM, MAE, and MSE are reported per image, by specimen
  group, and against the unprocessed noisy-input baseline.

Training writes `split_manifest.json`, `training_history.json`, and
`test_metrics.json`. Run one grouped cross-validation fold with:

```bash
python train.py --cv_folds 5 --cv_fold 0 --seed 42 --save_dir models/cv_fold_0
```

Repeat `--cv_fold` from `0` through `4`. Cross-validation is opt-in because
training five U-Nets is computationally expensive.

Evaluate a PyTorch or ONNX model with paired targets:

```bash
python evaluate_model.py \
  --model models/deploy/model.onnx \
  --data-dir data \
  --manifest models/split_manifest.json \
  --split test \
  --output-dir reports/model_validation
```

The report contains confidence intervals, per-group results, baseline deltas,
and worst-case panels arranged as noisy input, prediction, clean target, and
absolute error. These panels are the relevant explainability aid for image
restoration. Confusion matrices, classification reports, ROC-AUC, PR-AUC, and
class feature importance are not applicable because there are no class labels.

### Current deployment candidate

The bundled ONNX model is a 0.88-million-parameter residual U-Net trained for
60 epochs at 128x128 resolution with seed 42. The specimen-level split contains
55 training, 30 validation, and 20 test pairs. The test set contains two
acquisition groups that appear in neither training nor validation.

| Held-out measure | Model | Noisy-input baseline |
|---|---:|---:|
| Mean PSNR | 22.04 dB | 10.34 dB |
| Mean SSIM | 0.781 | 0.458 |
| Mean MAE | 0.094 | 0.324 |
| Images with PSNR improvement | 19 / 20 | - |
| Images with SSIM improvement | 20 / 20 | - |

ONNX output matches the source PyTorch model with maximum absolute error below
`4e-7`. One held-out PVD image lost 1.70 dB PSNR despite improving in SSIM, so
the model must not be assumed to improve every image. The full report is
generated by `evaluate_model.py`; checkpoint hashes and export parity are stored
next to `models/deploy/model.onnx`.

### Reliability and deployment changes

The previous deployment artifact was traced to a five-image overfit experiment,
not a complete dataset training run. Its retrospective 105-pair evaluation
improved mean PSNR on 84 images but degraded 21, with particularly poor results
on an unseen July PVD acquisition group. Quantization and 128x128 conversion
accounted for only a small quality difference; the checkpoint provenance and
domain overfitting were the primary causes of unreliable output.

The corrected pipeline now includes:

- deterministic specimen/session-level holdout and grouped cross-validation;
- duplicate clean-target checks to prevent cross-split leakage;
- synchronized augmentation on training pairs and no augmentation in validation
  or test datasets;
- a differentiable PyTorch SSIM loss that remains connected to autograd;
- per-image PSNR, SSIM, MAE, and MSE plus noisy-input baseline deltas;
- per-group summaries, confidence intervals, and worst-case error panels;
- checkpoint architecture, seed, split manifest, and held-out metrics;
- export gates that reject checkpoints without validation provenance;
- ONNX/PyTorch numerical parity verification before deployment;
- bounded image upload validation and thread-safe repeated inference.

The production artifact is a 3.4 MB, opset-17 ONNX model. It replaced the older
54 MB quantized model and 215 MB floating-point export, reducing Render memory
pressure without dynamic quantization. After deployment, the health endpoint
reported `ready: true` at 128x128 model resolution, and two consecutive live
dataset uploads returned HTTP 200. This directly verifies the repeated-request
failure mode that previously produced HTTP 502 responses.

### Remaining limitations

- Only 105 pairs from 14 specimen/session groups are available locally.
- Seventy clean files are duplicates across acquisition conditions, reducing
  the number of independent targets.
- The held-out test covers only two groups; grouped cross-validation has not yet
  been run for this checkpoint.
- Deployment uses 128x128 inference internally to fit Render's 512 MB tier,
  then resizes the result to the uploaded dimensions.
- PSNR and SSIM do not prove biological fidelity. External datasets and expert
  review are required before quantitative microscopy use.
- The grayscale model may fail on unseen modalities, structures, or noise.

The web UI reports output-to-input similarity only. True denoising quality can
be measured only when a registered clean target is available.

---

## 🔒 Security Features

### Input Validation
- **File Type Enforcement**: Strict whitelist of allowed image formats
- **Size Limits**: Configurable maximum file size (default: 50MB)
- **Content Validation**: Image format verification before processing
- **Path Sanitization**: Secure file path handling to prevent directory traversal

### Error Handling
- **Graceful Degradation**: Fallback mechanisms for model loading failures
- **Comprehensive Logging**: Detailed error tracking for debugging
- **User-Friendly Messages**: Clear error messages without exposing internals
- **Exception Isolation**: Request-level error handling to prevent system crashes

### Resource Management
- **Memory Limits**: Configurable memory constraints for large images
- **Timeout Protection**: Request timeout handling for long-running operations
- **Thread Safety**: Lock-based concurrent access control
- **Resource Cleanup**: Automatic cleanup of temporary files

### Environment Security
- **Secrets Management**: Environment-based configuration for sensitive data
- **No Hardcoded Credentials**: All secrets loaded from environment variables
- **Secure Defaults**: Safe default configurations for production use

---

## ⚡ Performance Optimizations

### Model Optimization
- **GroupNorm Implementation**: Batch-size independent normalization for stability
- **Inplace Operations**: Memory-efficient ReLU activations
- **Lazy Loading**: Model loaded only on first request
- **ONNX Runtime**: Optional deployment-optimized inference engine

### Training Optimizations
- **Learning Rate Scheduling**: ReduceLROnPlateau for adaptive learning rates
- **Early Stopping**: Automatic termination when validation metrics plateau
- **Deterministic Seeds**: Reproducible Python, NumPy, PyTorch, split, and loader randomness
- **Clean Evaluation**: Augmentation is disabled for validation and test data

### Inference Optimizations
- **Thread-Safe Service**: Concurrent request handling with proper locking
- **Model Caching**: Single model instance shared across requests
- **Image Preprocessing**: Optimized numpy-based pipeline
- **Low-Memory ONNX Session**: Sequential single-thread execution without arena or prepacking

### Memory Management
- **Compact Model**: 0.88M parameters and a 3.4 MB ONNX artifact
- **Bounded Uploads**: File-size and decoded-pixel limits reject unsafe inputs
- **128x128 Model Input**: Predictable memory use on Render's 512 MB free tier
- **Lazy Initialization**: Model memory is allocated only when inference is requested

---

## 🧩 Challenges & Solutions

### Challenge 1: Small Batch Training Stability
**Problem**: Traditional BatchNorm fails with small batch sizes common in microscopy due to memory constraints.

**Solution**: Implemented GroupNorm with dynamic group calculation based on channel count, ensuring stable training regardless of batch size.

### Challenge 2: Preserving Fine Structural Details
**Problem**: Standard L1 loss can blur fine cellular structures during denoising.

**Solution**: Combined L1 + SSIM loss function (0.7:0.3 ratio) to balance pixel-level accuracy with structural similarity preservation.

### Challenge 3: Memory Constraints During Inference
**Problem**: Large microscopy images can exceed available GPU memory.

**Solution**: Retrained a compact 128x128 residual U-Net and exported a 3.4 MB ONNX model using a low-memory sequential runtime configuration.

### Challenge 4: Dataset Pair Matching
**Problem**: CARE datasets require exact filename matching between noisy and clean images.

**Solution**: Automated filename-based matching with validation, reporting unmatched files, and flexible pattern support.

### Challenge 5: Production Deployment Complexity
**Problem**: PyTorch models have large memory footprint and deployment overhead.

**Solution**: Implemented ONNX runtime support for deployment-optimized inference, reducing memory requirements and improving startup time.

---

## 🚀 Future Enhancements

### Short-term Roadmap
- [ ] **Model Ensemble**: Combine multiple U-Net variants for improved performance
- [ ] **3D Image Support**: Extend architecture for volumetric microscopy data
- [ ] **Advanced Augmentation**: Add more sophisticated data augmentation techniques
- [ ] **Docker Support**: Containerized deployment for reproducible environments
- [ ] **API Authentication**: Add secure authentication for production deployments

### Medium-term Goals
- [ ] **Real-time Processing**: Optimize for near real-time inference on moderate hardware
- [ ] **Mobile Support**: Create mobile-friendly interface and lightweight models
- [ ] **Cloud Integration**: Add support for cloud storage and distributed processing
- [ ] **Advanced Metrics**: Implement additional quality assessment metrics
- [ ] **Transfer Learning**: Support for pre-trained models on public datasets

### Long-term Vision
- [ ] **Multi-modal Support**: Handle various microscopy modalities (confocal, SEM, TEM)
- [ ] **Self-supervised Learning**: Implement training without paired clean images
- [ ] **Interactive Refinement**: Allow user-guided denoising parameter adjustment
- [ ] **Publication-Ready Outputs**: Generate publication-quality figures and reports
- [ ] **Community Models**: Platform for sharing and comparing denoising models

---

## 🎓 Learning Outcomes

### Technical Skills Demonstrated
- **Deep Learning Architecture Design**: Custom U-Net implementations with advanced features
- **Production Software Engineering**: REST API design, error handling, and logging
- **Computer Vision**: Image preprocessing, augmentation, and quality assessment
- **Training Pipeline Development**: Loss function design, optimization, and validation
- **Performance Optimization**: Memory management, concurrent processing, and deployment

### Engineering Practices
- **Modular Architecture**: Clean separation of concerns across multiple modules
- **Configuration Management**: Environment-based configuration for flexibility
- **Testing Strategy**: Unit testing and integration testing approaches
- **Documentation**: Comprehensive code documentation and user guides
- **Version Control**: Git workflow and branch management

### Research Implementation
- **Paper Reproduction**: Implementation of CARE methodology with custom improvements
- **Experimentation**: Systematic hyperparameter tuning and architecture comparison
- **Metrics Analysis**: Understanding and implementing image quality metrics
- **Result Validation**: Quantitative and qualitative assessment of model performance

---

## 🌟 Why This Project Stands Out

### Engineering Excellence
- **Deployable Architecture**: Research code with Flask, Streamlit, and CLI interfaces
- **Multiple Deployment Options**: Flask web app, Streamlit UI, and CLI tools for different use cases
- **Comprehensive Error Handling**: Robust error management for real-world reliability
- **Performance Optimization**: Multiple optimization strategies for efficient inference
- **Clean Code Practices**: Modular design with clear separation of concerns

### Technical Sophistication
- **Advanced Model Architectures**: Multiple U-Net variants with custom improvements
- **Combined Loss Functions**: Innovative L1 + SSIM loss for balanced reconstruction
- **Training Pipeline Features**: Learning rate scheduling, early stopping, and comprehensive validation
- **Thread-Safe Inference**: Serialized model loading and inference safeguards
- **Flexible Deployment**: Support for both PyTorch and ONNX runtime

### Research Foundation
- **CARE Methodology**: Based on peer-reviewed research with custom enhancements
- **Auditable Evaluation**: Per-image and specimen-group restoration metrics
- **Systematic Experimentation**: Structured approach to hyperparameter optimization
- **Reproducible Results**: Consistent preprocessing and training pipelines

### User Experience
- **Intuitive Interface**: Modern, responsive web interface with real-time feedback
- **Multiple Use Cases**: Supports both researchers and clinicians with different interfaces
- **Comprehensive Documentation**: Detailed guides for installation, usage, and API
- **Flexible Configuration**: Environment-based setup for different deployment scenarios

---

## 👤 Author

**Samiksha**  
*Full Stack Developer & AI Engineer*

- 🎯 Specializing in deep learning applications for scientific imaging
- 💡 Passionate about bridging research and production software
- 🚀 Experienced in building end-to-end ML systems
- 📧 [LinkedIn](https://linkedin.com/in/yourprofile) | [GitHub](https://github.com/Sam-wan30) | [Portfolio](https://yourportfolio.com)

### Acknowledgments
- Inspired by the [CARE (Content-Aware Image Restoration)](https://arxiv.org/abs/1811.03675) framework
- Built with PyTorch, Flask, and modern web technologies
- Designed for microscopy researchers and image processing professionals

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue for bug reports and feature requests.

### Development Guidelines
- Follow PEP 8 style guidelines
- Write comprehensive docstrings
- Add tests for new features
- Update documentation as needed

---

## 📞 Support

For questions, issues, or collaboration opportunities:
- Open an Issue on GitHub
- Contact: [your.email@example.com]
- Documentation: [Project Wiki](https://github.com/YOUR_USERNAME/AI-Image-Denoising-In-Microscopy/wiki)

---

<div align="center">

**Built with ❤️ for the microscopy research community**

[⬆ Back to Top](#neuroscope)

</div>
