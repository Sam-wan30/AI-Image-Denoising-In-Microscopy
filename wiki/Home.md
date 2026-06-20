# FluoClean AI - AI Microscopy Image Denoising

**Advanced Deep Learning Pipeline for Microscopy Image Restoration**

---

## Project Overview

FluoClean AI is a research prototype for microscopy image denoising with a deployable web demonstration. It is inspired by CARE (Content-Aware Image Restoration) and is not validated for diagnostic or quantitative scientific use.

### Technology Stack
- **Deep Learning**: PyTorch 2.2+ with custom U-Net architectures
- **Backend**: Flask 3.x REST API with optional ONNX Runtime support
- **Frontend**: Modern responsive web interface + Streamlit UI prototype
- **Image Processing**: OpenCV, Pillow, NumPy with advanced preprocessing
- **Training**: CARE-style paired dataset with L1+SSIM combined loss

---

## Problem Statement

Microscopy imaging is fundamental to biological research, medical diagnostics, and materials science. However, acquired images frequently contain significant noise that:

- Obscures critical cellular structures and organelles
- Reduces the accuracy of quantitative measurements  
- Limits the effectiveness of automated image analysis
- Requires longer exposure times that can damage samples
- Increases computational burden for downstream processing

Traditional denoising methods often fail to preserve fine structural details, making AI-based approaches essential for high-quality microscopy imaging.

---

## Motivation

This project was developed to bridge the gap between academic research and practical application in microscopy imaging:

- **Research Foundation**: Based on peer-reviewed CARE methodology with custom improvements
- **Practical Application**: Deployable research demonstration for microscopy workflows
- **Open Source**: Accessible to researchers and developers worldwide
- **Performance**: Optimized for both accuracy and computational efficiency
- **Usability**: Intuitive interfaces for both technical and non-technical users

---

## Key Features

### 🧠 Advanced Deep Learning Models
- **Multiple U-Net Architectures**: Standard, Enhanced (with residual blocks), and Residual variants
- **Combined Loss Function**: L1 + SSIM (0.7:0.3) for balanced structural preservation
- **Flexible Training**: Support for CARE-style paired datasets with optional augmentation
- **Model Optimization**: GroupNorm for small-batch stability, configurable depth and channels

### 🌐 Application Interfaces
- **Flask REST API**: Robust web service for image processing
- **Streamlit UI**: Interactive interface for rapid prototyping
- **CLI Tools**: Command-line utilities for batch processing
- **Responsive Design**: Mobile-friendly web interface

### 📊 Comprehensive Evaluation
- **Evaluation Metrics**: PSNR and SSIM against aligned clean targets; the web UI shows input similarity only
- **Visualization**: Side-by-side comparisons with quantitative feedback
- **Training Monitoring**: TensorBoard integration for real-time progress tracking
- **Validation Tracking**: Automatic best-model saving based on validation metrics

### 🔧 Development Tools
- **Flexible Configuration**: Environment-based settings for different deployment scenarios
- **Error Handling**: Comprehensive exception handling and logging
- **ONNX Runtime Support**: Optional deployment-optimized inference
- **Cross-Platform**: Compatible with Linux, macOS, and Windows

---

## Quick Start

### Installation
```bash
git clone https://github.com/Sam-wan30/AI-Image-Denoising-In-Microscopy.git
cd AI-Image-Denoising-In-Microscopy
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Running the Application
```bash
# Flask web application
python application.py
# Visit http://localhost:5000

# Streamlit interface
streamlit run app.py
# Visit http://localhost:8501
```

### CLI Usage
```bash
python inference.py \
  --model models/deploy/model.pt \
  --input path/to/noisy_image.png \
  --output results/
```

---

## Wiki Navigation

- [Project Architecture](Project-Architecture) - System design and data flow
- [Dataset Documentation](Dataset-Documentation) - Data sources and preprocessing
- [Model Documentation](Model-Documentation) - Model architectures and training
- [Installation Guide](Installation-Guide) - Detailed setup instructions
- [User Guide](User-Guide) - How to use the application
- [API Documentation](API-Documentation) - REST API reference
- [Development Guide](Development-Guide) - Contributing and development workflow
- [Research Background](Research-Background) - Scientific context and motivation
- [Performance Evaluation](Performance-Evaluation) - Metrics and benchmarks
- [Roadmap](Roadmap) - Project status and future plans
- [FAQ](FAQ) - Common questions and troubleshooting
- [Contributing](Contributing) - Contribution guidelines and workflow

---

## Technical Specifications

### Model Performance
- **Input Resolution**: 256×256 pixels (configurable)
- **Supported Formats**: PNG, JPEG, TIFF, WebP, BMP
- **Processing Speed**: ~100ms per image (CPU), ~20ms (GPU)
- **Memory Usage**: ~2GB RAM for inference, ~8GB for training

### System Requirements
- **Python**: 3.11 or higher
- **RAM**: 8GB+ recommended for training, 4GB for inference
- **Storage**: 2GB+ for dependencies and models
- **GPU**: Optional CUDA support for accelerated training

---

## Use Cases

### Biological Research
- Fluorescence microscopy image enhancement
- Live-cell imaging noise reduction
- Confocal microscopy post-processing
- Time-lapse sequence denoising

### Medical Diagnostics
- Pathology image enhancement
- Medical microscopy quality improvement
- Diagnostic support through image restoration
- Telemedicine image quality enhancement

### Materials Science
- Scanning Electron Microscopy (SEM) denoising
- Transmission Electron Microscopy (TEM) enhancement
- Atomic force microscopy improvement
- Materials characterization support

---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{neuroscope2024,
  title={FluoClean AI: AI Microscopy Image Denoising System},
  author={Samiksha},
  year={2024},
  url={https://github.com/Sam-wan30/AI-Image-Denoising-In-Microscopy}
}
```

---

## Acknowledgments

This project is inspired by the groundbreaking CARE (Content-Aware Image Restoration) framework and incorporates methodologies from peer-reviewed research in deep learning for microscopy imaging.

- [CARE Paper](https://arxiv.org/abs/1811.03675) - Original CARE methodology
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Flask](https://flask.palletsprojects.com/) - Web framework
- [OpenCV](https://opencv.org/) - Computer vision library

---

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/Sam-wan30/AI-Image-Denoising-In-Microscopy/blob/deploy-ready/LICENSE) file for details.

---

## Contact & Support

For questions, issues, or collaboration opportunities:
- **Issues**: [GitHub Issues](https://github.com/Sam-wan30/AI-Image-Denoising-In-Microscopy/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Sam-wan30/AI-Image-Denoising-In-Microscopy/discussions)
- **Email**: [Contact Author](mailto:your.email@example.com)

---

<div align="center">

**Built with ❤️ for the microscopy research community**

[⬆ Back to Wiki Home](Home)

</div>
