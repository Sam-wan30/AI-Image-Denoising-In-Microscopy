# FAQ

**Common Questions and Troubleshooting**

---

## Overview

This FAQ addresses common questions from users, developers, and researchers about FluoClean AI. It covers installation, usage, technical issues, and best practices to help you resolve problems quickly and effectively.

### FAQ Categories
- **User Questions**: Usage, installation, and general usage
- **Developer Questions**: Development, contribution, and technical issues
- **Research Questions**: Methodology, evaluation, and scientific aspects
- **Troubleshooting**: Common problems and solutions

---

## User Questions

### Installation and Setup

#### Q: What are the system requirements for FluoClean AI?
**A:** 
- **Operating System**: Linux (Ubuntu 18.04+), macOS (10.15+), Windows 10+
- **Python**: 3.11 or higher
- **RAM**: 8GB minimum, 16GB+ recommended for training
- **Storage**: 5GB minimum, 10GB+ recommended
- **GPU**: Optional but recommended for training (NVIDIA GPU with 8GB+ VRAM)

#### Q: Can I run FluoClean AI without a GPU?
**A:** Yes, FluoClean AI can run on CPU for inference, though it will be slower. Training on CPU is possible but significantly slower. For the best experience, an NVIDIA GPU with CUDA support is recommended.

#### Q: How do I install FluoClean AI on Windows?
**A:**
1. Install Python 3.11 or higher from python.org
2. Clone the repository: `git clone https://github.com/Sam-wan30/AI-Image-Denoising-In-Microscopy.git`
3. Create virtual environment: `python -m venv .venv`
4. Activate environment: `.venv\Scripts\activate`
5. Install dependencies: `pip install -r requirements.txt`
6. Run the application: `python application.py`

#### Q: I'm getting import errors when trying to import modules. What should I do?
**A:** This is usually a virtual environment issue. Ensure:
- Your virtual environment is activated
- Dependencies are installed: `pip install -r requirements.txt`
- Python version is 3.11 or higher
- You're in the correct directory with the repository

### Usage and Functionality

#### Q: What image formats does FluoClean AI support?
**A:** FluoClean AI supports the following formats:
- PNG (recommended for lossless quality)
- JPEG/JPG (smaller file size, some quality loss)
- TIFF/TIFF (high-quality microscopy format)
- WebP (modern web format)
- BMP (uncompressed format)

#### Q: What is the maximum image size I can process?
**A:** The default maximum file size is 50MB, but this can be configured via the `MAX_UPLOAD_MB` environment variable. For very large images (16K×16K+), consider processing in patches or using a system with more memory.

#### Q: Which denoising mode should I use?
**A:**
- **Auto** (Recommended for most users): Automatically selects the best method
- **U-Net**: Best for general-purpose denoising with deep learning
- **Salt-Pepper**: Best for images with salt-and-pepper (impulse) noise
- **Brightfield**: Best for brightfield microscopy images

#### Q: How long does it take to denoise an image?
**A:** Processing time depends on hardware and image size:
- **CPU**: ~120ms per image (512×512)
- **GPU (RTX 3060)**: ~25ms per image (512×512)
- **GPU (V100)**: ~15ms per image (512×512)
Larger images take proportionally longer.

#### Q: Can I process multiple images at once?
**A:** Yes, use the CLI tool for batch processing:
```bash
python inference.py \
  --model models/deploy/model.pt \
  --input_dir path/to/images/ \
  --output_dir results/ \
  --batch
```

#### Q: What do the quality metrics mean?
**A:**
- In the web UI, PSNR and SSIM compare the output with the noisy input. They
  indicate how much the output changed, not whether denoising is correct.
- With an aligned clean reference, higher PSNR and SSIM generally indicate a
  closer reconstruction, but neither proves biological fidelity.
- **PSNR (Peak Signal-to-Noise Ratio)**: Higher is better
  - >40 dB: Excellent quality
  - 30-40 dB: Good quality
  - <30 dB: May need improvement
- **SSIM (Structural Similarity Index)**: Closer to 1.0 is better
  - >0.90: Excellent structural preservation
  - 0.70-0.90: Good structural preservation
  - <0.70: Poor structural preservation

#### Q: The denoised output looks worse than the input. What should I do?
**A:** This could indicate:
- Wrong denoising mode selected (try Auto mode)
- Image type doesn't match training distribution
- Model needs more training on similar data
- Input image quality is too low

Try different modes and ensure your images are similar to the training data type.

### Models and Training

#### Q: Do I need to train my own model, or can I use pre-trained models?
**A:** You can use pre-trained models if available. For best results with your specific microscopy setup, training on similar data is recommended. Pre-trained models work well for general fluorescence microscopy but may need fine-tuning for specialized applications.

#### Q: How do I train my own model?
**A:** Follow these steps:
1. Prepare paired noisy/clean images in CARE format
2. Run training: `python train.py --data_dir data --epochs 50 --save_dir models`
3. Export for inference: `python scripts/export_inference_checkpoint.py --input models/best_model.pth --output models/deploy/model.pt`
4. Update configuration to use your model

#### Q: What dataset do I need for training?
**A:** You need paired noisy/clean microscopy images in CARE format:
```
data/
└── train/
    ├── noisy/    # Noisy microscopy images
    └── clean/    # Clean reference images
```
Both directories must have matching filenames for corresponding pairs.

#### Q: How many images do I need for training?
**A:** Minimum requirements:
- **Standard U-Net**: 100+ image pairs
- **Enhanced U-Net**: 200+ image pairs
- **Residual U-Net**: 500+ image pairs
For best results, 1,000+ pairs are recommended.

#### Q: How long does training take?
**A:** Training time depends on hardware and dataset size:
- **CPU**: ~4 hours for 50 epochs on 1,000 pairs
- **GPU (RTX 3060)**: ~25 minutes for 50 epochs on 1,000 pairs
- **GPU (V100)**: ~8 minutes for 50 epochs on 1,000 pairs

#### Q: What are the differences between the model variants?
**A:**
- **Standard U-Net** (~31M parameters): Good baseline, fast training
- **Enhanced U-Net** (~38M parameters): Better feature extraction, medium training time
- **Residual U-Net** (~42M parameters): Best quality, longer training time

---

## Developer Questions

### Development Environment

#### Q: What development tools do I need?
**A:** For development, you'll need:
- Python 3.11+ with virtual environment
- Git for version control
- Code editor (VS Code, PyCharm, etc.)
- Optional: Docker for containerization

Recommended development tools:
- `black` for code formatting
- `flake8` for linting
- `mypy` for type checking
- `pytest` for testing

#### Q: How do I set up a development environment?
**A:**
```bash
# Clone repository
git clone https://github.com/Sam-wan30/AI-Image-Denoising-In-Microscopy.git
cd AI-Image-Denoising-In-Microscopy

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements_torch.txt

# Install development tools
pip install black flake8 mypy pytest pytest-cov
```

#### Q: How do I run tests?
**A:**
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_metrics.py

# Run with coverage
pytest --cov=src --cov=services --cov=utils
```

#### Q: What is the code style for this project?
**A:** The project follows PEP 8 style guidelines:
- Line length: maximum 100 characters
- Indentation: 4 spaces (no tabs)
- Naming: snake_case for variables/functions, PascalCase for classes
- Docstrings: Google-style docstrings
- Type hints: Required for function signatures

### Code and Architecture

#### Q: How do I add a new denoising mode?
**A:** To add a new denoising mode:
1. Implement the denoising function in `utils/` or `services/`
2. Add mode to available modes in configuration
3. Add mode selection in application.py and app.py
4. Add tests for the new mode
5. Update documentation

#### Q: How do I modify the model architecture?
**A:** Model architectures are in `src/unet_model.py`. To modify:
1. Create new model class inheriting from nn.Module
2. Add model type to create_unet_model() factory function
3. Update training script to support new model
4. Add configuration options
5. Test and document changes

#### Q: How do I integrate FluoClean AI into my existing pipeline?
**A:** You can integrate via:
- **REST API**: Use the Flask API endpoints
- **Python SDK**: Import the DenoiserService class
- **CLI**: Use the inference.py script
- **Library**: Import specific modules as needed

See the [API Documentation](API-Documentation) for details.

### API and Integration

#### Q: How do I use the REST API?
**A:** The REST API provides endpoints for denoising:
```python
import requests

with open('noisy.png', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/api/denoise',
        files={'image': f},
        data={'mode': 'auto'}
    )
result = response.json()
```

See [API Documentation](API-Documentation) for complete reference.

#### Q: How do I handle rate limiting?
**A:** Rate limiting is not currently implemented but is recommended for production. You can add Flask-Limiter:
```python
from flask_limiter import Limiter
limiter = Limiter(app, key_func=get_remote_address)
```

#### Q: How do I deploy the API in production?
**A:** For production deployment:
1. Use gunicorn instead of Flask dev server: `gunicorn application:app --workers 4`
2. Set up environment variables for production
3. Implement authentication
4. Set up monitoring and logging
5. Use reverse proxy (nginx) for production

---

## Research Questions

### Methodology and Evaluation

#### Q: Why do you use combined L1 + SSIM loss?
**A:** The combination provides balanced performance:
- **L1 loss**: Effective at pixel-level noise reduction
- **SSIM loss**: Preserves structural information
- **0.7:0.3 ratio**: Empirically determined for microscopy images

This combination was chosen because single metrics (L1, L2, or SSIM alone) don't provide optimal results for preserving fine biological structures while removing noise.

#### Q: How does FluoClean AI compare to CARE?
**A:** FluoClean AI is inspired by CARE but includes several enhancements:
- Multiple U-NET architectures (Standard, Enhanced, Residual)
- Production-ready web applications and API
- Multiple denoising modes for different use cases
- Comprehensive documentation and tools
- Focus on usability and deployment

#### Q: What is the validation methodology?
**A:** Validation uses:
- **Train/Validation Split**: 80/20 split standard
- **Metrics**: PSNR and SSIM on validation set
- **Early Stopping**: Patience-based on validation PSNR
- **Visual Inspection**: Sample outputs during training
- **Cross-validation**: Optional for small datasets

#### Q: How do you handle different microscopy modalities?
**A:** Currently, FluoClean AI is optimized for fluorescence microscopy but includes:
- **Brightfield mode**: Specialized processing for brightfield
- **Auto mode**: Attempts automatic mode selection
- **Configurable preprocessing**: Can be adapted for modalities
- **Future plans**: Specific support for SEM, TEM, confocal, etc.

### Performance and Limitations

#### Q: What are the limitations of the current approach?
**A:** Known limitations:
- **Training Data Dependency**: Performance best on data similar to training set
- **Resolution Constraints**: Fixed 256×256 training resolution
- **Memory Requirements**: Large images may exceed available memory
- **Modalality Specificity**: Optimized for fluorescence microscopy
- **Processing Speed**: CPU inference significantly slower than GPU

#### Q: How do you ensure the model doesn't overfit?
**A:** Overfitting prevention through:
- **Early Stopping**: Stop when validation PSNR plateaus
- **Data Augmentation**: Random flips and rotations
- **Regularization**: Combined loss function, dropout (if added)
- **Validation Monitoring**: Track validation metrics during training
- **Model Complexity**: Appropriate architecture for dataset size

#### Q: How do you evaluate the biological plausibility of results?
**A:** Biological plausibility is evaluated through:
- **Domain Expert Review**: Microscopy researchers review results
- **Structure Preservation**: SSIM metric and visual inspection
- **Quantitative Analysis**: Impact on downstream analysis
- **Comparative Studies**: Comparison with ground truth when available

---

## Troubleshooting

### Installation Issues

#### Q: I get "Module not found" errors during installation. What should I do?
**A:** 
1. Ensure virtual environment is activated: `source .venv/bin/activate`
2. Try upgrading pip: `python -m pip install --upgrade pip`
3. Install dependencies again: `pip install -r requirements.txt --force-reinstall`
4. Check Python version: `python --version` (must be 3.11+)

#### Q: GPU not detected even though I have one. What should I do?
**A:**
1. Check NVIDIA driver: `nvidia-smi`
2. Check CUDA installation: `nvcc --version`
3. Reinstall PyTorch with CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
4. Verify with: `python -c "import torch; print(torch.cuda.is_available())"`

#### Q: OpenCV import error: "libGL.so.1: cannot open shared object file"
**A:** This is a common issue with headless environments. Install system libraries:
```bash
# Ubuntu/Debian
sudo apt-get install libgl1-mesa-glx libglib2.0-0

# Or use headless version
pip install opencv-python-headless
```

### Runtime Issues

#### Q: Model status shows "not ready" but I have the model file. What should I do?
**A:**
1. Verify model file exists at the correct path
2. Check file permissions
3. Verify model file integrity (not corrupted)
4. Check application logs for specific error messages
5. Try re-exporting the model checkpoint

#### Q: Denoising takes too long. How can I speed it up?
**A:** 
1. Use GPU if available: Set `DEVICE=cuda` in environment
2. Try smaller images if possible
3. Use simpler denoising mode (salt-pepper is fastest)
4. Check system resources and other processes
5. Consider batch processing for better efficiency

#### Q: Memory errors during processing. What should I do?
**A:**
1. Reduce image size or process in patches
2. Use CPU instead of GPU if GPU memory is limited
3. Reduce batch size for training
4. Close other applications consuming memory
5. Increase system RAM if possible

#### Q: Poor denoising quality. How can I improve results?
**A:**
1. Try different denoising mode
2. Ensure input image quality is adequate
3. Check if image type matches training distribution
4. Try different model variant (Residual U-Net has best quality)
5. Consider training a model on similar data

### API Issues

#### Q: API returns 404 error for download endpoint. What should I do?
**A:**
1. Verify the filename is correct
2. Check that the file exists in the output directory
3. Ensure the file was successfully generated during denoising
4. Check file permissions and path configuration
5. Review API logs for specific error details

#### Q: API timeout errors during large file uploads. What should I do?
**A:**
1. Increase timeout configuration
2. Process smaller images if possible
3. Check network connectivity
4. Verify server is not overloaded
5. Consider chunked upload for very large files

---

## Best Practices

### Usage Best Practices

#### Image Preparation
- Use lossless formats (PNG, TIFF) when possible
- Maintain consistent bit depth across images
- Ensure adequate signal-to-noise ratio in input
- Avoid multiple compression cycles
- Keep backup copies of original images

#### Model Selection
- Use Auto mode for general use
- Use specific modes for known noise types
- Choose model variant based on quality vs. speed needs
- Test different modes on sample images
- Consider training custom models for specialized applications

#### Processing Workflow
- Process sample images first before batch processing
- Monitor quality metrics for unexpected results
- Verify output integrity before use in production
- Keep organized directory structures
- Document processing parameters for reproducibility

### Development Best Practices

#### Code Quality
- Follow PEP 8 style guidelines
- Add type hints to function signatures
- Write comprehensive docstrings
- Add unit tests for new functionality
- Review code before committing

#### Testing
- Test with diverse image types and sizes
- Include edge cases in test suites
- Test both CPU and GPU execution
- Verify error handling works correctly
- Performance test with large datasets

#### Documentation
- Keep documentation updated with code changes
- Provide examples for new features
- Document configuration options
- Include troubleshooting information
- Maintain API documentation current

---

## Getting Help

### Support Channels

#### Official Channels
- **GitHub Issues**: Report bugs and request features
- **GitHub Discussions**: Ask questions and share experiences
- **Documentation**: Check wiki pages for answers first

#### Community Resources
- **Wiki**: Comprehensive documentation
- **README**: Project overview and quick start
- **Code Comments**: Inline documentation
- **Examples**: Sample code and usage examples

### When to Ask for Help

#### Before Asking
1. Check documentation (wiki, README)
2. Search existing issues and discussions
3. Try the troubleshooting steps in this FAQ
4. Check your environment and configuration

#### When Asking
- After exhausting self-help options
- When encountering unexpected behavior
- For clarification on documentation
- To report bugs not already reported
- To suggest new features or improvements

### How to Ask Effective Questions

#### Good Questions Include
- Clear description of what you're trying to do
- Exact error messages or symptoms
- Your environment details (OS, Python version, etc.)
- Steps you've already tried
- Expected vs. actual behavior

#### Example Good Question
```
"I'm trying to train a model on 500 fluorescence images using Python 3.11 on Ubuntu 22.04.
When I run `python train.py --data_dir data --epochs 50`, I get this error:
'CUDA out of memory'. I have an RTX 3060 with 12GB VRAM and I'm using batch_size=8.
I've tried reducing to batch_size=4 but still get the same error.
How can I resolve this?"
```

---

## Common Mistakes

### User Mistakes

#### ❌ Using Compressed Images
**Problem**: Highly compressed JPEG images lose quality before denoising
**Solution**: Use lossless formats (PNG, TIFF) or minimal compression

#### ❌ Ignoring Model Status
**Problem**: Processing before model is ready
**Solution**: Check model status before processing, wait if needed

#### ❌ Wrong Mode Selection
**Problem**: Using salt-pepper mode for Gaussian noise
**Solution**: Use Auto mode for automatic selection, or choose appropriate mode

### Developer Mistakes

#### ❌ Not Using Virtual Environment
**Problem**: System-wide Python installation causes conflicts
**Solution**: Always use virtual environments for development

#### ❌ Ignoring Type Hints
**Problem**: Code becomes hard to understand and maintain
**Solution**: Add type hints to all function signatures

#### ❌ Skipping Tests
**Problem**: Bugs slip into production
**Solution**: Write tests for all new functionality

---

## Advanced Questions

### Research and Development

#### Q: Can I use FluoClean AI for 3D microscopy data?
**A:** Currently, FluoClean AI processes 2D images. 3D support is planned for future releases. For now, you can process 3D data by processing each Z-slice independently, but temporal consistency won't be preserved.

#### Q: Can FluoClean AI handle time-lapse imaging?
**A:** For individual time-lapse frames, yes. For time-aware processing that maintains temporal consistency, this feature is planned for future development.

#### Q: How does FluoClean AI handle different noise characteristics?
**A:** The Auto mode attempts to detect noise characteristics and select the appropriate denoising method. For best results with known noise types, use the specific mode (e.g., salt-pepper for impulse noise).

#### Q: Can I fine-tune a pre-trained model on my data?
**A:** Yes, you can fine-tune pre-trained models:
```python
# Load pre-trained checkpoint
checkpoint = torch.load('pretrained_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Fine-tune with lower learning rate
optimizer = optim.Adam(model.parameters(), lr=0.0001)
```

---

## Legal and Licensing

#### Q: Can I use FluoClean AI for commercial purposes?
**A:** Yes, FluoClean AI is released under the MIT License, which permits commercial use, modification, and distribution. See the LICENSE file for details.

#### Q: Can I use the models trained with FluoClean AI in commercial products?
**A:** Yes, the MIT License allows using the code and trained models in commercial products. However, if you use the training data (which may have its own license), ensure compliance with the dataset's license terms.

#### Q: Do I need to cite FluoClean AI if I use it in research?
**A:** Citing is appreciated but not required by the license. If you publish research using FluoClean AI, please cite:
```bibtex
@software{neuroscope2024,
  title={FluoClean AI: AI Microscopy Image Denoising System},
  author={Samiksha},
  year={2024},
  url={https://github.com/Sam-wan30/AI-Image-Denoising-In-Microscopy}
}
```

---

## Future Updates

#### Q: When will new features be released?
**A:** Follow the [Roadmap](Roadmap) for planned features and timelines. Major releases are typically quarterly, with minor updates as needed.

#### Q: How can I stay updated on FluoClean AI developments?
**A:** 
- Star the repository on GitHub
- Watch releases on GitHub
- Follow development commits
- Join GitHub Discussions
- Check the Wiki for updates

---

<div align="center">

**This FAQ covers the most common questions - check here first when encountering issues**

[⬆ Back to Wiki Home](Home) | [← Roadmap](Roadmap) | [Contributing](Contributing) →

</div>
