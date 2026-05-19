# Complete Guide to Run Microscopy Image Denoising Project

## 📋 Prerequisites

Make sure you have Python 3.8+ and conda/pip installed.

## 🚀 Quick Start Commands

### 1. Navigate to Project Directory
```bash
cd "/Users/samiksha/AI Image Denoising In Microscopy"
```

### 2. Install Dependencies
```bash
# Install basic dependencies
pip install -r requirements.txt

# Install PyTorch dependencies
pip install -r requirements_torch.txt

# Install Streamlit for web app
pip install streamlit
```

### 3. Dataset Processing (if you have raw CARE dataset)
```bash
# Process CARE dataset
python3 scripts/process_microscopy_dataset.py --input_dir /path/to/CARE_dataset --output_dir data

# Example with specific dataset location
python3 scripts/process_microscopy_dataset.py --input_dir "/Users/samiksha/Downloads/CARE (2D)" --output_dir data
```

### 4. Train the Model
```bash
# Basic training
python3 train.py

# Custom training parameters
python3 train.py --epochs 100 --batch_size 8 --lr 0.001 --model_type enhanced

# Training with specific data directory
python3 train.py --data_dir data --epochs 50 --batch_size 4 --save_dir my_models
```

### 5. Run Inference
```bash
# Single image inference
python3 inference.py --model models/best_model.pth --input path/to/noisy_image.png --output results

# Batch processing
python3 inference.py --model models/best_model.pth --input data/train/noisy --output results --ground_truth data/train/clean

# Interactive mode with comparison
python3 inference.py --model models/best_model.pth --input image.png --output results/ --show
```

### 6. Launch Streamlit Web App
```bash
# Start the web application
streamlit run app.py

# The app will open at: http://localhost:8501
```

## 📁 Project Structure Commands

### View Project Structure
```bash
# Show complete project structure
tree -a

# Or use ls for simple view
ls -la
```

### Check Model Files
```bash
# List saved models
ls -la models/

# Check model file size
du -h models/best_model.pth
```

### Check Dataset
```bash
# Verify dataset structure
ls -la data/train/noisy/
ls -la data/train/clean/

# Count images
find data/train/noisy -name "*.png" | wc -l
find data/train/clean -name "*.png" | wc -l
```

## 🧪 Testing Commands

### Test Individual Components
```bash
# Test metrics module
python3 utils/metrics.py

# Test dataset loading
python3 -c "
from src.care_dataset_simple import CAREDatasetSimple
dataset = CAREDatasetSimple(root_dir='data')
print(f'Dataset size: {len(dataset)}')
"

# Test model creation
python3 -c "
from src.unet_model import create_unet_model
model = create_unet_model()
print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
"

# Test Streamlit app components
python3 test_app.py
```

### Quick Training Test
```bash
# Test training with 2 epochs
python3 train.py --epochs 2 --batch_size 2 --no_samples
```

### Quick Inference Test
```bash
# Test inference on a few images
python3 inference.py --model models/best_model.pth --input data/train/noisy --output test_inference --no_comparison
```

## 🛠️ Development Commands

### Create Virtual Environment
```bash
# Using conda
conda create -n microscopy_denoising python=3.12
conda activate microscopy_denoising

# Using venv
python3 -m venv microscopy_env
source microscopy_env/bin/activate  # On Mac/Linux
# microscopy_env\Scripts\activate  # On Windows
```

### Install Development Tools
```bash
# Install Jupyter for experimentation
pip install jupyter notebook

# Install additional monitoring tools
pip install tensorboard
pip install matplotlib seaborn
```

### Run with GPU (if available)
```bash
# Check GPU availability
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Force GPU usage in inference
python3 inference.py --model models/best_model.pth --input image.png --device cuda --output results
```

## 📊 Monitoring Commands

### Monitor Training
```bash
# View training logs
tail -f logs/training.log

# Monitor GPU usage (if using GPU)
watch -n 1 nvidia-smi

# Monitor CPU usage
top -p $(pgrep -f python3)
```

### Check Results
```bash
# View training curves
open models/training_curves.png

# View inference results
open inference_test/

# List all saved models with details
ls -lah models/*.pth
```

## 🌐 Web App Commands

### Run Streamlit App
```bash
# Basic run
streamlit run app.py

# Run on specific port
streamlit run app.py --server.port 8501

# Run with network access
streamlit run app.py --server.address 0.0.0.0

# Run with debug mode
streamlit run app.py --logger.level debug
```

### Test Web App Components
```bash
# Validate app components
python3 test_app.py

# Check Streamlit version
streamlit --version
```

## 🔄 Complete Workflow Example

### Full Pipeline from Start to Finish
```bash
# 1. Navigate to project
cd "/Users/samiksha/AI Image Denoising In Microscopy"

# 2. Install dependencies
pip install -r requirements.txt
pip install -r requirements_torch.txt
pip install streamlit

# 3. Process dataset (if needed)
python3 scripts/process_microscopy_dataset.py --input_dir "/path/to/CARE_dataset" --output_dir data

# 4. Train model
python3 train.py --epochs 50 --batch_size 8 --lr 0.001

# 5. Test inference
python3 inference.py --model models/best_model.pth --input data/train/noisy --output test_results

# 6. Launch web app
streamlit run app.py
```

## 🚨 Troubleshooting Commands

### Check Dependencies
```bash
# Check Python version
python3 --version

# Check installed packages
pip list | grep -E "(torch|opencv|streamlit)"

# Check PyTorch CUDA support
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Fix Common Issues
```bash
# If numpy/h5py issues occur
conda install -c conda-forge numpy=1.26.4 h5py=3.11.0

# If OpenCV issues occur
pip install opencv-python-headless

# If Streamlit issues occur
pip install --upgrade streamlit
```

### Reset Project
```bash
# Clean generated files
rm -rf models/
rm -rf logs/
rm -rf inference_test/
rm -rf __pycache__/
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## 📱 Mobile/Remote Access

### Run Web App Remotely
```bash
# Run with external access
streamlit run app.py --server.address 0.0.0.0 --server.port 8501

# Access from other devices using: http://YOUR_IP:8501
```

## 🎯 Quick Validation Commands

### Validate Complete Setup
```bash
# Run all tests
python3 utils/metrics.py && python3 test_app.py && echo "✅ All tests passed!"

# Quick training test
python3 train.py --epochs 1 --batch_size 2 --no_samples && echo "✅ Training works!"

# Quick inference test
python3 inference.py --model models/best_model.pth --input data/train/noisy --output quick_test --no_comparison && echo "✅ Inference works!"
```

---

## 📝 Notes

1. **Model Training**: Takes 30-60 minutes for 100 epochs on CPU, faster on GPU
2. **Web App**: Requires trained model in `models/best_model.pth`
3. **Dataset**: Process CARE dataset first if you have raw data
4. **Memory**: Recommend 8GB+ RAM for training, 4GB+ for inference
5. **GPU**: Optional but recommended for faster training

## 🆘 Help Commands

```bash
# Get help for any script
python3 train.py --help
python3 inference.py --help
python3 scripts/process_microscopy_dataset.py --help

# Check project README
cat README.md
```
