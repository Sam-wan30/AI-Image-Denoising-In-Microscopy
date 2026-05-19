<div align="center">

# AI Microscopy Image Denoising System

### Deep learning restoration for fluorescence and microscopy imaging

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

**NeuroScope** — upload noisy microscopy images, denoise with a trained U-Net, compare results, and export restored outputs with quantitative metrics.

[Features](#features) · [Demo](#demo) · [Installation](#installation) · [Usage](#usage) · [Deployment](#deployment) · [Project Structure](#project-structure)

</div>

---

## Project Overview

This repository implements an end-to-end **AI microscopy image denoising** pipeline inspired by the [CARE (Content-Aware Image Restoration)](https://arxiv.org/abs/1811.03675) framework. A **PyTorch U-Net** is trained on paired noisy/clean microscopy data and served through a production **Flask** web application with a custom **HTML/CSS/JavaScript** interface.

The system targets **grayscale fluorescence microscopy** and related modalities, preserving fine cellular structures while suppressing photon and sensor noise. Users can upload images, run inference, view side-by-side comparisons, inspect **PSNR** and **SSIM**, and download denoised PNG outputs.

---

## Features

| Capability | Description |
|------------|-------------|
| **Deep denoising** | U-Net family models (standard, enhanced, residual) with skip connections |
| **Web application** | Flask API + responsive NeuroScope UI (upload, compare, download) |
| **Quality metrics** | PSNR and SSIM computed after inference |
| **Multi-mode inference** | U-Net, auto routing, salt-and-pepper median filter, brightfield mask |
| **Training pipeline** | `train.py` with L1 + SSIM loss, validation tracking, checkpointing |
| **Batch inference** | CLI script for single images or folders (`inference.py`) |
| **Cloud-ready** | Gunicorn, Render.com config, lean checkpoint export for deployment |

---

## Demo

<!-- Replace with your live URL once Render service is Live -->
**Live app (Render):** configure deployment per [DEPLOY.md](DEPLOY.md) — e.g. `https://neuroscope-denoising.onrender.com`

**Local web UI:**

```bash
python application.py
# → http://localhost:5000
```

**Optional Streamlit prototype (local only):**

```bash
pip install streamlit
streamlit run app.py
```

---

## Tech Stack

| Layer | Technologies |
|-------|----------------|
| **Deep learning** | PyTorch, torchvision |
| **Vision / I/O** | OpenCV, Pillow, NumPy |
| **Backend** | Flask, Gunicorn, Werkzeug |
| **Frontend** | HTML5, CSS3, JavaScript (vanilla) |
| **Metrics** | Custom PSNR / SSIM utilities |
| **Deployment** | Render.com (free tier), `render.yaml`, `Procfile` |

---

## Model Architecture

The core model is an **encoder–decoder U-Net** designed for **1-channel (grayscale)** microscopy images.

| Variant | Description |
|---------|-------------|
| **MicroscopyUNet** | Classic U-Net with GroupNorm and skip connections (~31M parameters) |
| **EnhancedMicroscopyUNet** | Additional residual blocks in the encoder path |
| **ResidualMicroscopyUNet** | Residual-block U-Net wrapper for stronger feature reuse |

**Training objective:** combined **L1 + SSIM** loss (`utils/losses.py`).

**Inference pipeline:** images are converted to grayscale, resized to **256×256**, normalized to `[0, 1]`, passed through the network, and resized back to the original resolution (`utils/preprocessing.py`).

Checkpoints are saved during training; a lean **inference-only** export strips optimizer state for deployment (~120 MB):

```bash
python scripts/export_inference_checkpoint.py
# → models/deploy/model.pt
```

---

## Dataset Information

Training expects **paired noisy and clean** microscopy images in CARE-style layout:

```text
data/
└── train/
    ├── noisy/    # Noisy acquisitions
    └── clean/    # Ground-truth or high-SNR references
```

**Preprocessing** (from raw CARE or similar sources):

```bash
python scripts/process_microscopy_dataset.py \
  --input_dir /path/to/CARE_dataset \
  --output_dir data
```

Supported training formats: PNG, TIFF, and other formats handled by OpenCV. The `CAREDatasetSimple` loader applies optional augmentation and consistent 256×256 resizing.

---

## Project Structure

```text
AI Image Denoising In Microscopy/
├── application.py          # Flask production app (Render / Gunicorn)
├── app.py                  # Streamlit UI (optional, local dev)
├── config.py               # Environment-based configuration
├── train.py                # Model training
├── inference.py            # CLI batch / single-image inference
├── requirements.txt        # Production + training dependencies
├── requirements_torch.txt  # Extended stack for local training experiments
├── render.yaml             # Render.com blueprint
├── Procfile                # Gunicorn process definition
├── build.sh                # Render build script
├── DEPLOY.md               # Deployment guide
│
├── src/
│   ├── unet_model.py       # U-Net architectures
│   ├── care_dataset.py
│   └── care_dataset_simple.py
│
├── services/               # Flask inference service layer
│   ├── denoiser.py
│   ├── bootstrap.py
│   └── model_utils.py
│
├── utils/
│   ├── preprocessing.py
│   ├── metrics.py
│   ├── losses.py
│   ├── salt_pepper.py
│   └── brightfield.py
│
├── templates/              # Flask HTML templates
├── static/                 # CSS + JavaScript
├── ui/                     # Streamlit UI components
├── scripts/
│   ├── process_microscopy_dataset.py
│   └── export_inference_checkpoint.py
│
├── models/                 # Saved checkpoints (not committed by default)
├── data/                   # Processed dataset
├── uploads/                # Runtime uploads (created automatically)
└── outputs/                # Denoised downloads (created automatically)
```

---

## Installation

### Prerequisites

- Python **3.11+**
- pip
- (Optional) CUDA for faster local training

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-microscopy-denoising.git
cd ai-microscopy-denoising
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

**Production / web app:**

```bash
pip install -r requirements.txt
```

**Full local training stack (if needed):**

```bash
pip install -r requirements_torch.txt
```

### 4. Prepare model weights

Train your own model (see [Training](#training)) **or** place a checkpoint at `models/deploy/model.pt`:

```bash
python scripts/export_inference_checkpoint.py \
  --input models/overfit/best_model.pth \
  --output models/deploy/model.pt
```

---

## Usage

### Web application (recommended)

```bash
python application.py
```

Open **http://localhost:5000**, upload a microscopy image, click **Start denoising**, then review metrics and download the result.

**Production server:**

```bash
gunicorn application:app --bind 0.0.0.0:5000 --workers 1 --timeout 180
```

### Training

```bash
python train.py \
  --data_dir data \
  --epochs 50 \
  --batch_size 8 \
  --lr 0.001 \
  --save_dir models
```

Checkpoints and training curves are written to the `--save_dir` folder.

### CLI inference

```bash
python inference.py \
  --model models/deploy/model.pt \
  --input path/to/noisy_image.png \
  --output results/
```

See `python train.py --help` and `python inference.py --help` for all options.

### Validate setup

```bash
python test_app.py
```

---

## Results

After inference, the web UI reports:

- **PSNR** (dB) — peak signal-to-noise ratio between input and denoised output  
- **SSIM** — structural similarity index (0–1 scale)

Training runs additionally log validation PSNR/SSIM and save comparison panels under the model output directory (e.g. `models/training_curves.png`, epoch sample images).

> Metrics are computed on the **uploaded noisy image vs. model output** for interactive feedback. For rigorous evaluation against ground-truth clean images, use the training/validation split and `inference.py` with `--ground_truth`.

---

## Deployment

Free hosting on **Render.com** is supported out of the box.

| File | Purpose |
|------|---------|
| [DEPLOY.md](DEPLOY.md) | Step-by-step Render setup |
| `render.yaml` | Infrastructure blueprint |
| `build.sh` | Install deps + optional `MODEL_URL` download |
| `.env.example` | Local environment template |

**Required environment variables (production):**

```bash
MODEL_PATH=models/deploy/model.pt
DEVICE=cpu
SECRET_KEY=<random-secret>
# MODEL_URL=<direct-download-link>   # if weights are not in the repo
```

---

## Future Improvements

- [ ] Ground-truth evaluation mode in the web UI (when clean reference is uploaded)
- [ ] GPU-accelerated deployment tier and mixed-precision inference
- [ ] Model versioning and A/B comparison in the dashboard
- [ ] Docker image for reproducible local and cloud runs
- [ ] Automated tests for API routes and preprocessing parity
- [ ] Integration with public microscopy datasets (e.g. CARE sample subsets)

---

## Author

**Samiksha**

- Research-focused deep learning project for computational microscopy  
- Built for portfolio, internship, and research demonstrations  

<!-- Optional: add your links when ready -->
<!-- [GitHub](https://github.com/YOUR_USERNAME) · [LinkedIn](https://linkedin.com/in/YOUR_PROFILE) · [Email](mailto:you@example.com) -->

---

## Acknowledgments

- Inspired by the **CARE** denoising framework for microscopy  
- U-Net architecture based on Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*

---

## License

This project is provided for educational and research purposes. Add a `LICENSE` file (e.g. MIT) before public distribution if not already present.
