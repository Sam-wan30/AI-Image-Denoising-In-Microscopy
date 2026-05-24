#!/usr/bin/env bash
# Render build script — install deps and optionally fetch the model
set -euo pipefail

echo "==> Installing Python dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [ -n "${MODEL_URL:-}" ]; then
  echo "==> Downloading model from MODEL_URL..."
  mkdir -p models/deploy
  python -c "
import os, urllib.request
from pathlib import Path
url = os.environ['MODEL_URL']
dest = Path(os.environ.get('MODEL_PATH', 'models/deploy/model.onnx'))
if url.lower().endswith('.onnx') and dest.suffix != '.onnx':
    print(f'WARNING: MODEL_URL points to .onnx but MODEL_PATH is set to {dest}.')
    print('         This may cause the app to load the file as a PyTorch checkpoint.')
dest.parent.mkdir(parents=True, exist_ok=True)
print(f'Downloading {url} -> {dest}')
urllib.request.urlretrieve(url, dest)
print(f'Done ({dest.stat().st_size // (1024*1024)} MB)')
"
else
  MODEL_FILE=${MODEL_PATH:-models/deploy/model.onnx}
  if [ ! -f "$MODEL_FILE" ]; then
    echo "WARNING: $MODEL_FILE not found and MODEL_URL not set."
    echo "         Deploy will start but U-Net denoising will fail until a model is provided."
  else
    echo "==> Found $MODEL_FILE"
  fi
fi

# Optionally quantize ONNX model to reduce memory usage on small hosts
MODEL_FILE=${MODEL_FILE:-models/deploy/model.onnx}
if [ "${QUANTIZE_ONNX:-0}" = "1" ]; then
  if [ -f "${MODEL_FILE}" ]; then
    echo "==> Quantizing ONNX model to reduce memory footprint..."
    python scripts/quantize_onnx.py --input "$MODEL_FILE" --output "${MODEL_FILE%.*}.quant.onnx"
    if [ -f "${MODEL_FILE%.*}.quant.onnx" ]; then
      mv "${MODEL_FILE%.*}.quant.onnx" "$MODEL_FILE"
      echo "==> Replaced model with quantized version: $MODEL_FILE"
    else
      echo "WARNING: Quantization failed or output not found."
    fi
  else
    echo "NOTE: QUANTIZE_ONNX=1 but model file not present to quantize."
  fi
fi

echo "==> Build complete."
