#!/usr/bin/env bash
# Render build script — install deps and optionally fetch the model
set -euo pipefail

echo "==> Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

if [ -n "${MODEL_URL:-}" ]; then
  echo "==> Downloading model from MODEL_URL..."
  mkdir -p models/deploy
  python -c "
import os, urllib.request
from pathlib import Path
url = os.environ['MODEL_URL']
dest = Path(os.environ.get('MODEL_PATH', 'models/deploy/model.pt'))
dest.parent.mkdir(parents=True, exist_ok=True)
print(f'Downloading {url} -> {dest}')
urllib.request.urlretrieve(url, dest)
print(f'Done ({dest.stat().st_size // (1024*1024)} MB)')
"
else
  if [ ! -f "models/deploy/model.pt" ]; then
    echo "WARNING: models/deploy/model.pt not found and MODEL_URL not set."
    echo "         Deploy will start but U-Net denoising will fail until a model is provided."
  else
    echo "==> Found models/deploy/model.pt"
  fi
fi

echo "==> Build complete."
