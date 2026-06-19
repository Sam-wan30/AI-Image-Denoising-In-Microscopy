#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

model_path="${MODEL_PATH:-models/deploy/model.onnx}"
if [[ -f "$model_path" ]]; then
  echo "Model already present at $model_path"
elif [[ -n "${MODEL_URL:-}" ]]; then
  echo "Downloading model to $model_path"
  MODEL_DESTINATION="$model_path" python - <<'PY'
import os
import tempfile
import urllib.request
from pathlib import Path

url = os.environ["MODEL_URL"]
destination = Path(os.environ["MODEL_DESTINATION"])
destination.parent.mkdir(parents=True, exist_ok=True)

with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
    temporary = Path(handle.name)

try:
    urllib.request.urlretrieve(url, temporary)
    if temporary.stat().st_size == 0:
        raise RuntimeError("Downloaded model is empty")
    temporary.replace(destination)
finally:
    temporary.unlink(missing_ok=True)

print(f"Downloaded {destination.stat().st_size / 1024 / 1024:.1f} MB")
PY
else
  echo "ERROR: $model_path is missing and MODEL_URL is not set." >&2
  exit 1
fi
