# Deploy FluoClean AI on Render

The production application is the Flask entry point in `application.py`. It uses
the ONNX checkpoint so the server does not need PyTorch.

## Prerequisites

- Push this repository to GitHub.
- The ONNX checkpoint is hosted in the repository's `V2.3.4` GitHub Release.
  It is intentionally excluded from Git because it is larger than GitHub's
  normal file limit.

## Deploy

1. In Render, create a new Blueprint and select this repository.
2. Apply the Blueprint. Render reads `render.yaml`, installs the production
   dependencies, downloads and quantizes the model, and starts Gunicorn.
3. Open `/health` on the deployed service. The process reports `status: ok` even
   before the model is loaded; `model.ready` becomes `true` after the first AI
   denoising request.

Uploads and generated outputs use ephemeral disk and disappear when the service
restarts. This is expected because the browser receives the result immediately.

## Local production check

```bash
MODEL_PATH=models/deploy/model.onnx \
gunicorn application:app --bind 127.0.0.1:5000 --workers 1 --threads 2 --timeout 180
```

Then open `http://127.0.0.1:5000` or request `http://127.0.0.1:5000/health`.
