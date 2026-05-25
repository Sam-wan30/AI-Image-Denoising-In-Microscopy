# Deploy NeuroScope to Render.com (ONNX-first, free tier)

This guide shows how to prepare and deploy the app to Render using an ONNX model (recommended for small hosts).

## Why ONNX?
- ONNXRuntime provides a much smaller deploy footprint than PyTorch and avoids installing `torch` on Render.
- The repo is updated to prefer `.onnx` deploy checkpoints. If you only have a `.pt` checkpoint, export it to ONNX (steps below).

---

## Quick summary
1. Export an ONNX model from your `.pt` checkpoint (or upload an existing `.onnx`).
2. Push code to GitHub (don't commit large model files — use `MODEL_URL` or Git LFS).
3. Create a Render Web Service using the provided `render.yaml` or manual settings.

---

## 1. Export PyTorch `.pt` → ONNX (local machine)

From the project root, run:

```bash
# (optional) activate your virtualenv
pip install -r requirements_torch.txt   # only if you need torch locally
python scripts/export_to_onnx.py --input models/deploy/model.pt --output models/deploy/model.onnx
```

Notes:
- The export requires PyTorch. If your environment lacks `torch`, install it locally (see `requirements_torch.txt`).
- If the exported ONNX is large (>100 MB), prefer hosting it externally (Hugging Face, Google Drive, S3) and set `MODEL_URL` in Render rather than committing the file.

---

## 2. Host the ONNX model (if large)

Options:
- Git LFS: `git lfs track "models/deploy/model.onnx"` then commit & push (recommended for private repos).
- Public URL: upload the ONNX file to a static host (Hugging Face, S3, Google Drive) and set `MODEL_URL` in Render.

If using `MODEL_URL`, Render's `build.sh` will download it into `MODEL_PATH` during build.

---

## 3. Render setup (beginner)

1. Push this repo to **GitHub** (include code; model via LFS or `MODEL_URL`).
2. Go to [render.com](https://render.com) → **Sign up** (free).
3. **New +** → **Web Service** → connect your repo.
4. Settings:

   - Runtime: `Python`
   - Build command:
     ```bash
     chmod +x build.sh && ./build.sh
     ```
   - Start command:
     ```bash
     gunicorn application:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 180
     ```
   - Plan: `Free`

5. Environment variables (Render → Environment):

   - `MODEL_PATH=models/deploy/model.onnx` (or `models/deploy/model.pt` if you insist on PyTorch)
   - `DEVICE=cpu`
   - `MAX_UPLOAD_MB=50`
   - `SECRET_KEY=<random secret>`

   If the model is not committed to the repo, also add:

   - `MODEL_URL=https://your-direct-download-link/model.onnx`

6. Click **Create Web Service**.

Render will run `build.sh`, which installs `requirements.txt` and will download `MODEL_URL` into `MODEL_PATH` if provided.

---

## 4. Local production test

```bash
pip install -r requirements.txt
# If you need PyTorch for export, use requirements_torch.txt locally
python application.py
```

Visit: http://localhost:5000

Or with gunicorn:

```bash
gunicorn application:app --bind 0.0.0.0:5000 --workers 1 --timeout 180
```

---

## 5. Troubleshooting

- Build fails because `onnxruntime` missing: ensure `onnxruntime>=1.15.0` is in `requirements.txt` (already included).
- `Model not found`: make sure `models/deploy/model.onnx` exists or set `MODEL_URL`.
- Worker timeout: increase gunicorn `--timeout` to `180`.
- Out of memory on free tier: use an ONNX model and single worker; reduce model size with quantization if needed.

---

## What I changed in the repo to make this easier
- Made the server `ONNX`-first to avoid installing PyTorch on Render.
- Updated `requirements.txt`, `Procfile`, and `render.yaml` for a Gunicorn/Flask deployment.

---

If you want, I can export your `models/deploy/model.pt` to ONNX now and create a small upload-ready artifact — tell me whether you want me to commit the ONNX file (not recommended) or upload it to a provided public URL.
