# Deploy NeuroScope to Render.com (FREE)

## If you see "Not Found" (plain text, 404)

Render returns **"Not Found"** with header `x-render-routing: no-server` when **no web service is running** at that URL.

**Fix:**
1. Log in to [dashboard.render.com](https://dashboard.render.com)
2. Check whether a service named `neuroscope-denoising` exists and status is **Live** (green)
3. If missing → create it (steps below)
4. If **Failed deploy** or **Suspended** → open **Logs**, fix the error, click **Manual Deploy**
5. Confirm the URL under **Settings → URL** matches the link you open

The Flask app is fine; the server simply is not deployed or not running yet.

---

This project ships a **Flask + HTML/CSS/JS** production app (`application.py`).  
The original Streamlit app (`app.py`) still works locally but is **not** used on Render.

---

## Before you deploy

### 1. Export a lean model (~120 MB)

```bash
cd "/Users/samiksha/AI Image Denoising In Microscopy"
pip install -r requirements.txt
python scripts/export_inference_checkpoint.py
```

Creates: `models/deploy/model.pt` (weights only, no optimizer).

### 2. Host the model file (required)

GitHub rejects files **> 100 MB**. Pick one:

| Option | Steps |
|--------|--------|
| **A. Git LFS** | `git lfs track "models/deploy/model.pt"` then commit & push |
| **B. Public URL** | Upload `model.pt` to Google Drive / Hugging Face / GitHub Release → copy direct download link → set `MODEL_URL` in Render |

---

## Render setup (beginner)

1. Push this repo to **GitHub** (include code; model via LFS or `MODEL_URL`).
2. Go to [render.com](https://render.com) → **Sign up** (free).
3. **New +** → **Web Service** → connect your repo.
4. Settings:

   | Field | Value |
   |-------|--------|
   | **Runtime** | Python 3 |
   | **Build Command** | `pip install -r requirements.txt` |
   | **Start Command** | `gunicorn application:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 180` |
   | **Plan** | Free |

5. **Environment variables** (Render → Environment):

   ```
   MODEL_PATH=models/deploy/model.pt
   DEVICE=cpu
   MAX_UPLOAD_MB=50
   SECRET_KEY=<click Generate or use a long random string>
   ```

   If the model is **not** in git, add:

   ```
   MODEL_URL=https://your-direct-download-link/model.pt
   ```

6. Click **Create Web Service**. First deploy takes ~10–15 minutes (PyTorch install).

7. Open your live URL: `https://neuroscope-denoising.onrender.com` (name varies).

---

## Blueprint deploy (optional)

If Render asks for a blueprint, use the included `render.yaml`:

**New +** → **Blueprint** → select repo → Render reads `render.yaml`.

---

## Local production test

```bash
pip install -r requirements.txt
python scripts/export_inference_checkpoint.py
python application.py
```

Visit: http://localhost:5000

Or with gunicorn:

```bash
gunicorn application:app --bind 0.0.0.0:5000 --workers 1 --timeout 180
```

---

## Free tier limits (important)

- **512 MB RAM** — uses CPU PyTorch + ~120 MB weights. Close to the limit; if the service crashes on boot, use `MODEL_URL` and ensure only **one** gunicorn worker.
- **Cold starts** — free apps sleep after ~15 min idle; first visit may take 30–60 s while the model loads.
- **No GPU** on free tier — inference is CPU-only (slower but free).

---

## Health check

- `GET /health` — returns model status (used by Render).
- `GET /api/status` — JSON model info for the UI.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Build fails on torch | Ensure `requirements.txt` has the `--extra-index-url` CPU line |
| `Model not found` | Run export script; set `MODEL_PATH` or `MODEL_URL` |
| Worker timeout | Increase gunicorn `--timeout` to `180` |
| Out of memory | Use `models/deploy/model.pt` (not 600MB residual checkpoint) |
| Upload fails | Max 50 MB; use PNG/JPG/TIFF |

---

## What users can do on the live site

1. Upload a microscopy image  
2. Run denoising  
3. Compare original vs denoised  
4. View **PSNR** and **SSIM**  
5. Download the denoised PNG  
