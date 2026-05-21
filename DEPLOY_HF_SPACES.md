Hugging Face Spaces deployment guide (Streamlit runtime)

1) Push the `deploy-ready` branch to GitHub (already done).

2) Create a new Space on Hugging Face:
   - Choose the "Streamlit" SDK.
   - Connect to your GitHub repository and select the `deploy-ready` branch.
   - Set the "Space SDK" to `streamlit` and `app.py` as the entrypoint.

3) Add repository secrets (in the Space settings -> "Secrets"):
   - `MODEL_PATH` (optional if you commit a small model)
   - `MODEL_URL` (recommended: public URL to the model or use HF Hub download link)
   - `DEVICE=cpu`
   - `MAX_UPLOAD_MB=50`

4) Using the Hugging Face Hub to host the model (recommended):
   - Upload `models/deploy/model.pt` as an artifact to the HF Hub or create a small repo with the model and use the raw file URL.
   - If using private model storage, you can provide an `HF_TOKEN` secret and adapt `services/bootstrap.py` to use the HF API to download.

5) Once configured, Spaces will build the environment from `requirements.txt` and `runtime.txt`. Check build logs for errors (missing packages or version conflicts).

Notes:
- Spaces enforces storage limits; prefer using the HF Hub or external hosting for large checkpoints.
- For private models, use `HF_TOKEN` and modify `build.sh` or `services/bootstrap.py` to fetch with authentication.
