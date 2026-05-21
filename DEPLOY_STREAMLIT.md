Streamlit Cloud deployment guide

1) Push the `deploy-ready` branch to GitHub (already done).

2) On Streamlit Cloud:
   - Create a new app -> "From GitHub" -> select your repository and branch `deploy-ready`.
   - Set the app entry point to `app.py`.
   - In "Secrets & config vars" add the following keys (see `.streamlit/secrets.example`):
     - `MODEL_PATH` (e.g. `models/deploy/model.pt`) OR `MODEL_URL` (public URL)
     - `DEVICE` (set to `cpu` on free hosts)
     - `MAX_UPLOAD_MB` (e.g. `50`)
     - `SECRET_KEY` (a random secret)

3) If your model is large:
   - Upload `models/deploy/model.pt` to a publicly accessible URL (S3, GDrive with direct link, or Hugging Face Hub) and set `MODEL_URL` to that URL.
   - The app's `services/bootstrap.py` will attempt to download `MODEL_URL` to `models/deploy/model.pt` on first run.

4) Deploy and monitor logs in the Streamlit Cloud dashboard. Use the "Logs" tab to view download and load messages.

Troubleshooting:
- If the app fails to load due to model path errors, check that `MODEL_PATH` exists or `MODEL_URL` is set.
- For large models (>100MB) consider hosting the model on the Hugging Face Hub and using a direct download URL.
