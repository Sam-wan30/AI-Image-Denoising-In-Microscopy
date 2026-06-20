(function () {
  const form = document.getElementById("denoise-form");
  const fileInput = document.getElementById("image-input");
  const denoiseBtn = document.getElementById("denoise-btn");
  const overlay = document.getElementById("loading-overlay");
  const alertBox = document.getElementById("alert-box");
  const statusDot = document.getElementById("status-dot");
  const statusText = document.getElementById("status-text");
  const origImg = document.getElementById("original-img");
  const denImg = document.getElementById("denoised-img");
  const origPlaceholder = document.getElementById("original-placeholder");
  const denPlaceholder = document.getElementById("denoised-placeholder");
  const origMeta = document.getElementById("original-meta");
  const denMeta = document.getElementById("denoised-meta");
  const metricsSection = document.getElementById("metrics-section");
  const psnrVal = document.getElementById("psnr-value");
  const ssimVal = document.getElementById("ssim-value");
  const psnrBar = document.getElementById("psnr-bar");
  const ssimBar = document.getElementById("ssim-bar");
  const downloadBtn = document.getElementById("download-btn");
  const resultsSection = document.getElementById("results-section");

  let downloadUrl = null;
  let previewUrl = null;

  function generateTimestampedFilename() {
    const now = new Date();
    const year = now.getFullYear();
    const month = String(now.getMonth() + 1).padStart(2, '0');
    const day = String(now.getDate()).padStart(2, '0');
    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');
    const seconds = String(now.getSeconds()).padStart(2, '0');
    return `denoised-image-${year}${month}${day}-${hours}${minutes}${seconds}.png`;
  }

  async function downloadImage(base64Data, filename) {
    try {
      // Convert base64 to blob
      const byteCharacters = atob(base64Data);
      const byteNumbers = new Array(byteCharacters.length);
      for (let i = 0; i < byteCharacters.length; i++) {
        byteNumbers[i] = byteCharacters.charCodeAt(i);
      }
      const byteArray = new Uint8Array(byteNumbers);
      const blob = new Blob([byteArray], { type: 'image/png' });

      // Create download link
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();

      // Cleanup
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Download failed:', err);
      showAlert('Download failed. Please try again.');
    }
  }

  async function downloadFromUrl(url, filename) {
    try {
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    } catch (err) {
      console.error('Download failed:', err);
      showAlert('Download failed. Please try again.');
    }
  }

  async function handleDownload(denoisedB64, downloadUrl, originalFilename) {
    try {
      const filename = generateTimestampedFilename();
      
      // Try to use the download URL from backend first
      if (downloadUrl && downloadUrl.startsWith('/api/download/')) {
        await downloadFromUrl(downloadUrl, filename);
      } else if (denoisedB64) {
        // Fallback to Base64 data if download URL is not available
        await downloadImage(denoisedB64, filename);
      } else {
        showAlert('No denoised image available for download.');
      }
    } catch (err) {
      console.error('Download failed:', err);
      showAlert('Download failed. Please try again.');
    }
  }

  downloadBtn.addEventListener('click', async (e) => {
    e.preventDefault();
    const currentDownloadUrl = downloadBtn.href;
    const denoisedSrc = denImg.src;
    const denoisedB64 = denoisedSrc ? denoisedSrc.replace('data:image/png;base64,', '') : null;
    
    await handleDownload(denoisedB64, currentDownloadUrl, 'denoised-image.png');
  });

  function showAlert(msg) {
    alertBox.textContent = msg;
    alertBox.classList.add("show");
  }

  function hideAlert() {
    alertBox.classList.remove("show");
  }

  function setLoading(on) {
    overlay.classList.toggle("active", on);
    denoiseBtn.disabled = on;
  }

  function setPreview(file) {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    if (!file) return;
    previewUrl = URL.createObjectURL(file);
    origImg.src = previewUrl;
    origImg.classList.remove("hidden");
    origPlaceholder.classList.add("hidden");
    origMeta.textContent = "—";
  }

  async function checkStatus() {
    try {
      const res = await fetch("/api/status");
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }
      const contentType = res.headers.get("content-type");
      if (!contentType || !contentType.includes("application/json")) {
        throw new Error("Response is not JSON");
      }
      const text = await res.text();
      if (!text.trim()) {
        throw new Error("Empty response body");
      }
      const data = JSON.parse(text);
      if (data.ready) {
        statusDot.classList.remove("offline");
        statusText.textContent = "Model online · " + (data.type || "U-Net");
      } else if (data.error) {
        statusDot.classList.add("offline");
        statusText.textContent = "Model offline";
        showAlert(data.error);
      } else {
        statusText.textContent = "Model loading…";
      }
    } catch (err) {
      console.error("Status check failed:", err);
      statusDot.classList.add("offline");
      statusText.textContent = "Server unreachable";
    }
  }

  fileInput.addEventListener("change", () => {
    hideAlert();
    const file = fileInput.files[0];
    setPreview(file);
    denImg.classList.add("hidden");
    denPlaceholder.classList.remove("hidden");
    denMeta.textContent = "—";
    metricsSection.classList.add("hidden");
    resultsSection.classList.add("hidden");
    downloadBtn.classList.add("hidden");
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    hideAlert();

    const file = fileInput.files[0];
    if (!file) {
      showAlert("Please choose a microscopy image first.");
      return;
    }

    const formData = new FormData();
    formData.append("image", file);
    formData.append("mode", document.getElementById("mode-select").value);

    setLoading(true);
    try {
      const res = await fetch("/api/denoise", { method: "POST", body: formData });
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }
      const contentType = res.headers.get("content-type");
      if (!contentType || !contentType.includes("application/json")) {
        throw new Error("Response is not JSON");
      }
      const text = await res.text();
      if (!text.trim()) {
        throw new Error("Empty response body");
      }
      const data = JSON.parse(text);

      if (!data.success) {
        throw new Error(data.error || "Denoising failed.");
      }

      origImg.src = "data:image/png;base64," + data.original_b64;
      denImg.src = "data:image/png;base64," + data.denoised_b64;
      origImg.classList.remove("hidden");
      denImg.classList.remove("hidden");
      origPlaceholder.classList.add("hidden");
      denPlaceholder.classList.add("hidden");

      const resLabel = data.width && data.height ? data.width + " × " + data.height : "—";
      origMeta.textContent = resLabel;
      denMeta.textContent = resLabel;

      psnrVal.textContent = data.psnr.toFixed(2);
      ssimVal.textContent = data.ssim.toFixed(3);
      psnrBar.style.width = Math.min(100, (data.psnr / 40) * 100) + "%";
      ssimBar.style.width = Math.min(100, data.ssim * 100) + "%";

      // Store download data for the download button
      downloadUrl = data.download_url;
      downloadBtn.classList.remove("hidden");
      metricsSection.classList.remove("hidden");
      resultsSection.classList.remove("hidden");
    } catch (err) {
      showAlert(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  });

  checkStatus();
  setInterval(checkStatus, 30000);
})();
