# API Documentation

**REST API Reference, Endpoints, and Usage Examples**

---

## Overview

NeuroScope provides a comprehensive REST API for programmatic image denoising. The API is designed to be simple, intuitive, and compatible with standard web service practices. This documentation covers all available endpoints, request/response formats, and best practices for integration.

### API Base URL
- **Local Development**: `http://localhost:5000`
- **Production**: `https://your-domain.com` (configure as needed)

### API Version
- **Current Version**: v1.0
- **Versioning**: URL-based versioning planned for future releases

---

## Authentication

### Current Status
The current implementation does not require authentication for local development. For production deployments, authentication should be implemented using one of the following methods:

### Recommended Authentication Methods

#### 1. API Key Authentication (Future)
```http
GET /api/denoise
X-API-Key: your-secret-api-key
```

#### 2. JWT Authentication (Future)
```http
GET /api/denoise
Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
```

---

## Endpoints

### Health Check

#### GET /health

Check the health status of the service and model readiness.

**Request**
```http
GET /health HTTP/1.1
Host: localhost:5000
```

**Response (200 OK)**
```json
{
  "status": "ok",
  "service": "neuroscope-denoising",
  "model": {
    "ready": true,
    "type": "ResidualMicroscopyUNet",
    "parameters": 42000000,
    "device": "cpu"
  }
}
```

**Response (200 OK - Model Not Ready)**
```json
{
  "status": "ok",
  "service": "neuroscope-denoising",
  "model": {
    "ready": false,
    "error": "Model file not found"
  }
}
```

**Use Cases**
- Health monitoring
- Load balancer health checks
- Service availability verification
- Model status monitoring

---

### Model Status

#### GET /api/status

Get detailed information about the loaded model and system status.

**Request**
```http
GET /api/status HTTP/1.1
Host: localhost:5000
```

**Response (200 OK)**
```json
{
  "ready": true,
  "error": null,
  "model_path": "models/deploy/model.pt",
  "device": "cpu",
  "type": "ResidualMicroscopyUNet",
  "parameters": 42000000,
  "epoch": 50,
  "val_psnr": 35.2,
  "val_ssim": 0.92
}
```

**Response (500 Internal Server Error)**
```json
{
  "ready": false,
  "error": "Status check failed"
}
```

**Use Cases**
- Model information retrieval
- System monitoring
- Debugging model loading issues
- Performance benchmarking

---

### Denoise Image

#### POST /api/denoise

Process a microscopy image through the denoising pipeline.

**Request**
```http
POST /api/denoise HTTP/1.1
Host: localhost:5000
Content-Type: multipart/form-data; boundary=----WebKitFormBoundary

------WebKitFormBoundary
Content-Disposition: form-data; name="image"; filename="noisy.png"
Content-Type: image/png

[binary image data]
------WebKitFormBoundary
Content-Disposition: form-data; name="mode"

auto
------WebKitFormBoundary--
```

**cURL Example**
```bash
curl -X POST http://localhost:5000/api/denoise \
  -F "image=@noisy.png" \
  -F "mode=auto"
```

**Python Example**
```python
import requests

with open('noisy.png', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/api/denoise',
        files={'image': f},
        data={'mode': 'auto'}
    )
result = response.json()
```

**Response (200 OK)**
```json
{
  "success": true,
  "psnr": 35.2,
  "ssim": 0.92,
  "download_url": "/api/download/denoised_noisy.png",
  "original_b64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "denoised_b64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "width": 512,
  "height": 512
}
```

**Response (400 Bad Request - No Image)**
```json
{
  "success": false,
  "error": "No image uploaded."
}
```

**Response (400 Bad Request - Unsupported Format)**
```json
{
  "success": false,
  "error": "Unsupported format. Allowed: .bmp, .jpeg, .jpg, .png, .tif, .tiff, .webp"
}
```

**Response (413 Payload Too Large)**
```json
{
  "success": false,
  "error": "File too large (max 50 MB)."
}
```

**Response (503 Service Unavailable - Model Not Ready)**
```json
{
  "success": false,
  "error": "Model is still loading. Try again shortly."
}
```

**Response (507 Insufficient Storage)**
```json
{
  "success": false,
  "error": "Insufficient memory. Try a smaller image."
}
```

**Response (500 Internal Server Error)**
```json
{
  "success": false,
  "error": "Internal server error."
}
```

**Parameters**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image` | File | Yes | Image file to denoise |
| `mode` | String | No | Denoising mode (auto, unet, salt_pepper, brightfield) |

**Denoising Modes**

| Mode | Description | Best For |
|------|-------------|----------|
| `auto` | Automatic mode selection | General use |
| `unet` | U-Net deep learning | Standard denoising |
| `salt_pepper` | Median filtering | Impulse noise |
| `brightfield` | Brightfield processing | Brightfield microscopy |

**Response Fields**

| Field | Type | Description |
|-------|------|-------------|
| `success` | Boolean | Operation success status |
| `psnr` | Float | Peak Signal-to-Noise Ratio (dB) |
| `ssim` | Float | Structural Similarity Index |
| `download_url` | String | URL to download denoised image |
| `original_b64` | String | Base64-encoded original image |
| `denoised_b64` | String | Base64-encoded denoised image |
| `width` | Integer | Image width in pixels |
| `height` | Integer | Image height in pixels |

**Use Cases**
- Web application image processing
- Batch processing workflows
- Integration with existing systems
- Automated image processing pipelines

---

### Download Result

#### GET /api/download/<filename>

Download a processed image file.

**Request**
```http
GET /api/download/denoised_noisy.png HTTP/1.1
Host: localhost:5000
```

**Response (200 OK)**
```
[binary image data]
Content-Type: image/png
Content-Disposition: attachment; filename="denoised_noisy.png"
```

**Response (404 Not Found)**
```json
{
  "error": "File not found."
}
```

**Parameters**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `filename` | String | Yes | Name of the file to download |

**Use Cases**
- Retrieving processed images
- Saving results to local storage
- Integration with download workflows

---

## Error Handling

### Error Response Format

All error responses follow a consistent format:

```json
{
  "success": false,
  "error": "Error message describing the issue"
}
```

### HTTP Status Codes

| Code | Description | Common Causes |
|------|-------------|---------------|
| `200 OK` | Success | Request processed successfully |
| `400 Bad Request` | Invalid request | Missing parameters, invalid file format |
| `404 Not Found` | Resource not found | File not found, invalid endpoint |
| `413 Payload Too Large` | File too large | Exceeds size limit |
| `500 Internal Server Error` | Server error | Unexpected server error |
| `503 Service Unavailable` | Service unavailable | Model not ready, server overloaded |
| `507 Insufficient Storage` | Memory error | Insufficient memory for processing |

### Error Handling Best Practices

#### Client-Side Error Handling

```python
import requests

try:
    response = requests.post(
        'http://localhost:5000/api/denoise',
        files={'image': open('noisy.png', 'rb')},
        data={'mode': 'auto'},
        timeout=30
    )
    response.raise_for_status()
    
    result = response.json()
    if result['success']:
        print(f"PSNR: {result['psnr']:.2f} dB")
        print(f"SSIM: {result['ssim']:.3f}")
    else:
        print(f"Error: {result['error']}")
        
except requests.exceptions.HTTPError as e:
    print(f"HTTP Error: {e}")
except requests.exceptions.Timeout:
    print("Request timed out")
except requests.exceptions.RequestException as e:
    print(f"Request Error: {e}")
```

#### Retry Logic

```python
import time
import requests

def denoise_with_retry(image_path, max_retries=3):
    for attempt in range(max_retries):
        try:
            with open(image_path, 'rb') as f:
                response = requests.post(
                    'http://localhost:5000/api/denoise',
                    files={'image': f},
                    data={'mode': 'auto'},
                    timeout=30
                )
                response.raise_for_status()
                return response.json()
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            wait_time = 2 ** attempt  # Exponential backoff
            time.sleep(wait_time)
```

---

## Rate Limiting

### Current Implementation
Rate limiting is not currently implemented but is recommended for production deployments.

### Recommended Rate Limiting

```python
# Example Flask rate limiting
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)

@app.route('/api/denoise', methods=['POST'])
@limiter.limit("10 per minute")
def api_denoise():
    # Implementation
    pass
```

### Rate Limiting Headers (Future)

```http
HTTP/1.1 200 OK
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1625097600
```

---

## Request/Response Examples

### Complete Workflow Example

#### Step 1: Check Service Health

```bash
curl http://localhost:5000/health
```

**Response**
```json
{
  "status": "ok",
  "service": "neuroscope-denoising",
  "model": {
    "ready": true,
    "type": "ResidualMicroscopyUNet",
    "parameters": 42000000,
    "device": "cpu"
  }
}
```

#### Step 2: Get Model Status

```bash
curl http://localhost:5000/api/status
```

**Response**
```json
{
  "ready": true,
  "error": null,
  "model_path": "models/deploy/model.pt",
  "device": "cpu",
  "type": "ResidualMicroscopyUNet",
  "parameters": 42000000,
  "epoch": 50,
  "val_psnr": 35.2,
  "val_ssim": 0.92
}
```

#### Step 3: Denoise Image

```bash
curl -X POST http://localhost:5000/api/denoise \
  -F "image=@noisy.png" \
  -F "mode=unet"
```

**Response**
```json
{
  "success": true,
  "psnr": 35.2,
  "ssim": 0.92,
  "download_url": "/api/download/denoised_noisy.png",
  "original_b64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "denoised_b64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "width": 512,
  "height": 512
}
```

#### Step 4: Download Result

```bash
curl -O http://localhost:5000/api/download/denoised_noisy.png
```

### Batch Processing Example

```python
import requests
import os
from pathlib import Path

def batch_denoise(input_dir, output_dir, mode='auto'):
    """Process all images in a directory"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for image_file in input_path.glob('*.png'):
        print(f"Processing {image_file.name}...")
        
        try:
            with open(image_file, 'rb') as f:
                response = requests.post(
                    'http://localhost:5000/api/denoise',
                    files={'image': f},
                    data={'mode': mode},
                    timeout=60
                )
            
            result = response.json()
            
            if result['success']:
                # Save denoised image
                output_file = output_path / f"denoised_{image_file.name}"
                download_url = f"http://localhost:5000{result['download_url']}"
                download_response = requests.get(download_url)
                
                with open(output_file, 'wb') as out_f:
                    out_f.write(download_response.content)
                
                print(f"✓ {image_file.name}: PSNR={result['psnr']:.2f}, SSIM={result['ssim']:.3f}")
            else:
                print(f"✗ {image_file.name}: {result['error']}")
                
        except Exception as e:
            print(f"✗ {image_file.name}: {str(e)}")

# Usage
batch_denoise('input_images', 'output_images', mode='auto')
```

---

## Integration Examples

### Python Integration

#### Simple Integration

```python
import requests
import base64
from PIL import Image
from io import BytesIO

class NeuroScopeClient:
    def __init__(self, base_url='http://localhost:5000'):
        self.base_url = base_url
    
    def denoise_image(self, image_path, mode='auto'):
        """Denoise an image"""
        url = f'{self.base_url}/api/denoise'
        
        with open(image_path, 'rb') as f:
            response = requests.post(
                url,
                files={'image': f},
                data={'mode': mode},
                timeout=60
            )
        
        response.raise_for_status()
        return response.json()
    
    def denoise_image_pil(self, image_path, mode='auto'):
        """Denoise image and return PIL Image"""
        result = self.denoise_image(image_path, mode)
        
        # Convert base64 to PIL Image
        image_data = base64.b64decode(result['denoised_b64'])
        image = Image.open(BytesIO(image_data))
        
        return image, result
    
    def check_health(self):
        """Check service health"""
        url = f'{self.base_url}/health'
        response = requests.get(url)
        return response.json()
    
    def get_model_status(self):
        """Get model status"""
        url = f'{self.base_url}/api/status'
        response = requests.get(url)
        return response.json()

# Usage
client = NeuroScopeClient()

# Check health
health = client.check_health()
print(f"Service status: {health['status']}")

# Denoise image
denoised_image, metrics = client.denoise_image_pil('noisy.png', mode='unet')
print(f"PSNR: {metrics['psnr']:.2f} dB")
print(f"SSIM: {metrics['ssim']:.3f}")

# Save result
denoised_image.save('denoised.png')
```

### JavaScript Integration

#### Browser Integration

```javascript
class NeuroScopeAPI {
    constructor(baseUrl = 'http://localhost:5000') {
        this.baseUrl = baseUrl;
    }
    
    async denoiseImage(imageFile, mode = 'auto') {
        const formData = new FormData();
        formData.append('image', imageFile);
        formData.append('mode', mode);
        
        const response = await fetch(`${this.baseUrl}/api/denoise`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    }
    
    async checkHealth() {
        const response = await fetch(`${this.baseUrl}/health`);
        return await response.json();
    }
    
    async getModelStatus() {
        const response = await fetch(`${this.baseUrl}/api/status`);
        return await response.json();
    }
    
    base64ToImage(base64String) {
        const imageBytes = atob(base64String);
        const byteArray = new Uint8Array(imageBytes.length);
        
        for (let i = 0; i < imageBytes.length; i++) {
            byteArray[i] = imageBytes.charCodeAt(i);
        }
        
        const blob = new Blob([byteArray], { type: 'image/png' });
        return URL.createObjectURL(blob);
    }
}

// Usage
const api = new NeuroScopeAPI();

// Denoise image from file input
document.getElementById('imageInput').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    
    try {
        const result = await api.denoiseImage(file, 'auto');
        
        // Display results
        const denoisedUrl = api.base64ToImage(result.denoised_b64);
        document.getElementById('resultImage').src = denoisedUrl;
        
        document.getElementById('psnr').textContent = result.psnr.toFixed(2);
        document.getElementById('ssim').textContent = result.ssim.toFixed(3);
        
    } catch (error) {
        console.error('Denoising failed:', error);
        alert('Denoising failed: ' + error.message);
    }
});
```

### Node.js Integration

```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

class NeuroScopeClient {
    constructor(baseUrl = 'http://localhost:5000') {
        this.baseUrl = baseUrl;
        this.client = axios.create({
            baseURL: baseUrl,
            timeout: 60000
        });
    }
    
    async denoiseImage(imagePath, mode = 'auto') {
        const form = new FormData();
        form.append('image', fs.createReadStream(imagePath));
        form.append('mode', mode);
        
        const response = await this.client.post('/api/denoise', form, {
            headers: form.getHeaders()
        });
        
        return response.data;
    }
    
    async downloadImage(filename, outputPath) {
        const response = await this.client.get(`/api/download/${filename}`, {
            responseType: 'arraybuffer'
        });
        
        fs.writeFileSync(outputPath, response.data);
    }
    
    async checkHealth() {
        const response = await this.client.get('/health');
        return response.data;
    }
    
    async getModelStatus() {
        const response = await this.client.get('/api/status');
        return response.data;
    }
}

// Usage
async function main() {
    const client = new NeuroScopeClient();
    
    try {
        // Check health
        const health = await client.checkHealth();
        console.log('Service status:', health.status);
        
        // Denoise image
        const result = await client.denoiseImage('noisy.png', 'auto');
        console.log('PSNR:', result.psnr.toFixed(2));
        console.log('SSIM:', result.ssim.toFixed(3));
        
        // Download result
        const filename = result.download_url.split('/').pop();
        await client.downloadImage(filename, 'denoised.png');
        console.log('Image saved successfully');
        
    } catch (error) {
        console.error('Error:', error.message);
    }
}

main();
```

---

## Performance Considerations

### Request Size Limits

| Parameter | Default | Maximum |
|-----------|---------|---------|
| **File Size** | 50MB | 100MB (configurable) |
| **Image Resolution** | 8192×8192 | 16384×16384 (with sufficient memory) |
| **Request Timeout** | 60 seconds | 300 seconds (configurable) |

### Performance Optimization

#### Client-Side Optimization

```python
# Use connection pooling
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

session = requests.Session()
retry = Retry(total=3, backoff_factor=1)
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)

# Use session for all requests
response = session.post('http://localhost:5000/api/denoise', ...)
```

#### Batch Processing Optimization

```python
# Process multiple images concurrently
import concurrent.futures
import requests

def denoise_single(image_path):
    with open(image_path, 'rb') as f:
        response = requests.post(
            'http://localhost:5000/api/denoise',
            files={'image': f},
            timeout=60
        )
    return response.json()

# Process images in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    image_paths = ['img1.png', 'img2.png', 'img3.png']
    results = list(executor.map(denoise_single, image_paths))
```

---

## Testing

### Testing with cURL

```bash
# Test health endpoint
curl http://localhost:5000/health

# Test model status
curl http://localhost:5000/api/status

# Test denoising
curl -X POST http://localhost:5000/api/denoise \
  -F "image=@test_image.png" \
  -F "mode=auto"

# Test download
curl -O http://localhost:5000/api/download/denoised_test_image.png
```

### Testing with Python

```python
import requests
import unittest

class TestNeuroScopeAPI(unittest.TestCase):
    def setUp(self):
        self.base_url = 'http://localhost:5000'
    
    def test_health(self):
        response = requests.get(f'{self.base_url}/health')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'ok')
    
    def test_model_status(self):
        response = requests.get(f'{self.base_url}/api/status')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('ready', data)
    
    def test_denoise(self):
        with open('test_image.png', 'rb') as f:
            response = requests.post(
                f'{self.base_url}/api/denoise',
                files={'image': f},
                data={'mode': 'auto'}
            )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data['success'])
        self.assertIn('psnr', data)
        self.assertIn('ssim', data)

if __name__ == '__main__':
    unittest.main()
```

---

## Troubleshooting API Issues

### Common API Problems

#### Issue 1: Connection Refused

**Symptoms**: `ConnectionRefusedError`

**Solutions**:
- Verify server is running: `python application.py`
- Check port is correct: Default is 5000
- Check firewall settings
- Verify URL is correct

#### Issue 2: Timeout Errors

**Symptoms**: `ReadTimeout` or `ConnectTimeout`

**Solutions**:
- Increase timeout value
- Check server performance
- Reduce image size
- Use batch processing for large jobs

#### Issue 3: Model Not Ready

**Symptoms**: 503 error with "Model is still loading"

**Solutions**:
- Wait for model to load (check status endpoint)
- Verify model file exists
- Check model file integrity
- Restart server if needed

#### Issue 4: File Upload Failures

**Symptoms**: 400 error with file upload issues

**Solutions**:
- Check file size limit
- Verify file format is supported
- Ensure file is not corrupted
- Check network connectivity

---

<div align="center">

**RESTful API design enables seamless integration with existing systems and workflows**

[⬆ Back to Wiki Home](Home) | [← User Guide](User-Guide) | [Development Guide](Development-Guide) →

</div>
