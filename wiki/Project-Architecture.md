# Project Architecture

**System Design, Data Flow, and Implementation Details**

---

## System Architecture Overview

NeuroScope implements a modular, production-ready architecture designed for scalability, maintainability, and performance. The system follows a layered architecture pattern with clear separation of concerns across presentation, application, business logic, and data layers.

### High-Level Architecture

```mermaid
graph TB
    subgraph "Presentation Layer"
        A[Flask Web App]
        B[Streamlit UI]
        C[CLI Tool]
    end
    
    subgraph "Application Layer"
        D[REST API Endpoints]
        E[Request Handlers]
        F[Authentication Middleware]
    end
    
    subgraph "Service Layer"
        G[Denoiser Service]
        H[Model Manager]
        I[Metrics Calculator]
        J[Image Processor]
    end
    
    subgraph "ML Engine Layer"
        K[U-Net Models]
        L[Training Pipeline]
        M[Inference Engine]
        N[ONNX Runtime]
    end
    
    subgraph "Data Layer"
        O[Image Storage]
        P[Model Checkpoints]
        Q[Training Dataset]
        R[Configuration]
    end
    
    A --> D
    B --> D
    C --> E
    D --> G
    E --> G
    F --> D
    G --> H
    H --> K
    G --> I
    J --> G
    K --> M
    L --> K
    M --> N
    L --> Q
    J --> O
    H --> P
    G --> R
    
    style A fill:#4CAF50
    style B fill:#2196F3
    style C fill:#FF9800
    style K fill:#9C27B0
```

### Component Interaction

```mermaid
sequenceDiagram
    participant User
    participant WebApp
    participant API
    participant Service
    participant Model
    participant Storage
    
    User->>WebApp: Upload Image
    WebApp->>API: POST /api/denoise
    API->>Service: process_image()
    Service->>Model: load_model()
    Model-->>Service: model_instance
    Service->>Storage: save_input()
    Service->>Model: inference()
    Model-->>Service: denoised_output
    Service->>Storage: save_output()
    Service->>Service: calculate_metrics()
    Service-->>API: results + metrics
    API-->>WebApp: JSON response
    WebApp-->>User: Display results
```

---

## Data Flow Architecture

### Image Processing Pipeline

```mermaid
graph LR
    subgraph "Input Processing"
        A[Raw Image] --> B[Grayscale Conversion]
        B --> C[Resize to 256x256]
        C --> D[Normalize to 0-1]
    end
    
    subgraph "Model Inference"
        D --> E[U-Net Encoder]
        E --> F[Bottleneck]
        F --> G[U-Net Decoder]
        G --> H[Skip Connections]
    end
    
    subgraph "Output Processing"
        H --> I[Postprocessing]
        I --> J[Resize to Original]
        J --> K[Denormalize]
        K --> L[Quality Metrics]
    end
    
    L --> M[Final Output]
    
    style B fill:#E3F2FD
    style E fill:#F3E5F5
    style I fill:#E8F5E9
```

### Training Data Flow

```mermaid
graph TB
    subgraph "Data Loading"
        A[Raw Dataset] --> B[Pair Matching]
        B --> C[Validation]
    end
    
    subgraph "Preprocessing"
        C --> D[Grayscale Conversion]
        D --> E[Resize & Normalize]
        E --> F[Data Augmentation]
    end
    
    subgraph "Training Loop"
        F --> G[Batch Loading]
        G --> H[Forward Pass]
        H --> I[Loss Calculation]
        I --> J[Backward Pass]
        J --> K[Optimizer Step]
    end
    
    subgraph "Validation"
        K --> L[Validation Set]
        L --> M[Metrics Calculation]
        M --> N[Early Stopping]
        N --> O[Model Checkpoint]
    end
    
    style A fill:#FFF3E0
    style H fill:#E8F5E9
    style M fill:#E3F2FD
```

---

## Model Pipeline

### U-Net Architecture

```mermaid
graph TB
    subgraph "Encoder Path"
        A[Input 1x256x256] --> B[DoubleConv 64]
        B --> C[MaxPool]
        C --> D[DoubleConv 128]
        D --> E[MaxPool]
        E --> F[DoubleConv 256]
        F --> G[MaxPool]
        G --> H[DoubleConv 512]
        H --> I[MaxPool]
        I --> J[DoubleConv 1024]
    end
    
    subgraph "Decoder Path"
        J --> K[UpConv 512]
        K --> L[Concat Skip]
        L --> M[DoubleConv 512]
        M --> N[UpConv 256]
        N --> O[Concat Skip]
        O --> P[DoubleConv 256]
        P --> Q[UpConv 128]
        Q --> R[Concat Skip]
        R --> S[DoubleConv 128]
        S --> T[UpConv 64]
        T --> U[Concat Skip]
        U --> V[DoubleConv 64]
    end
    
    subgraph "Output"
        V --> W[Output Conv 1x1]
        W --> X[Output 1x256x256]
    end
    
    style A fill:#4CAF50
    style X fill:#2196F3
```

### Model Variants

| Architecture | Parameters | Key Features | Use Case |
|--------------|------------|--------------|----------|
| **Standard U-Net** | ~31M | Classic encoder-decoder with skip connections | General-purpose denoising |
| **Enhanced U-Net** | ~38M | Additional residual blocks in encoder | Better feature extraction |
| **Residual U-Net** | ~42M | Residual learning with deeper architecture | Complex noise patterns |

---

## Training Workflow

### Training Pipeline Architecture

```mermaid
graph TB
    subgraph "Initialization"
        A[Load Configuration] --> B[Initialize Model]
        B --> C[Setup Optimizer]
        C --> D[Configure LR Scheduler]
    end
    
    subgraph "Data Preparation"
        D --> E[Load Dataset]
        E --> F[Split Train/Val]
        F --> G[Create DataLoaders]
    end
    
    subgraph "Training Loop"
        G --> H[For Each Epoch]
        H --> I[Train Epoch]
        I --> J[Validate Epoch]
        J --> K[Calculate Metrics]
        K --> L[Update LR Scheduler]
    end
    
    subgraph "Model Management"
        L --> M[Check Performance]
        M --> N[Save Best Model]
        M --> O[Early Stopping]
        N --> P[Continue Training]
        O --> Q[Stop Training]
    end
    
    subgraph "Logging"
        K --> R[TensorBoard Logging]
        P --> S[Sample Visualization]
        Q --> T[Final Metrics Report]
    end
    
    style A fill:#FFF3E0
    style I fill:#E8F5E9
    style N fill:#E3F2FD
```

### Training Components

#### Dataset Loading
- **CAREDataset**: Paired noisy/clean image loader with automatic matching
- **Data Augmentation**: Random flips, rotations, and color transformations
- **Batch Processing**: Configurable batch sizes with memory pinning
- **Validation Split**: Automatic train/validation dataset splitting

#### Loss Function
- **Combined Loss**: 0.7 × L1 + 0.3 × (1 - SSIM)
- **L1 Component**: Pixel-level accuracy for noise removal
- **SSIM Component**: Structural similarity for preserving fine details
- **Balanced Training**: Prevents over-smoothing while maintaining noise reduction

#### Optimization Strategy
- **Optimizer**: Adam with configurable learning rate
- **LR Scheduling**: ReduceLROnPlateau based on validation PSNR
- **Early Stopping**: Patience-based stopping to prevent overfitting
- **Gradient Handling**: Optional gradient clipping for training stability

---

## Inference Workflow

### Inference Pipeline Architecture

```mermaid
graph TB
    subgraph "Request Processing"
        A[Receive Request] --> B[Validate Input]
        B --> C[Load Image]
        C --> D[Preprocess]
    end
    
    subgraph "Model Execution"
        D --> E[Load Model]
        E --> F[Warm-up Check]
        F --> G[Run Inference]
    end
    
    subgraph "Post-processing"
        G --> H[Postprocess Output]
        H --> I[Calculate Metrics]
        I --> J[Generate Comparison]
    end
    
    subgraph "Response"
        J --> K[Encode Response]
        K --> L[Return JSON]
        L --> M[Save Files]
    end
    
    style A fill:#4CAF50
    style G fill:#9C27B0
    style L fill:#2196F3
```

### Inference Service Architecture

#### Thread-Safe Model Loading
- **Lazy Loading**: Model loaded only on first request
- **Singleton Pattern**: Single model instance shared across requests
- **Thread Safety**: Lock-based concurrent access control
- **Warm-up**: Optional pre-warming for reduced latency

#### Processing Modes
- **U-Net Mode**: Deep learning-based denoising
- **Auto Mode**: Automatic mode selection based on image analysis
- **Salt-Pepper**: Traditional median filtering for impulse noise
- **Brightfield**: Object mask processing for brightfield microscopy

#### Error Handling
- **Validation**: Input validation and sanitization
- **Fallback**: Graceful degradation on model failures
- **Logging**: Comprehensive error tracking and debugging
- **User Messages**: Clear error messages without exposing internals

---

## Service Layer Architecture

### Denoiser Service

```mermaid
classDiagram
    class DenoiserService {
        -model: Any
        -onnx_session: Any
        -model_lock: Lock
        -load_error: str
        +is_ready: bool
        +status: dict
        +warm_up(): void
        +process_upload(): dict
        +denoise(): np.ndarray
    }
    
    class ModelManager {
        +load_model(): Any
        +detect_type(): str
        +validate_checkpoint(): bool
    }
    
    class MetricsCalculator {
        +calculate_psnr(): float
        +calculate_ssim(): float
        +calculate_all(): dict
    }
    
    class ImageProcessor {
        +preprocess(): Tensor
        +postprocess(): np.ndarray
        +resize(): np.ndarray
        +normalize(): np.ndarray
    }
    
    DenoiserService --> ModelManager
    DenoiserService --> MetricsCalculator
    DenoiserService --> ImageProcessor
```

### Bootstrap Service

```mermaid
graph TB
    subgraph "Startup Process"
        A[Application Start] --> B[Create Directories]
        B --> C[Load Configuration]
        C --> D[Initialize Services]
        D --> E[Health Check]
    end
    
    subgraph "Directory Management"
        B --> F[Upload Directory]
        B --> G[Output Directory]
        B --> H[Model Directory]
    end
    
    subgraph "Service Initialization"
        D --> I[Denoiser Service]
        D --> J[Model Manager]
        D --> K[Metrics Calculator]
    end
    
    subgraph "Health Monitoring"
        E --> L[Model Status]
        E --> M[Resource Check]
        E --> N[API Endpoints]
    end
    
    style A fill:#4CAF50
    style I fill:#9C27B0
    style L fill:#2196F3
```

---

## API Architecture

### REST API Design

```mermaid
graph TB
    subgraph "API Gateway"
        A[Flask Application]
        B[Route Handlers]
        C[Middleware]
    end
    
    subgraph "Endpoints"
        B --> D[GET /health]
        B --> E[GET /api/status]
        B --> F[POST /api/denoise]
        B --> G[GET /api/download]
    end
    
    subgraph "Request Processing"
        D --> H[Health Check]
        E --> I[Status Query]
        F --> J[Image Processing]
        G --> K[File Download]
    end
    
    subgraph "Response Handling"
        H --> L[JSON Response]
        I --> M[JSON Response]
        J --> N[JSON with Images]
        K --> O[Binary File]
    end
    
    style A fill:#2196F3
    style J fill:#9C27B0
    style N fill:#4CAF50
```

### API Layer Components

#### Route Handlers
- **Health Endpoint**: Service availability and model readiness
- **Status Endpoint**: Detailed model information and loading state
- **Denoise Endpoint**: Image processing with multiple modes
- **Download Endpoint**: Secure file download with validation

#### Middleware
- **Error Handling**: Global exception handling and logging
- **Request Validation**: Input sanitization and format validation
- **Rate Limiting**: Configurable request rate limiting
- **CORS Headers**: Cross-origin resource sharing configuration

---

## Storage Architecture

### File System Organization

```mermaid
graph TB
    subgraph "Project Root"
        A[AI Image Denoising In Microscopy]
    end
    
    subgraph "Runtime Directories"
        A --> B[uploads/]
        A --> C[outputs/]
        A --> D[logs/]
    end
    
    subgraph "Storage"
        B --> E[Input Images]
        C --> F[Processed Images]
        D --> G[TensorBoard Logs]
        D --> H[Training Artifacts]
    end
    
    subgraph "Model Storage"
        A --> I[models/]
        I --> J[deploy/]
        I --> K[checkpoints/]
    end
    
    subgraph "Data Storage"
        A --> L[data/]
        L --> M[train/]
        M --> N[noisy/]
        M --> O[clean/]
    end
    
    style A fill:#4CAF50
    style E fill:#E3F2FD
    style F fill:#E8F5E9
    style J fill:#F3E5F5
```

### Storage Management

#### Runtime Storage
- **Upload Directory**: Temporary storage for input images
- **Output Directory**: Storage for processed results
- **Automatic Cleanup**: Configurable file retention policies
- **Access Control**: Secure file access with validation

#### Model Storage
- **Checkpoints**: Training checkpoints with metadata
- **Deployment Models**: Optimized models for production
- **Version Control**: Model versioning and rollback capability
- **Compression**: Optional model compression for storage efficiency

---

## Performance Architecture

### Optimization Strategies

```mermaid
graph TB
    subgraph "Model Optimization"
        A[GroupNorm] --> B[Small Batch Stability]
        C[Inplace Operations] --> D[Memory Efficiency]
        E[ONNX Runtime] --> F[Deployment Speed]
    end
    
    subgraph "Training Optimization"
        G[LR Scheduling] --> H[Adaptive Learning]
        I[Early Stopping] --> J[Overfitting Prevention]
        K[Mixed Precision] --> L[GPU Acceleration]
    end
    
    subgraph "Inference Optimization"
        M[Model Caching] --> N[Reduced Loading]
        O[Thread Safety] --> P[Concurrent Processing]
        Q[Image Chunking] --> R[Memory Management]
    end
    
    subgraph "System Optimization"
        S[Memory Pinning] --> T[Data Loading Speed]
        U[Garbage Collection] --> V[Resource Cleanup]
        W[Stream Processing] --> X[Large File Handling]
    end
    
    style A fill:#9C27B0
    style G fill:#4CAF50
    style M fill:#2196F3
    style S fill:#FF9800
```

### Performance Metrics

#### Training Performance
- **Batch Processing**: 8-16 images per batch (configurable)
- **GPU Utilization**: 80-95% on modern GPUs
- **Memory Efficiency**: GroupNorm reduces memory by 30%
- **Training Speed**: ~2-3 sec/batch on V100 GPU

#### Inference Performance
- **Latency**: ~100ms per image (CPU), ~20ms (GPU)
- **Throughput**: ~10 images/sec (CPU), ~50 images/sec (GPU)
- **Memory Usage**: ~2GB RAM for single inference
- **Scalability**: Concurrent request handling with thread safety

---

## Security Architecture

### Security Layers

```mermaid
graph TB
    subgraph "Input Security"
        A[File Type Validation] --> B[Size Limits]
        B --> C[Content Validation]
        C --> D[Path Sanitization]
    end
    
    subgraph "Process Security"
        D --> E[Memory Limits]
        E --> F[Timeout Protection]
        F --> G[Exception Isolation]
    end
    
    subgraph "Output Security"
        G --> H[Response Sanitization]
        H --> I[File Access Control]
        I --> J[Error Message Filtering]
    end
    
    subgraph "Environment Security"
        J --> K[Secrets Management]
        K --> L[Secure Defaults]
        L --> M[Configuration Validation]
    end
    
    style A fill:#F44336
    style E fill:#FF9800
    style H fill:#4CAF50
    style K fill:#2196F3
```

---

## Deployment Architecture

### Deployment Options

```mermaid
graph TB
    subgraph "Local Development"
        A[Python venv] --> B[Flask Dev Server]
        B --> C[Streamlit UI]
    end
    
    subgraph "Production Deployment"
        D[Gunicorn WSGI] --> E[Flask Production]
        E --> F[ONNX Runtime]
    end
    
    subgraph "Container Deployment"
        G[Docker Container] --> H[Port Mapping]
        H --> I[Volume Mounts]
    end
    
    subgraph "Cloud Deployment"
        J[Render.com] --> K[Auto-scaling]
        K --> L[Load Balancing]
    end
    
    style A fill:#4CAF50
    style D fill:#2196F3
    style G fill:#FF9800
    style J fill:#9C27B0
```

---

## Monitoring & Logging

### Logging Architecture

```mermaid
graph TB
    subgraph "Application Logging"
        A[Debug Logs] --> B[Training Logs]
        B --> C[Error Logs]
    end
    
    subgraph "Performance Logging"
        C --> D[Latency Metrics]
        D --> E[Memory Usage]
        E --> F[Throughput Stats]
    end
    
    subgraph "Model Logging"
        F --> G[Training Curves]
        G --> H[Validation Metrics]
        H --> I[Checkpoint Events]
    end
    
    subgraph "System Logging"
        I --> J[Health Checks]
        J --> K[Resource Monitoring]
        K --> L[Security Events]
    end
    
    style A fill:#4CAF50
    style D fill:#2196F3
    style G fill:#FF9800
    style J fill:#9C27B0
```

---

<div align="center">

**Architecture designed for scalability, maintainability, and production deployment**

[⬆ Back to Wiki Home](Home) | [Dataset Documentation](Dataset-Documentation) →

</div>
