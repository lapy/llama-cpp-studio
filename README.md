# llama.cpp Studio

A professional AI model management platform for llama.cpp models and versions, designed for modern AI workflows with comprehensive GPU support (NVIDIA CUDA, AMD Vulkan/ROCm, Metal, OpenBLAS).

## Features

### Model Management
- **Search & Download**: Search HuggingFace for GGUF models with comprehensive metadata and size information for each quantization
- **Multi-Quantization Support**: Download and manage multiple quantizations of the same model
- **Model Library**: Manage downloaded models with start/stop/delete functionality
- **Smart Configuration**: Auto-generate optimal llama.cpp parameters based on GPU capabilities
- **VRAM Estimation**: Real-time VRAM usage estimation with warnings for memory constraints
- **Metadata Extraction**: Rich model information including parameters, architecture, license, tags, and more

### llama.cpp Version Management
- **Release Installation**: Download and install pre-built binaries from GitHub releases
- **Source Building**: Build from source with optional patches from GitHub PRs
- **Custom Build Configuration**: Customize GPU backends (CUDA, Vulkan, Metal, OpenBLAS), build type, and compiler flags
- **Update Checking**: Check for updates to both releases and source code
- **Version Management**: Install, update, and delete multiple llama.cpp versions
- **Build Validation**: Automatic validation of built binaries to ensure they work correctly

### GPU Support
- **Multi-GPU Support**: Automatic detection and configuration for NVIDIA, AMD, and other GPUs
- **NVIDIA CUDA**: Full support for CUDA compute capabilities, flash attention, and multi-GPU
- **AMD GPU Support**: Vulkan and ROCm support for AMD GPUs
- **Apple Metal**: Support for Apple Silicon GPUs
- **OpenBLAS**: CPU acceleration with optimized BLAS routines
- **VRAM Monitoring**: Real-time GPU memory usage and temperature monitoring
- **NVLink Detection**: Automatic detection of NVLink connections and topology analysis

### Multi-Model Serving
- **Concurrent Execution**: Run multiple models simultaneously via llama-swap proxy
- **OpenAI-Compatible API**: Standard API format for easy integration
- **Port 2000**: All models served through a single unified endpoint
- **Automatic Lifecycle Management**: Seamless starting/stopping of models

### Web Interface
- **Modern UI**: Vue.js 3 with PrimeVue components
- **Real-time Updates**: WebSocket-based progress tracking and system monitoring
- **Responsive Design**: Works on desktop and mobile devices
- **System Status**: CPU, memory, disk, and GPU monitoring
- **Dark Mode**: Built-in theme support

## Quick Start

### Using Docker Compose

1. Clone the repository:
```bash
git clone <repository-url>
cd llama-cpp-studio
```

2. Start the application:
```bash
# CPU-only mode
docker-compose -f docker-compose.cpu.yml up -d

# GPU mode (NVIDIA)
docker-compose up -d

# Vulkan/AMD GPU mode
docker-compose -f docker-compose.vulkan.yml up -d

# ROCm mode
docker-compose -f docker-compose.rocm.yml up -d
```

3. Access the web interface at `http://localhost:8080`

### Manual Docker Build

1. Build the image:
```bash
docker build -t llama-cpp-studio .
```

2. Run the container:
```bash
# With GPU support
docker run -d \
  --name llama-cpp-studio \
  --gpus all \
  -p 8080:8080 \
  -v ./data:/app/data \
  llama-cpp-studio

# CPU-only
docker run -d \
  --name llama-cpp-studio \
  -p 8080:8080 \
  -v ./data:/app/data \
  llama-cpp-studio
```

## Configuration

### Environment Variables
- `CUDA_VISIBLE_DEVICES`: GPU device selection (default: all, set to "" for CPU-only)
- `PORT`: Web server port (default: 8080)
- `HUGGINGFACE_API_KEY`: HuggingFace API token for model search and download (optional)

### Volume Mounts
- `/app/data`: Persistent storage for models, configurations, and database

### HuggingFace API Key

To enable model search and download functionality, you need to set your HuggingFace API key. You can do this in several ways:

#### Option 1: Docker Compose Environment Variable
Uncomment and set the token in your `docker-compose.yml`:
```yaml
environment:
  - CUDA_VISIBLE_DEVICES=all
  - HUGGINGFACE_API_KEY=your_huggingface_token_here
```

#### Option 2: .env File
Create a `.env` file in your project root:
```bash
HUGGINGFACE_API_KEY=your_huggingface_token_here
```

Then uncomment the `env_file` section in `docker-compose.yml`:
```yaml
env_file:
  - .env
```

#### Option 3: System Environment Variable
Set the environment variable before running Docker Compose:
```bash
export HUGGINGFACE_API_KEY=your_huggingface_token_here
docker-compose up -d
```

#### Getting Your HuggingFace Token
1. Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Create a new token with "Read" permissions
3. Copy the token and use it in one of the methods above

**Note**: When the API key is set via environment variable, it cannot be modified through the web UI for security reasons.

### GPU Requirements
- **NVIDIA**: NVIDIA GPU with CUDA support, NVIDIA Container Toolkit installed
- **AMD**: AMD GPU with Vulkan/ROCm drivers
- **Apple**: Apple Silicon with Metal support
- **CPU**: OpenBLAS for CPU acceleration (included in Docker image)
- Minimum 8GB VRAM recommended for most models

## Usage

### 1. Model Management

#### Search Models
- Use the search bar to find GGUF models on HuggingFace
- Filter by tags, parameters, or model name
- View comprehensive metadata including downloads, likes, tags, and file sizes

#### Download Models
- Click download on any quantization to start downloading
- Multiple quantizations of the same model are automatically grouped
- Progress tracking with real-time updates via WebSocket

#### Configure Models
- Set llama.cpp parameters or use Smart Auto for optimal settings
- View VRAM estimation before starting
- Configure context size, batch sizes, temperature, and more

#### Run Models
- Start/stop models with one click
- Multiple models can run simultaneously
- View running instances and resource usage

### 2. llama.cpp Versions

#### Check Updates
- View available releases and source updates
- See commit history and release notes

#### Install Release
- Download pre-built binaries from GitHub
- Automatic verification and installation

#### Build from Source
- Compile from source with custom configuration
- Select GPU backends (CUDA, Vulkan, Metal, OpenBLAS)
- Configure build type (Release, Debug, RelWithDebInfo)
- Add custom CMake flags and compiler options
- Apply patches from GitHub PRs
- Automatic validation of built binaries

#### Manage Versions
- Delete old versions to free up space
- View installation details and build configuration

### 3. System Monitoring
- **Overview**: CPU, memory, disk, and GPU usage
- **GPU Details**: Individual GPU information and utilization
- **Running Instances**: Active model instances with resource usage
- **WebSocket**: Real-time updates for all metrics

## Multi-Model Serving

llama-cpp-studio uses llama-swap to serve multiple models simultaneously on port 2000.

### Starting Models

Simply start any model from the Model Library. All models run on port 2000 simultaneously.

### OpenAI-Compatible API

```bash
curl http://localhost:2000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3-2-1b-instruct-iq2-xs",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Model names are shown in System Status after starting a model.

### Features

- Multiple models run concurrently
- No loading time - instant switching between models
- Standard OpenAI API format
- Automatic lifecycle management
- Single unified endpoint

### Troubleshooting

- Check available models: `http://localhost:2000/v1/models`
- Check proxy health: `http://localhost:2000/health`
- View logs: `docker logs llama-cpp-studio`

## Build Customization

### GPU Backends

Enable specific GPU backends during source builds:

- **CUDA**: NVIDIA GPU acceleration with cuBLAS
- **Vulkan**: AMD/Intel GPU acceleration with Vulkan compute
- **Metal**: Apple Silicon GPU acceleration
- **OpenBLAS**: CPU optimization with OpenBLAS routines

### Build Configuration

Customize your build with:

- **Build Type**: Release (optimal), Debug (development), RelWithDebInfo
- **Custom CMake Flags**: Additional CMake configuration
- **Compiler Flags**: CFLAGS and CXXFLAGS for optimization
- **Git Patches**: Apply patches from GitHub PRs

### Example Build Configuration

```json
{
  "commit_sha": "master",
  "patches": [
    "https://github.com/ggerganov/llama.cpp/pull/1234.patch"
  ],
  "build_config": {
    "build_type": "Release",
    "enable_cuda": true,
    "enable_vulkan": false,
    "enable_metal": false,
    "enable_openblas": true,
    "custom_cmake_args": "-DGGML_CUDA_CUBLAS=ON",
    "cflags": "-O3 -march=native",
    "cxxflags": "-O3 -march=native"
  }
}
```

## Smart Auto Configuration

The Smart Auto feature automatically generates optimal llama.cpp parameters based on:

- **GPU Capabilities**: VRAM, compute capability, multi-GPU support
- **NVLink Topology**: Automatic detection and optimization for NVLink clusters
- **Model Architecture**: Detected from model name (Llama, Mistral, etc.)
- **Available Resources**: CPU cores, memory, disk space
- **Performance Optimization**: Flash attention, tensor parallelism, batch sizing

### NVLink Optimization Strategies

The system automatically detects NVLink topology and applies appropriate strategies:

- **Unified NVLink**: All GPUs connected via NVLink - uses aggressive tensor splitting and higher parallelism
- **Clustered NVLink**: Multiple NVLink clusters - optimizes for the largest cluster
- **Partial NVLink**: Some GPUs connected via NVLink - uses hybrid approach
- **PCIe Only**: No NVLink detected - uses conservative PCIe-based configuration

### Supported Parameters
- Context size, batch sizes, GPU layers
- Temperature, top-k, top-p, repeat penalty
- CPU threads, parallel sequences
- RoPE scaling, YaRN factors
- Multi-GPU tensor splitting
- Custom arguments via YAML config

## API Endpoints

### Models
- `GET /api/models` - List all models
- `POST /api/models/search` - Search HuggingFace
- `POST /api/models/download` - Download model
- `GET /api/models/{id}/config` - Get model configuration
- `PUT /api/models/{id}/config` - Update configuration
- `POST /api/models/{id}/auto-config` - Generate smart configuration
- `POST /api/models/{id}/start` - Start model
- `POST /api/models/{id}/stop` - Stop model
- `DELETE /api/models/{id}` - Delete model

### llama.cpp Versions
- `GET /api/llama-versions` - List installed versions
- `GET /api/llama-versions/check-updates` - Check for updates
- `GET /api/llama-versions/build-capabilities` - Get build capabilities
- `POST /api/llama-versions/install-release` - Install release
- `POST /api/llama-versions/build-source` - Build from source
- `DELETE /api/llama-versions/{id}` - Delete version

### System
- `GET /api/status` - System status
- `GET /api/gpu-info` - GPU information
- `WebSocket /ws` - Real-time updates

## Database Migration

If upgrading from an older version, you may need to migrate your database:

```bash
# Run migration to support multi-quantization
python migrate_db.py
```

## Troubleshooting

### Common Issues

1. **GPU Not Detected**
   - Ensure NVIDIA Container Toolkit is installed (for NVIDIA)
   - Check `nvidia-smi` output
   - Verify `--gpus all` flag in docker run
   - For AMD: Check Vulkan/ROCm drivers

2. **Build Failures**
   - Check CUDA version compatibility (for NVIDIA)
   - Ensure sufficient disk space (at least 10GB free)
   - Verify internet connectivity for downloads
   - For Vulkan builds: Ensure `glslang-tools` is installed
   - Check build logs for specific errors

3. **Memory Issues**
   - Use Smart Auto configuration
   - Reduce context size or batch size
   - Enable memory mapping
   - Check available system RAM and VRAM

4. **Model Download Failures**
   - Check HuggingFace connectivity
   - Verify model exists and is public
   - Ensure sufficient disk space
   - Set HUGGINGFACE_API_KEY if using private models

5. **Validation Failed**
   - Binary exists and is executable
   - Binary runs `--version` successfully
   - Output contains "llama" or "version:" string

### Logs
- Application logs: `docker logs llama-cpp-studio`
- Model logs: Available in the web interface
- Build logs: Shown during source compilation
- WebSocket logs: DEBUG level for detailed connection info

## Development

### Backend
- FastAPI with async support
- SQLAlchemy for database management
- WebSocket for real-time updates
- Background tasks for long operations
- Llama-swap integration for multi-model serving

### Frontend
- Vue.js 3 with Composition API
- PrimeVue component library
- Pinia for state management
- Vite for build tooling
- Dark mode support

### Database
- SQLite for simplicity
- Models, versions, and instances tracking
- Configuration storage
- Multi-quantization support

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2024 llama.cpp Studio

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the API documentation

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - The core inference engine
- [llama-swap](https://github.com/mostlygeek/llama-swap) - Multi-model serving proxy
- [HuggingFace](https://huggingface.co) - Model hosting and search
- [Vue.js](https://vuejs.org) - Frontend framework
- [FastAPI](https://fastapi.tiangolo.com) - Backend framework
