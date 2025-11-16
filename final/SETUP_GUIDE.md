# Fish Speech TTS - Setup Guide

## Quick Start

### 1. Clone and Install Dependencies

```bash
# Clone repository
git clone <your-repo-url>
cd final

# Create virtual environment
python -m venv venv312
source venv312/bin/activate  # Linux/macOS
# or
venv312\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example configuration
cp .env_example .env

# Edit .env with your preferred settings
nano .env  # or use any text editor
```

### 3. Hardware-Specific Setup

#### Option A: NVIDIA GPU (Recommended for best performance)

**Prerequisites:**
- NVIDIA GPU with CUDA support
- NVIDIA drivers installed

**Install CUDA Toolkit:**

**Windows:**
1. Download from: https://developer.nvidia.com/cuda-downloads
2. Install with default settings
3. Verify: `nvcc --version`

**Linux (Ubuntu/Debian):**
```bash
# For CUDA 12.1 (RTX 3000/4000 series)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-1

# Add to PATH (add to ~/.bashrc)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
source ~/.bashrc
```

**Install PyTorch with CUDA:**
```bash
# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8 (older GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Verify CUDA Installation:**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

**Configure .env:**
```bash
DEVICE=cuda
MIXED_PRECISION=fp16
QUANTIZATION=none  # For 12GB+ VRAM, or int8 for <12GB
ENABLE_TORCH_COMPILE=auto
```

#### Option B: Apple Silicon (M1/M2/M3)

**Prerequisites:**
- macOS with Apple Silicon
- Python 3.10+

**Install PyTorch with MPS:**
```bash
pip install torch torchvision torchaudio
```

**Verify MPS:**
```bash
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

**Configure .env:**
```bash
DEVICE=mps
MIXED_PRECISION=fp16
QUANTIZATION=false # INT8 quantization is not supported on MPS
ENABLE_TORCH_COMPILE=false  # Important: MPS is unstable with torch.compile
```

#### Option C: CPU Only (No GPU)

**Install PyTorch (CPU):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Configure .env:**
```bash
DEVICE=cpu
MIXED_PRECISION=fp32
QUANTIZATION=int8
OMP_NUM_THREADS=8  # Set to your CPU core count
ENABLE_TORCH_COMPILE=false
```

**Note:** CPU-only mode is 10-20x slower than GPU. Only use if no GPU is available.

### 4. Download Model

```bash
# Create checkpoints directory
mkdir -p checkpoints

# Download Fish Speech model (openaudio-s1-mini)
# Follow instructions from Fish Speech repository
# Place model in: checkpoints/openaudio-s1-mini/
```

### 5. Start Backend

```bash
# Windows
.\start_backend.bat

# Linux/macOS
python backend/app.py
```

### 6. Start UI (Optional)

**Streamlit UI:**
```bash
# Windows
.\start_streamlit.bat

# Linux/macOS
streamlit run ui/streamlit_app.py
```

**Gradio UI:**
```bash
# Windows
.\start_gradio.bat

# Linux/macOS
python ui/gradio_app.py
```

## Multi-GPU Setup

If you have multiple GPUs, select which one to use:

```bash
# Use first GPU (default)
export CUDA_VISIBLE_DEVICES=0

# Use second GPU
export CUDA_VISIBLE_DEVICES=1

# Use third GPU
export CUDA_VISIBLE_DEVICES=2
```

Add to your `.env`:
```bash
CUDA_VISIBLE_DEVICES=1
```

## Troubleshooting

### CUDA Issues

**"CUDA not available"**
```bash
# Check NVIDIA drivers
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**"CUDA out of memory"**
- Enable quantization: `QUANTIZATION=int8`
- Reduce text length: `MAX_TEXT_LENGTH=600`
- Use smaller batch size

### macOS Issues

**"Process hangs during synthesis"**
- Ensure `ENABLE_TORCH_COMPILE=false` in `.env`
- torch.compile is unstable on MPS

**"MPS not available"**
- Check macOS version (requires macOS 12.3+)
- Verify Apple Silicon (M1/M2/M3)

### CPU-Only Issues

**"Very slow performance"**
- Expected: CPU is 10-20x slower than GPU
- Increase threads: `OMP_NUM_THREADS=<core_count>`
- Enable quantization: `QUANTIZATION=int8`

### General Issues

**"Model not found"**
- Check model path: `MODEL_PATH=checkpoints/openaudio-s1-mini`
- Verify model files exist

**"Out of memory"**
- Reduce memory budget: `MEMORY_BUDGET_GB=6`
- Enable quantization: `QUANTIZATION=int8`
- Close other applications

## Performance Expectations

| Hardware | RTF | Quality | Notes |
|----------|-----|---------|-------|
| RTX 4090 (24GB) | 0.8x | Excellent | No quantization needed |
| RTX 3090 (24GB) | 0.8x | Excellent | No quantization needed |
| V100 (16GB) | 0.8x | Excellent | No quantization needed |
| RTX 3060 (12GB) | 1.2x | Very Good | INT8 quantization |
| RTX 4060 (8GB) | 1.2x | Very Good | INT8 quantization |
| GTX 1660 (6GB) | 2.0x | Good | INT8 + reduced text length |
| M1 Air (8GB) | 2.0x | Good | INT8 quantization |
| M2 Pro (16GB) | 1.5x | Very Good | INT8 quantization |
| CPU (8 cores) | 10-20x | Good | Very slow, INT8 quantization |

**RTF (Real-Time Factor):**
- 0.8x = Faster than real-time (generates 10s audio in 8s)
- 1.0x = Real-time (generates 10s audio in 10s)
- 2.0x = 2x slower (generates 10s audio in 20s)

## Advanced Configuration

See `.env_example` for all available options and detailed explanations.

## Support

For issues and questions:
1. Check this guide first
2. Review `.env_example` for configuration options
3. Check `changelog_thesis.md` for recent changes
4. Review logs in `metrics/` directory
