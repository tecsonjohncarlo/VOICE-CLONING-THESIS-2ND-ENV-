# Device Selection Guide - Quick Reference

## 🎯 Quick Start

### Automatic (Recommended)
```bash
DEVICE=auto
```
The system automatically detects and uses the best available device.

### Manual Selection
```bash
# Windows/Linux with NVIDIA GPU
DEVICE=cuda

# macOS with Apple Silicon
DEVICE=mps

# Any system (slowest)
DEVICE=cpu
```

## 🖥️ Which Device Should I Use?

### I have a Windows PC with NVIDIA GPU
```bash
DEVICE=cuda
```
**Performance**: Fastest (2-5 seconds)
**Supported**: RTX 20/30/40 series, GTX 16 series

### I have a MacBook with M1/M2/M3/M4
```bash
DEVICE=mps
```
**Performance**: Fast (3-7 seconds)
**Supported**: All Apple Silicon Macs

### I have an Intel Mac or no GPU
```bash
DEVICE=cpu
```
**Performance**: Slow (10-30 seconds)
**Supported**: All systems

## 📊 Performance Comparison

| Device | Example Hardware | Speed | Quality | Power |
|--------|-----------------|-------|---------|-------|
| CUDA | RTX 3060 | ⚡⚡⚡⚡⚡ | ⭐⭐⭐⭐⭐ | 🔋🔋🔋 |
| MPS | M3 MacBook | ⚡⚡⚡⚡ | ⭐⭐⭐⭐⭐ | 🔋 |
| CPU | i7 Desktop | ⚡⚡ | ⭐⭐⭐⭐⭐ | 🔋🔋 |

## 🔧 How to Configure

### Step 1: Copy environment file
```bash
copy .env.example .env
```

### Step 2: Edit .env file
Open `.env` and set your device:
```bash
DEVICE=auto  # or cuda, mps, cpu
```

### Step 3: Restart backend
```bash
python backend/app.py
```

## 🎨 Device-Specific Tips

### NVIDIA GPU (CUDA)

**Best Settings**:
```bash
DEVICE=cuda
MIXED_PRECISION=auto  # Uses BF16 on RTX 30/40, FP16 on RTX 20/GTX 16
QUANTIZATION=none     # Or int8 for lower VRAM
```

**If you get "Out of Memory"**:
```bash
QUANTIZATION=int8
CHUNK_SIZE=4096
```

### Apple Silicon (MPS)

**Best Settings**:
```bash
DEVICE=mps
MIXED_PRECISION=auto  # Uses FP16
QUANTIZATION=none
```

**Note**: Some operations may fall back to CPU automatically. This is normal.

### CPU Only

**Best Settings**:
```bash
DEVICE=cpu
MIXED_PRECISION=fp32
QUANTIZATION=none
```

**Speed Tips**:
- Use shorter reference audio (<15 seconds)
- Close other applications
- Use shorter text inputs

## 🚀 Optimization by Device

### RTX 3060/4060 (6-8GB VRAM)
```bash
DEVICE=cuda
MIXED_PRECISION=auto
QUANTIZATION=none
```
✅ Perfect balance of speed and quality

### RTX 3050 (4GB VRAM)
```bash
DEVICE=cuda
MIXED_PRECISION=fp16
QUANTIZATION=int8
```
✅ Reduces VRAM usage

### MacBook M1/M2 (8GB)
```bash
DEVICE=mps
MIXED_PRECISION=auto
QUANTIZATION=none
```
✅ Good performance, efficient power

### MacBook M3/M4 (16GB+)
```bash
DEVICE=mps
MIXED_PRECISION=auto
QUANTIZATION=none
```
✅ Excellent performance

### Desktop CPU (16GB+ RAM)
```bash
DEVICE=cpu
MIXED_PRECISION=fp32
QUANTIZATION=none
```
✅ Reliable but slower

## ❓ FAQ

### Q: How do I know which device is being used?
**A**: Check the backend startup logs:
```
Device: cuda
Detected NVIDIA GPU with 12.0GB VRAM
```

### Q: Can I use AMD GPU?
**A**: Not yet. AMD GPU support (ROCm) is planned for future releases.

### Q: Why is MPS slower than CUDA?
**A**: Fish Speech inference scripts don't support MPS directly, so they fall back to CPU. The system still uses MPS for PyTorch operations where possible.

### Q: My GPU has 2GB VRAM, will it work?
**A**: It might work with heavy quantization, but CPU is recommended:
```bash
DEVICE=cpu  # More reliable for low VRAM
```

### Q: Does auto-detection always pick the best device?
**A**: Yes, it follows this priority:
1. CUDA (if GPU has ≥3.5GB VRAM)
2. MPS (if on Apple Silicon)
3. CPU (fallback)

### Q: Can I switch devices without reinstalling?
**A**: Yes! Just change `DEVICE` in `.env` and restart the backend.

## 🔍 Troubleshooting

### CUDA not detected
```bash
# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### MPS not detected
```bash
# Check if MPS is available
python -c "import torch; print(torch.backends.mps.is_available())"

# If False:
# 1. Update macOS to 12.3+
# 2. Update PyTorch: pip install --upgrade torch
```

### Slow performance
```bash
# Check current device
# Look at backend logs when starting

# Force faster device if available
DEVICE=cuda  # or mps
```

## 📚 More Information

For comprehensive device documentation, see:
- [DEVICE_SUPPORT.md](DEVICE_SUPPORT.md) - Complete technical documentation
- [README.md](README.md) - General setup guide
- [.env.example](.env.example) - All configuration options

## 🎯 Recommended Configurations

### Production Server (Best Performance)
```bash
DEVICE=cuda
MIXED_PRECISION=auto
QUANTIZATION=none
ENABLE_TORCH_COMPILE=False
```
**Hardware**: NVIDIA RTX 3060 or better

### Development Laptop (Good Balance)
```bash
DEVICE=mps
MIXED_PRECISION=auto
QUANTIZATION=none
ENABLE_TORCH_COMPILE=False
```
**Hardware**: MacBook with M2 or better

### Budget/Testing (Works Everywhere)
```bash
DEVICE=cpu
MIXED_PRECISION=fp32
QUANTIZATION=none
ENABLE_TORCH_COMPILE=False
```
**Hardware**: Any modern CPU

---

**Need help?** Check the full [DEVICE_SUPPORT.md](DEVICE_SUPPORT.md) documentation for detailed information about each device type.
