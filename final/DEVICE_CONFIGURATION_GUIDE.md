# Device Configuration Guide - Cross-Platform

This guide explains how to configure device preferences for different operating systems: macOS, Windows, and Linux.

## Quick Start

1. **Copy the example configuration:**
```bash
cp .env.example .env
```

2. **Edit `.env` to set your device preference:**
```bash
# Edit .env and change the DEVICE setting:
DEVICE=auto    # Intelligent auto-selection (recommended)
DEVICE=cpu     # Force CPU only
DEVICE=cuda    # Force NVIDIA GPU (Windows/Linux only)
DEVICE=mps     # Force Apple Silicon GPU (macOS M1/M2/M3/M4 only)
```

3. **Run the backend/UI and your preference will be respected:**
```bash
python3 backend/app.py  # FastAPI backend
# or
python3 ui/gradio_app.py  # Gradio UI
```

## Device Configuration Options

### 1. `DEVICE=auto` (Recommended - Default)
**Intelligent auto-selection with real-time monitoring**

✅ **Best for:** Most users, systems with mixed GPU/CPU workloads, cloud environments

```
How it works:
├─ Detects available hardware (GPU/CPU)
├─ Monitors GPU utilization every 5 seconds
├─ If GPU > 85% utilized → switches to CPU (frees GPU for other work)
├─ If GPU > 90% VRAM used → switches to CPU (prevents memory overflow)
├─ If CPU > 90% utilized AND GPU available → switches to GPU
└─ Auto-adjusts based on system load
```

**Performance expectations:**
- Initial inference: Uses GPU if available and not busy
- Under heavy load: Gracefully falls back to CPU
- No manual intervention needed

**Example logs:**
```
✅ Smart Adaptive Backend initialized
🤖 Smart auto-selection enabled (will optimize based on system load)
Device: cuda (GPU selected - 10GB VRAM available)
```

---

### 2. `DEVICE=cpu` (Force CPU Only)
**Lock to CPU, completely hide GPU**

✅ **Best for:**
- Low-power devices (Raspberry Pi, mobile)
- Debugging performance issues
- Freeing GPU for other applications
- Reducing power consumption
- Testing on CPU without GPU interference

```
How it works:
├─ Disables GPU backend completely
│  ├─ macOS: Sets PYTORCH_ENABLE_MPS_FALLBACK=0 (disables MPS)
│  └─ Windows/Linux: Sets CUDA_VISIBLE_DEVICES='' (hides CUDA GPUs)
├─ Clears GPU memory cache
└─ Forces all computation to CPU
```

**Performance expectations:**
- M1 MacBook Air: ~2.4-3x RTF (slower than real-time)
- Intel i5 CPU: ~6-8x RTF
- High-end CPU: ~2-3x RTF
- **Don't use on weak CPUs** (will be very slow)

**Example configuration:**
```bash
# .env
DEVICE=cpu
ENABLE_TORCH_COMPILE=False  # Recommended: CPU doesn't support torch.compile well
MIXED_PRECISION=fp32        # CPU works best with fp32 (not fp16/bf16)
```

**Example logs (should see GPU disabled message):**
```
✅ Forcing CPU mode (user preference)
🔒 MPS backend disabled - using CPU only          # macOS
🚫 CUDA disabled - GPU hidden, using CPU only     # Windows/Linux
Device: cpu
```

---

### 3. `DEVICE=cuda` (Force NVIDIA GPU - Windows/Linux only)
**Lock to NVIDIA GPU, fail if unavailable**

✅ **Best for:**
- Windows with NVIDIA GeForce/RTX GPU
- Linux workstations with NVIDIA GPU
- Cloud instances (AWS, Google Cloud, Azure with GPU)
- Maximum performance needed

❌ **Won't work on:** macOS (no CUDA support)

```
How it works:
├─ Requires: PyTorch with CUDA support (check: python3 -c "import torch; print(torch.cuda.is_available())")
├─ Auto-detects GPU capability
├─ Fails gracefully to CPU if CUDA not available
└─ Applies appropriate precision based on GPU capability
```

**Performance expectations:**
- NVIDIA RTX 3060 (12GB): ~0.8-1.2x RTF (faster than real-time!)
- NVIDIA RTX 2060 (6GB): ~1.5-2.0x RTF (real-time)
- NVIDIA GTX 1080 (8GB): ~0.8-1.0x RTF (faster than real-time)
- Older GPUs: Check compute capability (`nvidia-smi`)

**Example configuration:**
```bash
# .env
DEVICE=cuda
ENABLE_TORCH_COMPILE=False  # Triton compiler issues on Windows
MIXED_PRECISION=auto        # Auto-selects bf16/fp16 based on GPU capability
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128  # Prevents memory fragmentation
```

**Example logs:**
```
✅ Forcing CUDA mode (user preference)
Detected NVIDIA GPU with 6.0GB VRAM
Device: cuda
Mixed Precision: fp16 (RTX with compute capability 7.5+)
```

**Troubleshooting:**
- GPU showing 0% utilization: Check `nvidia-smi`, may indicate CPU fallback
- "CUDA out of memory": Try `QUANTIZATION=int8` or `DEVICE=cpu`
- Poor performance: May need `PYTORCH_CUDA_ALLOC_CONF` tuning

---

### 4. `DEVICE=mps` (Force Apple Silicon GPU - macOS only)
**Lock to Apple M-series GPU, fail if unavailable**

✅ **Best for:**
- macOS M1/M2/M3/M4 MacBook Pro/Air
- Apple Silicon devices only
- When you want guaranteed GPU performance

❌ **Won't work on:** Intel Macs, Windows, Linux

```
How it works:
├─ Requires: macOS 12.3+ with PyTorch MPS support
├─ Auto-detects M1/M2/M3/M4 chip
├─ Falls back to CPU if MPS unavailable
└─ Optimizes for unified memory architecture
```

**Performance expectations:**
- M1 Air (initial): ~2.4x RTF (real-time speech speed)
- M1 Pro: ~1.2-1.5x RTF
- M2/M3: ~1.0-1.2x RTF (faster than real-time)
- M4: ~0.8-1.0x RTF (very fast!)

**Performance degrades after ~10 minutes** on M1 Air due to passive cooling (fanless design).

**Example configuration:**
```bash
# .env for M1 Air
DEVICE=mps
ENABLE_TORCH_COMPILE=False  # Not supported on macOS
MIXED_PRECISION=fp16        # MPS optimal precision
QUANTIZATION=int8           # Reduces memory usage on 4GB GPU
```

**Example logs:**
```
✅ Forcing MPS mode (user preference)
Detected Apple Silicon (M-series chip)
Device: mps
Mixed Precision: fp16
GPU Memory: 4.8 GB
```

**Thermal management:**
```
⚠️  M1 MacBook Air detected. Performance will degrade after ~10 minutes
    of sustained load due to fanless design.
    
💡 Recommendations:
   - Use DEVICE=cpu for long sessions
   - Add external cooling if needed
   - Monitor thermal throttling in Activity Monitor
```

---

## Advanced Configuration

### Precision Settings (for GPU modes)

```bash
# .env
MIXED_PRECISION=auto   # Auto-select based on GPU capability (recommended)
MIXED_PRECISION=bf16   # Brain Float 16 (best for performance, requires Ampere+)
MIXED_PRECISION=fp16   # Float 16 (good balance, older GPU support)
MIXED_PRECISION=fp32   # Float 32 (safest, no precision loss, slowest)
```

**Recommendation by GPU:**
- NVIDIA Ampere (RTX 30 series): `bf16` (fastest)
- NVIDIA Turing (RTX 20 series): `fp16` (good balance)
- Older NVIDIA: `fp32` (safe)
- Apple Silicon: `fp16` (MPS optimized)
- CPU: Always `fp32` (no mixed precision support)

### Quantization Settings

```bash
# .env
QUANTIZATION=none   # No quantization (best quality, most memory)
QUANTIZATION=int8   # 8-bit quantization (good quality, less memory)
QUANTIZATION=4bit   # 4-bit quantization (lower quality, minimal memory)
```

**When to use quantization:**
- `none`: GPU with >8GB VRAM
- `int8`: GPU with 4-8GB VRAM, or CPU with limited RAM
- `4bit`: Laptops, mobile devices, very limited memory

---

## Platform-Specific Setup

### macOS Setup

**Installation:**
```bash
# 1. Clone repository
git clone <repo-url>
cd final

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install PyTorch with MPS support
pip install torch torchvision torchaudio

# 4. Install dependencies
pip install -r requirements.txt

# 5. Setup .env
cp .env.example .env
# Edit .env: Set DEVICE=auto, mps, or cpu
```

**Verify setup:**
```bash
python3 -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

**Run:**
```bash
source venv/bin/activate
python3 backend/app.py
```

---

### Windows Setup

**Installation:**
```bash
# 1. Clone repository
git clone <repo-url>
cd final

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. Install dependencies
pip install -r requirements.txt

# 5. Setup .env
copy .env.example .env
# Edit .env: Set DEVICE=auto, cuda, or cpu
```

**Verify setup:**
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

**Run (Option 1: Command Prompt):**
```bash
venv\Scripts\activate
python backend/app.py
```

**Run (Option 2: PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
python backend/app.py
```

**Troubleshooting:**
- "CUDA not available": Reinstall with correct index URL for your CUDA version
- "Command not found": Make sure to activate venv first
- "Permission denied": Right-click terminal → Run as Administrator

---

### Linux Setup

**Installation:**
```bash
# 1. Clone repository
git clone <repo-url>
cd final

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install PyTorch with CUDA support (if NVIDIA GPU available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. Install dependencies
pip install -r requirements.txt

# 5. Setup .env
cp .env.example .env
# Edit .env: Set DEVICE=auto, cuda, or cpu
```

**Verify setup:**
```bash
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# For NVIDIA GPU, also check:
nvidia-smi
```

**Run:**
```bash
source venv/bin/activate
python3 backend/app.py
```

---

## Troubleshooting Device Configuration

### Problem: Device setting in .env is ignored

**Symptom:**
```
Set DEVICE=cpu in .env, but logs show "Device: cuda"
```

**Causes & Solutions:**

1. **Make sure .env is in the right location:**
   ```bash
   # Should be in the project root, same level as backend/ folder
   ls -la .env   # Should show the file
   ```

2. **Restart your Python process:**
   ```bash
   # Kill any running Python
   pkill -f "python3 backend/app.py"
   
   # Verify it's gone
   ps aux | grep python3
   
   # Start fresh
   python3 backend/app.py
   ```

3. **Check for hardcoded device in code:**
   ```bash
   # Search for hardcoded device settings
   grep -r "device.*=" backend/ | grep -v ".env"
   grep -r "DEVICE.*=" backend/ | grep -v "os.getenv"
   ```

4. **Verify .env is being loaded:**
   ```bash
   python3 -c "import os; from dotenv import load_dotenv; load_dotenv(); print('DEVICE:', os.getenv('DEVICE'))"
   ```

### Problem: GPU detected but very slow performance

**Symptom:**
```
DEVICE=cuda is set, CUDA shows available, but RTF is 20x+
```

**Causes & Solutions:**

1. **Gradient checkpointing still enabled** (most likely):
   ```bash
   # Check logs for:
   grep "use_gradient_checkpointing" logs.txt
   
   # If True: Run gradient checkpointing patch
   python3 patch_checkpoint.py
   ```

2. **GPU memory fragmentation:**
   ```bash
   # Add to .env:
   PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
   ```

3. **Quantization too aggressive:**
   ```bash
   # In .env, try:
   QUANTIZATION=none   # Remove aggressive quantization
   ```

4. **Thermal throttling** (laptop with limited cooling):
   ```bash
   # Check GPU temperature:
   watch -n 1 nvidia-smi
   
   # If throttling: Use CPU mode or improve cooling
   DEVICE=cpu
   ```

### Problem: CPU mode still uses GPU

**Symptom:**
```
Set DEVICE=cpu, but GPU utilization shows 100%
```

**This was the critical bug fixed in v2.6.2!**

**Solution:** Update to latest version:
```bash
git pull origin configurable-device
```

**Verify the fix:**
```bash
python3 backend/app.py 2>&1 | grep -E "CUDA disabled|MPS backend disabled"
```

You should see:
- macOS: `🔒 MPS backend disabled - using CPU only`
- Windows/Linux: `🚫 CUDA disabled - GPU hidden, using CPU only`

---

## Performance Comparison

| Platform | Device | RTF | Notes |
|----------|--------|-----|-------|
| M1 Air | auto (2-4GB GPU VRAM) | 2.4-3.0x | Initial speed, degrades with sustained use |
| M1 Air | cpu | 2.4x | Stable, recommended for long sessions |
| M1 Pro | auto | 1.2-1.5x | Good sustained performance |
| M2/M3 | auto | 1.0-1.2x | Faster than real-time |
| Intel i5 | cpu | 6-8x | Slow but works |
| Intel i7 | cpu | 4-5x | Better but still slow |
| RTX 3060 | cuda | 0.8-1.2x | Faster than real-time |
| RTX 2060 | cuda | 1.5-2.0x | Real-time |

**RTF (Real-time Factor):**
- 1.0x = Same speed as spoken audio
- < 1.0x = Faster than real-time (great!)
- > 1.0x = Slower than real-time (acceptable < 5x, good < 2x)

---

## Summary

✅ **Device configuration is now cross-platform and flexible:**
- Set `DEVICE=auto/cpu/cuda/mps` in `.env`
- All platforms respect your choice
- No hardcoded device values
- Intelligent auto-selection available

✅ **Critical fixes applied:**
- [v2.6.1] Fixed macOS device preference ignored (MPS backend not disabling)
- [v2.6.2] Fixed Windows/Linux device preference ignored (CUDA not disabling)

✅ **Recommendations:**
- Use `DEVICE=auto` for most users (intelligent optimization)
- Use `DEVICE=cpu` on M1 Air for sustained use (avoids thermal throttling)
- Use `DEVICE=cuda` on Windows with NVIDIA GPU for best performance
- Always verify the correct device in startup logs
