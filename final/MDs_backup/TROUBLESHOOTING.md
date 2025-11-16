# Troubleshooting Guide

## Common Issues and Solutions

### 0. NumPy Warnings on Windows

**Warning**:
```
Warning: Numpy built with MINGW-W64 on Windows 64 bits is experimental
RuntimeWarning: invalid value encountered in exp2
```

**Status**: ⚠️ **HARMLESS - Can be ignored**

These are known NumPy warnings on Windows and don't affect functionality. The backend will continue to work normally.

**To suppress** (optional):
The warnings are now automatically suppressed in `start_backend.bat`. If you see them, just press any key to continue.

---

### 1. "Fish Speech installation not found"

**Error**:
```
Fish Speech installation not found!
```

**Solution**:
```bash
# Run the installer
install_fish_speech.bat

# Choose option 1 (clone to final folder)
```

---

### 2. "ModuleNotFoundError: No module named 'hydra'"

**Error**:
```
Traceback (most recent call last):
  File "...\fish_speech\models\dac\inference.py", line 4, in <module>
    import hydra
ModuleNotFoundError: No module named 'hydra'
```

**Cause**: Fish Speech dependencies not fully installed

**Solution**:
```bash
# Run the dependency fix script
fix_fish_speech_deps.bat
```

**Or manually install**:
```bash
# Activate virtual environment
venv\Scripts\activate

# Install missing dependencies
pip install hydra-core omegaconf pyrootutils loguru click

# Install Fish Speech
cd fish-speech
pip install -e .
cd ..
```

---

### 3. "ModuleNotFoundError: No module named 'fish_speech'"

**Error**:
```
ModuleNotFoundError: No module named 'fish_speech'
```

**Solution**:
```bash
cd fish-speech
pip install -e .
cd ..
```

---

### 4. VQ Extraction Failed

**Error**:
```
Error: TTS generation failed: TTS failed: VQ extraction failed
```

**Possible Causes**:
1. Missing dependencies
2. Incorrect Fish Speech path
3. Model files not found
4. Audio file issues

**Solutions**:

**Check 1: Dependencies**
```bash
fix_fish_speech_deps.bat
```

**Check 2: Fish Speech Path**
Open `.env` and verify:
```bash
FISH_SPEECH_DIR=C:\Users\VM02\Desktop\THESIS (SALAS)\SECOND PHASE ENV\final\fish-speech
```

**Check 3: Model Files**
Verify model exists:
```bash
dir checkpoints\openaudio-s1-mini\codec.pth
```

If missing, download:
```bash
huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini
```

**Check 4: Test Inference Script**
```bash
cd fish-speech
python fish_speech\models\dac\inference.py --help
```

If this fails, reinstall Fish Speech.

---

### 5. "CUDA out of memory"

**Error**:
```
RuntimeError: CUDA out of memory
```

**Solutions**:

**Option 1: Enable Quantization**
Edit `.env`:
```bash
QUANTIZATION=int8
```

**Option 2: Reduce Chunk Size**
```bash
CHUNK_SIZE=4096
```

**Option 3: Use CPU**
```bash
DEVICE=cpu
```

---

### 6. Slow Performance on CPU

**Issue**: Generation takes 20+ seconds

**Solutions**:

**Check 1: Verify Device**
Look at backend startup logs:
```
Device: cpu
```

**Check 2: Force GPU if Available**
Edit `.env`:
```bash
DEVICE=cuda  # or mps for Mac
```

**Check 3: Optimize for CPU**
```bash
DEVICE=cpu
MIXED_PRECISION=fp32
QUANTIZATION=none
```

**Check 4: Use Shorter Audio**
- Keep reference audio under 15 seconds
- Use shorter text inputs

---

### 7. "No such file or directory: inference.py"

**Error**:
```
FileNotFoundError: [Errno 2] No such file or directory: 
'...\fish_speech\models\dac\inference.py'
```

**Solution**:

**Check 1: Verify Fish Speech Installation**
```bash
dir fish-speech\fish_speech\models\dac\inference.py
```

**Check 2: Reinstall Fish Speech**
```bash
# Remove old installation
rmdir /s /q fish-speech

# Reinstall
install_fish_speech.bat
```

---

### 8. Import Errors

**Error**:
```
ImportError: cannot import name 'xxx' from 'yyy'
```

**Solution**:

**Reinstall all dependencies**:
```bash
# Activate venv
venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Reinstall requirements
pip install -r requirements.txt --upgrade

# Reinstall Fish Speech
cd fish-speech
pip install -e . --upgrade
cd ..
```

---

### 9. Backend Won't Start

**Error**: Backend crashes immediately

**Solutions**:

**Check 1: Python Version**
```bash
python --version
# Should be 3.10 or 3.11
```

**Check 2: Virtual Environment**
```bash
# Recreate venv
rmdir /s /q venv
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Check 3: Check Logs**
Look for specific error messages in terminal

---

### 10. Model Download Failed

**Error**:
```
Error downloading model
```

**Solutions**:

**Option 1: Use HuggingFace CLI**
```bash
pip install huggingface-hub
huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini
```

**Option 2: Manual Download**
1. Go to https://huggingface.co/fishaudio/openaudio-s1-mini
2. Download files manually
3. Place in `checkpoints/openaudio-s1-mini/`

**Option 3: Use Python**
```python
from huggingface_hub import snapshot_download
snapshot_download('fishaudio/openaudio-s1-mini', local_dir='checkpoints/openaudio-s1-mini')
```

---

## Quick Diagnostic Commands

### Check Python Environment
```bash
python --version
pip list | findstr torch
pip list | findstr fish
```

### Check File Structure
```bash
dir fish-speech\fish_speech\models\dac\inference.py
dir checkpoints\openaudio-s1-mini\codec.pth
dir .env
```

### Test Fish Speech
```bash
cd fish-speech
python -c "import fish_speech; print('OK')"
cd ..
```

### Test Dependencies
```bash
python -c "import hydra; print('hydra OK')"
python -c "import omegaconf; print('omegaconf OK')"
python -c "import pyrootutils; print('pyrootutils OK')"
```

---

## Complete Reinstall (Nuclear Option)

If nothing else works:

```bash
# 1. Backup .env
copy .env .env.backup

# 2. Remove everything
rmdir /s /q venv
rmdir /s /q fish-speech
rmdir /s /q checkpoints

# 3. Start fresh
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 4. Install Fish Speech
install_fish_speech.bat

# 5. Download model
huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini

# 6. Restore config
copy .env.backup .env

# 7. Start backend
start_backend.bat
```

---

## Getting Help

### Before Asking for Help

Collect this information:

1. **Error Message**: Full error traceback
2. **Python Version**: `python --version`
3. **OS**: Windows version
4. **GPU**: NVIDIA/AMD/None
5. **Steps**: What you did before the error
6. **Logs**: Backend startup logs

### Check Logs

**Backend Logs**: Look in terminal where you ran `start_backend.bat`

**Key Information**:
- Device detected
- Fish Speech directory
- Model path
- Error messages

### Common Log Messages

**Good**:
```
✓ Engine initialized successfully
Device: cuda
Fish Speech found in: ...\fish-speech
```

**Bad**:
```
Fish Speech installation not found!
ModuleNotFoundError: No module named 'hydra'
FileNotFoundError: codec.pth not found
```

---

## Prevention Tips

### 1. Always Use Virtual Environment
```bash
venv\Scripts\activate
```

### 2. Keep Dependencies Updated
```bash
pip install -r requirements.txt --upgrade
```

### 3. Verify Installation
```bash
# After installing Fish Speech
cd fish-speech
python -c "import fish_speech; print('OK')"
cd ..
```

### 4. Check .env Configuration
```bash
type .env
```

### 5. Use Absolute Paths
In `.env`, use full paths:
```bash
FISH_SPEECH_DIR=C:\Users\...\final\fish-speech
MODEL_DIR=C:\Users\...\final\checkpoints\openaudio-s1-mini
```

---

## Platform-Specific Issues

### Windows

**Issue**: Path with spaces
```bash
# Use quotes in .env
FISH_SPEECH_DIR="C:\Program Files\fish-speech"
```

**Issue**: Permission denied
```bash
# Run as administrator or change folder permissions
```

### macOS

**Issue**: MPS not detected
```bash
# Update macOS to 12.3+
# Update PyTorch: pip install --upgrade torch
```

### Linux

**Issue**: CUDA not found
```bash
# Install CUDA toolkit
sudo apt-get install cuda-11-8

# Reinstall PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## Still Having Issues?

1. **Run the fix script**: `fix_fish_speech_deps.bat`
2. **Check this guide**: Look for your specific error
3. **Review logs**: Read error messages carefully
4. **Try clean install**: Follow "Complete Reinstall" section
5. **Check documentation**: Read DEVICE_SUPPORT.md and README.md

---

## Success Checklist

✅ Virtual environment activated
✅ All dependencies installed (`pip list`)
✅ Fish Speech installed (`cd fish-speech && python -c "import fish_speech"`)
✅ Model downloaded (`dir checkpoints\openaudio-s1-mini\codec.pth`)
✅ `.env` configured correctly
✅ Backend starts without errors
✅ Can access http://localhost:8000/docs

If all checked, you're ready to generate speech! 🎉
