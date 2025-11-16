# Setup Fix - Fish Speech Path Configuration

## Issue
The backend was failing with:
```
Error: can't open file 'fish_speech\models\dac\inference.py': No such file or directory
```

## Root Cause
The optimization engine (`backend/opt_engine.py`) was trying to run Fish Speech inference scripts from the current directory, but they don't exist in the `final` folder. The Fish Speech codebase is located in a different directory.

## Solution
Added `FISH_SPEECH_DIR` configuration to point to the actual Fish Speech installation.

### Changes Made

1. **backend/opt_engine.py**
   - Added `FISH_SPEECH_DIR` configuration variable
   - Updated all three subprocess calls to:
     - Use absolute paths to Fish Speech inference scripts
     - Run with `cwd=FISH_SPEECH_DIR` (working directory)
     - Look for output files in Fish Speech directory

2. **.env.example**
   - Added `FISH_SPEECH_DIR` configuration
   - Default path points to existing installation:
     ```
     FISH_SPEECH_DIR=C:\Users\VM02\Desktop\THESIS (SALAS)\SECOND PHASE ENV\voice cloning (FINAL)\scripts\fish-speech
     ```

3. **README.md**
   - Updated setup instructions to mention FISH_SPEECH_DIR configuration
   - Added note about requiring Fish Speech codebase access

## Setup Instructions

### 1. Copy and configure environment file
```bash
copy .env.example .env
```

### 2. Edit .env file
Open `.env` and verify/update the `FISH_SPEECH_DIR` path:

```bash
# Fish Speech Installation Directory
FISH_SPEECH_DIR=C:\Users\VM02\Desktop\THESIS (SALAS)\SECOND PHASE ENV\voice cloning (FINAL)\scripts\fish-speech
```

**Important**: Use the full absolute path to your Fish Speech installation.

### 3. Verify Fish Speech installation
Make sure these files exist in your Fish Speech directory:
- `fish_speech/models/dac/inference.py`
- `fish_speech/models/text2semantic/inference.py`
- Model files in `checkpoints/openaudio-s1-mini/`

### 4. Start the backend
```bash
python backend/app.py
```

The backend should now successfully call Fish Speech inference scripts!

## Technical Details

### Updated Code Flow

**Before (Broken):**
```python
cmd = [sys.executable, "fish_speech/models/dac/inference.py", ...]
subprocess.run(cmd)  # Looks in current directory (final/)
```

**After (Fixed):**
```python
inference_script = Path(FISH_SPEECH_DIR) / "fish_speech" / "models" / "dac" / "inference.py"
cmd = [sys.executable, str(inference_script), ...]
subprocess.run(cmd, cwd=FISH_SPEECH_DIR)  # Runs from Fish Speech directory
```

### Output File Handling

The Fish Speech scripts generate output files (`fake.npy`, `fake.wav`, `codes_*.npy`) in their working directory. The fix:

1. Runs subprocess with `cwd=FISH_SPEECH_DIR`
2. Looks for output files in `FISH_SPEECH_DIR`
3. Moves them to the temp directory for processing

**Example:**
```python
# Find output in Fish Speech directory
fake_npy = Path(FISH_SPEECH_DIR) / "fake.npy"
if fake_npy.exists():
    shutil.move(str(fake_npy), output_path)
```

## Verification

After applying the fix, test with:

```bash
# Start backend
python backend/app.py

# In another terminal, test API
curl -X POST "http://localhost:8000/tts" \
  -F "text=Hello world" \
  --output test.wav
```

You should see:
- ✅ No "file not found" errors
- ✅ VQ extraction completes successfully
- ✅ Semantic generation works
- ✅ Audio synthesis produces output.wav

## Alternative: Copy Fish Speech to Final Folder

If you prefer, you can copy the Fish Speech codebase into the `final` folder:

```bash
# Copy Fish Speech to final folder
xcopy "C:\Users\VM02\Desktop\THESIS (SALAS)\SECOND PHASE ENV\voice cloning (FINAL)\scripts\fish-speech" "final\fish-speech" /E /I

# Update .env
FISH_SPEECH_DIR=fish-speech
```

This makes the project self-contained but increases folder size.

## Troubleshooting

### Issue: "Module not found" errors
**Solution**: Make sure Fish Speech dependencies are installed:
```bash
cd "C:\Users\VM02\Desktop\THESIS (SALAS)\SECOND PHASE ENV\voice cloning (FINAL)\scripts\fish-speech"
pip install -r requirements.txt
```

### Issue: Path with spaces causes problems
**Solution**: Use raw strings in .env:
```bash
FISH_SPEECH_DIR=C:\Users\VM02\Desktop\THESIS (SALAS)\SECOND PHASE ENV\voice cloning (FINAL)\scripts\fish-speech
```
No quotes needed in .env files.

### Issue: "Permission denied" on output files
**Solution**: Make sure Fish Speech directory is writable. The scripts need to create temporary files there.

## Summary

The fix enables the optimized backend to correctly interface with the existing Fish Speech installation by:
1. Using absolute paths to inference scripts
2. Running subprocesses from the Fish Speech directory
3. Properly handling output files generated in that directory

This maintains the subprocess-based architecture (stable, isolated) while correctly locating and executing the Fish Speech codebase.
