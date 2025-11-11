# Quick Startup Guide

## ✅ Recommended: Use the All-in-One Launcher

The easiest way to start everything:

```bash
run_all.bat
```

This will:
1. ✅ Activate virtual environment
2. ✅ Start backend server in a new window
3. ✅ Start Gradio UI in a new window  
4. ✅ Open browser to http://localhost:7860

---

## 🔧 Manual Startup (If Needed)

### Option 1: Separate Terminals

**Terminal 1 - Backend:**
```bash
venv\Scripts\activate
set PYTHONWARNINGS=ignore
python backend/app.py
```

**Terminal 2 - Gradio UI:**
```bash
venv\Scripts\activate
set PYTHONWARNINGS=ignore
python ui/gradio_app.py
```

### Option 2: Using Batch Files

**Important**: The batch files need to be run from CMD, not PowerShell!

```cmd
# Open CMD (not PowerShell)
cmd

# Then run:
start_backend.bat
start_gradio.bat
```

---

## 🐛 Troubleshooting

### Issue: "Press any key to continue" and exits

**Cause**: Batch files don't work well in PowerShell

**Solution**: Use `run_all.bat` instead, or run from CMD:
```bash
# Open CMD
cmd

# Navigate to folder
cd "C:\Users\VM02\Desktop\THESIS (SALAS)\SECOND PHASE ENV\final"

# Run
start_backend.bat
```

### Issue: Unicode errors (✓ ✗ symbols)

**Status**: ✅ FIXED

All Unicode characters have been replaced with `[OK]` and `[ERROR]` text.

### Issue: NumPy warnings

**Status**: ⚠️ HARMLESS

These warnings are normal on Windows and don't affect functionality. They're now suppressed in the batch files.

### Issue: Backend not responding

**Check**:
1. Is it actually running? Look for the CMD window
2. Test: http://localhost:8000/health
3. Check for error messages in the backend window

---

## 📊 How to Verify Everything is Working

### 1. Check Backend
Open browser: http://localhost:8000/docs

You should see the FastAPI documentation.

### 2. Check Gradio UI  
Open browser: http://localhost:7860

You should see the Fish Speech interface.

### 3. Test TTS Generation
1. Enter text: "Hello, this is a test"
2. (Optional) Upload reference audio
3. Click "Generate Speech"
4. Wait for audio output

---

## 🎯 Expected Output

### Backend Window Should Show:
```
INFO:     Started server process [XXXX]
INFO:     Waiting for application startup.
[OK] Engine initialized successfully
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Gradio Window Should Show:
```
Running on local URL:  http://0.0.0.0:7860
```

---

## 🚀 Quick Commands Reference

| Task | Command |
|------|---------|
| Start Everything | `run_all.bat` |
| Backend Only | `python backend/app.py` |
| Gradio Only | `python ui/gradio_app.py` |
| Test Imports | `python test_imports.py` |
| Fix Dependencies | `fix_fish_speech_deps.bat` |
| Install Fish Speech | `install_fish_speech.bat` |

---

## 💡 Pro Tips

1. **Use `run_all.bat`** - It's the easiest and most reliable method

2. **Keep windows open** - Don't close the CMD windows, they need to stay running

3. **Check ports** - Make sure ports 8000 and 7860 aren't already in use

4. **Use CMD not PowerShell** - Batch files work better in CMD

5. **Wait for backend** - Give the backend 5-10 seconds to start before opening Gradio

---

## 📝 File Structure

```
final/
├── run_all.bat              ← Use this to start everything!
├── start_backend.bat        ← Start backend only
├── start_gradio.bat         ← Start Gradio only  
├── test_imports.py          ← Test if everything is installed
├── fix_fish_speech_deps.bat ← Fix missing dependencies
└── install_fish_speech.bat  ← Install Fish Speech
```

---

## ✨ Success Checklist

Before generating speech, verify:

- ✅ Virtual environment activated
- ✅ Backend running (http://localhost:8000/health returns OK)
- ✅ Gradio UI accessible (http://localhost:7860 loads)
- ✅ Model downloaded (checkpoints/openaudio-s1-mini/codec.pth exists)
- ✅ Fish Speech installed (fish-speech folder exists)
- ✅ GPU detected (check backend logs for "Detected NVIDIA GPU")

If all checked, you're ready to generate speech! 🎉
