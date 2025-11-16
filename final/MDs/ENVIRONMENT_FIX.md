# Environment Fix Guide

## 🔴 The Problem

Your system has **TWO Python environments** competing:

1. **Anaconda Base** (Python 3.13) - ❌ WRONG
   - Path: `C:\Users\VM02\anaconda3\`
   - Has broken NumPy with MINGW-W64 issues
   - Causes crashes

2. **venv312** (Python 3.12) - ✅ CORRECT
   - Path: `C:\Users\VM02\Desktop\THESIS (SALAS)\SECOND PHASE ENV\final\venv312\`
   - Has working NumPy and all dependencies
   - Same environment Fish Speech uses

**When you run `python`, Windows uses Anaconda instead of venv312!**

---

## ✅ The Solution

### **Step 1: Always Deactivate Conda First**

```powershell
# In PowerShell
conda deactivate
```

### **Step 2: Activate venv312**

```powershell
cd "C:\Users\VM02\Desktop\THESIS (SALAS)\SECOND PHASE ENV\final"
.\venv312\Scripts\activate
```

### **Step 3: Verify Environment**

```powershell
# Check Python version (should be 3.12.x)
python --version

# Check Python location (should show venv312)
where python

# Check NumPy location (should show venv312)
python -c "import numpy; print(numpy.__file__)"

# Or use the check script
python check_environment.py
```

**Expected output:**
```
Python 3.12.x
C:\...\final\venv312\Scripts\python.exe
C:\...\final\venv312\Lib\site-packages\numpy\...
```

---

## 🚀 Running Your Optimized System

### **Option 1: Use Batch Files (Easiest)**

All batch files now automatically deactivate conda and use venv312:

```powershell
# Start everything
.\run_all.bat

# Or start individually
.\start_backend.bat
.\start_gradio.bat
```

### **Option 2: Manual Start (More Control)**

**Terminal 1 - Backend:**
```powershell
conda deactivate
cd "C:\Users\VM02\Desktop\THESIS (SALAS)\SECOND PHASE ENV\final"
.\venv312\Scripts\activate
python backend/app.py
```

**Terminal 2 - Gradio UI:**
```powershell
conda deactivate
cd "C:\Users\VM02\Desktop\THESIS (SALAS)\SECOND PHASE ENV\final"
.\venv312\Scripts\activate
python ui/gradio_app.py
```

### **Option 3: Direct Python Call (No Activation Needed)**

```powershell
# Backend
.\venv312\Scripts\python.exe backend/app.py

# Gradio
.\venv312\Scripts\python.exe ui/gradio_app.py
```

---

## 🔧 Install Missing Dependencies

If you get `ModuleNotFoundError`, install in venv312:

```powershell
# Make sure you're in venv312
.\venv312\Scripts\activate

# Install missing packages
pip install psutil pynvml

# Or reinstall all requirements
pip install -r requirements.txt
```

---

## 🎯 Why This Happens

### **PATH Priority**

When both Conda and venv are active, Windows PATH looks like:

```
1. C:\Users\VM02\anaconda3\Scripts\         ← Conda (WRONG)
2. C:\Users\VM02\anaconda3\
3. ...\final\venv312\Scripts\               ← venv312 (CORRECT)
```

Windows finds Conda's Python first, so `python` runs Anaconda Python 3.13.

### **The Fix**

Deactivating Conda removes it from PATH:

```
1. ...\final\venv312\Scripts\               ← venv312 (CORRECT)
2. C:\Users\VM02\anaconda3\Scripts\         ← Conda (ignored)
```

Now `python` runs venv312's Python 3.12.

---

## 📊 Comparison

| Command | Without Fix | With Fix |
|---------|-------------|----------|
| `python --version` | Python 3.13 (Conda) ❌ | Python 3.12 (venv312) ✅ |
| `where python` | Anaconda first ❌ | venv312 first ✅ |
| NumPy location | Anaconda ❌ | venv312 ✅ |
| NumPy errors | MINGW-W64 crashes ❌ | Works perfectly ✅ |
| Fish Speech | Fails ❌ | Works ✅ |

---

## 🛠️ Troubleshooting

### **"Still getting NumPy errors"**

```powershell
# 1. Check which Python you're using
where python
# First line MUST be venv312, not anaconda3

# 2. If anaconda3 appears first, deactivate it
conda deactivate

# 3. Verify again
where python
```

### **"conda deactivate doesn't work"**

```powershell
# Force deactivate
conda deactivate
conda deactivate  # Run twice if needed

# Or close terminal and open new one
```

### **"venv312 activation fails"**

```powershell
# Use full path
C:\Users\VM02\Desktop\THESIS` (SALAS)\SECOND` PHASE` ENV\final\venv312\Scripts\activate.bat

# Or use direct Python call
.\venv312\Scripts\python.exe your_script.py
```

### **"ModuleNotFoundError: No module named 'X'"**

```powershell
# Install in venv312
.\venv312\Scripts\activate
pip install X

# Or reinstall all
pip install -r requirements.txt
```

---

## ✅ Quick Checklist

Before running anything:

- [ ] Deactivate conda: `conda deactivate`
- [ ] Activate venv312: `.\venv312\Scripts\activate`
- [ ] Verify Python: `python --version` shows 3.12.x
- [ ] Verify location: `where python` shows venv312 first
- [ ] Check environment: `python check_environment.py`

---

## 🎉 Success Indicators

You'll know it's working when:

1. ✅ No NumPy MINGW-W64 warnings
2. ✅ No "invalid value encountered" errors
3. ✅ `where python` shows venv312 first
4. ✅ Backend starts without errors
5. ✅ TTS generation works

---

## 📝 Summary

**Problem:** Conda Python 3.13 with broken NumPy was being used instead of venv312 Python 3.12

**Solution:** Always deactivate conda before activating venv312

**Result:** Everything works because you're using the correct Python environment

**Remember:** `conda deactivate` → `.\venv312\Scripts\activate` → profit! 🚀
