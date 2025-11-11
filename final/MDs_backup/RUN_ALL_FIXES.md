# run_all.ps1 Fixes 🔧

## Issues Found and Fixed

### **Issue 1: Missing Working Directory**
**Problem:** Script didn't change to its own directory, causing file not found errors.

**Fix:**
```powershell
# Change to script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir
Write-Host "Working directory: $scriptDir" -ForegroundColor Gray
```

**Why:** Ensures script always runs from the `final` directory regardless of where it's called from.

---

### **Issue 2: Conda Deactivation Errors**
**Problem:** `conda deactivate` would fail if conda wasn't active or installed.

**Fix:**
```powershell
# Deactivate conda (if active)
try {
    conda deactivate 2>$null
} catch {
    # Conda not active or not installed
}
```

**Why:** Gracefully handles cases where conda is not available.

---

### **Issue 3: Relative Paths in Subprocesses**
**Problem:** Backend and UI scripts used relative paths that might not work in new PowerShell windows.

**Fix:**
```powershell
$backendScript = @"
Set-Location '$scriptDir'
try { conda deactivate 2>`$null } catch { }
& '$scriptDir\$venvPath\Scripts\Activate.ps1'
`$env:PYTHONWARNINGS='ignore'
Write-Host 'Backend starting on http://localhost:8000' -ForegroundColor Green
python backend/app.py
"@
```

**Why:** Uses absolute paths (`$scriptDir`) to ensure scripts work from any location.

---

### **Issue 4: No File Verification**
**Problem:** Script would fail silently if required files were missing.

**Fix:**
```powershell
# Verify required files exist
Write-Host "`nVerifying files..." -ForegroundColor Gray
$requiredFiles = @("backend\app.py", "ui\gradio_app.py")
$missingFiles = @()

foreach ($file in $requiredFiles) {
    if (-not (Test-Path $file)) {
        $missingFiles += $file
        Write-Host "  ❌ Missing: $file" -ForegroundColor Red
    } else {
        Write-Host "  ✅ Found: $file" -ForegroundColor Gray
    }
}

if ($missingFiles.Count -gt 0) {
    Write-Host "`n❌ ERROR: Missing required files!" -ForegroundColor Red
    Write-Host "Make sure you're running this script from the 'final' directory" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}
```

**Why:** Catches missing files early with clear error messages.

---

## How to Use

### **Method 1: Right-click (Recommended)**
```
1. Navigate to: C:\Users\VM02\Desktop\THESIS (SALAS)\SECOND PHASE ENV\final
2. Right-click on run_all.ps1
3. Select "Run with PowerShell"
```

### **Method 2: PowerShell Terminal**
```powershell
cd "C:\Users\VM02\Desktop\THESIS (SALAS)\SECOND PHASE ENV\final"
.\run_all.ps1
```

### **Method 3: From Any Location**
```powershell
# Works from anywhere now!
& "C:\Users\VM02\Desktop\THESIS (SALAS)\SECOND PHASE ENV\final\run_all.ps1"
```

---

## Expected Output

```
========================================
Fish Speech TTS - Complete Startup
========================================

Working directory: C:\Users\VM02\Desktop\THESIS (SALAS)\SECOND PHASE ENV\final

Checking for existing processes...
  No existing processes found

✅ Found venv312 (Python 3.12)

Verifying files...
  ✅ Found: backend\app.py
  ✅ Found: ui\gradio_app.py

Starting Backend Server...
Waiting for backend to start...

Starting Gradio UI...

========================================
Services Started!
========================================
Backend API: http://localhost:8000
Gradio UI: http://localhost:7860

Press Enter to open Gradio UI in browser...
```

---

## Troubleshooting

### **Error: "No virtual environment found"**
```powershell
# Create venv312
uv venv --python 3.12 venv312

# Or use standard venv
python -m venv venv312
```

### **Error: "Missing required files"**
```
Make sure you're in the correct directory:
cd "C:\Users\VM02\Desktop\THESIS (SALAS)\SECOND PHASE ENV\final"
```

### **Error: "Port already in use"**
```
The script automatically kills existing processes on ports 7860 and 8000.
If this fails, manually kill them:

# Find processes
netstat -ano | findstr ":8000"
netstat -ano | findstr ":7860"

# Kill by PID
taskkill /PID <PID> /F
```

### **Backend/UI won't start**
```
Check the spawned PowerShell windows for error messages.
Common issues:
- Missing dependencies: pip install -r requirements.txt
- Wrong Python version: Use Python 3.12
- Model not found: Check MODEL_DIR in .env
```

---

## What Changed

| Line | Before | After | Why |
|------|--------|-------|-----|
| 5-9 | No directory change | Added `Set-Location $scriptDir` | Ensures correct working directory |
| 50-55 | Direct `conda deactivate` | Wrapped in try-catch | Handles conda not installed |
| 72-91 | No file verification | Added verification loop | Catches missing files early |
| 75-81 | Relative paths in scripts | Absolute paths with `$scriptDir` | Works from any location |
| 95-102 | No status messages | Added "Backend starting..." | Better user feedback |

---

## Benefits

✅ **Robust:** Works from any directory
✅ **Clear:** Shows exactly what's happening
✅ **Safe:** Verifies files before starting
✅ **Helpful:** Clear error messages
✅ **Flexible:** Handles conda/no-conda setups

---

## Testing

```powershell
# Test 1: Run from script directory
cd "C:\Users\VM02\Desktop\THESIS (SALAS)\SECOND PHASE ENV\final"
.\run_all.ps1

# Test 2: Run from parent directory
cd "C:\Users\VM02\Desktop\THESIS (SALAS)\SECOND PHASE ENV"
.\final\run_all.ps1

# Test 3: Run from anywhere
& "C:\Users\VM02\Desktop\THESIS (SALAS)\SECOND PHASE ENV\final\run_all.ps1"

# All should work now! ✅
```

---

## Status: ✅ FIXED

All issues resolved:
- [x] Working directory set correctly
- [x] Conda deactivation error handling
- [x] Absolute paths in subprocesses
- [x] File verification added
- [x] Better error messages
- [x] User feedback improved

**The script should now work reliably from any location!** 🚀
