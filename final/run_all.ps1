Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Fish Speech TTS - Complete Startup" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Change to script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir
Write-Host "Working directory: $scriptDir" -ForegroundColor Gray
Write-Host ""

# Kill existing processes on ports 7860 and 8000
Write-Host "Checking for existing processes..." -ForegroundColor Yellow

$pidsToKill = @()

# Check port 7860 (Gradio)
$port7860 = netstat -ano | findstr ":7860" | findstr "LISTENING"
if ($port7860) {
    $pid = ($port7860 -split '\s+')[-1]
    $pidsToKill += $pid
    Write-Host "  Found process on port 7860 (PID: $pid)" -ForegroundColor Gray
}

# Check port 8000 (Backend)
$port8000 = netstat -ano | findstr ":8000" | findstr "LISTENING"
if ($port8000) {
    $pid = ($port8000 -split '\s+')[-1]
    $pidsToKill += $pid
    Write-Host "  Found process on port 8000 (PID: $pid)" -ForegroundColor Gray
}

# Kill processes if found
if ($pidsToKill.Count -gt 0) {
    Write-Host "  Stopping existing processes..." -ForegroundColor Yellow
    foreach ($pid in $pidsToKill) {
        try {
            Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
        } catch {
            # Ignore errors
        }
    }
    Start-Sleep -Seconds 2
    Write-Host "  Existing processes stopped" -ForegroundColor Green
} else {
    Write-Host "  No existing processes found" -ForegroundColor Green
}

Write-Host ""

# Deactivate conda (if active)
try {
    conda deactivate 2>$null
} catch {
    # Conda not active or not installed
}

# Find virtual environment
$venvPath = $null
if (Test-Path "venv312\Scripts\Activate.ps1") {
    $venvPath = "venv312"
    Write-Host "✅ Found venv312 (Python 3.12)" -ForegroundColor Green
} elseif (Test-Path "venv\Scripts\Activate.ps1") {
    $venvPath = "venv"
    Write-Host "✅ Found venv" -ForegroundColor Yellow
} else {
    Write-Host "❌ ERROR: No virtual environment found!" -ForegroundColor Red
    Write-Host "Please run: uv venv --python 3.12 venv312" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

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

# Start Backend
Write-Host "`nStarting Backend Server..." -ForegroundColor Yellow
$backendScript = @"
Set-Location '$scriptDir'
try { conda deactivate 2>`$null } catch { }
& '$scriptDir\$venvPath\Scripts\Activate.ps1'
`$env:PYTHONWARNINGS='ignore'
Write-Host 'Backend starting on http://localhost:8000' -ForegroundColor Green
python backend/app.py
"@

Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendScript -WindowStyle Normal

Write-Host "Waiting for backend to start..." -ForegroundColor Gray
Start-Sleep -Seconds 5

# Start Gradio UI
Write-Host "`nStarting Gradio UI..." -ForegroundColor Yellow
$uiScript = @"
Set-Location '$scriptDir'
try { conda deactivate 2>`$null } catch { }
& '$scriptDir\$venvPath\Scripts\Activate.ps1'
`$env:PYTHONWARNINGS='ignore'
Write-Host 'Gradio UI starting on http://localhost:7860' -ForegroundColor Green
python ui/gradio_app.py
"@

Start-Process powershell -ArgumentList "-NoExit", "-Command", $uiScript -WindowStyle Normal

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Services Started!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Backend API: http://localhost:8000" -ForegroundColor White
Write-Host "Gradio UI: http://localhost:7860" -ForegroundColor White
Write-Host "`nPress Enter to open Gradio UI in browser..." -ForegroundColor Gray
Read-Host
Start-Process "http://localhost:7860"
