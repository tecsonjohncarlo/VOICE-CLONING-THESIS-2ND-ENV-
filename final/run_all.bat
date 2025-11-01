@echo off
echo ========================================
echo Fish Speech TTS - Complete Startup
echo ========================================
echo.

REM Deactivate conda if active
call conda deactivate 2>nul

REM Check for venv312 (Python 3.12)
if exist venv312\Scripts\activate.bat (
    set VENV_PATH=venv312
    echo Using venv312 (Python 3.12)
) else if exist venv\Scripts\activate.bat (
    set VENV_PATH=venv
    echo Using venv
) else (
    echo ERROR: No virtual environment found!
    echo Please create venv312 first.
    pause
    exit /b 1
)

echo.
echo Starting Backend Server...
echo.
start "Fish Speech Backend" cmd /k "call conda deactivate 2>nul && %VENV_PATH%\Scripts\activate.bat && set PYTHONWARNINGS=ignore && python backend/app.py"

echo Waiting for backend to start...
timeout /t 5 /nobreak >nul

echo.
echo Starting Gradio UI...
echo.
start "Fish Speech Gradio UI" cmd /k "call conda deactivate 2>nul && %VENV_PATH%\Scripts\activate.bat && set PYTHONWARNINGS=ignore && python ui/gradio_app.py"

echo.
echo ========================================
echo Services Started!
echo ========================================
echo.
echo Backend API: http://localhost:8000
echo Gradio UI: http://localhost:7860
echo.
echo Press any key to open Gradio UI in browser...
pause >nul
start http://localhost:7860
