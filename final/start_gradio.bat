@echo off
echo Starting Gradio UI...
echo.

REM Deactivate conda if active
call conda deactivate 2>nul

REM Activate venv312 (correct Python 3.12)
if exist venv312\Scripts\activate.bat (
    call venv312\Scripts\activate.bat
    echo Using venv312 (Python 3.12)
) else if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
    echo Using venv
) else (
    echo ERROR: No virtual environment found!
    pause
    exit /b 1
)

REM Check if backend is running
echo Checking backend connection...
curl -s http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: Backend is not running!
    echo Please start the backend first using start_backend.bat
    echo.
    pause
    exit /b 1
)

echo Backend is running. Starting Gradio UI...

REM Suppress NumPy warnings
set PYTHONWARNINGS=ignore

python ui/gradio_app.py
if errorlevel 1 (
    echo.
    echo ERROR: Gradio UI failed to start!
    echo Check the error messages above.
    echo.
)

pause
