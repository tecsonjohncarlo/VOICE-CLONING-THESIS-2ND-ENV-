@echo off
echo Starting Optimized Fish Speech Backend...
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
    echo Please create venv312 first.
    pause
    exit /b 1
)

REM Load environment variables
if exist .env (
    echo Loading environment variables from .env
)

REM Suppress NumPy warnings
set PYTHONWARNINGS=ignore

REM Start FastAPI server
echo.
echo Starting server...
python backend/app.py
if errorlevel 1 (
    echo.
    echo ERROR: Backend failed to start!
    echo Check the error messages above.
    echo.
)

pause
