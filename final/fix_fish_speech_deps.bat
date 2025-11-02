@echo off
echo ========================================
echo Fish Speech Dependencies Fix
echo ========================================
echo.
echo This will install missing Fish Speech dependencies
echo.

REM Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

echo.
echo Installing missing dependencies...
echo.

REM Install core dependencies that are commonly missing
pip install hydra-core omegaconf pyrootutils loguru click

echo.
echo Installing Fish Speech package...
cd fish-speech
pip install -e .
cd ..

echo.
echo ========================================
echo Fix Complete!
echo ========================================
echo.
echo Try running the backend again:
echo   start_backend.bat
echo.
pause
