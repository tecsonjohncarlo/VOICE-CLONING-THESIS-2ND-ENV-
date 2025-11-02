@echo off
echo ========================================
echo Fish Speech Installation Script
echo ========================================
echo.

REM Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

echo.
echo Choose installation method:
echo.
echo 1. Clone Fish Speech to final folder (Recommended)
echo 2. Install Fish Speech as Python package
echo.
set /p choice="Enter choice (1 or 2): "

if "%choice%"=="1" goto clone
if "%choice%"=="2" goto package
echo Invalid choice!
pause
exit /b 1

:clone
echo.
echo Cloning Fish Speech repository...
echo This will download ~500MB of code
echo.

REM Check if git is installed
git --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Git is not installed!
    echo Please install Git from: https://git-scm.com/download/win
    pause
    exit /b 1
)

REM Clone repository
if exist fish-speech (
    echo Fish Speech folder already exists!
    set /p overwrite="Overwrite? (y/n): "
    if /i not "%overwrite%"=="y" (
        echo Installation cancelled.
        pause
        exit /b 0
    )
    echo Removing old installation...
    rmdir /s /q fish-speech
)

echo Cloning repository...
git clone https://github.com/fishaudio/fish-speech.git

if errorlevel 1 (
    echo ERROR: Failed to clone repository!
    pause
    exit /b 1
)

echo.
echo Installing Fish Speech dependencies...
cd fish-speech

echo Installing core dependencies...
pip install hydra-core omegaconf pyrootutils loguru click soundfile torchaudio

echo Installing Fish Speech package...
pip install -e .

if errorlevel 1 (
    echo WARNING: Some dependencies failed to install
    echo Trying alternative installation...
    pip install -r pyproject.toml 2>nul
)

cd ..

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Fish Speech is now installed in: %CD%\fish-speech
echo.
echo The .env file will be automatically updated...
echo.

REM Update .env file
if exist .env (
    echo Backing up .env to .env.backup...
    copy .env .env.backup >nul
    
    REM Remove old FISH_SPEECH_DIR line if exists
    findstr /v "FISH_SPEECH_DIR" .env > .env.tmp
    move /y .env.tmp .env >nul
    
    REM Add new FISH_SPEECH_DIR
    echo FISH_SPEECH_DIR=%CD%\fish-speech>> .env
    echo Updated .env file
) else (
    echo Creating .env file...
    copy .env.example .env
    echo FISH_SPEECH_DIR=%CD%\fish-speech>> .env
)

echo.
echo You can now start the backend:
echo   start_backend.bat
echo.
pause
exit /b 0

:package
echo.
echo Installing Fish Speech as Python package...
echo This will download and install Fish Speech globally
echo.

pip install git+https://github.com/fishaudio/fish-speech.git

if errorlevel 1 (
    echo ERROR: Installation failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Fish Speech is now installed as a Python package
echo The system will auto-detect it automatically
echo.
echo You can now start the backend:
echo   start_backend.bat
echo.
pause
exit /b 0
