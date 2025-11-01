@echo off
echo Starting Streamlit UI...
echo.

REM Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
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

echo Backend is running. Starting Streamlit UI...
streamlit run ui/streamlit_app.py

pause
