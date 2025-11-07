@echo off
echo Starting Streamlit UI...
echo.

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

REM Check if streamlit is installed in venv312
if exist venv312\Scripts\streamlit.exe (
    echo Using streamlit from venv312
    venv312\Scripts\streamlit.exe run ui/streamlit_app.py
) else (
    echo Streamlit not found in venv312, installing...
    venv312\Scripts\pip.exe install streamlit
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to install streamlit
        echo Please run: venv312\Scripts\pip.exe install streamlit
        pause
        exit /b 1
    )
    venv312\Scripts\streamlit.exe run ui/streamlit_app.py
)

pause
