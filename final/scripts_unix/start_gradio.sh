#!/bin/bash
# Start Fish Speech Gradio UI

echo "========================================"
echo "Fish Speech Gradio UI"
echo "========================================"
echo ""

# Deactivate conda if active
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    conda deactivate 2>/dev/null || true
fi

# Find and activate virtual environment
if [ -d "venv312" ]; then
    VENV_PATH="venv312"
    echo "Using venv312 (Python 3.12)"
elif [ -d "venv" ]; then
    VENV_PATH="venv"
    echo "Using venv"
else
    echo "ERROR: No virtual environment found!"
    echo "Please run: python3.12 -m venv venv312"
    exit 1
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Set Python warnings
export PYTHONWARNINGS='ignore'

# Check if Gradio is already running
if lsof -Pi :7860 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "WARNING: Gradio UI is already running on port 7860"
    echo "Stop it first with: ./scripts_unix/stop_all.sh"
    exit 1
fi

# Check if backend is running
echo "Checking backend connection..."
if ! lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "WARNING: Backend is not running on port 8000"
    echo "Start it first with: ./scripts_unix/start_backend.sh"
    echo "Or use: ./scripts_unix/run_all.sh to start both"
    exit 1
fi

echo "Backend is running. Starting Gradio UI..."
echo ""
echo "Gradio UI will be available at: http://localhost:7860"
echo ""
python ui/gradio_app.py
