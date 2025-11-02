#!/bin/bash
# Start Fish Speech Backend API

echo "========================================"
echo "Fish Speech Backend API"
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

# Check if backend is already running
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    echo "WARNING: Backend is already running on port 8000"
    echo "Stop it first with: ./scripts_unix/stop_all.sh"
    exit 1
fi

# Start backend
echo ""
echo "Starting Backend API on http://localhost:8000"
echo ""
python backend/app.py
