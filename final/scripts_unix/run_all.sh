#!/bin/bash
# Fish Speech TTS - Complete Startup Script for macOS/Linux

echo "========================================"
echo "Fish Speech TTS - Complete Startup"
echo "========================================"
echo ""

# Kill existing processes on ports 7860 and 8000
echo "Checking for existing processes..."

# Check and kill port 7860 (Gradio)
if lsof -Pi :7860 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    pid=$(lsof -Pi :7860 -sTCP:LISTEN -t)
    echo "  Found process on port 7860 (PID: $pid)"
    kill -9 $pid 2>/dev/null
    echo "  Stopped process on port 7860"
fi

# Check and kill port 8000 (Backend)
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    pid=$(lsof -Pi :8000 -sTCP:LISTEN -t)
    echo "  Found process on port 8000 (PID: $pid)"
    kill -9 $pid 2>/dev/null
    echo "  Stopped process on port 8000"
fi

echo "  Process cleanup complete"
echo ""

# Deactivate conda if active
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "Deactivating conda environment..."
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

# Start Backend in background
echo ""
echo "Starting Backend Server..."
python backend/app.py > backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend started (PID: $BACKEND_PID)"

# Wait for backend to start
echo "Waiting for backend to start..."
sleep 5

# Check if backend is running
if ! ps -p $BACKEND_PID > /dev/null; then
    echo "ERROR: Backend failed to start. Check backend.log for details."
    exit 1
fi

# Start Gradio UI in background
echo ""
echo "Starting Gradio UI..."
python ui/gradio_app.py > gradio.log 2>&1 &
GRADIO_PID=$!
echo "Gradio UI started (PID: $GRADIO_PID)"

echo ""
echo "========================================"
echo "Services Started!"
echo "========================================"
echo "Backend API: http://localhost:8000"
echo "Gradio UI: http://localhost:7860"
echo ""
echo "Backend PID: $BACKEND_PID"
echo "Gradio PID: $GRADIO_PID"
echo ""
echo "Logs:"
echo "  Backend: backend.log"
echo "  Gradio: gradio.log"
echo ""
echo "To stop services, run: ./scripts_unix/stop_all.sh"
echo ""

# Open browser (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Opening Gradio UI in browser..."
    sleep 2
    open http://localhost:7860
fi

# Open browser (Linux with xdg-open)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if command -v xdg-open &> /dev/null; then
        echo "Opening Gradio UI in browser..."
        sleep 2
        xdg-open http://localhost:7860 2>/dev/null
    fi
fi

echo "Press Ctrl+C to stop all services"
echo ""

# Wait for user interrupt
trap "echo ''; echo 'Stopping services...'; kill $BACKEND_PID $GRADIO_PID 2>/dev/null; exit 0" INT TERM

# Keep script running
wait
