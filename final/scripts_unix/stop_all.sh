#!/bin/bash
# Stop all Fish Speech TTS services

echo "========================================"
echo "Stopping Fish Speech TTS Services"
echo "========================================"
echo ""

# Kill processes on port 7860 (Gradio)
if lsof -Pi :7860 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    pid=$(lsof -Pi :7860 -sTCP:LISTEN -t)
    echo "Stopping Gradio UI (PID: $pid)..."
    kill -9 $pid 2>/dev/null
    echo "  Gradio UI stopped"
else
    echo "  Gradio UI not running"
fi

# Kill processes on port 8000 (Backend)
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    pid=$(lsof -Pi :8000 -sTCP:LISTEN -t)
    echo "Stopping Backend API (PID: $pid)..."
    kill -9 $pid 2>/dev/null
    echo "  Backend API stopped"
else
    echo "  Backend API not running"
fi

# Kill processes on port 8501 (Streamlit)
if lsof -Pi :8501 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    pid=$(lsof -Pi :8501 -sTCP:LISTEN -t)
    echo "Stopping Streamlit UI (PID: $pid)..."
    kill -9 $pid 2>/dev/null
    echo "  Streamlit UI stopped"
else
    echo "  Streamlit UI not running"
fi

echo ""
echo "All services stopped"
