#!/bin/bash

# cyclingECG Server Startup Script
# Starts the FastAPI server with the correct environment

set -e  # Exit on error

PROJECT_DIR="/home/user/cyclingECG"
VENV_DIR="$PROJECT_DIR/.venv"

# Navigate to project directory
cd "$PROJECT_DIR"

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Virtual environment not found. Please run ./setup.sh first"
    exit 1
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Check if dependencies are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "❌ Dependencies not installed. Please run ./setup.sh first"
    exit 1
fi

echo "🚀 Starting FastAPI server..."
echo "   Server will be available at http://0.0.0.0:8000"
echo "   API docs at http://0.0.0.0:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
