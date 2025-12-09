#!/bin/bash

# cyclingECG Setup Script
# This script sets up the Python environment and starts the FastAPI server

set -e  # Exit on error

PROJECT_DIR="/Users/aaronsolomon/Documents/LocalCode/cyclingECG"
VENV_DIR="$PROJECT_DIR/.venv"

echo "🚀 cyclingECG Setup Script"
echo "=========================="

# Check if we're in the project directory, if not, navigate there
if [ "$PWD" != "$PROJECT_DIR" ]; then
    echo "📁 Navigating to project directory: $PROJECT_DIR"
    cd "$PROJECT_DIR"
fi

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv .venv
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Check if requirements are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "📦 Installing dependencies from requirements.txt..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "✓ Dependencies already installed"
    echo "  (Run 'pip install -r requirements.txt' to update)"
fi

# Display environment info
echo ""
echo "✅ Setup complete!"
echo ""
echo "Environment information:"
echo "  Python: $(python --version)"
echo "  Virtual environment: $VENV_DIR"
echo "  Project directory: $PROJECT_DIR"
echo ""
echo "To start the server, run:"
echo "  ./start_server.sh"
echo ""
echo "Or manually:"
echo "  source .venv/bin/activate"
echo "  uvicorn app.main:app --host 0.0.0.0 --port 8000"
