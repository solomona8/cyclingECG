#!/bin/bash
# Quick script to test 30-day averages locally

echo "Starting backend server locally..."
echo "Make sure you have installed requirements: pip install -r requirements.txt"
echo ""
echo "Starting server on http://localhost:8000"
echo "Update your iOS app API URL to: http://YOUR_LOCAL_IP:8000"
echo "Find your local IP with: ipconfig getifaddr en0 (macOS)"
echo ""

cd /home/user/cyclingECG
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
