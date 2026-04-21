#!/bin/bash
echo "================================================"
echo "  Disease Prediction System - Mac/Linux Setup"
echo "================================================"

echo ""
echo "[Step 1] Creating virtual environment..."
python3 -m venv venv

echo ""
echo "[Step 2] Activating virtual environment..."
source venv/bin/activate

echo ""
echo "[Step 3] Installing required libraries..."
pip install -r requirements.txt

echo ""
echo "[Step 4] Training the ML model..."
python backend/train_model.py

echo ""
echo "[Step 5] Starting Flask server..."
echo "   Open your browser at: http://127.0.0.1:5000"
echo ""
python backend/app.py
