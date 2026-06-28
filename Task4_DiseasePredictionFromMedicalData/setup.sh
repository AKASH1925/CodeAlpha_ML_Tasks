#!/bin/bash
# Quick setup for Disease Prediction project
set -e
echo "==================================="
echo " Disease Prediction - Setup"
echo "==================================="
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 not installed."
    exit 1
fi
echo "[1/3] Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate
echo "[2/3] Installing dependencies..."
pip install -r requirements.txt --quiet
echo "[3/3] Setup complete!"
echo ""
echo "Run with:"
echo "  python predict.py"
echo ""
echo "Or specific dataset:"
echo "  python predict.py --dataset heart"
echo "  python predict.py --dataset diabetes"
echo "  python predict.py --dataset breast_cancer"