#!/bin/bash

echo "================================================================="
echo "   DiffLocks Studio - Launcher (Linux/Mac)"
echo "================================================================="

# 1. Create environment
if [ ! -d "venv" ]; then
    echo "[INFO] Creating virtual environment..."
    python3 -m venv venv
fi

# 2. Activate
source venv/bin/activate

# 3. Install
if [ ! -f "venv/installed.flag" ]; then
    echo "[INFO] Installing dependencies..."
    pip install --upgrade pip
    pip install "numpy<2.0.0"
    pip install -r requirements.txt
    touch venv/installed.flag
fi

# 4. Run
echo "[INFO] Starting App..."
python3 app.py
