@echo off
setlocal

:: Window Title
title DiffLocks Studio - Launcher

echo =================================================================
echo    DiffLocks Studio - Auto-Installer & Launcher
echo =================================================================
echo.

:: 1. Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found.
    echo Please install Python 3.10 or 3.11 from python.org
    echo IMPORTANT: Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b
)

:: 2. Create Virtual Environment (if missing)
if not exist "venv" (
    echo [INFO] Creating virtual environment (venv folder)...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Could not create virtual environment.
        pause
        exit /b
    )
)

:: 3. Activate Environment
call venv\Scripts\activate.bat

:: 4. Install Dependencies
if not exist "venv\installed.flag" (
    echo.
    echo [INFO] First time setup: Installing dependencies...
    echo        (This may take a few minutes depending on your internet)
    echo.
    
    :: Upgrade pip
    python -m pip install --upgrade pip

    :: Force NumPy compatibility first
    pip install "numpy<2.0.0"

    :: Install project requirements
    pip install -r requirements.txt
    
    :: Install PyTorch with CUDA support (Specific for Windows)
    echo [INFO] Installing PyTorch with CUDA support...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install dependencies.
        pause
        exit /b
    )
    
    :: Create flag file
    echo installed > venv\installed.flag
)

:: 5. Launch App
echo.
echo =================================================================
echo    Starting DiffLocks Studio...
echo    Wait for the URL to appear below (e.g., http://127.0.0.1:7860)
echo =================================================================
echo.

python app.py

pause
