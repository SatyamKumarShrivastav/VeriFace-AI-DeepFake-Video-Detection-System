@echo off
title DeepFake Detection System - S. K. Shrivastav
color 0B
echo.
echo  ============================================================
echo   DeepFake Video Detection System
echo   Welcome! Mr. S. K. Shrivastav
echo  ============================================================
echo.

cd /d "%~dp0"

:: Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found. Please install Python 3.10+
    pause
    exit /b 1
)

:: Install dependencies if needed
echo  [1/3] Checking dependencies...
pip install -r backend\requirements.txt -q --disable-pip-version-check
if errorlevel 1 (
    echo  [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo  [2/3] Starting backend server...
echo  [3/3] Open your browser at: http://127.0.0.1:8000
echo.
echo  Press Ctrl+C to stop the server.
echo  ============================================================
echo.

cd backend
python main.py

pause
