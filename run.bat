@echo off
setlocal enabledelayedexpansion

echo [INFO] Starting KokoroGUI...

:: Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    pause
    exit /b 1
)

:: Set up Virtual Environment if not exists
if not exist .venv (
    echo [INFO] Creating virtual environment...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

:: Activate Venv
call .venv\Scripts\activate

:: Install/Update Requirements
if exist requirements.txt (
    echo [INFO] Checking requirements...
    pip install -r requirements.txt --quiet
)

:: Start Application
echo [INFO] Launching GUI...
python main.py

pause
