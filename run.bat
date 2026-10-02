@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo [INFO] Starting KokoroGUI...

:: Find Python. The py launcher wins when it has a Python 3, else python on PATH.
set "PY="
where py >nul 2>&1
if !errorlevel! equ 0 (
    py -3 --version >nul 2>&1
    if !errorlevel! equ 0 set "PY=py -3"
)
if not defined PY (
    where python >nul 2>&1
    if !errorlevel! equ 0 set "PY=python"
)
if not defined PY (
    echo [ERROR] Python is not installed or not in PATH.
    pause
    exit /b 1
)

set "VENV_PY=%~dp0.venv\Scripts\python.exe"

:: Set up Virtual Environment if not exists
if not exist "%VENV_PY%" (
    echo [INFO] Creating virtual environment...
    %PY% -m venv .venv
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

:: Install/Update Requirements
if exist requirements.txt (
    echo [INFO] Checking requirements...
    "%VENV_PY%" -m pip install -r requirements.txt --quiet
)

:: Start Application
echo [INFO] Launching GUI...
"%VENV_PY%" main.py

pause
