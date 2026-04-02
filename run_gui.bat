@echo off
set "VENV_PYTHON=.venv\Scripts\python.exe"
if exist "%VENV_PYTHON%" (
    echo [INFO] Launching LingBot-World GUI in venv...
    "%VENV_PYTHON%" gui.py
) else (
    echo [ERROR] Virtual environment not found. Please ensure the setup is complete.
    pause
)
