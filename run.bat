@echo off
setlocal
title Dynamo Magician
color 0A
cd /d "%~dp0"

echo.
echo  +---------------------------------------------------+
echo  ^|            DYNAMO MAGICIAN  -  Starting           ^|
echo  +---------------------------------------------------+
echo.

REM Did they run setup first?
if not exist ".venv\Scripts\activate.bat" (
    color 0C
    echo  Setup has not been run yet.
    echo  Please double-click setup.bat first, then try again.
    echo.
    pause
    exit /b 1
)

REM Are the AI model files here?
if not exist "models_t5_umt5-xxl-enc-bf16.pth" (
    color 0C
    echo  AI model files are missing.
    echo  Please double-click setup.bat first, then try again.
    echo.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

echo  Starting server...
echo  Your browser will open automatically once it is ready.
echo.
echo  NOTE: Do not close this window while using Dynamo Magician.
echo        Close this window to stop the app.
echo.

REM Open the browser once the server is actually ready (polls /health)
start /b "" .venv\Scripts\python _wait_and_open.py

REM Launch the server  (python -m avoids broken .exe entry-point launchers)
.venv\Scripts\python -m uvicorn server:app --host 127.0.0.1 --port 8000

echo.
echo  Dynamo Magician has stopped.
pause
endlocal

