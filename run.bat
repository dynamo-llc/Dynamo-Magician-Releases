@echo off
setlocal
title DYNAMO MAGICIAN
color 0A
cd /d "%~dp0"

echo.
echo  =====================================================
echo           DYNAMO MAGICIAN  --  STARTING...
echo  =====================================================
echo.

REM Did they run setup first?
if not exist ".venv\Scripts\activate.bat" (
    color 0C
    echo  It looks like setup has not been run yet.
    echo.
    echo  Please double-click  setup.bat  first,
    echo  then come back and run this file again.
    echo.
    pause
    exit /b 1
)

REM Are the AI model files here?
if not exist "models_t5_umt5-xxl-enc-bf16.pth" (
    color 0C
    echo  The AI model files are missing.
    echo.
    echo  Please double-click  setup.bat  first,
    echo  then come back and run this file again.
    echo.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

echo  Starting up...
echo  Your browser will open automatically in a few seconds.
echo.
echo  -------------------------------------------------------
echo   DO NOT CLOSE THIS WINDOW while using Dynamo Magician.
echo   To stop the app, just close this window.
echo  -------------------------------------------------------
echo.

REM Open browser after a short delay
start /b cmd /c "timeout /t 4 /nobreak >nul && start http://localhost:8000"

REM Launch the server
uvicorn server:app --host 127.0.0.1 --port 8000

echo.
echo  Dynamo Magician has stopped.
pause
endlocal

