@echo off
setlocal
title Dynamo Magician - Setup
color 0A
cd /d "%~dp0"

echo.
echo  +---------------------------------------------------+
echo  ^|         DYNAMO MAGICIAN  -  First-Time Setup      ^|
echo  +---------------------------------------------------+
echo.
echo  This will install everything you need to run
echo  Dynamo Magician.  It only takes a few minutes.
echo.


REM --- 1. Find Python ----------------------------------------------------------

set PYTHON=
py -3.12 --version >nul 2>&1 && set "PYTHON=py -3.12" && goto :python_ok
py -3    --version >nul 2>&1 && set "PYTHON=py -3"    && goto :python_ok
python   --version >nul 2>&1 && set "PYTHON=python"   && goto :python_ok

REM Python not found
color 0E
echo  Python is not installed on this computer.
echo  Downloading the Python 3.12 installer now...
echo.
powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.12.9/python-3.12.9-amd64.exe' -OutFile '%TEMP%\python_setup.exe' -UseBasicParsing"
if errorlevel 1 (
    color 0C
    echo.
    echo  ERROR: Could not download the Python installer.
    echo  Please visit https://www.python.org/downloads/ and install Python 3.12.
    echo  Make sure to check "Add Python to PATH" before clicking Install Now.
    echo  Then run setup.bat again.
    echo.
    pause
    exit /b 1
)
echo  The Python installer is about to open.
echo.
echo  IMPORTANT: Check the box "Add Python to PATH" before clicking Install Now.
echo.
pause
start /wait "" "%TEMP%\python_setup.exe"
echo.
echo  Done.  Close this window and double-click setup.bat again.
echo.
pause
exit /b 0

:python_ok
color 0A
for /f "tokens=*" %%V in ('%PYTHON% --version 2^>^&1') do echo  [OK] %%V found
echo.


REM --- 2. Create virtual environment -------------------------------------------

if exist ".venv\Scripts\activate.bat" (
    echo  [OK] Python environment already exists
) else (
    echo  [ ] Creating Python environment...
    %PYTHON% -m venv .venv
    if errorlevel 1 (
        color 0C
        echo.
        echo  ERROR: Could not create the Python environment.
        echo  Close this window and run setup.bat again.
        echo.
        pause
        exit /b 1
    )
    echo  [OK] Python environment created
)
echo.
call .venv\Scripts\activate.bat


REM --- 3. Install packages -----------------------------------------------------

echo  Installing required packages  (one-time, about 5-15 minutes)
echo.

echo  [1/3] Upgrading pip...
python -m pip install --upgrade pip --quiet --progress-bar off
if errorlevel 1 goto :pkg_error
echo  [1/3] Done

echo  [2/3] Installing PyTorch with CUDA 12.8 support  (about 3 GB)...
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --quiet --progress-bar off
if errorlevel 1 goto :pkg_error
echo  [2/3] Done

echo  [3/3] Installing remaining packages...
python -m pip install -r requirements.txt --quiet --progress-bar off
if errorlevel 1 goto :pkg_error
echo  [3/3] Done

echo.
goto :pkg_done

:pkg_error
color 0C
echo.
echo  ERROR: Package installation failed.
echo  Check your internet connection and run setup.bat again.
echo.
pause
exit /b 1

:pkg_done


REM --- 4. Open graphical installer ---------------------------------------------

echo  All packages installed.  Opening the setup window...
echo.
python _installer_gui.py

exit /b 0
