@echo off
setlocal enabledelayedexpansion
title DYNAMO MAGICIAN - First Time Setup
color 0A
cd /d "%~dp0"

echo.
echo  =====================================================
echo        DYNAMO MAGICIAN  --  FIRST TIME SETUP
echo  =====================================================
echo.
echo  Welcome! This will get everything installed for you.
echo.
echo  HOW LONG WILL THIS TAKE?
echo    Step 1-2  :  about 5-15 minutes
echo    Step 3    :  30-60 minutes  (downloads ~30 GB)
echo.
echo  You can leave this window open and come back later.
echo  It is safe to close and re-run if anything goes wrong.
echo.
echo  Press any key to begin...
pause >nul
echo.


REM =====================================================
echo  =====================================================
echo    STEP 1 of 4  --  Checking Python
echo  =====================================================
echo.
REM =====================================================

REM Try py launcher first, then python
set PYTHON=
py -3 --version >nul 2>&1
if not errorlevel 1 ( set PYTHON=py -3 & goto :python_ok )
python --version >nul 2>&1
if not errorlevel 1 ( set PYTHON=python & goto :python_ok )

REM Python not found -- download the installer
color 0E
echo  Python is not installed on this computer.
echo  Downloading the Python installer now...
echo.
powershell -NoProfile -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.12.9/python-3.12.9-amd64.exe' -OutFile '%TEMP%\python_setup.exe' -UseBasicParsing" >nul 2>&1
if errorlevel 1 (
    echo  Could not download Python automatically.
    echo  Please go to  https://www.python.org/downloads/  and install Python 3.12.
    echo  Make sure to check the box that says "Add Python to PATH"!
    echo.
    pause
    exit /b 1
)
echo  The Python installer is about to open.
echo.
echo  IMPORTANT  --  In the installer, before clicking anything:
echo    Check the box at the bottom:  "Add Python to PATH"
echo    Then click  "Install Now"
echo.
echo  After the installer finishes, close it and
echo  run this setup file again.
echo.
pause
start /wait "" "%TEMP%\python_setup.exe"
echo.
echo  If Python installed successfully, close this window
echo  and double-click  setup.bat  again to continue.
echo.
pause
exit /b 0

:python_ok
color 0A
echo  Python is ready!
echo.


REM =====================================================
echo  =====================================================
echo    STEP 2 of 4  --  Installing required software
echo  =====================================================
echo  (Lots of text will scroll by -- that is normal!)
echo  =====================================================
echo.
REM =====================================================

if exist ".venv\Scripts\activate.bat" (
    echo  Previous install found, using it.
    echo.
) else (
    echo  Setting up isolated software environment...
    %PYTHON% -m venv .venv
    if errorlevel 1 (
        color 0C
        echo.
        echo  Something went wrong setting up the environment.
        echo  Try running this file again. If it keeps failing,
        echo  please contact support.
        echo.
        pause
        exit /b 1
    )
)

call .venv\Scripts\activate.bat

echo  Updating package installer...
python -m pip install --upgrade pip -q --progress-bar off

echo  Installing PyTorch (GPU support -- this is the big one)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 -q --progress-bar off
if errorlevel 1 (
    color 0C
    echo.
    echo  Installation failed. Please check your internet connection
    echo  and run this file again.
    echo.
    pause
    exit /b 1
)

echo  Installing everything else...
pip install -r requirements.txt -q --progress-bar off
if errorlevel 1 (
    color 0C
    echo.
    echo  Installation failed. Please check your internet connection
    echo  and run this file again.
    echo.
    pause
    exit /b 1
)

echo  Software installed successfully!
echo.


REM =====================================================
echo  =====================================================
echo    STEP 3 of 4  --  Downloading AI model files
echo  =====================================================
echo  (About 30 GB total  --  this will take a while)
echo  (If it stops, just run this file again to resume)
echo  =====================================================
echo.
REM =====================================================

set HF_REPO=cahlen/lingbot-world-base-cam-nf4
set NEED_DL=0
if not exist "models_t5_umt5-xxl-enc-bf16.pth"              set NEED_DL=1
if not exist "Wan2.1_VAE.pth"                                set NEED_DL=1
if not exist "high_noise_model_bnb_nf4\model.safetensors"    set NEED_DL=1
if not exist "low_noise_model_bnb_nf4\model.safetensors"     set NEED_DL=1
if not exist "tokenizer\tokenizer.json"                       set NEED_DL=1

if "%NEED_DL%"=="0" (
    echo  AI files are already downloaded!
    echo.
    goto :models_done
)

echo  Downloading now. Go grab a coffee -- this takes a while!
echo.

hf download %HF_REPO% models_t5_umt5-xxl-enc-bf16.pth             --local-dir .
hf download %HF_REPO% Wan2.1_VAE.pth                               --local-dir .
hf download %HF_REPO% high_noise_model_bnb_nf4/model.safetensors   --local-dir .
hf download %HF_REPO% high_noise_model_bnb_nf4/config.json         --local-dir .
hf download %HF_REPO% high_noise_model_bnb_nf4/quantization_meta.json --local-dir .
hf download %HF_REPO% low_noise_model_bnb_nf4/model.safetensors    --local-dir .
hf download %HF_REPO% low_noise_model_bnb_nf4/config.json          --local-dir .
hf download %HF_REPO% low_noise_model_bnb_nf4/quantization_meta.json  --local-dir .
hf download %HF_REPO% tokenizer/tokenizer.json                     --local-dir .
hf download %HF_REPO% tokenizer/tokenizer_config.json              --local-dir .
hf download %HF_REPO% tokenizer/special_tokens_map.json            --local-dir .

if errorlevel 1 (
    color 0C
    echo.
    echo  Download did not finish. This is usually an internet issue.
    echo  Just run this file again -- it will pick up where it left off.
    echo.
    pause
    exit /b 1
)
echo  All AI files downloaded!
echo.

:models_done


REM =====================================================
echo  =====================================================
echo    STEP 4 of 4  --  Final check
echo  =====================================================
echo.
REM =====================================================

set ALL_OK=1
if not exist "models_t5_umt5-xxl-enc-bf16.pth"           set ALL_OK=0
if not exist "Wan2.1_VAE.pth"                             set ALL_OK=0
if not exist "high_noise_model_bnb_nf4\model.safetensors" set ALL_OK=0
if not exist "low_noise_model_bnb_nf4\model.safetensors"  set ALL_OK=0

if "%ALL_OK%"=="0" (
    color 0C
    echo  Some files seem to be missing.
    echo  Close this window and run  setup.bat  again.
    echo.
    pause
    exit /b 1
)

echo  Everything looks great!
echo.


color 0A
echo.
echo  =====================================================
echo.
echo        SETUP IS COMPLETE!  YOU ARE READY TO GO!
echo.
echo    To use Dynamo Magician, double-click:
echo.
echo           run.bat
echo.
echo  =====================================================
echo.
pause
exit /b 0
