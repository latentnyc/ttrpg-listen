@echo off
setlocal enabledelayedexpansion
title TTRPG Listen - Startup
color 0A

echo ============================================
echo   TTRPG Listen - Starting up...
echo ============================================
echo.

:: ---- Locate Python ----
set "PYTHON="
where python >nul 2>&1 && set "PYTHON=python"
if not defined PYTHON (
    where python3 >nul 2>&1 && set "PYTHON=python3"
)
if not defined PYTHON (
    echo [ERROR] Python not found on PATH.
    echo         Install Python 3.11+ from https://python.org
    pause
    exit /b 1
)

:: Verify version >= 3.11
for /f "tokens=2 delims= " %%v in ('%PYTHON% --version 2^>^&1') do set "PYVER=%%v"
echo   Python: %PYVER%

:: ---- Virtual environment ----
set "VENV_DIR=%~dp0.venv"

if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo.
    echo   Creating virtual environment...
    %PYTHON% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo   Virtual environment created.
    set "FRESH_VENV=1"
) else (
    set "FRESH_VENV=0"
)

call "%VENV_DIR%\Scripts\activate.bat"
echo   Activated: %VENV_DIR%

:: ---- Install / update dependencies ----
:: Always install on fresh venv; otherwise check if package is installed
if "%FRESH_VENV%"=="1" (
    goto :install
)

python -c "import ttrpglisten" >nul 2>&1
if errorlevel 1 goto :install

:: Check if pyproject.toml is newer than the egg-info
:: (crude staleness check - reinstall if source changed)
for /f %%A in ('dir /b /od "%~dp0pyproject.toml" "%VENV_DIR%\Lib\site-packages\ttrpg_listen*" 2^>nul ^| findstr /i "pyproject.toml"') do (
    if "%%A"=="pyproject.toml" goto :install
)
goto :skip_install

:install
echo.
echo   Installing dependencies (this may take a while on first run)...
echo   - PyTorch, Whisper, pyannote-audio, PySide6, etc.
echo.

:: Install PyTorch 2.8 with CUDA if missing, wrong version, or CPU-only
set "NEED_TORCH=0"
for /f "delims=" %%V in ('python -c "import torch; v=torch.__version__; c=torch.cuda.is_available(); print('OK' if v.startswith('2.8') and c else 'NEED')" 2^>^&1') do (
    if "%%V"=="NEED" set "NEED_TORCH=1"
)
python -c "import torch" >nul 2>&1
if errorlevel 1 set "NEED_TORCH=1"

if "!NEED_TORCH!"=="1" (
    echo   Installing PyTorch 2.8 with CUDA 12.6 ^(required by whisperx^)...
    pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126 --force-reinstall --quiet
    if errorlevel 1 (
        echo   [WARN] CUDA PyTorch install failed, trying default...
        pip install torch==2.8.0 torchaudio==2.8.0 --quiet
    )
)

:: Install the package in editable mode
:: Stderr is captured because pyannote emits a harmless torchcodec wall of text
pip install -e "%~dp0." --quiet >nul 2>"%~dp0.pip-stderr.tmp"
if errorlevel 1 (
    echo.
    echo [ERROR] Dependency installation failed:
    type "%~dp0.pip-stderr.tmp"
    del "%~dp0.pip-stderr.tmp" >nul 2>&1
    pause
    exit /b 1
)
del "%~dp0.pip-stderr.tmp" >nul 2>&1
echo   Dependencies installed.

:skip_install

:: ---- GPU check ----
echo.
for /f "delims=" %%G in ('python -c "import torch; p=torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None; print(str(p.name)+' ('+str(round(p.total_memory/1073741824,1))+' GB)' if p else 'None (CPU mode)')" 2^>^&1') do (
    echo   GPU: %%G
)

:: ---- Ensure transcripts directory exists ----
if not exist "%~dp0transcripts" mkdir "%~dp0transcripts"

:: ---- Launch ----
echo.
echo ============================================
echo   Launching TTRPG Listen...
echo   Close the window or press Ctrl+C to stop.
echo ============================================
echo.

:: PID file is written by the app itself and cleaned up via atexit
python -m ttrpglisten
set "EXIT_CODE=%errorlevel%"

if %EXIT_CODE% neq 0 (
    echo.
    echo   TTRPG Listen exited with code %EXIT_CODE%.
    pause
)

endlocal
