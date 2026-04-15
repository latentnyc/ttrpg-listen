@echo off
setlocal enabledelayedexpansion
title TTRPG Listen - Shutdown

echo ============================================
echo   TTRPG Listen - Shutting down...
echo ============================================
echo.

:: ---- Strategy 1: Use saved PID ----
set "PID_FILE=%~dp0.pid"
if exist "%PID_FILE%" (
    set /p APP_PID=<"%PID_FILE%"
    echo   Found PID file: !APP_PID!

    tasklist /fi "PID eq !APP_PID!" 2>nul | findstr /i "python" >nul
    if not errorlevel 1 (
        echo   Sending graceful shutdown to PID !APP_PID!...
        taskkill /PID !APP_PID! >nul 2>&1

        for /l %%i in (1,1,15) do (
            timeout /t 1 /nobreak >nul
            tasklist /fi "PID eq !APP_PID!" 2>nul | findstr /i "python" >nul
            if errorlevel 1 (
                echo   Process exited cleanly.
                del "%PID_FILE%" >nul 2>&1
                goto :done
            )
        )

        echo   [WARN] Process did not exit in 15s, forcing...
        taskkill /F /PID !APP_PID! >nul 2>&1
        del "%PID_FILE%" >nul 2>&1
        echo   Forced shutdown complete.
        goto :done
    ) else (
        echo   PID !APP_PID! is no longer running.
        del "%PID_FILE%" >nul 2>&1
    )
)

:: ---- Strategy 2: Find by window title ----
echo   Searching for TTRPG Listen processes...

set "FOUND=0"
for /f "tokens=2" %%p in ('tasklist /fi "WINDOWTITLE eq TTRPG Listen*" /fo list 2^>nul ^| findstr /i "PID:"') do (
    echo   Found process: PID %%p
    taskkill /PID %%p >nul 2>&1
    set "FOUND=1"
)

if "!FOUND!"=="1" (
    timeout /t 5 /nobreak >nul
    echo   Shutdown signal sent.
    goto :done
)

:: ---- Strategy 3: Find any python running ttrpglisten ----
for /f "tokens=2 delims=," %%p in ('wmic process where "commandline like '%%ttrpglisten%%'" get processid /format:csv 2^>nul ^| findstr /r "[0-9]"') do (
    if "%%p" neq "" (
        echo   Found ttrpglisten process: PID %%p
        taskkill /PID %%p >nul 2>&1
        set "FOUND=1"
    )
)

if "!FOUND!"=="1" (
    timeout /t 5 /nobreak >nul
    echo   Shutdown signal sent.
    goto :done
)

echo   No running TTRPG Listen process found.

:done
echo.
echo   Shutdown complete.
echo.

endlocal
