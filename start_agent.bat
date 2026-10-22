@echo off
chcp 65001 >nul
REM Keep this CMD file ASCII with CRLF line endings for reliable Windows parsing.
REM Check the old Agent, activate Conda, and launch Gradio.
REM Carets join the PowerShell command; never insert comments between continued lines.

title Local Personal Agent


echo ========================================
echo       Starting Local Personal Agent
echo ========================================


echo.
echo [1/4] Checking existing Agent process...


REM Stop only the app.py process listening on 7860, never all Python processes.
REM Refuse to stop unrelated programs using the port.
REM Running this launcher restarts an existing Agent; this is not a read-only check.
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$connection = @(Get-NetTCPConnection -LocalPort 7860 -State Listen -ErrorAction SilentlyContinue)[0]; " ^
  "if ($connection) { " ^
  "  $process = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $connection.OwningProcess); " ^
  "  if ($process.CommandLine -notmatch 'app\.py') { " ^
  "    Write-Host ('Port 7860 is occupied by another process: ' + $process.CommandLine); exit 2 " ^
  "  }; " ^
  "  Write-Host ('Stopping existing Local Personal Agent PID ' + $connection.OwningProcess + '...'); " ^
  "  $null = & taskkill.exe /PID $connection.OwningProcess /T /F; " ^
  "  $deadline = (Get-Date).AddSeconds(10); " ^
  "  do { Start-Sleep -Milliseconds 250; $remaining = Get-NetTCPConnection -LocalPort 7860 -State Listen -ErrorAction SilentlyContinue } " ^
  "  while ($remaining -and (Get-Date) -lt $deadline); " ^
  "  if ($remaining) { Write-Host 'Port 7860 was not released.'; exit 3 } " ^
  "}"

REM Stop on preparation failure instead of launching with an occupied port.
if errorlevel 1 (
    echo Unable to prepare port 7860 safely.
    pause
    exit /b 1
)


echo.
echo [2/4] Activating Conda environment...


REM CALL returns from activation while retaining the environment variables.
call C:\ProgramData\anaconda3\Scripts\activate.bat qwen_agent


REM Do not continue after activation failure.
if errorlevel 1 (
    echo Failed to activate conda environment.
    pause
    exit /b
)


echo.
echo [3/4] Switching project directory...


REM Switch both drive and directory so relative application paths resolve correctly.
cd /d D:\46-Agent


REM Do not continue if the project directory is unavailable.
if errorlevel 1 (
    echo Project directory not found.
    pause
    exit /b
)


echo.
echo [4/4] Starting Agent...


REM Run in the foreground; show the stopped message only after the app exits.
python app.py


echo.
echo ========================================
echo       Agent stopped
echo ========================================


pause
