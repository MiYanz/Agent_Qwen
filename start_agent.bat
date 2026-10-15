@echo off
chcp 65001 >nul

title Local Personal Agent


echo ========================================
echo       Starting Local Personal Agent
echo ========================================


echo.
echo [1/4] Checking existing Agent process...


REM 只停止占用 7860 且命令行为 app.py 的旧 Agent。
REM 不再使用 taskkill /IM python.exe，以免终止其他 Python 工作。
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

if errorlevel 1 (
    echo Unable to prepare port 7860 safely.
    pause
    exit /b 1
)


echo.
echo [2/4] Activating Conda environment...


call C:\ProgramData\anaconda3\Scripts\activate.bat qwen_agent


if errorlevel 1 (
    echo Failed to activate conda environment.
    pause
    exit /b
)


echo.
echo [3/4] Switching project directory...


cd /d D:\46-Agent


if errorlevel 1 (
    echo Project directory not found.
    pause
    exit /b
)


echo.
echo [4/4] Starting Agent...


python app.py


echo.
echo ========================================
echo       Agent stopped
echo ========================================


pause
