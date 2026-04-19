@echo off
setlocal

REM ============================================================
REM Stop backend/frontend dev services started by start_all.bat
REM - Try kill by window title first
REM - Fallback: kill process occupying common dev ports
REM ============================================================

echo Stopping windows started by start_all.bat...

taskkill /FI "WINDOWTITLE eq heat-backend*" /T /F >nul 2>nul
taskkill /FI "WINDOWTITLE eq heat-frontend*" /T /F >nul 2>nul

set BACKEND_PORT=8000
set FRONTEND_PORT=5173

echo Checking port %BACKEND_PORT%...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%BACKEND_PORT% ^| findstr LISTENING') do (
  echo Killing PID %%a on port %BACKEND_PORT%
  taskkill /PID %%a /T /F >nul 2>nul
)

echo Checking port %FRONTEND_PORT%...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :%FRONTEND_PORT% ^| findstr LISTENING') do (
  echo Killing PID %%a on port %FRONTEND_PORT%
  taskkill /PID %%a /T /F >nul 2>nul
)

echo Done. If any window remains, close it manually.
endlocal
