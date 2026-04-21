@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM Universal quick-start script for this project (Windows)
REM - Starts FastAPI backend
REM - Starts Vue frontend
REM - Opens browser automatically
REM ============================================================

set ROOT_DIR=%~dp0
cd /d "%ROOT_DIR%"

set BACKEND_PORT=8000
set FRONTEND_URL=http://127.0.0.1:5173

REM Resolve Python command robustly (PyCharm may show invalid SDK)
set PY_CMD=
where python >nul 2>nul && set PY_CMD=python
if "%PY_CMD%"=="" (
  where py >nul 2>nul && set PY_CMD=py -3
)
if "%PY_CMD%"=="" (
  if exist "%ROOT_DIR%.venv\Scripts\python.exe" set PY_CMD="%ROOT_DIR%.venv\Scripts\python.exe"
)
if "%PY_CMD%"=="" (
  if exist "%ROOT_DIR%venv\Scripts\python.exe" set PY_CMD="%ROOT_DIR%venv\Scripts\python.exe"
)

if "%PY_CMD%"=="" (
  echo [ERROR] Python not found. Please install Python 3.10+ or create .venv in project root.
  echo [TIP] In PyCharm: Settings -^> Project -^> Python Interpreter, select a valid interpreter first.
  pause
  exit /b 1
)

echo Using Python command: %PY_CMD%

where npm >nul 2>nul
if errorlevel 1 (
  echo [ERROR] npm not found in PATH.
  echo Please install Node.js (includes npm) and add it to PATH.
  pause
  exit /b 1
)

echo [1/4] Preparing backend dependencies...
call %PY_CMD% -m pip install -r backend\requirements.txt
if errorlevel 1 (
  echo [ERROR] Failed to install backend requirements.
  pause
  exit /b 1
)

echo [2/4] Preparing frontend dependencies...
if not exist "frontend\node_modules" (
  cd /d "%ROOT_DIR%frontend"
  call npm install
  if errorlevel 1 (
    echo [ERROR] npm install failed.
    pause
    exit /b 1
  )
  cd /d "%ROOT_DIR%"
) else (
  echo frontend\node_modules exists, skip npm install.
)

echo [3/4] Starting backend on port %BACKEND_PORT%...
start "heat-backend" cmd /k "cd /d "%ROOT_DIR%" && call %PY_CMD% -m uvicorn backend.app:app --reload --port %BACKEND_PORT%"

echo [4/4] Starting frontend dev server...
start "heat-frontend" cmd /k "cd /d "%ROOT_DIR%frontend" && npm run dev"

echo Waiting for services to start...
timeout /t 6 /nobreak >nul

echo Opening browser: %FRONTEND_URL%
start "" "%FRONTEND_URL%"

echo Done. Keep the two terminal windows running while previewing.
echo To stop: close the backend and frontend terminal windows or run stop_all.bat.
endlocal
