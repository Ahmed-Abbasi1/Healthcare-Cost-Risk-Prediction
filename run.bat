@echo off
echo ========================================
echo Healthcare Cost Prediction System
echo ========================================
echo.

REM Activate virtual environment
echo [1/2] Activating virtual environment...
call venv\Scripts\activate.bat

REM Start FastAPI server
echo [2/2] Starting web application...
echo.
echo Opening in browser: http://127.0.0.1:8000
echo Press CTRL+C to stop the server
echo.

cd app
start http://127.0.0.1:8000
uvicorn api:app --host 127.0.0.1 --port 8000
