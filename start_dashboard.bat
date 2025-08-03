@echo off
echo Starting Accident Detection MLOps Dashboard...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Install requirements if needed
echo Installing requirements...
pip install -r requirements_streamlit.txt

REM Start FastAPI server in background
echo Starting FastAPI server...
start /B python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

REM Wait a moment for API to start
timeout /t 3 /nobreak >nul

REM Start Streamlit dashboard
echo Starting Streamlit dashboard...
streamlit run streamlit_app.py --server.port 8501

echo.
echo Both applications are now running:
echo - FastAPI: http://localhost:8000
echo - Streamlit: http://localhost:8501
echo.
pause
