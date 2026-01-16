@echo off
echo ================================================
echo Domain-Specific Summarization Research System
echo Starting Backend Server
echo ================================================
echo.

cd backend

echo Activating virtual environment...
call .venv\Scripts\activate

echo.
echo Starting FastAPI server...
echo Backend will be available at: http://localhost:8000
echo API documentation at: http://localhost:8000/docs
echo.

python main.py

pause
