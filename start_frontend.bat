@echo off
echo ================================================
echo Domain-Specific Summarization Research System
echo Starting Frontend Development Server
echo ================================================
echo.

cd frontend

echo Installing dependencies (if needed)...
call npm install

echo.
echo Starting Next.js development server...
echo Frontend will be available at: http://localhost:3000
echo.

call npm run dev

pause
