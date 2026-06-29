@echo off
echo ==========================================
echo  Seq2Seq French-to-English Translator
echo ==========================================
echo.

echo Starting Backend Server (FastAPI)...
start "Backend" cmd /k "cd backend && python main.py"

echo Waiting for backend to start...
timeout /t 3 /nobreak > nul

echo Starting Frontend (Next.js)...
start "Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo ==========================================
echo  Backend:  http://localhost:8000
echo  Frontend: http://localhost:3000
echo  API Docs: http://localhost:8000/docs
echo ==========================================
echo.
echo Press any key to stop both servers...
pause > nul
taskkill /FI "WINDOWTITLE eq Backend*" /F > nul 2>&1
taskkill /FI "WINDOWTITLE eq Frontend*" /F > nul 2>&1
