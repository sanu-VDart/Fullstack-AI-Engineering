@echo off
echo Starting Mechanical AI Assistant...

start cmd /k "echo Starting Backend... && cd backend && python -m uvicorn src.main:app --reload --port 8000"
start cmd /k "echo Starting Frontend... && cd frontend && npm run dev"

echo Services are starting in separate windows.
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
