@echo off
echo Starting Mechanical AI Assistant (Combined Service)...

start cmd /k "echo Starting Backend ^& Frontend... && cd backend && python -m uvicorn src.main:app --reload --port 8000"

echo Service is starting.
echo Access the application at: http://localhost:8000
