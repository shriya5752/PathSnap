@echo off
cd /d "%~dp0"
python -B demo.py
if errorlevel 1 goto finish
echo.
echo API documentation: http://127.0.0.1:8000/docs
echo Press Ctrl+C to stop the server.
python -B -m uvicorn main:app --host 127.0.0.1 --port 8000
:finish
pause
