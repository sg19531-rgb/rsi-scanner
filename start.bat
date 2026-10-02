@echo off
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Starting RSI Scanner...
echo Open your browser at: http://localhost:5000
echo Press Ctrl+C to stop.
echo.
python app.py
pause
