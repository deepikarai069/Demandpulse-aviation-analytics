@echo off
REM One-time setup on Windows: virtual env + dependencies + full pipeline
python -m venv .venv || goto :err
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt || goto :err
python run_pipeline.py || goto :err
echo.
echo Setup complete. Start the dashboard with run_dashboard.bat
pause
exit /b 0
:err
echo Something went wrong - see the messages above.
pause
exit /b 1
