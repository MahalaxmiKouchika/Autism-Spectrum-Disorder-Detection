@echo off
echo =======================================================
echo STARTING STREAMLIT WEB APP
echo =======================================================

echo.
echo Installing all required libraries...
pip install -r requirements.txt

echo.
echo Launching the web app...
python -m streamlit run app.py
pause
