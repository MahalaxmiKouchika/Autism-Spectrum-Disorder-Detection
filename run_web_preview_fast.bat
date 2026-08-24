@echo off
echo =======================================================
echo STARTING STREAMLIT WEB APP (FAST DEMO)
echo =======================================================

echo.
echo Installing lightweight UI libraries only...
pip install streamlit==1.11.1 Pillow

echo.
echo Launching the web app...
python -m streamlit run app_fast.py
pause
