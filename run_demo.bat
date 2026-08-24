@echo off
echo =======================================================
echo RUNNING AUTISM SPECTRUM PREDICTION - 2D DEMO
echo =======================================================

echo.
echo Installing requirements if necessary...
pip install -r requirements.txt

echo.
echo -------------------------------------------------------
echo 1. Training the 2D CNN Prototype...
echo -------------------------------------------------------
python train.py

echo.
echo -------------------------------------------------------
echo 2. Running Machine Learning Pipeline...
echo -------------------------------------------------------
python ml_pipeline.py

echo.
echo Done!
pause
