@echo off
echo =======================================================
echo RUNNING AUTISM SPECTRUM PREDICTION - 3D FULL PROJECT
echo =======================================================

echo.
echo Installing requirements if necessary...
pip install -r requirements.txt

echo.
echo -------------------------------------------------------
echo 1. Downloading ABIDE 3D fMRI data...
echo -------------------------------------------------------
python src\download_abide.py

echo.
echo -------------------------------------------------------
echo 2. Preprocessing data...
echo -------------------------------------------------------
python src\preprocessing.py

echo.
echo -------------------------------------------------------
echo 3. Training the 3D CNN Pipeline...
echo -------------------------------------------------------
python src\train_cnn.py

echo.
echo Done!
pause
