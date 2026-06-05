@echo off
REM Optional Python tests — saves results under build\test_out\
REM Not required to compile or run the C++ demo.
cd /d "%~dp0\.."

python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found. Install Python 3 and opencv-python, or skip this script.
    exit /b 1
)

echo Running detection tests...
python scripts\test_construction.py
python scripts\test_warning.py
python scripts\test_service.py
python scripts\test_regulatory.py
echo Done. Check build\test_out\
