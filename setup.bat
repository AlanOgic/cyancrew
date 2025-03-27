@echo off
:: Setup script for the Product Research and Sales Guide CrewAI Project
:: For Windows systems

:: Print header
echo.
echo ==================================================
echo Product Research and Sales Guide CrewAI Project Setup
echo ==================================================
echo.

:: Check Python version
echo Checking Python version...
python --version 2>nul | find "Python 3." > nul
if errorlevel 1 (
    echo Error: Python 3.8+ is required.
    echo Python not found or version too old.
    exit /b 1
)

:: Extract Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set python_version=%%i
for /f "tokens=2 delims=." %%i in ("%python_version%") do set minor_version=%%i
if %minor_version% LSS 8 (
    echo Error: Python 3.8+ is required.
    echo Current Python version: %python_version%
    exit /b 1
) else (
    echo Python version check passed: %python_version%
)

:: Create virtual environment
echo.
echo Creating virtual environment...
if exist venv (
    echo Virtual environment already exists.
) else (
    python -m venv venv
    if errorlevel 1 (
        echo Error creating virtual environment.
        exit /b 1
    ) else (
        echo Virtual environment created successfully.
    )
)

:: Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate
if errorlevel 1 (
    echo Error activating virtual environment.
    exit /b 1
) else (
    echo Virtual environment activated.
)

:: Install dependencies
echo.
echo Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error installing dependencies.
    exit /b 1
) else (
    echo Dependencies installed successfully.
)

:: Print success message
echo.
echo ==================================================
echo Setup completed successfully!
echo ==================================================
echo.

:: Print instructions
echo To use the project:
echo 1. Activate the virtual environment (if not already activated):
echo    venv\Scripts\activate
echo 2. Run the script:
echo    python crew-1.py
echo 3. When finished, deactivate the virtual environment:
echo    deactivate
echo.

:: Deactivate virtual environment
deactivate
